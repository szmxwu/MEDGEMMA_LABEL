"""
MedGemma X-ray 标注系统 - 本地GPU推理版 (多卡数据并行, 多模型支持)
支持 MedGemma / HuluMed / MedDR2(InternVL) 三种模型后端
torchrun多卡数据并行加速，支持batch推理

用法 (conda env: /mnt/hpfs/hkust/shebd/conda_env/torch26):
    # 单卡测试 (MedGemma, 默认)
    python LLM_label_local.py --limit 10

    # 双卡并行
    torchrun --nproc_per_node=2 LLM_label_local.py --limit 10

    # 指定模型类型 + batch推理
    torchrun --nproc_per_node=2 LLM_label_local.py --model-type meddr2 --checkpoint /path/to/meddr2 --batch-size 4
    torchrun --nproc_per_node=2 LLM_label_local.py --model-type hulu --checkpoint /path/to/hulu

    # 断点续跑 (自动从checkpoint恢复)
    torchrun --nproc_per_node=2 LLM_label_local.py --resume

    # 仅启动复核
    python LLM_label_local.py --review-only

    # 修复乳房标签 (单卡，无需torchrun)
    python LLM_label_local.py --fix-breast --dry-run          # 预览
    python LLM_label_local.py --fix-breast                    # 实际修复
    python LLM_label_local.py --fix-breast --input processed_labels_v3.xlsx --output fixed.xlsx
"""
from __future__ import annotations

import os

# 使用4号卡和5号卡
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5")
import sys
import glob
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Heavy imports deferred so --help / --review-only work without GPU
torch = None
Image = None


def _ensure_gpu_imports():
    """延迟导入GPU依赖 (torch + PIL)"""
    global torch, Image
    if torch is None:
        import torch as _torch
        torch = _torch
    if Image is None:
        from PIL import Image as _Img
        Image = _Img


# Import stateless projection_matcher module
from projection_matcher import (
    MatchResult, STANDARD_VIEWS, BREAST_VIEWS, is_breast_body_part,
    parse_scores_from_response, solve_assignment_with_confidence,
    calculate_confidence,
)

# ================= 配置区 =================
DEFAULT_CHECKPOINT = "/mnt/hpfs/hkust/shebd/0_Pretrained/medgemma-27b-it/"
IMAGE_ROOT = "/mnt/hpfs/hkust/shebd/MingxiangWU/data"
CONFIG_FILE = "part_exam_orientation.json"
EXCEL_FILE = "selected_samples.xlsx"
OUTPUT_FILE = "processed_labels_v3.xlsx"
CHECKPOINT_FILE = "checkpoint_local.jsonl"
CONFIDENCE_THRESHOLD = 0.6
DEFAULT_BATCH_SIZE = 1  # 默认逐条，可通过 --batch-size 提高

# ================= 日志配置 =================
os.makedirs("logs", exist_ok=True)


def setup_logging(rank: int = 0):
    log_file = f"logs/local_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}.log"
    handlers = [logging.FileHandler(log_file, encoding='utf-8')]
    if rank == 0:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [R{rank}] %(levelname)s - %(message)s',
        handlers=handlers,
    )
    return logging.getLogger(__name__)


# ================= 中英文映射 (from LLM_lable.py) =================
CN_EN_MAP = {
    "左": "left", "右": "right", "双": "bilateral",
    "头尾位": "cephalocaudal", "定点压迫位": "spot compression", "腋尾位": "mediolateral oblique",
    "正位": "frontal", "立位": "frontal", "卧位": "frontal", "柯氏位": "frontal", "瓦氏位": "frontal",
    "开口位": "frontal", "闭口位": "frontal", "穿胸位": "frontal", "后前位": "frontal", "仰卧位": "frontal",
    "左右弯曲正位": "frontal", "左右侧屈正位": "frontal", "俯卧位": "frontal", "冠状位": "frontal",
    "负重正位": "frontal", "侧卧水平正位": "frontal", "舟骨位": "frontal", "穴位": "frontal",
    "蝶位": "frontal", "蛙氏位": "frontal", "蝶式位": "frontal",
    "侧位": "lateral", "动力位": "lateral", "过伸位": "lateral", "过屈位": "lateral",
    "左侧位": "lateral", "右侧位": "lateral", "前弓位": "lateral", "双侧位": "lateral",
    "负重侧位": "lateral", "仰卧水平侧位": "lateral",
    "斜位": "oblique", "Y位": "oblique", "侧斜位": "oblique", "切线位": "oblique",
    "双斜位": "oblique", "后斜位": "oblique", "闭孔斜位": "oblique",
    "轴位": "axial", "侧轴位": "axial,lateral",
    "特殊": "special",
    "骸顶位": "special", "华氏位": "special", "梅氏位": "special", "汤氏位": "special",
    "Broden位": "special", "颅底位": "special", "颧弓位": "special", "许氏位": "special",
    "斯氏位": "special", "瑞氏位": "special", "梅伦氏位": "special", "劳式位": "special",
    "薄骨位": "special", "尺偏位": "special", "出入口位": "special", "劳梅氏位": "special",
    "颈椎": "Cervical Spine", "胸椎": "Thoracic Spine", "腰椎": "Lumbar Spine",
    "骶尾椎": "Sacrum/Coccyx", "脊柱": "Spine", "肩部": "Shoulder", "肘关节": "Elbow",
    "腕关节": "Wrist", "手": "Hand", "上臂": "Upper Arm", "前臂": "Forearm",
    "盆部": "Pelvis", "骶髂关节": "Sacroiliac Joint", "大腿": "Thigh", "膝关节": "Knee",
    "小腿": "Calf", "踝关节": "Ankle", "跟骨": "Calcaneus", "足部": "Foot",
    "下肢": "Lower Limb", "胸部": "Chest", "腹部": "Abdomen", "乳房": "Breast",
    "颅脑": "Skull", "颜面": "Facial Bones",
}

SUB_PART_OPTIONS = {
    "脊柱": ["Cervical Spine", "Thoracic Spine", "Lumbar Spine", "Full Spine"],
    "下肢": ["Thigh", "Calf", "Full Lower Limb"],
}


# ================= 工具函数 =================

def parse_medgemma_response(response: str, num_images: int) -> List[Optional[str]]:
    results = [None] * num_images
    if not response:
        return results
    lines = response.strip().split('\n')
    for line in lines:
        for i in range(num_images):
            prefix = f"Image {i+1}:"
            if prefix in line:
                val = line.split(prefix)[1].strip()
                val = val.rstrip('.').rstrip('。')
                results[i] = val
    return results


def normalize_projection_label(label: str) -> str:
    if not label:
        return "unknown"
    label = label.lower().replace(" position", "").strip()
    for view in ["frontal", "lateral", "oblique", "axial", "special"]:
        if view in label:
            return view
    return label


def _get_review_reason(proj_conf: float, ori_conf: float, method: str) -> str:
    reasons = []
    if proj_conf < CONFIDENCE_THRESHOLD:
        reasons.append(f"low_projection_confidence({proj_conf:.2f})")
    if ori_conf < CONFIDENCE_THRESHOLD:
        reasons.append(f"low_orientation_confidence({ori_conf:.2f})")
    if method == "free_classification":
        reasons.append("no_excel_reference")
    if method == "greedy_excess":
        reasons.append("excess_image_unmatched")
    return "; ".join(reasons) if reasons else ""


def load_images(paths: List[str]) -> list:
    _ensure_gpu_imports()
    logger = logging.getLogger(__name__)
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            # 极端尺寸(如1xN)会导致Gemma3图像处理器通道歧义崩溃
            if w < 10 or h < 10:
                logger.warning(f"图片尺寸异常 {p}: {w}x{h}, 使用占位图")
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            images.append(img)
        except Exception as e:
            logger.error(f"加载图片失败 {p}: {e}")
            images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
    return images


def build_prompt_text(num_images: int, question: str,
                      options: Optional[List[str]] = None) -> str:
    """构建文本prompt (不含图像标记，由各backend自行添加)"""
    prompt = f"\n{question}\n"
    if options:
        prompt += "Please choose STRICTLY from the following options for each image:\n"
        prompt += f"{json.dumps(options, ensure_ascii=False)}\n"
    prompt += "\nAnswer format strictly as:\n"
    for idx in range(num_images):
        prompt += f"Image {idx+1}: [Your Choice]\n"
    return prompt


# ================= 分布式工具 =================

class InferenceSampler:
    """数据分片采样器 (from mmrag/src/eval/utils.py)"""

    def __init__(self, size: int):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def get_rank() -> int:
    _ensure_gpu_imports()
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    _ensure_gpu_imports()
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


# ================= Checkpoint (JSONL) =================

def _read_jsonl(path: str) -> List[dict]:
    """读取单个JSONL文件的所有记录"""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_checkpoint(path: str) -> Tuple[set, List[dict]]:
    """加载checkpoint，读取主文件 + 所有per-rank文件。
    Returns:
        (done_image_ids, all_records)
        done_image_ids: 已完成的image_id集合 (用于过滤)
        all_records: 所有记录行 (用于重建DataFrame，一个image_id可能有多行)
    """
    all_records = []
    # 读取主checkpoint
    all_records.extend(_read_jsonl(path))
    # 读取所有per-rank checkpoint
    base = path.replace('.jsonl', '')
    for rank_file in sorted(glob.glob(f"{base}_rank*.jsonl")):
        all_records.extend(_read_jsonl(rank_file))

    # 按 (image_id, filename) 去重 (防止主文件和rank文件重复)
    seen = set()
    deduped = []
    for rec in all_records:
        key = (rec.get("image_id", ""), rec.get("filename", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(rec)

    done_ids = {rec["image_id"] for rec in deduped if "image_id" in rec}
    return done_ids, deduped


def append_checkpoint(path: str, records: List[dict]):
    with open(path, 'a', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        f.flush()


# ================= 数据结构 =================

@dataclass
class SampleData:
    index: int
    image_id: str
    std_part: str
    excel_pos: Optional[str]
    excel_proj: Optional[str]
    img_paths: List[str]
    num_imgs: int = 0
    breast_matches: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    needs_task_a: bool = False
    task_b_mode: str = ""
    task_c_branch: str = ""
    expected_labels_en: List[str] = field(default_factory=list)
    allowed_projections: List[str] = field(default_factory=list)
    final_parts: List[str] = field(default_factory=list)
    final_orientations: List[Optional[str]] = field(default_factory=list)
    orientation_confidences: List[float] = field(default_factory=list)
    final_projections: List[Optional[str]] = field(default_factory=list)
    projection_confidences: List[float] = field(default_factory=list)
    projection_needs_review: List[bool] = field(default_factory=list)
    projection_raw_scores: List[Dict] = field(default_factory=list)
    match_methods: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ================= InternVL图像预处理工具 (for MedDR2) =================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _build_internvl_transform(input_size):
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


# ================= 推理后端基类 =================

class ModelBackend:
    """推理后端基类。
    统一接口: infer_single(img_paths, question, options) -> str
              infer_batch(items, desc) -> List[str]
    items = List[(img_paths, question, options)]
    """

    def __init__(self, checkpoint: str, batch_size: int = 1):
        self.batch_size = batch_size

    def infer_single(self, img_paths: List[str], question: str,
                     options: Optional[List[str]] = None) -> str:
        raise NotImplementedError

    def _process_chunk(self, chunk: list) -> List[str]:
        """处理一个batch chunk，默认逐条"""
        return [self.infer_single(paths, q, opts) for paths, q, opts in chunk]

    def infer_batch(self, items: list, desc: str = "Inference") -> List[str]:
        """批量推理，按 batch_size 分chunk，带进度条和OOM回退"""
        if not items:
            return []
        bs = self.batch_size
        results = []

        chunks = [items[i:i + bs] for i in range(0, len(items), bs)]
        iterator = chunks
        if is_main_process() and len(chunks) > 1:
            from tqdm import tqdm
            iterator = tqdm(chunks, desc=desc, unit="batch", dynamic_ncols=True)

        for chunk in iterator:
            try:
                chunk_results = self._process_chunk(chunk)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    logging.getLogger(__name__).warning(
                        f"OOM (batch={len(chunk)}), 回退到逐条推理")
                else:
                    logging.getLogger(__name__).warning(
                        f"Batch RuntimeError: {e}, 回退到逐条推理")
                chunk_results = self._sequential_fallback(chunk)
            except Exception as e:
                # 非RuntimeError的batch错误(如图片格式问题)也回退到逐条
                logging.getLogger(__name__).warning(
                    f"Batch异常: {e}, 回退到逐条推理")
                chunk_results = self._sequential_fallback(chunk)
            results.extend(chunk_results)
        return results

    def _sequential_fallback(self, chunk):
        """逐条推理回退"""
        results = []
        for item in chunk:
            try:
                results.append(self.infer_single(*item))
            except Exception as e:
                logging.getLogger(__name__).error(f"逐条推理也失败: {e}")
                results.append("")
        return results


# ================= MedGemma 后端 =================

class MedGemmaBackend(ModelBackend):
    """MedGemma-27B-IT 推理后端 (参考 eval_251120.py)"""

    def __init__(self, checkpoint: str, batch_size: int = 1):
        super().__init__(checkpoint, batch_size)
        _ensure_gpu_imports()
        from transformers import AutoProcessor, AutoModelForImageTextToText
        logger = logging.getLogger(__name__)
        logger.info(f"[MedGemma] 加载模型: {checkpoint}")

        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16,
        ).eval().cuda()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info(f"[MedGemma] 加载完成, device={self.model.device}")

    def _build_messages(self, pil_images, question, options):
        num_images = len(pil_images)
        user_content = [{"type": "text", "text": "Here are the X-ray images for analysis:\n"}]
        for idx, img in enumerate(pil_images):
            user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": f"\n[Image {idx+1} Above]\n"})
        user_content.append({"type": "text", "text": build_prompt_text(num_images, question, options)})
        return [{"role": "user", "content": user_content}]

    def infer_single(self, img_paths, question, options=None):
        pil_images = load_images(img_paths)
        messages = self._build_messages(pil_images, question, options)
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                gen = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                gen = gen[0][input_len:]
            return self.processor.decode(gen, skip_special_tokens=True).strip()
        except Exception as e:
            logging.getLogger(__name__).error(f"[MedGemma] 推理失败: {e}")
            return ""

    def _process_chunk(self, chunk):
        if len(chunk) == 1:
            return [self.infer_single(*chunk[0])]
        # 批量: apply_chat_template(tokenize=False) → processor(text, images, padding) → generate
        texts = []
        all_images = []
        for img_paths, question, options in chunk:
            pil_images = load_images(img_paths)
            messages = self._build_messages(pil_images, question, options)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            all_images.append(pil_images)

        inputs = self.processor(
            text=texts, images=all_images,
            return_tensors="pt", padding=True,
        ).to(self.model.device)
        # 确保bfloat16
        if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=512, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        results = []
        for i in range(len(chunk)):
            resp = self.processor.decode(output_ids[i][input_len:], skip_special_tokens=True)
            results.append(resp.strip())
        return results


# ================= HuluMed 后端 =================

class HuluMedBackend(ModelBackend):
    """HuluMed 推理后端 (参考 eval_251120.py evaluate_hulu)"""

    def __init__(self, checkpoint: str, batch_size: int = 1):
        # HuluMed 不支持原生batch，batch_size强制为1
        super().__init__(checkpoint, batch_size=1)
        _ensure_gpu_imports()
        from transformers import AutoModelForCausalLM, AutoProcessor
        logger = logging.getLogger(__name__)
        logger.info(f"[HuluMed] 加载模型: {checkpoint}")

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        ).eval().cuda()
        for param in self.model.parameters():
            param.requires_grad = False

        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True,
            max_tokens=256 * 2 * 2,
        )
        logger.info(f"[HuluMed] 加载完成, device={self.model.device}")

    def infer_single(self, img_paths, question, options=None):
        num_images = len(img_paths)
        # HuluMed消息格式: image用 {"image_path": path}
        user_content = [{"type": "text", "text": "Here are the X-ray images for analysis:\n"}]
        for idx, path in enumerate(img_paths):
            user_content.append({"type": "image", "image": {"image_path": path}})
            user_content.append({"type": "text", "text": f"\n[Image {idx+1} Above]\n"})
        user_content.append({"type": "text", "text": build_prompt_text(num_images, question, options)})

        conversation = [{"role": "user", "content": user_content}]
        try:
            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=512, do_sample=False)

            decoded = self.processor.batch_decode(
                output_ids, skip_special_tokens=True, use_think=False)
            return decoded[0].strip() if decoded else ""
        except Exception as e:
            logging.getLogger(__name__).error(f"[HuluMed] 推理失败: {e}")
            return ""


# ================= MedDR2 / InternVL 后端 =================

class MedDR2Backend(ModelBackend):
    """MedDR2 / InternVL 推理后端 (参考 eval_251120.py evaluate_internvl)
    支持 model.chat() 和 model.batch_chat() 原生batch推理
    """

    def __init__(self, checkpoint: str, batch_size: int = 1, image_size: int = 448):
        super().__init__(checkpoint, batch_size)
        _ensure_gpu_imports()
        from transformers import AutoModel, AutoTokenizer
        logger = logging.getLogger(__name__)
        logger.info(f"[MedDR2] 加载模型: {checkpoint}")

        self.image_size = image_size
        self.transform = _build_internvl_transform(image_size)

        self.model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().cuda()
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, use_fast=False)
        self.tokenizer.model_max_length = 20490

        # 位置编码插值 (如果需要)
        model_image_size = getattr(self.model.config, 'force_image_size', None) or \
                           getattr(getattr(self.model.config, 'vision_config', None), 'image_size', image_size)
        if model_image_size != image_size:
            logger.info(f"[MedDR2] 位置编码插值: {model_image_size} -> {image_size}")
            self.model.num_image_token = int(
                (image_size // self.model.patch_size) ** 2 * (self.model.downsample_ratio ** 2))
            self.model.vision_model.resize_pos_embeddings(model_image_size, image_size, self.model.patch_size)
            self.model.config.force_image_size = image_size

        self.gen_config = dict(
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.0,
        )
        logger.info(f"[MedDR2] 加载完成, device={self.model.device}")

    def _preprocess_images(self, img_paths):
        """预处理图片 → (pixel_values_tensor, num_patches_list)"""
        all_patches = []
        num_patches_list = []
        for path in img_paths:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            patches = _dynamic_preprocess(img, image_size=self.image_size, use_thumbnail=True)
            pixel_values = [self.transform(p) for p in patches]
            all_patches.extend(pixel_values)
            num_patches_list.append(len(patches))
        pixel_values = torch.stack(all_patches)
        return pixel_values, num_patches_list

    def _build_question(self, num_images, question, options):
        """构建InternVL格式的问题文本 (使用<image>标记)"""
        parts = []
        for idx in range(num_images):
            parts.append(f"<image>\n[Image {idx+1} Above]\n")
        parts.append(build_prompt_text(num_images, question, options))
        return "".join(parts)

    def infer_single(self, img_paths, question, options=None):
        pixel_values, num_patches_list = self._preprocess_images(img_paths)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        q_text = self._build_question(len(img_paths), question, options)
        try:
            response, _ = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=q_text,
                generation_config=self.gen_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
            return response.strip()
        except Exception as e:
            logging.getLogger(__name__).error(f"[MedDR2] 推理失败: {e}")
            return ""

    def _process_chunk(self, chunk):
        if len(chunk) == 1:
            return [self.infer_single(*chunk[0])]
        # batch_chat: 拼接所有pixel_values, 每个样本独立的num_patches_list
        all_pixel_values = []
        all_questions = []
        all_num_patches = []
        for img_paths, question, options in chunk:
            pv, npl = self._preprocess_images(img_paths)
            all_pixel_values.append(pv)
            all_num_patches.append(npl)
            all_questions.append(self._build_question(len(img_paths), question, options))

        cat_pv = torch.cat(all_pixel_values, dim=0).to(torch.bfloat16).cuda()
        responses = self.model.batch_chat(
            tokenizer=self.tokenizer,
            pixel_values=cat_pv,
            questions=all_questions,
            generation_config=self.gen_config,
            num_patches_list=all_num_patches,
            history=None,
            return_history=False,
        )
        return [r.strip() for r in responses]


# ================= 模型工厂 =================

def detect_model_type(checkpoint: str) -> str:
    """从checkpoint路径自动检测模型类型"""
    name = checkpoint.lower().rstrip('/').split('/')[-1]
    if "medgemma" in name:
        return "medgemma"
    if "hulu" in name:
        return "hulu"
    if any(k in name for k in ("meddr", "internvl", "medintern")):
        return "meddr2"
    return "medgemma"  # 默认


def create_model_backend(model_type: str, checkpoint: str,
                         batch_size: int = 1) -> ModelBackend:
    logger = logging.getLogger(__name__)
    logger.info(f"创建推理后端: type={model_type}, batch_size={batch_size}")
    if model_type == "medgemma":
        return MedGemmaBackend(checkpoint, batch_size=batch_size)
    elif model_type == "hulu":
        return HuluMedBackend(checkpoint, batch_size=batch_size)
    elif model_type == "meddr2":
        return MedDR2Backend(checkpoint, batch_size=batch_size)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


# ================= 投影打分 prompt 构建 =================

def build_standard_scoring_prompt(num_imgs: int, body_part: str) -> str:
    return (
        f"Analyze these {num_imgs} {body_part} X-ray images.\n"
        "For EACH image, rate the likelihood (0-10) of it being one of the standard geometric views:\n"
        "- Frontal (AP/PA/Townes/Caldwell)\n"
        "- Lateral (Lat/Dynamic/Flex/Ext)\n"
        "- Oblique (Left/Right/Mortise)\n"
        "- Axial (Skyline/Sunrise)\n\n"
        "Output format strict example:\n"
        "Image 1: Frontal=9, Lateral=1, Oblique=0, Axial=0\n"
        "Image 2: Frontal=2, Lateral=8, Oblique=0, Axial=0\n"
        "..."
    )


def build_breast_scoring_prompt(num_imgs: int) -> str:
    return (
        f"Analyze these {num_imgs} breast mammography images.\n"
        "For EACH image, rate the likelihood (0-10) of it being one of the standard mammographic views:\n"
        "- Cephalocaudal (CC view, top-to-bottom compression)\n"
        "- Mediolateral oblique (MLO view, angled compression)\n"
        "- Spot compression (focused compression view)\n\n"
        "Output format strict example:\n"
        "Image 1: Cephalocaudal=9, Mediolateral=1, Spot=0\n"
        "Image 2: Cephalocaudal=2, Mediolateral=8, Spot=0\n"
        "..."
    )


# ================= Phase 0: 预扫描 =================

def phase0_prescan(df: pd.DataFrame, part_config: Dict) -> List[SampleData]:
    logger = logging.getLogger(__name__)
    if is_main_process():
        logger.info("=" * 60)
        logger.info("Phase 0: 预扫描")
        logger.info("=" * 60)

    samples = []
    skipped = 0

    for idx, row in df.iterrows():
        img_id = str(row['影像号'])
        std_part = row['标准化部位']
        excel_pos = row.get('Position_orientation', None)
        excel_proj = row.get('exam_projection', None)
        if pd.isna(excel_pos):
            excel_pos = None
        if pd.isna(excel_proj):
            excel_proj = None

        img_dir = os.path.join(IMAGE_ROOT, img_id)
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

        if not img_paths:
            skipped += 1
            samples.append(SampleData(
                index=idx, image_id=img_id, std_part=std_part,
                excel_pos=excel_pos, excel_proj=excel_proj,
                img_paths=[], num_imgs=0,
                error=f"No images found in {img_dir}"))
            continue

        num_imgs = len(img_paths)
        s = SampleData(
            index=idx, image_id=img_id, std_part=std_part,
            excel_pos=excel_pos, excel_proj=excel_proj,
            img_paths=img_paths, num_imgs=num_imgs)

        std_part_en = CN_EN_MAP.get(std_part, std_part)
        s.final_parts = [std_part_en] * num_imgs
        s.final_orientations = [None] * num_imgs
        s.orientation_confidences = [1.0] * num_imgs
        s.final_projections = [None] * num_imgs
        s.projection_confidences = [0.0] * num_imgs
        s.projection_needs_review = [True] * num_imgs
        s.projection_raw_scores = [{} for _ in range(num_imgs)]
        s.match_methods = ["unknown"] * num_imgs

        # 乳房文件名快速匹配
        if std_part == "乳房":
            for i, img_path in enumerate(img_paths):
                fn = os.path.basename(img_path).upper()
                if "_L_CC" in fn:
                    s.breast_matches[i] = ("left", "cephalocaudal")
                elif "_R_CC" in fn:
                    s.breast_matches[i] = ("right", "cephalocaudal")
                elif "_L_MLO" in fn:
                    s.breast_matches[i] = ("left", "mediolateral oblique")
                elif "_R_MLO" in fn:
                    s.breast_matches[i] = ("right", "mediolateral oblique")
            for i, (ori, proj) in s.breast_matches.items():
                s.final_orientations[i] = ori
                s.orientation_confidences[i] = 0.95
                s.final_projections[i] = proj
                s.projection_confidences[i] = 0.95
                s.projection_needs_review[i] = False
                s.match_methods[i] = "filename_breast"

        # 路由: Task A
        s.needs_task_a = (std_part in ("脊柱", "下肢") and num_imgs > 2)

        # 路由: Task B
        part_cfg = part_config.get(std_part, {})
        allowed_orientations = part_cfg.get("Position_orientation", [])
        if not allowed_orientations:
            s.task_b_mode = "na"
            s.final_orientations = ["Not Applicable"] * num_imgs
        else:
            mapped_excel_pos = CN_EN_MAP.get(excel_pos, excel_pos) if excel_pos else None
            if mapped_excel_pos in ("left", "right"):
                s.task_b_mode = "excel"
                s.final_orientations = [mapped_excel_pos] * num_imgs
                s.orientation_confidences = [0.95] * num_imgs
            else:
                s.task_b_mode = "llm"
        if std_part == "乳房" and s.task_b_mode == "llm":
            if all(i in s.breast_matches for i in range(num_imgs)):
                s.task_b_mode = "excel"

        # 路由: Task C
        s.allowed_projections = part_cfg.get("exam_projection", [])
        raw_proj_str = str(excel_proj) if excel_proj else ""
        excel_proj_list = [x.strip() for x in raw_proj_str.replace("，", ",").split(",") if x.strip()]
        expected_labels_en = []
        for p in excel_proj_list:
            mapped = CN_EN_MAP.get(p, p)
            if isinstance(mapped, str) and "," in mapped:
                expected_labels_en.extend([ss.strip() for ss in mapped.split(",") if ss.strip()])
            else:
                expected_labels_en.append(mapped)
        s.expected_labels_en = expected_labels_en

        if std_part == "乳房":
            unmatched = [i for i in range(num_imgs) if s.final_projections[i] is None]
            if not unmatched:
                s.task_c_branch = "done"
            elif expected_labels_en:
                s.task_c_branch = "B"
            else:
                s.task_c_branch = "C"
        elif num_imgs == 1 and len(expected_labels_en) == 1:
            s.task_c_branch = "A"
        elif expected_labels_en:
            s.task_c_branch = "B"
        else:
            s.task_c_branch = "C"

        samples.append(s)

    if is_main_process():
        logger.info(f"  扫描完成: {len(samples)} 个样本, {skipped} 个跳过(无图片)")
        ta = sum(1 for s in samples if s.needs_task_a and not s.error)
        tb = sum(1 for s in samples if s.task_b_mode == "llm" and not s.error)
        ba = sum(1 for s in samples if s.task_c_branch == "A")
        bb = sum(1 for s in samples if s.task_c_branch == "B")
        bc = sum(1 for s in samples if s.task_c_branch == "C")
        bd = sum(1 for s in samples if s.task_c_branch == "done")
        logger.info(f"  Task A: {ta}, Task B(LLM): {tb}")
        logger.info(f"  Task C: A={ba}, B={bb}, C={bc}, done={bd}")

    return samples


# ================= Phase-based batch processing =================

def process_phases_batch(
    local_samples: List[SampleData],
    backend: ModelBackend,
    part_config: Dict,
    rank_ckpt_path: str,
) -> List[dict]:
    """处理本rank分到的所有样本 (分阶段批量推理)"""
    logger = logging.getLogger(__name__)
    show = is_main_process()

    # === Branch A fast-path (no LLM) ===
    for s in local_samples:
        if s.task_c_branch == "A" and not s.error:
            label = normalize_projection_label(s.expected_labels_en[0])
            s.final_projections = [label]
            s.projection_confidences = [1.0]
            s.projection_needs_review = [False]
            s.projection_raw_scores = [{}]
            s.match_methods = ["fast_path"]

    # === Phase 1: Task A (部位细分) ===
    task_a_targets = [s for s in local_samples if s.needs_task_a and not s.error]
    if task_a_targets:
        if show:
            logger.info(f"Phase 1: 部位细分 ({len(task_a_targets)} 个样本)")
        items = []
        for s in task_a_targets:
            opts = SUB_PART_OPTIONS[s.std_part]
            q = f"Identify the specific sub-part for each image. Are they {', '.join(opts)}?"
            items.append((s.img_paths, q, opts))
        responses = backend.infer_batch(items, desc="Phase1-TaskA")
        for s, resp in zip(task_a_targets, responses):
            parsed = parse_medgemma_response(resp, s.num_imgs)
            for i, p in enumerate(parsed):
                if p:
                    s.final_parts[i] = p

    # === Phase 2: Task B (方位识别) ===
    task_b_items = []
    task_b_meta = []
    for s in local_samples:
        if s.task_b_mode != "llm" or s.error:
            continue
        part_cfg = part_config.get(s.std_part, {})
        allowed_ori = part_cfg.get("Position_orientation", [])
        need_idx = ([i for i in range(s.num_imgs) if i not in s.breast_matches]
                    if s.std_part == "乳房" else list(range(s.num_imgs)))
        if not need_idx:
            continue
        llm_paths = [s.img_paths[i] for i in need_idx]
        task_b_items.append((llm_paths, "Identify the side (laterality) for each image.", allowed_ori))
        task_b_meta.append((s, need_idx))

    if task_b_items:
        if show:
            logger.info(f"Phase 2: 方位识别 ({len(task_b_items)} 个样本)")
        responses = backend.infer_batch(task_b_items, desc="Phase2-TaskB")
        for (s, indices), resp in zip(task_b_meta, responses):
            parsed = parse_medgemma_response(resp, len(indices))
            for mi, img_idx in enumerate(indices):
                p = parsed[mi]
                if p:
                    s.final_orientations[img_idx] = p
                    s.orientation_confidences[img_idx] = 0.8
                else:
                    s.final_orientations[img_idx] = "unknown"
                    s.orientation_confidences[img_idx] = 0.0

    # === Phase 3: Task C ===
    scoring_items = []
    scoring_meta = []
    classify_items = []
    classify_meta = []

    for s in local_samples:
        if s.error or s.task_c_branch in ("A", "done", ""):
            continue
        is_breast = (s.std_part == "乳房")

        if is_breast:
            target_idx = [i for i in range(s.num_imgs) if s.final_projections[i] is None]
            if not target_idx:
                continue
            target_paths = [s.img_paths[i] for i in target_idx]
            prompt = build_breast_scoring_prompt(len(target_idx))
            scoring_items.append((target_paths, prompt, None))
            scoring_meta.append((s, target_idx, BREAST_VIEWS,
                                 s.expected_labels_en if s.task_c_branch == "B" else [], True))
        elif s.task_c_branch == "B":
            body = s.final_parts[0] if s.final_parts else CN_EN_MAP.get(s.std_part, s.std_part)
            prompt = build_standard_scoring_prompt(s.num_imgs, body)
            scoring_items.append((s.img_paths, prompt, None))
            scoring_meta.append((s, list(range(s.num_imgs)), STANDARD_VIEWS,
                                 s.expected_labels_en, False))
        else:  # Branch C non-breast
            opts = (s.allowed_projections + ["special"]) if s.allowed_projections else \
                   ["frontal", "lateral", "oblique", "axial", "special"]
            classify_items.append((s.img_paths, "Identify the projection view for each image.", opts))
            classify_meta.append((s, list(range(s.num_imgs))))

    # 3a: 打分推理
    if scoring_items:
        if show:
            logger.info(f"Phase 3a: 打分推理 ({len(scoring_items)} 个样本)")
        responses = backend.infer_batch(scoring_items, desc="Phase3a-Score")
        for (s, t_idx, views, labels, is_b), resp in zip(scoring_meta, responses):
            n = len(t_idx)
            scores = parse_scores_from_response(resp, n, valid_views=views)
            if labels:
                paths_m = [s.img_paths[i] for i in t_idx]
                mrs = solve_assignment_with_confidence(
                    scores, labels, paths_m, valid_views=views, is_breast=is_b)
            else:
                mrs = []
                for i in range(n):
                    bi = int(np.argmax(scores[i]))
                    bv = views[bi]
                    cf, lv = calculate_confidence(scores[i], bv, valid_views=views)
                    raw = {views[j]: float(scores[i, j]) for j in range(len(views))}
                    mrs.append(MatchResult(
                        label=bv, confidence=cf,
                        needs_review=(lv in ('low', 'very_low', 'medium')),
                        raw_scores=raw, match_method='greedy'))
            for mi, img_idx in enumerate(t_idx):
                mr = mrs[mi]
                s.final_projections[img_idx] = mr.label if is_b else normalize_projection_label(mr.label)
                s.projection_confidences[img_idx] = mr.confidence
                s.projection_needs_review[img_idx] = mr.needs_review
                s.projection_raw_scores[img_idx] = mr.raw_scores
                s.match_methods[img_idx] = mr.match_method

    # 3b: 分类推理
    if classify_items:
        if show:
            logger.info(f"Phase 3b: 分类推理 ({len(classify_items)} 个样本)")
        responses = backend.infer_batch(classify_items, desc="Phase3b-Classify")
        for (s, t_idx), resp in zip(classify_meta, responses):
            parsed = parse_medgemma_response(resp, len(t_idx))
            for mi, img_idx in enumerate(t_idx):
                p = parsed[mi]
                if p:
                    s.final_projections[img_idx] = normalize_projection_label(p)
                    s.projection_confidences[img_idx] = 0.6
                else:
                    s.final_projections[img_idx] = "unknown"
                    s.projection_confidences[img_idx] = 0.0
                s.projection_needs_review[img_idx] = True
                s.match_methods[img_idx] = "free_classification"

    # 序列化结果
    all_records = []
    for s in local_samples:
        records = sample_to_records(s)
        all_records.extend(records)
    if all_records:
        append_checkpoint(rank_ckpt_path, all_records)

    return all_records


# ================= 结果序列化 =================

def sample_to_records(s: SampleData) -> List[dict]:
    rows = []
    if s.error:
        return rows
    for i in range(s.num_imgs):
        proj_conf = s.projection_confidences[i]
        ori_conf = s.orientation_confidences[i]
        overall_conf = (proj_conf + ori_conf) / 2
        needs_review = s.projection_needs_review[i] or (overall_conf < CONFIDENCE_THRESHOLD)
        method = s.match_methods[i]
        raw_scores = s.projection_raw_scores[i] if i < len(s.projection_raw_scores) else {}
        rows.append({
            "image_id": s.image_id,
            "filename": os.path.basename(s.img_paths[i]),
            "original_part": s.std_part,
            "final_body_part": s.final_parts[i] if i < len(s.final_parts) else "",
            "final_orientation": s.final_orientations[i] if i < len(s.final_orientations) else None,
            "final_projection": s.final_projections[i] if i < len(s.final_projections) else None,
            "confidence_projection": round(proj_conf, 3),
            "confidence_orientation": round(ori_conf, 3),
            "confidence_overall": round(overall_conf, 3),
            "needs_review": needs_review,
            "review_reason": _get_review_reason(proj_conf, ori_conf, method),
            "match_method": method,
            "raw_scores_frontal": raw_scores.get("frontal", 0),
            "raw_scores_lateral": raw_scores.get("lateral", 0),
            "raw_scores_oblique": raw_scores.get("oblique", 0),
            "raw_scores_axial": raw_scores.get("axial", 0),
            "raw_excel_pos": s.excel_pos,
            "raw_excel_proj": s.excel_proj,
        })
    return rows


# ================= 主流程 =================

def run_pipeline(args):
    _ensure_gpu_imports()
    rank = get_rank()
    world_size = get_world_size()
    logger = logging.getLogger(__name__)

    model_type = args.model_type or detect_model_type(args.checkpoint)

    if is_main_process():
        logger.info("=" * 60)
        logger.info("X-ray 标注系统 (本地GPU版 - 多卡数据并行)")
        logger.info(f"模型类型: {model_type}")
        logger.info(f"模型路径: {args.checkpoint}")
        logger.info(f"Batch大小: {args.batch_size}")
        logger.info(f"Rank: {rank}/{world_size}")
        logger.info("=" * 60)

    # 1. 配置
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"配置文件未找到: {CONFIG_FILE}")
        return False
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        part_config = json.load(f)

    # 2. Excel
    if not os.path.exists(EXCEL_FILE):
        logger.error(f"Excel文件未找到: {EXCEL_FILE}")
        return False
    df = pd.read_excel(EXCEL_FILE)
    if args.limit:
        df = df.head(args.limit)
    if is_main_process():
        logger.info(f"待处理样本: {len(df)}")

    # 3. Phase 0
    all_samples = phase0_prescan(df, part_config)

    # 4. Checkpoint
    done_ids = set()
    checkpoint_records = []
    if args.resume:
        done_ids, checkpoint_records = load_checkpoint(CHECKPOINT_FILE)
        if is_main_process():
            logger.info(f"Checkpoint: 已完成 {len(done_ids)} 个样本, {len(checkpoint_records)} 条记录")

    pending_samples = [
        s for s in all_samples
        if not s.error and s.image_id not in done_ids
    ]
    if is_main_process():
        logger.info(f"本次需处理: {len(pending_samples)} 个样本 "
                    f"(跳过: {len(all_samples) - len(pending_samples)})")

    if not pending_samples:
        if is_main_process():
            _save_final_results(all_samples, checkpoint_records)
        return True

    # 5. 加载模型
    backend = create_model_backend(model_type, args.checkpoint, args.batch_size)

    # 6. 数据分片
    if world_size > 1:
        sampler = InferenceSampler(len(pending_samples))
        local_indices = list(sampler)
    else:
        local_indices = list(range(len(pending_samples)))

    local_samples = [pending_samples[i] for i in local_indices]
    logger.info(f"Rank {rank}: 分配到 {len(local_samples)} 个样本")

    # 7. 分阶段批量处理
    start_time = time.time()
    rank_ckpt = CHECKPOINT_FILE.replace('.jsonl', f'_rank{rank}.jsonl')
    local_records = process_phases_batch(local_samples, backend, part_config, rank_ckpt)

    elapsed = time.time() - start_time
    logger.info(f"Rank {rank}: 推理完成, {len(local_records)} 条记录, 耗时 {elapsed:.1f}s")

    # 释放模型显存，为 all_gather_object NCCL通信腾出空间
    del backend
    torch.cuda.empty_cache()

    # 8. 汇总
    if world_size > 1:
        torch.distributed.barrier()
        gathered = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            gathered, json.dumps(local_records, ensure_ascii=False))
        if is_main_process():
            all_records = []
            for g in gathered:
                all_records.extend(json.loads(g))
            logger.info(f"汇总: {len(all_records)} 条新记录来自 {world_size} 个rank")
    else:
        all_records = local_records

    # 9. 保存
    if is_main_process():
        final_records = checkpoint_records + all_records
        res_df = pd.DataFrame(final_records) if final_records else pd.DataFrame()
        if len(res_df) == 0:
            logger.warning("无有效结果")
            return False
        res_df.to_excel(OUTPUT_FILE, index=False)
        logger.info(f"结果已保存: {OUTPUT_FILE} ({len(res_df)} 条记录)")
        append_checkpoint(CHECKPOINT_FILE, all_records)

        if 'confidence_overall' in res_df.columns:
            hc = len(res_df[res_df['confidence_overall'] >= CONFIDENCE_THRESHOLD])
            nr = res_df['needs_review'].sum() if 'needs_review' in res_df.columns else 0
            logger.info(f"高置信度: {hc}/{len(res_df)} ({100*hc/len(res_df):.1f}%)")
            logger.info(f"需审核: {nr}/{len(res_df)} ({100*nr/len(res_df):.1f}%)")
        logger.info(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        logger.info("=" * 60)

    return True


def _save_final_results(all_samples, checkpoint_records):
    logger = logging.getLogger(__name__)
    final_records = list(checkpoint_records)
    if not final_records:
        logger.warning("无有效结果")
        return
    res_df = pd.DataFrame(final_records)
    res_df.to_excel(OUTPUT_FILE, index=False)
    logger.info(f"结果已保存 (从checkpoint): {OUTPUT_FILE} ({len(res_df)} 条记录)")


# ================= 乳房标签修复 (合并自 fix_breast_labels.py) =================

WRONG_BREAST_LABELS = {'frontal', 'lateral', 'oblique', 'axial',
                       'left', 'right', 'bilateral', 'unknown'}


def _find_breast_to_fix(df: pd.DataFrame) -> pd.DataFrame:
    """筛选需要修复的乳房样本: original_part==乳房 且 final_projection 为通用视角或空"""
    breast_df = df[df['original_part'] == '乳房'].copy()
    if len(breast_df) == 0:
        return breast_df
    mask = (
        breast_df['final_projection'].isna() |
        breast_df['final_projection'].str.lower().isin(WRONG_BREAST_LABELS)
    )
    return breast_df[mask].copy()


def _filename_breast_fix(filename: str) -> Optional[Dict[str, str]]:
    """根据文件名模式返回 (orientation, projection)，无法识别返回 None"""
    fn = filename.upper()
    if "_L_CC" in fn:
        return {"orientation": "left", "projection": "cephalocaudal"}
    if "_R_CC" in fn:
        return {"orientation": "right", "projection": "cephalocaudal"}
    if "_L_MLO" in fn:
        return {"orientation": "left", "projection": "mediolateral oblique"}
    if "_R_MLO" in fn:
        return {"orientation": "right", "projection": "mediolateral oblique"}
    return None


def fix_breast_labels(args):
    """
    修复已标注Excel中乳房样本的投影标签。
    合并自 fix_breast_labels.py，使用本地GPU模型替代远程API。

    流程:
    1. 读取已标注Excel
    2. 筛选 original_part==乳房 且 final_projection 为通用视角的记录
    3. 文件名快速匹配 (_L_CC, _R_CC, _L_MLO, _R_MLO)
    4. 剩余图片用本地模型做乳腺投影打分
    5. 更新Excel
    """
    _ensure_gpu_imports()
    logger = setup_logging(0)

    input_file = getattr(args, 'input', None) or OUTPUT_FILE
    output_file = getattr(args, 'output', None) or input_file.replace('.xlsx', '_fixed.xlsx')
    dry_run = getattr(args, 'dry_run', False)
    limit = getattr(args, 'limit', None)

    logger.info("=" * 60)
    logger.info("乳房标签修复模式 (--fix-breast)")
    logger.info(f"输入: {input_file}")
    logger.info(f"输出: {output_file}")
    logger.info(f"模式: {'模拟运行(dry-run)' if dry_run else '实际修复'}")
    logger.info("=" * 60)

    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return False

    df = pd.read_excel(input_file)
    logger.info(f"总计 {len(df)} 条记录")

    to_fix = _find_breast_to_fix(df)
    logger.info(f"发现 {len(to_fix)} 条需要修复的乳房记录")
    if len(to_fix) == 0:
        logger.info("无需修复，退出")
        return True

    # 错误标签分布
    logger.info("当前错误标签分布:")
    logger.info(to_fix['final_projection'].value_counts(dropna=False).to_string())

    # 统计文件名可直接修复的数量
    fn_fixable = sum(1 for _, row in to_fix.iterrows()
                     if _filename_breast_fix(row['filename']) is not None)
    logger.info(f"其中可通过文件名直接修复: {fn_fixable} 条")

    if dry_run:
        logger.info("\n[模拟运行] 列出前10条需要修复的记录:")
        for _, row in to_fix.head(10).iterrows():
            logger.info(f"  {row['image_id']}: {row['filename']} -> 当前: {row['final_projection']}")
        logger.info("实际修复请去掉 --dry-run")
        return True

    # 按 image_id 分组
    image_ids = to_fix['image_id'].unique()
    if limit:
        image_ids = image_ids[:limit]
    logger.info(f"开始处理 {len(image_ids)} 个影像号...")

    # 第一遍: 文件名匹配
    all_fixed = []
    needs_llm = {}  # image_id -> [(row_idx_in_to_fix, img_path, filename)]

    for img_id in image_ids:
        rows = to_fix[to_fix['image_id'] == img_id]
        for _, row in rows.iterrows():
            fname = row['filename']
            hit = _filename_breast_fix(fname)
            if hit:
                all_fixed.append({
                    'image_id': img_id,
                    'filename': fname,
                    'final_body_part': 'Breast',
                    'final_orientation': hit['orientation'],
                    'final_projection': hit['projection'],
                    'confidence_projection': 0.95,
                    'confidence_orientation': 0.95,
                    'confidence_overall': 0.95,
                    'needs_review': False,
                    'review_reason': '',
                    'match_method': 'filename_breast_fixed',
                })
            else:
                img_path = os.path.join(IMAGE_ROOT, str(img_id), fname)
                if os.path.exists(img_path):
                    needs_llm.setdefault(img_id, []).append((img_path, fname))

    logger.info(f"文件名修复: {len(all_fixed)} 条, 需LLM: {sum(len(v) for v in needs_llm.values())} 条")

    # 第二遍: LLM推理
    if needs_llm:
        torch.cuda.set_device(0)
        model_type = args.model_type or detect_model_type(args.checkpoint)
        backend = create_model_backend(model_type, args.checkpoint, args.batch_size)

        # 构建batch items: 按 image_id 分组，每组用乳腺打分 prompt
        batch_items = []
        batch_meta = []  # (image_id, [(img_path, filename), ...])
        for img_id, file_list in needs_llm.items():
            paths = [p for p, _ in file_list]
            prompt = build_breast_scoring_prompt(len(paths))
            batch_items.append((paths, prompt, None))
            batch_meta.append((img_id, file_list))

        logger.info(f"LLM乳腺打分推理: {len(batch_items)} 组")
        responses = backend.infer_batch(batch_items, desc="BreastFix-Score")

        for (img_id, file_list), resp in zip(batch_meta, responses):
            n = len(file_list)
            scores = parse_scores_from_response(resp, n, valid_views=BREAST_VIEWS)

            # 自由分类 (无excel标签约束)
            for i in range(n):
                bi = int(np.argmax(scores[i]))
                bv = BREAST_VIEWS[bi]
                cf, lv = calculate_confidence(scores[i], bv, valid_views=BREAST_VIEWS)
                img_path, fname = file_list[i]
                all_fixed.append({
                    'image_id': img_id,
                    'filename': fname,
                    'final_body_part': 'Breast',
                    'final_orientation': 'bilateral',
                    'final_projection': bv,
                    'confidence_projection': round(cf, 3),
                    'confidence_orientation': 0.5,
                    'confidence_overall': round(cf * 0.5 + 0.5 * 0.5, 3),
                    'needs_review': lv in ('low', 'very_low', 'medium'),
                    'review_reason': 'breast_relabeled' if lv in ('low', 'very_low', 'medium') else '',
                    'match_method': f'llm_breast_fixed_greedy',
                })

        del backend
        torch.cuda.empty_cache()

    # 更新 Excel
    logger.info(f"共 {len(all_fixed)} 条修复记录")
    if not all_fixed:
        logger.info("无修复记录，退出")
        return True

    out_df = df.copy()
    key_to_idx = {}
    for idx, row in out_df.iterrows():
        key_to_idx[(row['image_id'], row['filename'])] = idx

    updated = 0
    for rec in all_fixed:
        key = (rec['image_id'], rec['filename'])
        if key in key_to_idx:
            idx = key_to_idx[key]
            for col, val in rec.items():
                if col in out_df.columns:
                    out_df.at[idx, col] = val
            updated += 1

    out_df.to_excel(output_file, index=False)
    logger.info(f"已更新 {updated} 条记录 -> {output_file}")

    # 修复报告
    report_path = output_file.replace('.xlsx', '_report.json')
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_file': input_file,
        'output_file': output_file,
        'total_records': len(df),
        'breast_to_fix': len(to_fix),
        'filename_fixed': fn_fixable,
        'llm_fixed': len(all_fixed) - fn_fixable,
        'updated': updated,
        'details': all_fixed,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"修复报告: {report_path}")
    logger.info("=" * 60)
    return True


def start_review_server(port: int = 5000, auto_open: bool = True):
    logger = logging.getLogger(__name__)
    web_app_path = os.path.join(os.path.dirname(__file__), 'web', 'app.py')
    if not os.path.exists(web_app_path):
        logger.warning(f"Web应用不存在: {web_app_path}")
        return False
    import subprocess
    cmd = [sys.executable, web_app_path, '--input', OUTPUT_FILE, '--port', str(port)]
    if not auto_open:
        cmd.append('--no-open')
    subprocess.Popen(cmd, cwd=os.path.dirname(web_app_path))
    logger.info(f"Web服务器已启动: http://localhost:{port}")
    return True


# ================= 入口 =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='X-ray 标注系统 (本地GPU版, 多模型)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help=f'模型路径 (默认: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['medgemma', 'hulu', 'meddr2'],
                        help='模型类型 (默认: 自动检测)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'批量推理大小 (默认: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制处理样本数 (测试用)')
    parser.add_argument('--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                        help=f'置信度阈值 (默认: {CONFIDENCE_THRESHOLD})')
    parser.add_argument('--resume', action='store_true',
                        help='从checkpoint恢复')
    parser.add_argument('--review', action='store_true',
                        help='标注后启动复核Web服务器')
    parser.add_argument('--review-only', action='store_true',
                        help='仅启动复核服务器')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--no-open', action='store_true')
    # 乳房修复模式
    parser.add_argument('--fix-breast', action='store_true',
                        help='修复乳房样本的投影标签 (合并自fix_breast_labels.py)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅查看需要修复的样本，不实际执行 (配合 --fix-breast)')
    parser.add_argument('--input', type=str, default=None,
                        help='输入Excel路径 (--fix-breast 模式)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出Excel路径 (--fix-breast 模式)')

    args = parser.parse_args()
    CONFIDENCE_THRESHOLD = args.confidence

    if args.review_only:
        setup_logging(0)
        start_review_server(port=args.port, auto_open=not args.no_open)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        sys.exit(0)

    if args.fix_breast:
        # 乳房修复模式: 单卡即可，不需要 torchrun
        _ensure_gpu_imports()
        success = fix_breast_labels(args)
        sys.exit(0 if success else 1)

    _ensure_gpu_imports()
    rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size_env = int(os.getenv('WORLD_SIZE', '1'))

    if world_size_env > 1:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size_env,
            rank=int(os.getenv('RANK', '0')),
            timeout=timedelta(minutes=60),
        )
    else:
        torch.cuda.set_device(0)

    logger = setup_logging(rank)
    logger.info(f"Rank {rank}/{world_size_env}, device=cuda:{rank}")

    success = run_pipeline(args)

    if success and args.review and is_main_process():
        start_review_server(port=args.port, auto_open=not args.no_open)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("停止...")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port 11721 LLM_label_local.py --resume --batch-size 4
# CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port 11721 LLM_label_local.py --resume --batch-size 4 --limit 10

# CUDA_VISIBLE_DEVICES=2 python LLM_label_local.py --fix-breast

# # MedGemma (default), batch=1
#   torchrun --nproc_per_node=2 LLM_label_local.py --limit 10

#   # MedDR2/InternVL with batch=4
#   torchrun --nproc_per_node=2 LLM_label_local.py --model-type meddr2 \
#       --checkpoint /path/to/meddr2 --batch-size 4

#   # HuluMed
#   torchrun --nproc_per_node=2 LLM_label_local.py --model-type hulu \
#       --checkpoint /path/to/hulu
