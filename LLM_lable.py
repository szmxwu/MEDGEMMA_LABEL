"""
MedGemma X-ray 标注系统 - 改进版
支持多线程并发、置信度评估、错误重试和结构化日志
"""

import os
import sys
import glob
import json
import base64
import time
import logging
import requests
import pandas as pd
import numpy as np
import threading
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import asdict

# 引入投影匹配模块
import projection_matcher
from projection_matcher import MatchResult

# ================= 配置区 =================
BASE_URL = "https://smartlab.cse.ust.hk/smartcare/api/shebd/medgemma15"
API_URL = f"{BASE_URL}/v1/chat/completions"
MODEL_ID = "/data/shebd/0_Pretrained/medgemma-1.5-4b-it"

# 请求头配置
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer YOUR_TOKEN",
}

# 重试和流控配置
MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_DELAY = 1.5

# 多线程配置
MAX_WORKERS = 4  # 并发线程数，可根据API限制调整
BATCH_SIZE = 10  # 每批处理的样本数

# 文件路径配置
IMAGE_ROOT = "data"
CONFIG_FILE = "part_exam_orientation.json"
EXCEL_FILE = "selected_samples.xlsx"
OUTPUT_FILE = "processed_labels_v3.xlsx"
LOG_FILE = f"logs/processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
FAILED_LOG_FILE = f"logs/failed_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
CHECKPOINT_FILE = "processing_checkpoint.json"

# 置信度阈值
CONFIDENCE_THRESHOLD = 0.6

# ================= 日志配置 =================
def setup_logging():
    """配置结构化日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# 断点续跑检查点写入锁
CHECKPOINT_LOCK = threading.Lock()

# ================= 中英文映射字典 =================
CN_EN_MAP = {
    "左": "left", "右": "right", "双": "bilateral",
    # === 拍摄体位 ===
    "头尾位":"cephalocaudal", "定点压迫位":"spot compression", "腋尾位": "mediolateral oblique",
    "正位": "frontal", "立位":"frontal", "卧位":"frontal", "柯氏位":"frontal", "瓦氏位":"frontal",
    "开口位":"frontal", "闭口位":"frontal", "穿胸位":"frontal","后前位":"frontal","仰卧位":"frontal","左右弯曲正位": "frontal",
    "左右侧屈正位":"frontal","俯卧位":"frontal", "仰卧位":"frontal", "冠状位":"frontal","负重正位":"frontal",
    "侧卧水平正位": "frontal","舟骨位":"frontal","穴位": "frontal","蝶位": "frontal","蛙氏位": "frontal","蝶式位":"frontal",
    "侧位": "lateral","动力位":"lateral","过伸位":"lateral", "过屈位": "lateral","左侧位": "lateral", "右侧位": "lateral",
    "前弓位": "lateral","双侧位": "lateral", "负重侧位": "lateral","仰卧水平侧位": "lateral",
    "斜位": "oblique", "Y位":"oblique", "侧斜位":"oblique","切线位":"oblique",
    "双斜位": "oblique","后斜位": "oblique","闭孔斜位":"oblique",
    "轴位": "axial","侧轴位":"axial,lateral",
    "特殊": "special",
    "骸顶位": "special", "华氏位": "special", "梅氏位":  "special","汤氏位": "special",  "Broden位": "special",  
    "颅底位": "special",  "颧弓位": "special", "许氏位": "special", "斯氏位": "special", "瑞氏位": "special",  "梅伦氏位": "special", 
    "劳式位": "special", "薄骨位": "special", "尺偏位": "special", "出入口位": "special", "劳梅氏位": "special", 
    # === 部位映射  ===
    "颈椎": "Cervical Spine",
    "胸椎": "Thoracic Spine",
    "腰椎": "Lumbar Spine",
    "骶尾椎": "Sacrum/Coccyx",
    "脊柱": "Spine",
    "肩部": "Shoulder",
    "肘关节": "Elbow",
    "腕关节": "Wrist",
    "手": "Hand",
    "上臂": "Upper Arm",
    "前臂": "Forearm",
    "盆部": "Pelvis",
    "骶髂关节": "Sacroiliac Joint",
    "大腿": "Thigh",
    "膝关节": "Knee",
    "小腿": "Calf",
    "踝关节": "Ankle",
    "跟骨": "Calcaneus",
    "足部": "Foot",
    "下肢": "Lower Limb",
    "胸部": "Chest",
    "腹部": "Abdomen",
    "乳房": "Breast",
    "颅脑": "Skull",
    "颜面": "Facial Bones"
}

# 脊柱和下肢的细分选项
SUB_PART_OPTIONS = {
    "脊柱": ["Cervical Spine", "Thoracic Spine", "Lumbar Spine", "Full Spine"],
    "下肢": ["Thigh", "Calf", "Full Lower Limb"]
}

# ================= 核心工具函数 =================

class ProcessingError(Exception):
    """处理异常基类"""
    pass


class APIError(ProcessingError):
    """API调用异常"""
    pass


class ImageError(ProcessingError):
    """图片处理异常"""
    pass


def encode_image_to_base64(image_path: str) -> str:
    """将本地图片文件读取并转换为 base64 编码字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"图片编码失败 {image_path}: {e}")
        raise ImageError(f"Failed to encode image {image_path}: {e}")


def make_request_with_retry(payload: Dict[str, Any], max_retries: int = MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """带重试机制的 API 请求函数"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(RETRY_DELAY)
            else:
                time.sleep(REQUEST_DELAY)
            
            resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=600)
            
            if resp.status_code != 200:
                logger.warning(f"HTTP错误 {resp.status_code}: {resp.text[:200]}")
                continue
                
            data = resp.json()
            
            if "error" in data:
                logger.warning(f"API错误: {data['error']}")
                continue
                
            if "choices" in data and len(data["choices"]) > 0:
                return data
                
        except requests.exceptions.Timeout:
            logger.warning(f"请求超时 (尝试 {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.warning(f"请求异常: {e}")
            
    logger.error(f"请求失败，已重试{max_retries}次")
    return None


def call_medgemma(image_paths: List[str], prompt_question: str, options: List[str]) -> str:
    """构造多模态请求并调用 MedGemma"""
    content_blocks = []
    content_blocks.append({"type": "text", "text": "Here are the X-ray images for analysis:\n"})
    
    for idx, img_path in enumerate(image_paths):
        try:
            b64_str = encode_image_to_base64(img_path)
            content_blocks.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{b64_str}"}
            })
            content_blocks.append({"type": "text", "text": f"\n[Image {idx+1} Above]\n"})
        except ImageError:
            return ""

    prompt_text = f"\n{prompt_question}\n"
    if options:
        prompt_text += "Please choose STRICTLY from the following options for each image:\n"
        prompt_text += f"{json.dumps(options, ensure_ascii=False)}\n"
    prompt_text += "\nAnswer format strictly as:\n"
    for idx in range(len(image_paths)):
        prompt_text += f"Image {idx+1}: [Your Choice]\n"
    
    content_blocks.append({"type": "text", "text": prompt_text})

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content_blocks}],
        "max_tokens": 512,
        "temperature": 0.2
    }

    response_data = make_request_with_retry(payload)
    
    if response_data:
        try:
            content = response_data["choices"][0]["message"]["content"]
            logger.debug(f"API响应: {content[:200]}...")
            return content
        except (KeyError, IndexError) as e:
            logger.error(f"解析响应失败: {e}")
            return ""
    return ""


def parse_medgemma_response(response: str, num_images: int) -> List[Optional[str]]:
    """解析模型返回的文本，提取每张图片的标签"""
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
    """统一投影标签格式"""
    if not label:
        return "unknown"
    
    label = label.lower().replace(" position", "").strip()
    
    for view in ["frontal", "lateral", "oblique", "axial", "special"]:
        if view in label:
            return view
    
    return label


# ================= 单样本处理函数 =================

def process_single_sample(
    index: int,
    row: pd.Series,
    part_config: Dict,
    total_samples: int
) -> Tuple[List[Dict], Optional[str]]:
    """
    处理单个样本
    
    Returns:
        (results_list, error_message)
        如果成功，error_message为None
    """
    img_id = str(row['影像号'])
    std_part = row['标准化部位']
    excel_pos = row.get('Position_orientation', None)
    excel_proj = row.get('exam_projection', None)
    
    logger.info(f"[{index+1}/{total_samples}] 处理 {img_id} ({std_part})")
    
    try:
        # 查找图片
        img_dir = os.path.join(IMAGE_ROOT, img_id)
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        
        if not img_paths:
            logger.warning(f"  未找到图片: {img_dir}")
            return [], f"No images found in {img_dir}"
        
        num_imgs = len(img_paths)
        logger.info(f"  找到 {num_imgs} 张图片")
        
        # --- Task A: 部位细分 ---
        std_part_en = CN_EN_MAP.get(std_part, std_part)
        final_parts = [std_part_en] * num_imgs
        
        if std_part in ["脊柱", "下肢"] and num_imgs > 2:
            options = SUB_PART_OPTIONS[std_part]
            prompt = f"Identify the specific sub-part for each image. Are they {', '.join(options)}?"
            logger.info("  -> 调用模型进行部位细分...")
            
            resp = call_medgemma(img_paths, prompt, options)
            logger.debug(f"     响应: {repr(resp)}")
            
            parsed = parse_medgemma_response(resp, num_imgs)
            for i, p in enumerate(parsed):
                if p:
                    final_parts[i] = p
        
        # --- Task B: 左右方位 ---
        final_orientations = [None] * num_imgs
        orientation_confidences = [1.0] * num_imgs  # 默认值
        part_cfg = part_config.get(std_part, {})
        allowed_orientations = part_cfg.get("Position_orientation", [])
        
        if not allowed_orientations:
            final_orientations = ["Not Applicable"] * num_imgs
        else:
            mapped_excel_pos = CN_EN_MAP.get(excel_pos, excel_pos)
            
            if mapped_excel_pos in ["left", "right"]:
                # 明确指定，高置信度
                final_orientations = [mapped_excel_pos] * num_imgs
                orientation_confidences = [0.95] * num_imgs
            else:
                # 调用模型识别
                logger.info("  -> 调用模型识别左右方位...")
                prompt = "Identify the side (laterality) for each image."
                resp = call_medgemma(img_paths, prompt, allowed_orientations)
                logger.debug(f"     响应: {repr(resp)}")
                
                parsed = parse_medgemma_response(resp, num_imgs)
                for i, p in enumerate(parsed):
                    if p:
                        final_orientations[i] = p
                        orientation_confidences[i] = 0.8  # 模型预测中等置信度
                    else:
                        final_orientations[i] = "unknown"
                        orientation_confidences[i] = 0.0
        
        # --- Task B.5: 乳房特殊处理 (基于文件名快速打标) ---
        breast_projections = [None] * num_imgs  # 乳房部位的体位预测
        if std_part == "乳房":
            logger.info("  -> 乳房部位: 基于文件名快速打标...")
            for i, img_path in enumerate(img_paths):
                filename = os.path.basename(img_path).upper()
                
                # 根据文件名模式确定方位和体位
                if "_L_CC" in filename:
                    final_orientations[i] = "left"
                    orientation_confidences[i] = 0.95
                    breast_projections[i] = "cephalocaudal"
                    logger.info(f"     图片{i+1} ({filename}): 文件名包含_L_CC -> left, cephalocaudal")
                elif "_R_CC" in filename:
                    final_orientations[i] = "right"
                    orientation_confidences[i] = 0.95
                    breast_projections[i] = "cephalocaudal"
                    logger.info(f"     图片{i+1} ({filename}): 文件名包含_R_CC -> right, cephalocaudal")
                elif "_L_MLO" in filename:
                    final_orientations[i] = "left"
                    orientation_confidences[i] = 0.95
                    breast_projections[i] = "mediolateral oblique"
                    logger.info(f"     图片{i+1} ({filename}): 文件名包含_L_MLO -> left, mediolateral oblique")
                elif "_R_MLO" in filename:
                    final_orientations[i] = "right"
                    orientation_confidences[i] = 0.95
                    breast_projections[i] = "mediolateral oblique"
                    logger.info(f"     图片{i+1} ({filename}): 文件名包含_R_MLO -> right, mediolateral oblique")
                else:
                    # 文件名不包含关键词，保持原值，后续由LLM处理
                    breast_projections[i] = None
                    logger.info(f"     图片{i+1} ({filename}): 文件名无关键词，需LLM处理")
        
        # --- Task C: 拍摄体位 ---
        final_projections = [None] * num_imgs
        projection_confidences = [0.0] * num_imgs
        projection_needs_review = [True] * num_imgs
        projection_raw_scores = [{}] * num_imgs
        match_methods = ["unknown"] * num_imgs
        
        part_cfg = part_config.get(std_part, {})
        allowed_projections = part_cfg.get("exam_projection", [])
        
        # 解析Excel标签
        raw_proj_str = str(excel_proj) if excel_proj else ""
        excel_proj_list = [x.strip() for x in raw_proj_str.replace("，", ",").split(",") if x.strip()]
        # 先做中英文映射，再展开任何包含逗号的映射值（例如 "axial,lateral" -> ["axial","lateral"]）
        expected_labels_en = []
        for p in excel_proj_list:
            mapped = CN_EN_MAP.get(p, p)
            if isinstance(mapped, str) and "," in mapped:
                parts = [s.strip() for s in mapped.split(",") if s.strip()]
                expected_labels_en.extend(parts)
            else:
                expected_labels_en.append(mapped)
        
        # === 逻辑分支 0: 乳房文件名快速通道 ===
        if std_part == "乳房" and 'breast_projections' in locals():
            logger.info("  -> 乳房部位: 应用文件名快速打标结果...")
            for i in range(num_imgs):
                if breast_projections[i] is not None:
                    # 文件名匹配成功，使用预定义标签
                    final_projections[i] = breast_projections[i]
                    projection_confidences[i] = 0.95
                    projection_needs_review[i] = False
                    match_methods[i] = "filename_breast"
                    logger.info(f"     图片{i+1}: {final_projections[i]} (文件名匹配)")
                else:
                    # 文件名不匹配，需要LLM处理
                    # 这里先标记为None，后续由其他逻辑分支处理
                    final_projections[i] = None
        
        # === 逻辑分支 A: 极速通道 (单图单标签) ===
        elif num_imgs == 1 and len(expected_labels_en) == 1:
            label = normalize_projection_label(expected_labels_en[0])
            final_projections = [label]
            projection_confidences = [1]  # 高速通道高置信度
            projection_needs_review = [False]
            match_methods = ["fast_path"]
            logger.info(f"  -> 极速通道: {label}")

        # === 逻辑分支 B: 全局最优匹配 (改进版) ===
        elif expected_labels_en:
            # 对于乳房部位，只处理文件名不匹配的图片
            if std_part == "乳房":
                unmatched_indices = [i for i in range(num_imgs) if final_projections[i] is None]
                if unmatched_indices:
                    logger.info(f"  -> 乳房部位: {len(unmatched_indices)}张图片文件名不匹配，使用LLM...")
                    unmatched_images = [img_paths[i] for i in unmatched_indices]
                    
                    match_results: List[MatchResult] = projection_matcher.predict_projection_globally(
                        images=unmatched_images,
                        excel_labels=expected_labels_en,
                        body_part=final_parts[0],
                        api_call_func=call_medgemma
                    )
                    
                    for idx, match_result in enumerate(match_results):
                        i = unmatched_indices[idx]
                        final_projections[i] = normalize_projection_label(match_result.label)
                        projection_confidences[i] = match_result.confidence
                        projection_needs_review[i] = match_result.needs_review
                        projection_raw_scores[i] = match_result.raw_scores
                        match_methods[i] = match_result.match_method
                        
                        logger.info(f"     图片{i+1}: {final_projections[i]}, "
                                   f"置信度={match_result.confidence:.2f}, "
                                   f"需审核={match_result.needs_review}")
                else:
                    logger.info("  -> 乳房部位: 所有图片已通过文件名匹配，无需LLM调用")
            else:
                # 非乳房部位，正常处理
                logger.info(f"  -> 使用全局匹配，目标标签: {expected_labels_en}")
                
                match_results: List[MatchResult] = projection_matcher.predict_projection_globally(
                    images=img_paths,
                    excel_labels=expected_labels_en,
                    body_part=final_parts[0],
                    api_call_func=call_medgemma
                )
                
                for i, match_result in enumerate(match_results):
                    final_projections[i] = normalize_projection_label(match_result.label)
                    projection_confidences[i] = match_result.confidence
                    projection_needs_review[i] = match_result.needs_review
                    projection_raw_scores[i] = match_result.raw_scores
                    match_methods[i] = match_result.match_method
                    
                    logger.info(f"     图片{i+1}: {final_projections[i]}, "
                               f"置信度={match_result.confidence:.2f}, "
                               f"需审核={match_result.needs_review}")

        # === 逻辑分支 C: 自由分类 ===
        else:
            # 对于乳房部位，使用专用的投影匹配函数
            if std_part == "乳房":
                unmatched_indices = [i for i in range(num_imgs) if final_projections[i] is None]
                if unmatched_indices:
                    logger.info(f"  -> 乳房部位: {len(unmatched_indices)}张图片文件名不匹配，使用专用分类器...")
                    unmatched_images = [img_paths[i] for i in unmatched_indices]
                    
                    # 使用projection_matcher中的专用函数，传入空标签列表触发自由分类
                    match_results: List[MatchResult] = projection_matcher.predict_projection_globally(
                        images=unmatched_images,
                        excel_labels=[],  # 空标签列表触发自由分类模式
                        body_part="breast",
                        api_call_func=call_medgemma
                    )
                    
                    for idx, match_result in enumerate(match_results):
                        i = unmatched_indices[idx]
                        final_projections[i] = match_result.label
                        projection_confidences[i] = match_result.confidence
                        projection_needs_review[i] = match_result.needs_review
                        projection_raw_scores[i] = match_result.raw_scores
                        match_methods[i] = match_result.match_method
                        
                        logger.info(f"     图片{i+1}: {final_projections[i]}, "
                                   f"置信度={match_result.confidence:.2f}, "
                                   f"需审核={match_result.needs_review}")
                else:
                    logger.info("  -> 乳房部位: 所有图片已通过文件名匹配，无需LLM调用")
            else:
                # 非乳房部位，正常处理
                logger.info("  -> Excel无标签，使用自由分类...")
                prompt = "Identify the projection view for each image."
                options_with_special = allowed_projections + ["special"]
                
                resp = call_medgemma(img_paths, prompt, options_with_special)
                parsed = parse_medgemma_response(resp, num_imgs)
                
                for i, p in enumerate(parsed):
                    if p:
                        final_projections[i] = normalize_projection_label(p)
                        projection_confidences[i] = 0.6
                        projection_needs_review[i] = True
                    else:
                        final_projections[i] = "unknown"
                        projection_confidences[i] = 0.0
                        projection_needs_review[i] = True
                    match_methods[i] = "free_classification"
        
        # --- 汇总结果 ---
        results = []
        for i in range(num_imgs):
            # 综合置信度：体位和方位的平均
            overall_confidence = (projection_confidences[i] + orientation_confidences[i]) / 2
            
            results.append({
                "image_id": img_id,
                "filename": os.path.basename(img_paths[i]),
                "original_part": std_part,
                "final_body_part": final_parts[i],
                "final_orientation": final_orientations[i],
                "final_projection": final_projections[i],
                "confidence_projection": round(projection_confidences[i], 3),
                "confidence_orientation": round(orientation_confidences[i], 3),
                "confidence_overall": round(overall_confidence, 3),
                "needs_review": projection_needs_review[i] or (overall_confidence < CONFIDENCE_THRESHOLD),
                "review_reason": _get_review_reason(projection_confidences[i], orientation_confidences[i], match_methods[i]),
                "match_method": match_methods[i],
                "raw_scores_frontal": projection_raw_scores[i].get('frontal', 0),
                "raw_scores_lateral": projection_raw_scores[i].get('lateral', 0),
                "raw_scores_oblique": projection_raw_scores[i].get('oblique', 0),
                "raw_scores_axial": projection_raw_scores[i].get('axial', 0),
                "raw_excel_pos": excel_pos,
                "raw_excel_proj": excel_proj
            })
        
        logger.info(f"  ✓ 完成处理，生成 {len(results)} 条记录")
        return results, None
        
    except Exception as e:
        logger.error(f"  ✗ 处理失败: {e}", exc_info=True)
        return [], str(e)


def _get_review_reason(proj_conf: float, ori_conf: float, method: str) -> str:
    """生成需要审核的原因说明"""
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


# ================= 主处理流程 =================

def process_samples_parallel(max_workers: int = MAX_WORKERS, limit: Optional[int] = None):
    """
    并行处理样本
    
    Args:
        max_workers: 并发线程数
        limit: 限制处理的样本数（用于测试）
    """
    logger.info("=" * 60)
    logger.info("MedGemma X-ray 标注系统启动")
    logger.info(f"线程数: {max_workers}, 置信度阈值: {CONFIDENCE_THRESHOLD}")
    logger.info("=" * 60)
    
    # 1. 读取配置
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"配置文件未找到: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        part_config = json.load(f)
        
    # 2. 读取 Excel
    if not os.path.exists(EXCEL_FILE):
        logger.error(f"Excel文件未找到: {EXCEL_FILE}")
        return
        
    df = pd.read_excel(EXCEL_FILE)
    if limit:
        df = df.head(limit)
        
    total_samples = len(df)
    logger.info(f"待处理样本总数: {total_samples}")

    # 断点续跑：加载检查点
    processed_indices = set()
    all_results = []
    failed_samples = []
    processed_count = 0
    success_count = 0

    def save_checkpoint():
        """原子写入检查点文件"""
        tmp_path = CHECKPOINT_FILE + ".tmp"
        payload = {
            "processed_indices": list(processed_indices),
            "all_results": all_results,
            "failed_samples": failed_samples
        }
        with CHECKPOINT_LOCK:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, CHECKPOINT_FILE)

    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                ck = json.load(f)
            processed_indices = set(ck.get("processed_indices", []))
            all_results = ck.get("all_results", []) or []
            failed_samples = ck.get("failed_samples", []) or []
            # 仅保留当前 df 中存在的 index
            processed_indices = processed_indices.intersection(set(df.index))
            processed_count = len(processed_indices)
            success_count = max(processed_count - len(failed_samples), 0)
            logger.info(f"已加载检查点: {processed_count} 个样本已处理，继续处理剩余样本")
        except Exception as e:
            logger.warning(f"读取检查点失败，将从头开始: {e}")
            processed_indices = set()
            all_results = []
            failed_samples = []
            processed_count = 0
            success_count = 0
    
    # 3. 并行处理
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Worker") as executor:
        # 提交所有任务（跳过已处理的样本）
        future_to_idx = {}
        for idx, row in df.iterrows():
            if idx in processed_indices:
                continue
            future_to_idx[executor.submit(process_single_sample, idx, row, part_config, total_samples)] = idx

        if not future_to_idx:
            logger.info("没有待处理任务，检查点已包含全部样本")
        
        # 处理完成的任务
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results, error = future.result()
                processed_count += 1
                processed_indices.add(idx)
                
                if error:
                    failed_samples.append({
                        "index": idx,
                        "image_id": str(df.iloc[idx]['影像号']),
                        "error": error
                    })
                    save_checkpoint()
                else:
                    all_results.extend(results)
                    success_count += 1
                    save_checkpoint()
                
                # 进度报告
                if processed_count % 10 == 0 or processed_count == total_samples:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    eta = (total_samples - processed_count) / rate if rate > 0 else 0
                    logger.info(f"进度: {processed_count}/{total_samples} "
                               f"({100*processed_count/total_samples:.1f}%) | "
                               f"成功: {success_count} | 失败: {len(failed_samples)} | "
                               f"速率: {rate:.2f}样本/秒 | 预计剩余: {eta/60:.1f}分钟")
                
            except Exception as e:
                logger.error(f"任务执行异常 (index={idx}): {e}", exc_info=True)
                failed_samples.append({
                    "index": idx,
                    "image_id": str(df.iloc[idx]['影像号']),
                    "error": str(e)
                })
                processed_indices.add(idx)
                save_checkpoint()
    
    # 4. 保存结果
    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"处理完成!")
    logger.info(f"总耗时: {elapsed_total/60:.1f}分钟")
    logger.info(f"成功样本: {success_count}/{total_samples}")
    logger.info(f"失败样本: {len(failed_samples)}")
    logger.info(f"生成记录: {len(all_results)}")
    
    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df.to_excel(OUTPUT_FILE, index=False)
        logger.info(f"结果已保存: {OUTPUT_FILE}")
        
        # 统计置信度分布
        high_conf = len(res_df[res_df['confidence_overall'] >= CONFIDENCE_THRESHOLD])
        logger.info(f"高置信度记录: {high_conf}/{len(res_df)} ({100*high_conf/len(res_df):.1f}%)")
        
        needs_review = res_df['needs_review'].sum()
        logger.info(f"需要人工审核: {needs_review}/{len(res_df)} ({100*needs_review/len(res_df):.1f}%)")
    
    # 5. 保存失败记录
    if failed_samples:
        with open(FAILED_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        logger.info(f"失败记录已保存: {FAILED_LOG_FILE}")

    # 成功完成后清理检查点文件
    if os.path.exists(CHECKPOINT_FILE) and processed_count >= total_samples:
        try:
            os.remove(CHECKPOINT_FILE)
            logger.info("已处理完毕，检查点文件已删除")
        except Exception as e:
            logger.warning(f"删除检查点文件失败: {e}")
    
    logger.info("=" * 60)
    
    return len(all_results) > 0


def start_review_server(excel_path: str, port: int = 5000, auto_open: bool = True):
    """
    启动复核Web服务器
    
    Args:
        excel_path: 标注结果Excel文件路径
        port: 服务器端口
        auto_open: 是否自动打开浏览器
    """
    try:
        # 检查web/app.py是否存在
        web_app_path = os.path.join(os.path.dirname(__file__), 'web', 'app.py')
        if not os.path.exists(web_app_path):
            logger.warning(f"Web应用不存在: {web_app_path}")
            return False
        
        logger.info("=" * 60)
        logger.info("正在启动复核Web服务器...")
        logger.info("=" * 60)
        
        # 使用subprocess启动Flask服务器
        import subprocess
        import sys
        
        cmd = [
            sys.executable,
            web_app_path,
            '--input', excel_path,
            '--port', str(port)
        ]
        
        if not auto_open:
            cmd.append('--no-open')
        
        # 在新进程中启动服务器
        subprocess.Popen(cmd, cwd=os.path.dirname(web_app_path))
        
        logger.info(f"✓ Web服务器已启动，访问地址: http://localhost:{port}")
        return True
        
    except Exception as e:
        logger.error(f"启动Web服务器失败: {e}")
        return False


def process_samples_single():
    """单线程处理（向后兼容）"""
    logger.info("使用单线程模式")
    process_samples_parallel(max_workers=1)


# ================= 入口 =================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MedGemma X-ray 标注系统')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                       help=f'并发线程数 (默认: {MAX_WORKERS})')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理的样本数（用于测试）')
    parser.add_argument('--single', action='store_true',
                       help='使用单线程模式')
    parser.add_argument('--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                       help=f'置信度阈值 (默认: {CONFIDENCE_THRESHOLD})')
    parser.add_argument('--review', action='store_true',
                       help='标注完成后自动启动复核Web服务器')
    parser.add_argument('--review-port', type=int, default=5000,
                       help='复核服务器端口 (默认: 5000)')
    parser.add_argument('--no-open', action='store_true',
                       help='不自动打开浏览器')
    parser.add_argument('--review-only', action='store_true',
                       help='仅启动复核服务器，不运行标注')
    
    args = parser.parse_args()
    
    # 更新配置
    CONFIDENCE_THRESHOLD = args.confidence
    
    # 仅启动复核服务器
    if args.review_only:
        start_review_server(OUTPUT_FILE, port=args.review_port, auto_open=not args.no_open)
        # 保持主进程运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n正在停止...")
        sys.exit(0)
    
    # 运行标注
    success = False
    if args.single:
        process_samples_single()
        success = True
    else:
        success = process_samples_parallel(max_workers=args.workers, limit=args.limit)
    
    # 标注完成后启动复核服务器
    if success and args.review:
        logger.info("\n")
        start_review_server(OUTPUT_FILE, port=args.review_port, auto_open=not args.no_open)
        
        # 保持主进程运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n正在停止...")
