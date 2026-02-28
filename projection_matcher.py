"""
投影体位匹配模块 - 改进版
使用匈牙利算法进行全局最优匹配，支持置信度评估和文件名辅助提示
"""

import numpy as np
import re
import os
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass

# 定义基础几何视角 (Visual Geometry Standards)
STANDARD_VIEWS = ["frontal", "lateral", "oblique", "axial"]
BREAST_VIEWS = ["cephalocaudal", "mediolateral oblique", "spot compression"]  # 乳腺X光常见视角


def is_breast_body_part(body_part: str) -> bool:
    """检查是否为乳房/乳腺部位"""
    if not body_part:
        return False
    breast_keywords = ['乳房', '乳腺', 'breast', 'mammary', 'mammo']
    return any(kw in body_part.lower() for kw in breast_keywords)
# 置信度阈值
CONFIDENCE_THRESHOLD_HIGH = 0.7   # 高置信度，无需审核
CONFIDENCE_THRESHOLD_MEDIUM = 0.4  # 中等置信度，建议审核
CONFIDENCE_THRESHOLD_LOW = 0.3     # 低置信度，必须审核


@dataclass
class MatchResult:
    """匹配结果数据类"""
    label: str
    confidence: float
    needs_review: bool
    raw_scores: Dict[str, float]
    match_method: str  # 'hungarian', 'greedy', 'filename_hint', 'fallback'


def normalize_to_geometry(label: str) -> str:
    """
    将复杂的临床体位映射到基础几何视角用于视觉打分。
    """
    l = label.lower()
    if "frontal" in l or "正位" in l or "ap" in l or "pa" in l or "mouth" in l or "towne" in l or "caldwell" in l:
        return "frontal"
    if "lateral" in l or "侧位" in l or "dynamic" in l or "flexion" in l or "extension" in l:
        return "lateral"
    if "oblique" in l or "斜位" in l or "butterfly" in l:
        return "oblique"
    if "axial" in l or "轴位" in l or "sunrise" in l:
        return "axial"
    return "special"


def get_filename_hint(image_path: str, label: str) -> float:
    """
    从文件名提取体位提示作为辅助信号。
    
    Returns:
        0.0-1.0 的匹配分数
    """
    filename = os.path.basename(image_path).lower()
    label_lower = label.lower()
    target_geo = normalize_to_geometry(label)
    
    # 常见命名约定映射
    hints = {
        'lat': 'lateral',
        'lat.': 'lateral',
        '_lat_': 'lateral',
        'lateral': 'lateral',
        'ap': 'frontal',
        'ap.': 'frontal',
        '_ap_': 'frontal',
        'pa': 'frontal',
        'pa.': 'frontal',
        'frontal': 'frontal',
        'ob': 'oblique',
        'ob.': 'oblique',
        '_ob_': 'oblique',
        'oblique': 'oblique',
        'axi': 'axial',
        'axi.': 'axial',
        'axial': 'axial',
    }
    
    for hint_key, geo_type in hints.items():
        if hint_key in filename:
            if target_geo == geo_type:
                return 1.0
    
    return 0.0


def parse_scores_from_response(response: str, num_imgs: int, valid_views: List[str] = None) -> np.ndarray:
    """
    解析LLM返回的文本，提取打分矩阵。
    
    Args:
        response: LLM返回的文本
        num_imgs: 图片数量
        valid_views: 有效的视角列表，默认使用STANDARD_VIEWS
    
    Returns:
        Numpy矩阵 [num_imgs, len(valid_views)]
    """
    views = valid_views if valid_views else STANDARD_VIEWS
    scores = np.ones((num_imgs, len(views))) * 0.1
    
    if not response:
        return scores
    
    lines = response.strip().split('\n')
    for i in range(num_imgs):
        target_prefix = f"Image {i+1}"
        matched_line = None
        
        for line in lines:
            if target_prefix in line:
                matched_line = line
                break
        
        if not matched_line:
            continue
            
        line_lower = matched_line.lower()
        
        for col_idx, view in enumerate(views):
            # 对于多单词视角(如"mediolateral oblique"),使用第一个单词匹配
            view_key = view.split()[0] if ' ' in view else view
            pattern = view_key + r"[^0-9]*(\d+)"
            match = re.search(pattern, line_lower)
            if match:
                try:
                    score = float(match.group(1))
                    scores[i, col_idx] = max(score, 0.1)  # 确保最小值
                except ValueError:
                    pass
                    
    # 归一化
    row_sums = scores.sum(axis=1, keepdims=True)
    scores = scores / (row_sums + 1e-9)
    
    return scores


def calculate_confidence(
    scores: np.ndarray, 
    assigned_view: str, 
    filename_hint: float = 0.0,
    valid_views: List[str] = None
) -> Tuple[float, str]:
    """
    计算分配结果的置信度。
    
    Args:
        scores: 该图片在视角上的得分向量
        assigned_view: 被分配的视角名称
        filename_hint: 文件名提示信号 (0.0-1.0)
        valid_views: 有效的视角列表，默认使用STANDARD_VIEWS
    
    Returns:
        (confidence_score, confidence_level)
        confidence_level: 'high', 'medium', 'low', 'very_low'
    """
    views = valid_views if valid_views else STANDARD_VIEWS
    
    if assigned_view not in views:
        # 特殊标签给予中等置信度
        return 0.5, 'medium'
    
    view_idx = views.index(assigned_view)
    assigned_score = scores[view_idx]
    
    # 计算与其他最高分的差距 (margin)
    other_scores = np.delete(scores, view_idx)
    best_other_score = np.max(other_scores)
    margin = assigned_score - best_other_score
    
    # 基础置信度 = 分配得分 * (1 + 差距倍数)
    base_confidence = assigned_score * (1 + margin)
    
    # 融合文件名提示 (如果文件名提示强，提升置信度)
    if filename_hint > 0.8:
        confidence = min(base_confidence * 1.2, 1.0)
    elif filename_hint > 0:
        confidence = base_confidence * 0.9  # 轻微冲突时降低
    else:
        confidence = base_confidence
    
    confidence = float(np.clip(confidence, 0.0, 1.0))
    
    # 分级
    if confidence >= CONFIDENCE_THRESHOLD_HIGH:
        level = 'high'
    elif confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
        level = 'medium'
    elif confidence >= CONFIDENCE_THRESHOLD_LOW:
        level = 'low'
    else:
        level = 'very_low'
    
    return confidence, level


def solve_assignment_with_confidence(
    scores_matrix: np.ndarray, 
    expected_labels: List[str],
    image_paths: List[str],
    valid_views: List[str] = None,
    is_breast: bool = False
) -> List[MatchResult]:
    """
    使用贪心算法进行匹配，确保每张图片都分配到最合理的标签。
    
    策略：
    1. 计算每张图片对每个标签的匹配得分
    2. 按得分从高到低排序所有可能的匹配
    3. 贪心选择：优先分配得分高的匹配，每个标签只能用一次
    4. 未匹配到标签的图片使用模型最高分视角
    
    Args:
        scores_matrix: [N_images, N_views] 得分矩阵
        expected_labels: Excel中的标签列表
        image_paths: 图片路径列表
        valid_views: 有效的视角列表，默认使用STANDARD_VIEWS
        is_breast: 是否为乳房部位
    
    Returns:
        MatchResult列表
    """
    views = valid_views if valid_views else STANDARD_VIEWS
    num_imgs = scores_matrix.shape[0]
    num_labels = len(expected_labels)
    
    # 1. 将标签转换为几何类别 (乳房部位跳过此步骤)
    if is_breast:
        # 乳房部位：直接使用标签名作为几何类别
        target_geos = expected_labels
    else:
        target_geos = [normalize_to_geometry(lbl) for lbl in expected_labels]
    
    # 2. 计算所有可能的匹配得分
    # matches: list of (score, img_idx, lbl_idx, label, geo)
    matches = []
    for img_idx in range(num_imgs):
        for lbl_idx, (label, geo) in enumerate(zip(expected_labels, target_geos)):
            # 基础得分：模型在该几何类别的分数
            if geo in views:
                geo_idx = views.index(geo)
                base_score = scores_matrix[img_idx, geo_idx]
            else:
                base_score = 0.5  # special标签给中等分数
            
            # 文件名提示
            hint = get_filename_hint(image_paths[img_idx], label)
            
            # 加权组合
            combined_score = 0.7 * base_score + 0.3 * hint
            
            matches.append({
                'score': combined_score,
                'img_idx': img_idx,
                'lbl_idx': lbl_idx,
                'label': label,
                'geo': geo,
                'hint': hint
            })
    
    # 3. 按得分降序排序
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    # 4. 贪心分配
    used_images = set()
    used_labels = set()
    assignments = {}  # img_idx -> match_info
    
    for match in matches:
        img_idx = match['img_idx']
        lbl_idx = match['lbl_idx']
        
        # 如果图片或标签已经被使用，跳过
        if img_idx in used_images or lbl_idx in used_labels:
            continue
        
        # 分配
        assignments[img_idx] = match
        used_images.add(img_idx)
        used_labels.add(lbl_idx)
        
        # 如果所有标签都分配完了，停止
        if len(used_labels) >= num_labels:
            break
    
    # 5. 构建结果
    results = []
    for img_idx in range(num_imgs):
        # 构建raw_scores字典
        raw_scores = {views[j]: float(scores_matrix[img_idx, j]) for j in range(len(views))}
        
        if img_idx in assignments:
            # 分配到预期标签
            match = assignments[img_idx]
            label = match['label']
            geo = match['geo']
            hint = match['hint']
            
            confidence, level = calculate_confidence(
                scores_matrix[img_idx],
                geo,
                hint,
                valid_views=views
            )
            
            results.append(MatchResult(
                label=label,
                confidence=confidence,
                needs_review=(level in ['low', 'very_low']),
                raw_scores=raw_scores,
                match_method='greedy' if num_imgs == num_labels else 'greedy_partial'
            ))
        else:
            # 未分配到标签，使用模型最高分视角
            best_view_idx = int(np.argmax(scores_matrix[img_idx]))
            best_view_name = views[best_view_idx]
            
            confidence, level = calculate_confidence(
                scores_matrix[img_idx],
                best_view_name,
                valid_views=views
            )
            
            results.append(MatchResult(
                label=best_view_name,
                confidence=confidence,
                needs_review=True,  # 未匹配到预期标签，需要审核
                raw_scores=raw_scores,
                match_method='greedy_excess'
            ))
    
    return results


def predict_projection_globally(
    images: List[str], 
    excel_labels: List[str], 
    body_part: str, 
    api_call_func
) -> List[MatchResult]:
    """
    主入口函数 - 改进版，支持置信度评估
    
    Args:
        images: 图片路径列表
        excel_labels: Excel中提取并标准化的英文标签列表
        body_part: 部位名称
        api_call_func: call_medgemma函数引用
      
    Returns:
        MatchResult列表，包含置信度和审核标记
    """
    num_imgs = len(images)
    is_breast = is_breast_body_part(body_part)
    
    # 根据部位类型选择合适的视角和Prompt
    if is_breast:
        # 乳房部位：使用乳腺X光专用视角
        valid_views = BREAST_VIEWS
        prompt = (
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
    else:
        # 非乳房部位：使用标准几何视角
        valid_views = STANDARD_VIEWS
        prompt = (
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
    
    response = api_call_func(images, prompt, [])
    
    # 解析分数矩阵
    scores_matrix = parse_scores_from_response(response, num_imgs, valid_views=valid_views)
    
    # 如果Excel标签为空，退化为贪心算法
    if not excel_labels:
        results = []
        for i in range(num_imgs):
            best_view_idx = int(np.argmax(scores_matrix[i]))
            best_view_name = valid_views[best_view_idx]
            
            confidence, level = calculate_confidence(
                scores_matrix[i], 
                best_view_name,
                valid_views=valid_views
            )
            
            raw_scores = {valid_views[j]: float(scores_matrix[i, j]) for j in range(len(valid_views))}
            
            results.append(MatchResult(
                label=best_view_name,
                confidence=confidence,
                needs_review=(level in ['low', 'very_low', 'medium']),
                raw_scores=raw_scores,
                match_method='greedy'
            ))
        return results
    
    # 使用改进的贪心算法
    results = solve_assignment_with_confidence(
        scores_matrix, 
        excel_labels, 
        images,
        valid_views=valid_views,
        is_breast=is_breast
    )
    return results


# 向后兼容的旧接口
def solve_assignment(scores_matrix: np.ndarray, expected_labels: List[str]) -> List[str]:
    """
    向后兼容的简化接口，仅返回标签列表。
    注意：此接口不支持置信度评估，建议迁移到 predict_projection_globally
    """
    num_imgs = scores_matrix.shape[0]
    # 创建虚拟路径列表
    dummy_paths = [f"image_{i}.png" for i in range(num_imgs)]
    results = solve_assignment_with_confidence(scores_matrix, expected_labels, dummy_paths)
    return [r.label for r in results]
