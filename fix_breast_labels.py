"""
乳房标签数据修复脚本

问题：已完成的标注中，乳房部位的体位标签使用了通用视角(frontal/lateral/oblique/axial)
      而不是乳腺X光专用视角(cephalocaudal/mediolateral oblique/spot compression)

修复逻辑：
1. 读取 processed_labels_v3.xlsx
2. 筛选出 original_part == "乳房" 的样本
3. 进一步筛选出 final_projection 为通用视角（frontal/lateral/oblique/axial/left/right/bilateral）的样本
4. 对这些样本重新调用 LLM 进行打标
5. 更新 Excel 文件

使用方法：
    python fix_breast_labels.py [--dry-run] [--limit N]
    
参数：
    --dry-run: 只查看需要修复的样本，不实际执行修复
    --limit N: 限制只处理前 N 个样本（用于测试）
"""

import os
import sys
import json
import time
import glob
import argparse
import base64
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# 导入项目模块
import projection_matcher
from projection_matcher import MatchResult
# 数据目录配置
IMAGE_ROOT = "data"


def get_image_paths(image_id: str) -> List[str]:
    """获取指定影像号的所有图片路径"""
    img_dir = os.path.join(IMAGE_ROOT, str(image_id))
    if not os.path.isdir(img_dir):
        return []
    return sorted(glob.glob(os.path.join(img_dir, "*.png")))

# API 配置
BASE_URL = "https://smartlab.cse.ust.hk/smartcare/api/shebd/medgemma15"
API_URL = f"{BASE_URL}/v1/chat/completions"
MODEL_ID = "/data/shebd/0_Pretrained/medgemma-1.5-4b-it"

# 修复日志
LOG_FILE = "logs/breast_fix_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
os.makedirs("logs", exist_ok=True)


def log_message(msg: str, level: str = "INFO"):
    """记录日志到文件和控制台"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {msg}"
    print(log_line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def find_breast_samples_to_fix(df: pd.DataFrame) -> pd.DataFrame:
    """
    筛选需要修复的乳房样本
    
    修复条件：
    1. original_part == "乳房"
    2. final_projection 为通用视角（frontal/lateral/oblique/axial/left/right/bilateral）或空值
    """
    # 筛选乳房样本
    breast_df = df[df['original_part'] == '乳房'].copy()
    
    if len(breast_df) == 0:
        return breast_df
    
    # 错误的通用视角标签
    wrong_labels = ['frontal', 'lateral', 'oblique', 'axial', 'left', 'right', 'bilateral', 'unknown']
    
    # 筛选出 final_projection 为错误标签或为空的样本
    mask = (
        breast_df['final_projection'].isna() | 
        breast_df['final_projection'].str.lower().isin(wrong_labels)
    )
    
    to_fix = breast_df[mask].copy()
    
    return to_fix


def find_breast_samples_by_filename(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据文件名识别乳房样本，这些样本可以通过文件名直接修复
    
    文件名模式：
    - _L_CC -> left, cephalocaudal
    - _R_CC -> right, cephalocaudal
    - _L_MLO -> left, mediolateral oblique
    - _R_MLO -> right, mediolateral oblique
    """
    breast_df = df[df['original_part'] == '乳房'].copy()
    
    if len(breast_df) == 0:
        return breast_df
    
    # 筛选文件名包含特定关键词的样本
    filename_patterns = ['_L_CC', '_R_CC', '_L_MLO', '_R_MLO']
    mask = breast_df['filename'].str.upper().apply(
        lambda x: any(p in x for p in filename_patterns)
    )
    
    return breast_df[mask].copy()


def fix_single_sample(
    image_id: str, 
    filename: str, 
    img_paths: List[str]
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    对单个样本重新打标
    
    Returns:
        (results, error): results为包含打标结果的字典列表，error为错误信息
    """
    num_imgs = len(img_paths)
    if num_imgs == 0:
        return None, "无图片文件"
    
    try:
        # 首先尝试基于文件名打标
        results = []
        needs_llm_indices = []
        
        for i, img_path in enumerate(img_paths):
            fname = os.path.basename(img_path).upper()
            
            if "_L_CC" in fname:
                results.append({
                    'image_id': image_id,
                    'filename': os.path.basename(img_path),
                    'original_part': '乳房',
                    'final_body_part': 'Breast',
                    'final_orientation': 'left',
                    'final_projection': 'cephalocaudal',
                    'confidence_projection': 0.95,
                    'confidence_orientation': 0.95,
                    'confidence_overall': 0.95,
                    'needs_review': False,
                    'review_reason': '',
                    'match_method': 'filename_breast_fixed',
                    'raw_scores_frontal': 0,
                    'raw_scores_lateral': 0,
                    'raw_scores_oblique': 0,
                    'raw_scores_axial': 0,
                    'raw_excel_pos': None,
                    'raw_excel_proj': None
                })
            elif "_R_CC" in fname:
                results.append({
                    'image_id': image_id,
                    'filename': os.path.basename(img_path),
                    'original_part': '乳房',
                    'final_body_part': 'Breast',
                    'final_orientation': 'right',
                    'final_projection': 'cephalocaudal',
                    'confidence_projection': 0.95,
                    'confidence_orientation': 0.95,
                    'confidence_overall': 0.95,
                    'needs_review': False,
                    'review_reason': '',
                    'match_method': 'filename_breast_fixed',
                    'raw_scores_frontal': 0,
                    'raw_scores_lateral': 0,
                    'raw_scores_oblique': 0,
                    'raw_scores_axial': 0,
                    'raw_excel_pos': None,
                    'raw_excel_proj': None
                })
            elif "_L_MLO" in fname:
                results.append({
                    'image_id': image_id,
                    'filename': os.path.basename(img_path),
                    'original_part': '乳房',
                    'final_body_part': 'Breast',
                    'final_orientation': 'left',
                    'final_projection': 'mediolateral oblique',
                    'confidence_projection': 0.95,
                    'confidence_orientation': 0.95,
                    'confidence_overall': 0.95,
                    'needs_review': False,
                    'review_reason': '',
                    'match_method': 'filename_breast_fixed',
                    'raw_scores_frontal': 0,
                    'raw_scores_lateral': 0,
                    'raw_scores_oblique': 0,
                    'raw_scores_axial': 0,
                    'raw_excel_pos': None,
                    'raw_excel_proj': None
                })
            elif "_R_MLO" in fname:
                results.append({
                    'image_id': image_id,
                    'filename': os.path.basename(img_path),
                    'original_part': '乳房',
                    'final_body_part': 'Breast',
                    'final_orientation': 'right',
                    'final_projection': 'mediolateral oblique',
                    'confidence_projection': 0.95,
                    'confidence_orientation': 0.95,
                    'confidence_overall': 0.95,
                    'needs_review': False,
                    'review_reason': '',
                    'match_method': 'filename_breast_fixed',
                    'raw_scores_frontal': 0,
                    'raw_scores_lateral': 0,
                    'raw_scores_oblique': 0,
                    'raw_scores_axial': 0,
                    'raw_excel_pos': None,
                    'raw_excel_proj': None
                })
            else:
                # 文件名不匹配，需要LLM处理
                needs_llm_indices.append(i)
                results.append(None)  # 占位
        
        # 如果有需要LLM处理的图片
        if needs_llm_indices:
            log_message(f"  影像号 {image_id}: {len(needs_llm_indices)}/{num_imgs} 张图片需要LLM处理")
            
            llm_images = [img_paths[i] for i in needs_llm_indices]
            
            # 使用乳房专用投影匹配
            match_results: List[MatchResult] = projection_matcher.predict_projection_globally(
                images=llm_images,
                excel_labels=[],  # 自由分类
                body_part="breast",
                api_call_func=call_medgemma
            )
            
            # 填充LLM结果
            for idx, match_result in enumerate(match_results):
                i = needs_llm_indices[idx]
                results[i] = {
                    'image_id': image_id,
                    'filename': os.path.basename(img_paths[i]),
                    'original_part': '乳房',
                    'final_body_part': 'Breast',
                    'final_orientation': 'bilateral',  # 自由分类时方位设为bilateral
                    'final_projection': match_result.label,
                    'confidence_projection': round(match_result.confidence, 3),
                    'confidence_orientation': 0.5,
                    'confidence_overall': round(match_result.confidence * 0.5 + 0.5 * 0.5, 3),
                    'needs_review': match_result.needs_review,
                    'review_reason': 'breast_relabeled' if match_result.needs_review else '',
                    'match_method': f"llm_breast_fixed_{match_result.match_method}",
                    'raw_scores_frontal': match_result.raw_scores.get('frontal', 0),
                    'raw_scores_lateral': match_result.raw_scores.get('lateral', 0),
                    'raw_scores_oblique': match_result.raw_scores.get('oblique', 0),
                    'raw_scores_axial': match_result.raw_scores.get('axial', 0),
                    'raw_excel_pos': None,
                    'raw_excel_proj': None
                }
        
        return results, None
        
    except Exception as e:
        return None, str(e)


def update_excel_with_fixed_data(
    original_df: pd.DataFrame, 
    fixed_results: List[Dict],
    output_path: str
):
    """
    将修复后的数据更新回Excel
    
    策略：
    1. 根据 image_id 和 filename 定位记录
    2. 更新 final_projection 等相关字段
    3. 保留其他字段不变
    """
    df = original_df.copy()
    
    # 创建 (image_id, filename) -> index 的映射
    key_to_idx = {}
    for idx, row in df.iterrows():
        key = (row['image_id'], row['filename'])
        key_to_idx[key] = idx
    
    # 更新数据
    updated_count = 0
    for result in fixed_results:
        key = (result['image_id'], result['filename'])
        if key in key_to_idx:
            idx = key_to_idx[key]
            for col, val in result.items():
                if col in df.columns:
                    df.at[idx, col] = val
            updated_count += 1
    
    # 保存
    df.to_excel(output_path, index=False)
    log_message(f"已更新 {updated_count} 条记录到 {output_path}")
    
    return updated_count


def main():
    parser = argparse.ArgumentParser(description='修复乳房标签数据')
    parser.add_argument('--dry-run', action='store_true', 
                        help='只查看需要修复的样本，不实际执行修复')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制只处理前 N 个样本（用于测试）')
    parser.add_argument('--input', type=str, default='processed_labels_v3.xlsx',
                        help='输入Excel文件路径')
    parser.add_argument('--output', type=str, default='processed_labels_v3_fixed.xlsx',
                        help='输出Excel文件路径')
    args = parser.parse_args()
    
    log_message("=" * 60)
    log_message("乳房标签数据修复脚本启动")
    log_message(f"输入文件: {args.input}")
    log_message(f"输出文件: {args.output}")
    log_message(f"模式: {'模拟运行' if args.dry_run else '实际修复'}")
    log_message("=" * 60)
    
    # 读取数据
    log_message(f"读取 {args.input}...")
    df = pd.read_excel(args.input)
    log_message(f"总计 {len(df)} 条记录")
    
    # 筛选需要修复的样本
    to_fix = find_breast_samples_to_fix(df)
    log_message(f"发现 {len(to_fix)} 条需要修复的乳房记录")
    
    if len(to_fix) == 0:
        log_message("无需修复，退出")
        return
    
    # 查看当前错误分布
    log_message("\n当前错误标签分布:")
    log_message(to_fix['final_projection'].value_counts(dropna=False).to_string())
    
    # 查看文件名匹配情况
    filename_fixable = find_breast_samples_by_filename(df)
    log_message(f"\n其中可通过文件名直接修复: {len(filename_fixable)} 条")
    
    if args.dry_run:
        log_message("\n[模拟运行] 列出前10条需要修复的记录:")
        for idx, row in to_fix.head(10).iterrows():
            log_message(f"  - {row['image_id']}: {row['filename']} -> 当前标签: {row['final_projection']}")
        log_message("\n使用 --dry-run 查看，实际修复请去掉此参数")
        return
    
    # 限制处理数量
    if args.limit:
        to_fix = to_fix.head(args.limit)
        log_message(f"\n限制处理前 {args.limit} 条记录")
    
    # 按 image_id 分组处理
    image_ids = to_fix['image_id'].unique()
    log_message(f"\n开始处理 {len(image_ids)} 个影像号...")
    
    all_fixed_results = []
    success_count = 0
    fail_count = 0
    
    for i, img_id in enumerate(image_ids):
        log_message(f"\n[{i+1}/{len(image_ids)}] 处理影像号: {img_id}")
        
        # 获取该影像号的所有图片
        img_paths = get_image_paths(img_id)
        if not img_paths:
            log_message(f"  警告: 未找到图片文件", "WARN")
            fail_count += 1
            continue
        
        # 获取该影像号在to_fix中的filename列表
        fix_files = to_fix[to_fix['image_id'] == img_id]['filename'].tolist()
        log_message(f"  找到 {len(img_paths)} 张图片，需要修复 {len(fix_files)} 条记录")
        
        # 执行修复
        results, error = fix_single_sample(img_id, None, img_paths)
        
        if error:
            log_message(f"  修复失败: {error}", "ERROR")
            fail_count += 1
            continue
        
        # 只保留需要修复的那些文件的结果
        for r in results:
            if r['filename'] in fix_files:
                all_fixed_results.append(r)
                log_message(f"  修复: {r['filename']} -> {r['final_projection']} "
                           f"(置信度={r['confidence_projection']:.2f})")
        
        success_count += 1
        
        # 请求间隔，避免触发限流
        time.sleep(1.5)
    
    # 更新Excel
    log_message("\n" + "=" * 60)
    log_message(f"处理完成: 成功 {success_count}, 失败 {fail_count}")
    log_message(f"生成 {len(all_fixed_results)} 条修复记录")
    
    if all_fixed_results:
        updated = update_excel_with_fixed_data(df, all_fixed_results, args.output)
        log_message(f"已更新 {updated} 条记录到 {args.output}")
        
        # 生成修复报告
        report_path = args.output.replace('.xlsx', '_report.json')
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_file': args.input,
            'output_file': args.output,
            'total_samples': len(df),
            'breast_samples_to_fix': len(to_fix),
            'success_count': success_count,
            'fail_count': fail_count,
            'fixed_records': len(all_fixed_results),
            'fixed_details': all_fixed_results
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log_message(f"修复报告已保存: {report_path}")
    
    log_message("=" * 60)


if __name__ == "__main__":
    main()
