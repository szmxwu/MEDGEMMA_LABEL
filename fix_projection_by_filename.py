#!/usr/bin/env python3
"""
根据文件名中的关键词自动修正 final_projection，并标记 reviewed=True

使用方法:
    python fix_projection_by_filename.py [--dry-run] [--input FILE]

参数:
    --dry-run: 预览修改，不保存文件
    --input: 输入文件路径 (默认: processed_labels_v3.xlsx)
"""

import pandas as pd
import re
import os
import argparse
from pathlib import Path
from datetime import datetime

# 文件名关键词到 final_projection 的映射
# 参考 projection_matcher.py 中的 hints
PROJECTION_KEYWORDS = {
    # lateral 侧位
    'lateral': ['lat', '_lat_', 'lat.', 'lateral', '侧位', 'lat_'],
    
    # frontal 正位 (AP/PA)
    'frontal': ['ap', '_ap_', 'ap.', 'pa', '_pa_', 'pa.', 'frontal', '正位', 'ap_', 'pa_'],
    
    # oblique 斜位
    'oblique': ['ob', '_ob_', 'ob.', 'oblique', '斜位', 'ob_'],
    
    # axial 轴位
    'axial': ['axi', 'axi.', 'axial', '轴位', 'axi_'],
}

# 乳房X光专用关键词 (如果 original_part 包含乳房/乳腺/Breast)
BREAST_KEYWORDS = {
    'cephalocaudal': ['cc', '_cc_', 'cc.', 'cephalocaudal'],
    'axillary tail': ['mlo', '_mlo_', 'mlo.', 'axillary', 'tail'],
    'spot compression': ['spot', '_spot_', 'spot.', 'compression'],
}

def detect_projection_by_filename(filename, is_breast=False):
    """
    根据文件名检测拍摄体位
    
    Args:
        filename: 文件名
        is_breast: 是否为乳房/乳腺部位
        
    Returns:
        (projection, matched_keyword) 或 (None, None)
    """
    filename_lower = filename.lower()
    
    # 乳房部位使用专用关键词
    keywords_dict = BREAST_KEYWORDS if is_breast else PROJECTION_KEYWORDS
    
    matches = []
    for projection, keywords in keywords_dict.items():
        for keyword in keywords:
            if keyword in filename_lower:
                # 计算匹配质量（优先匹配更精确的关键词）
                # 例如：_lat_ 比 lat 更精确
                score = len(keyword) if keyword.startswith('_') else len(keyword) * 0.5
                matches.append((projection, keyword, score))
    
    if not matches:
        return None, None
    
    # 按匹配质量排序，返回最佳匹配
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[0][0], matches[0][1]

def is_breast_body_part(body_part):
    """检查是否为乳房/乳腺部位"""
    if not body_part:
        return False
    breast_keywords = ['乳房', '乳腺', 'breast', 'mammary', 'mammo']
    return any(kw in str(body_part).lower() for kw in breast_keywords)

def main():
    parser = argparse.ArgumentParser(description='根据文件名关键词自动修正 final_projection')
    parser.add_argument('--input', '-i', default='processed_labels_v3.xlsx', 
                        help='输入Excel文件路径 (默认: processed_labels_v3.xlsx)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='预览模式，不保存修改')
    parser.add_argument('--backup', '-b', action='store_true', default=True,
                        help='修改前创建备份 (默认: True)')
    args = parser.parse_args()
    
    excel_path = Path(args.input)
    if not excel_path.exists():
        print(f"❌ 文件不存在: {excel_path}")
        return
    
    print(f"📂 正在读取: {excel_path}")
    df = pd.read_excel(excel_path)
    
    # 确保必要列存在
    if 'final_projection' not in df.columns:
        df['final_projection'] = ''
    if 'reviewed' not in df.columns:
        df['reviewed'] = False
    if 'original_part' not in df.columns:
        df['original_part'] = ''
    if 'modified_at' not in df.columns:
        df['modified_at'] = ''
    if 'projection_modified' not in df.columns:
        df['projection_modified'] = False
    
    total_rows = len(df)
    modified_count = 0
    modified_rows = []
    
    print(f"📊 总行数: {total_rows}")
    print("🔍 开始分析文件名...")
    
    for idx, row in df.iterrows():
        filename = str(row.get('filename', ''))
        original_part = str(row.get('original_part', ''))
        current_projection = str(row.get('final_projection', ''))
        is_reviewed = row.get('reviewed', False)
        
        # 检测是否为乳房部位
        is_breast = is_breast_body_part(original_part)
        
        # 跳过乳腺样本，不进行自动修正
        if is_breast:
            continue
        
        # 根据文件名检测体位（非乳腺部位）
        detected_projection, matched_keyword = detect_projection_by_filename(filename, is_breast=False)
        
        if detected_projection:
            # 如果检测到的体位与当前不同，或者当前未设置
            if detected_projection != current_projection:
                modified_rows.append({
                    'index': idx,
                    'filename': filename,
                    'original_part': original_part,
                    'old_projection': current_projection,
                    'new_projection': detected_projection,
                    'matched_keyword': matched_keyword,
                    'was_reviewed': is_reviewed
                })
                
                # 更新数据
                df.at[idx, 'final_projection'] = detected_projection
                # df.at[idx, 'reviewed'] = True
                df.at[idx, 'projection_modified'] = True
                df.at[idx, 'modified_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                modified_count += 1
    
    # 显示结果
    print(f"\n{'='*80}")
    print(f"📈 修改统计")
    print(f"{'='*80}")
    print(f"总样本数: {total_rows}")
    print(f"修改数量: {modified_count}")
    print(f"修改比例: {modified_count/total_rows*100:.2f}%")
    
    if modified_rows:
        print(f"\n{'='*80}")
        print(f"📝 修改详情 (前20条)")
        print(f"{'='*80}")
        print(f"{'文件名':<40} {'部位':<10} {'原标签':<15} {'新标签':<15} {'关键词'}")
        print("-"*80)
        
        for row in modified_rows[:20]:
            filename_short = row['filename'][:38]
            print(f"{filename_short:<40} {row['original_part']:<10} {str(row['old_projection']):<15} {row['new_projection']:<15} {row['matched_keyword']}")
        
        if len(modified_rows) > 20:
            print(f"\n... 还有 {len(modified_rows) - 20} 条修改 ...")
    
    # 保存或预览
    if args.dry_run:
        print(f"\n{'='*80}")
        print("🔍 预览模式，未保存修改")
        print(f"{'='*80}")
        print(f"如需实际修改，请运行: python {__file__} --input {args.input}")
    else:
        # 创建备份
        if args.backup:
            backup_path = excel_path.with_suffix(f'.xlsx.bak_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            df_original = pd.read_excel(excel_path)
            df_original.to_excel(backup_path, index=False)
            print(f"\n💾 已创建备份: {backup_path}")
        
        # 保存修改
        df.to_excel(excel_path, index=False)
        print(f"\n✅ 已保存修改到: {excel_path}")
        print(f"   共修改 {modified_count} 行")

if __name__ == '__main__':
    main()
