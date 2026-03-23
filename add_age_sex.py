import json
import os
import glob
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import ast
import json as _json
from tqdm import tqdm
import re
from collections import Counter
import random

MAX_SAMPLES_PER_PART = 800
warnings.filterwarnings("ignore")

# project repo root (two levels up from this file)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'output'


def merge_excel_files(corpus_dir: str = None) -> pd.DataFrame:
    """Read and merge Excel files named like all_data_match*.xlsx under corpus_dir.

    If `corpus_dir` is None, resolve it relative to the project root based on this
    script's location (two levels up -> project root, then `radiology_data`). This
    makes the function behave the same regardless of current working directory.

    Optimizations:
    - Only read needed columns to reduce memory and parsing time.
    - Force '影像号' to string and strip whitespace so lookups match JSON values.
    - Drop duplicate '影像号' keeping the first occurrence.
    """
    # resolve default corpus_dir relative to this script to avoid cwd-dependency
    if corpus_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        corpus_dir = str(repo_root / 'Radiology_Entities/radiology_data')
    pattern = os.path.join(corpus_dir, "all_data_match*.xlsx")
    excel_files = glob.glob(pattern)

    if not excel_files:
        raise ValueError(f"未找到任何符合模式的Excel文件: {pattern}")

    print(f"📂 发现 {len(excel_files)} 个数据文件: {excel_files}")
    
    # 只读取需要的列，减少内存占用
    usecols = ['影像号', '年龄', '性别']
    dataframes = []
    for file in excel_files:
        try:
            print(f"正在读取文件: {file} ...")
            # 对于大文件，使用dtype和只读必要列
            df = pd.read_excel(
                file, 
                dtype={'影像号': str},
                usecols=usecols
            )
            # 去除影像号空白并去重
            df['影像号'] = df['影像号'].astype(str).str.strip()
            df = df.drop_duplicates(subset=['影像号'], keep='first')
            dataframes.append(df)
            print(f"✓ 读取完成: {len(df)} 条唯一记录")
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    if not dataframes:
        raise ValueError("未找到任何有效的Excel文件（或读取失败）")

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"✅ 合并完成: 共 {len(combined_df)} 条唯一影像号记录")
    return combined_df


def get_years_int(age_str: str):
    """Parse various age string formats and return years according to rules:

    - If a year value appears (e.g. '10岁', '10 years', '1y'), return the integer number of years (truncate if necessary).
    - Else if months appear (e.g. '1 M', '3个月', '1月3天'), return months/12 as a float.
    - Else if only weeks/days appear (e.g. '1 Day', '1 天'), return 0.
    - If empty/invalid, return None.

    Examples:
    - '1 M' or '1月3天' -> 1/12 (≈0.0833333)
    - '1 Day' or '1 天' -> 0
    - '10岁3个月' or '1 years 3 month' -> 10
    """
    if age_str is None:
        return None
    s = str(age_str).strip()
    if s == '' or s.lower() in {'nan', 'none', 'null'}:
        return None

    low = s.lower()

    # Check for year patterns first (explicit year units)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:y|yr|yrs|year|years|年|岁|歲)", low, flags=re.I)
    if m:
        try:
            y = float(m.group(1))
            return int(y)
        except Exception:
            pass

    # If no explicit years, check months (may appear as '1 M', '1月', '个月')
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|mo|mos|month|months|月|个月)", low, flags=re.I)
    if m:
        try:
            months = float(m.group(1))
            return months / 12.0
        except Exception:
            pass

    # Weeks or days -> treat as 0 years
    if re.search(r"\d+\s*(?:w|wk|wks|week|weeks|周|星期)", low, flags=re.I) or \
       re.search(r"\d+\s*(?:d|day|days|天|日)", low, flags=re.I):
        return 0

    # Fallback: if the string is purely digits, assume years
    m = re.match(r"^\s*(\d+)\s*$", s)
    if m:
        return int(m.group(1))

    # As a last resort, try to extract the first number and treat it as years
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))

    return None


def categorize_age(age):
    """将年龄数值分类为 child/adult/elderly
    
    Args:
        age: 年龄（年）
        
    Returns:
        'child' | 'adult' | 'elderly' | None
    """
    if age is None or pd.isna(age):
        return None
    try:
        a = float(age)
        if a < 18:
            return 'child'
        elif a >= 65:
            return 'elderly'
        else:
            return 'adult'
    except Exception:
        return None


def normalize_gender(val):
    """标准化性别值为 Female/Male"""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in {'男', 'm', 'male', 'man', '1', '先生'}:
        return 'Male'
    if s in {'女', 'f', 'female', 'woman', '0', '女士', '小姐'}:
        return 'Female'
    return None


def create_lookup_dict(combined_df):
    """创建高效的查找字典
    
    Returns:
        dict: {影像号: (age_category, sex)}
    """
    print("正在创建查找字典...")
    lookup = {}
    
    # 预处理年龄和性别
    combined_df['年龄数值'] = combined_df['年龄'].apply(get_years_int)
    combined_df['年龄分类'] = combined_df['年龄数值'].apply(categorize_age)
    combined_df['性别标准'] = combined_df['性别'].apply(normalize_gender)
    
    # 构建字典
    for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="构建索引"):
        image_id = str(row['影像号']).strip()
        age_cat = row['年龄分类']
        sex = row['性别标准']
        lookup[image_id] = (age_cat, sex)
    
    print(f"✅ 查找字典创建完成: {len(lookup)} 条记录")
    return lookup


def add_age_sex_to_processed(processed_file: str, combined_df: pd.DataFrame, output_file: str = None):
    """为 processed_labels_v3.xlsx 添加 age 和 sex 列
    
    Args:
        processed_file: 输入文件路径
        combined_df: 包含年龄性别的大表
        output_file: 输出文件路径（默认覆盖原文件）
    """
    print(f"\n📂 正在读取处理后的标签文件: {processed_file}")
    df = pd.read_excel(processed_file)
    print(f"✅ 读取完成: {len(df)} 条记录")
    
    # 创建查找字典（更高效的匹配方式）
    lookup = create_lookup_dict(combined_df)
    
    # 初始化新列
    df['age'] = None
    df['sex'] = None
    
    # 统计
    matched_count = 0
    unmatched_count = 0
    age_stats = Counter()
    sex_stats = Counter()
    
    print(f"\n🔍 开始匹配年龄和性别...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="匹配进度"):
        image_id = str(row.get('image_id', '')).strip()
        
        if image_id in lookup:
            age_cat, sex = lookup[image_id]
            df.at[idx, 'age'] = age_cat
            df.at[idx, 'sex'] = sex
            matched_count += 1
            
            if age_cat:
                age_stats[age_cat] += 1
            if sex:
                sex_stats[sex] += 1
        else:
            unmatched_count += 1
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"📊 匹配统计")
    print(f"{'='*60}")
    print(f"总记录数: {len(df)}")
    print(f"匹配成功: {matched_count} ({matched_count/len(df)*100:.2f}%)")
    print(f"匹配失败: {unmatched_count} ({unmatched_count/len(df)*100:.2f}%)")
    
    print(f"\n年龄分布:")
    for age_cat, count in sorted(age_stats.items()):
        print(f"  {age_cat}: {count} ({count/len(df)*100:.2f}%)")
    
    print(f"\n性别分布:")
    for sex, count in sorted(sex_stats.items()):
        print(f"  {sex}: {count} ({count/len(df)*100:.2f}%)")
    
    # 保存结果
    if output_file is None:
        output_file = processed_file
    
    # 创建备份
    backup_file = str(output_file).replace(".xlsx","_bak.xlsx") 
    print(f"\n💾 正在保存...")
    print(f"   备份: {backup_file}")
    df.to_excel(backup_file, index=False)
    
    print(f"   主文件: {output_file}")
    df.to_excel(output_file, index=False)
    
    print(f"\n✅ 完成！已添加 age 和 sex 列")
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='为 processed_labels_v3.xlsx 添加 age 和 sex 标签')
    parser.add_argument('--input', '-i', default='processed_labels_v3.xlsx',
                        help='输入文件路径 (默认: processed_labels_v3.xlsx)')
    parser.add_argument('--output', '-o', default=None,
                        help='输出文件路径 (默认覆盖输入文件)')
    parser.add_argument('--corpus-dir', '-d', default=None,
                        help='包含 all_data_match*.xlsx 文件的目录')
    
    args = parser.parse_args()
    
    # 读取大数据文件（合并多个 all_data_match*.xlsx）
    combined_df = merge_excel_files(args.corpus_dir)
    
    # 执行匹配和添加标签
    add_age_sex_to_processed(args.input, combined_df, args.output)
