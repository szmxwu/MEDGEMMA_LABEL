#!/usr/bin/env python3
"""
检查标注文件和图像是否匹配，验证图片是否存在
"""

import pandas as pd
from pathlib import Path
import os

def check_images(excel_path='processed_labels_v3.xlsx', data_dir='data'):
    """检查Excel中引用的图片是否都存在"""
    
    print(f"📂 读取Excel文件: {excel_path}")
    df = pd.read_excel(excel_path)
    
    print(f"📊 总记录数: {len(df)}")
    print(f"📁 检查数据目录: {data_dir}")
    
    # 统计
    total_images = 0
    missing_images = []
    exists_images = 0
    
    # 检查每个样本的图片
    for idx, row in df.iterrows():
        image_id = str(row.get('image_id', ''))
        filename = str(row.get('filename', ''))
        
        if not image_id or not filename:
            continue
            
        total_images += 1
        
        # 构建图片路径
        img_path = Path(data_dir) / image_id / filename
        
        if not img_path.exists():
            missing_images.append({
                'index': idx,
                'image_id': image_id,
                'filename': filename,
                'expected_path': str(img_path)
            })
        else:
            exists_images += 1
        
        # 每1000条打印进度
        if idx > 0 and idx % 1000 == 0:
            print(f"   已检查 {idx}/{len(df)} 条...")
    
    print(f"\n{'='*60}")
    print("📈 检查结果")
    print(f"{'='*60}")
    print(f"总记录数: {len(df)}")
    print(f"有效图片引用: {total_images}")
    print(f"✅ 存在的图片: {exists_images}")
    print(f"❌ 缺失的图片: {len(missing_images)}")
    
    if missing_images:
        print(f"\n{'='*60}")
        print("❌ 缺失图片列表 (前20条)")
        print(f"{'='*60}")
        for item in missing_images[:20]:
            print(f"  行{item['index']}: {item['image_id']}/{item['filename']}")
        
        if len(missing_images) > 20:
            print(f"\n  ... 还有 {len(missing_images) - 20} 条缺失 ...")
    
    return missing_images

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='检查标注文件和图像是否匹配')
    parser.add_argument('--input', '-i', default='processed_labels_v3.xlsx',
                        help='输入Excel文件路径')
    parser.add_argument('--data-dir', '-d', default='data',
                        help='图像数据目录')
    
    args = parser.parse_args()
    
    check_images(args.input, args.data_dir)
