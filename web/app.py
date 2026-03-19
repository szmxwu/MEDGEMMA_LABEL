#!/usr/bin/env python3
"""
Flask Web服务器 - 用于预览和人工复核X光标注结果
支持离线运行，所有静态资源已本地化
"""

import os
import sys
import webbrowser
import threading
import time
import shutil
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

EXCEL_LOCK = threading.Lock()


def safe_save_excel(df, excel_path, create_backup=True):
    """
    原子方式保存Excel文件，防止写入过程中断导致文件损坏
    
    策略:
    1. 写入临时文件
    2. 创建备份（可选）
    3. 原子重命名替换原文件
    4. 清理临时文件
    
    Args:
        df: pandas DataFrame
        excel_path: 目标文件路径
        create_backup: 是否创建备份文件
    
    Raises:
        Exception: 保存失败时抛出异常
    """
    excel_path = Path(excel_path)
    temp_path = excel_path.with_suffix('.tmp')
    backup_path = excel_path.with_suffix('.xlsx.bak')
    
    try:
        # 1. 写入临时文件
        df.to_excel(temp_path, index=False)
        
        # 验证临时文件写入成功
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise IOError("临时文件写入失败或为空")
        
        # 2. 创建备份（如果原文件存在）
        if create_backup and excel_path.exists():
            shutil.copy2(excel_path, backup_path)
        
        # 3. 原子重命名（操作系统保证这是原子操作）
        temp_path.replace(excel_path)
        
        # 4. 成功后保留最近10个备份，删除旧的
        if create_backup:
            cleanup_old_backups(excel_path.parent, keep_count=10)
            
    except Exception as e:
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        
        # 如果原文件丢失但备份存在，尝试恢复
        if not excel_path.exists() and backup_path.exists():
            try:
                backup_path.rename(excel_path)
                print(f"⚠️  已从备份恢复: {excel_path}")
            except Exception as restore_error:
                print(f"❌ 恢复备份失败: {restore_error}")
        
        raise Exception(f"保存Excel失败: {str(e)}")


def cleanup_old_backups(directory, keep_count=10):
    """清理旧备份文件，只保留最近的指定数量"""
    try:
        backup_files = sorted(
            Path(directory).glob('*.xlsx.bak'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        for old_backup in backup_files[keep_count:]:
            try:
                old_backup.unlink()
            except:
                pass
    except:
        pass  # 清理失败不影响主流程


def log_modification(data):
    """
    记录修改操作到JSONL日志，用于灾难恢复
    即使Excel文件损坏，也能从日志重建数据
    """
    try:
        import json
        log_path = OUTPUTS_DIR / 'modifications.jsonl'
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"⚠️  记录修改日志失败: {e}")  # 日志失败不影响主流程

# 配置
DATA_DIR = Path(__file__).parent / 'data'
OUTPUTS_DIR = Path(__file__).parent / 'outputs'

# 全局状态
app_status = {
    'samples': [],
    'modified_data': {},
    'excel_path': None,
    'total_samples': 0,
    'total_images': 0,
    'review_count': 0
}


def load_data_from_excel(excel_path):
    """从Excel加载标注数据"""
    import pandas as pd
    import math
    
    if not os.path.exists(excel_path):
        return []
    
    df = pd.read_excel(excel_path)
    
    # 确保必要列存在
    if 'confidence_projection' not in df.columns:
        df['confidence_projection'] = 0.8
    if 'confidence_orientation' not in df.columns:
        df['confidence_orientation'] = 0.8
    if 'confidence_overall' not in df.columns:
        df['confidence_overall'] = 0.8
    if 'needs_review' not in df.columns:
        df['needs_review'] = False
    if 'review_reason' not in df.columns:
        df['review_reason'] = ''
    if 'match_method' not in df.columns:
        df['match_method'] = 'unknown'
    if 'original_part' not in df.columns:
        df['original_part'] = df.get('final_body_part', 'unknown')
    if 'projection_modified' not in df.columns:
        df['projection_modified'] = False
    if 'orientation_modified' not in df.columns:
        df['orientation_modified'] = False
    if 'modified_at' not in df.columns:
        df['modified_at'] = ''
    if 'reviewed' not in df.columns:
        df['reviewed'] = False
    if 'reviewed_at' not in df.columns:
        df['reviewed_at'] = ''
    if 'is_invalid' not in df.columns:
        df['is_invalid'] = False
    if 'invalid_label' not in df.columns:
        df['invalid_label'] = ''
    if 'invalid_at' not in df.columns:
        df['invalid_at'] = ''
    
    # 按image_id分组
    grouped = df.groupby('image_id')
    samples = []

    def safe_str(v, default=''):
        if pd.isna(v):
            return default
        return str(v)

    def safe_bool(v, default=False):
        if pd.isna(v):
            return default
        return bool(v)
    
    for image_id, group in grouped:
        valid_group = group[~group['is_invalid'].fillna(False)]
        if valid_group.empty:
            continue

        sample = {
            'image_id': str(image_id),
            'original_part': safe_str(valid_group['original_part'].iloc[0], 'unknown'),
            'needs_review': safe_bool(valid_group['needs_review'].any()),
            'images': []
        }
        
        for _, row in valid_group.iterrows():
            img_path = f"/api/image/{image_id}/{row['filename']}"
            # 处理可能的 NaN 值，确保 JSON 可序列化（浏览器端 JSON.parse 无法解析 NaN）
            def safe_float(v, default=0.0):
                try:
                    if pd.isna(v):
                        return default
                    fv = float(v)
                    if math.isfinite(fv):
                        return fv
                    return default
                except Exception:
                    return default

            cp = safe_float(row.get('confidence_projection', 0.8), default=0.0)
            co = safe_float(row.get('confidence_orientation', 0.8), default=0.0)
            cov = safe_float(row.get('confidence_overall', 0.8), default=0.0)

            sample['images'].append({
                'filename': safe_str(row.get('filename', ''), ''),
                'final_body_part': safe_str(row.get('final_body_part', ''), ''),
                'final_orientation': safe_str(row.get('final_orientation', 'unknown'), 'unknown'),
                'final_projection': safe_str(row.get('final_projection', 'unknown'), 'unknown'),
                'confidence_projection': cp,
                'confidence_orientation': co,
                'confidence_overall': cov,
                'needs_review': safe_bool(row.get('needs_review', False)),
                'review_reason': safe_str(row.get('review_reason', ''), ''),
                'match_method': safe_str(row.get('match_method', ''), ''),
                'projection_modified': safe_bool(row.get('projection_modified', False)),
                'orientation_modified': safe_bool(row.get('orientation_modified', False)),
                'modified_at': safe_str(row.get('modified_at', ''), ''),
                'reviewed': safe_bool(row.get('reviewed', False)),
                'reviewed_at': safe_str(row.get('reviewed_at', ''), ''),
                'image_url': img_path
            })
        
        samples.append(sample)
    
    return samples


@app.route('/')
def index():
    """主页面"""
    return render_template('review.html')


@app.route('/api/samples')
def get_samples():
    """获取所有样本数据"""
    return jsonify(app_status['samples'])


@app.route('/api/stats')
def get_stats():
    """获取统计信息"""
    return jsonify({
        'total_samples': len(app_status['samples']),
        'total_images': sum(len(s['images']) for s in app_status['samples']),
        'review_count': sum(sum(1 for img in s['images'] if img['needs_review']) 
                             for s in app_status['samples']),
        'modified_count': len(app_status['modified_data'])
    })


@app.route('/api/save', methods=['POST'])
def save_modification():
    """保存修改"""
    data = request.json or {}
    if not app_status.get('excel_path'):
        return jsonify({'success': False, 'message': '未找到Excel数据文件'}), 500

    if not data.get('image_id') or not data.get('filename'):
        return jsonify({'success': False, 'message': '参数缺失'}), 400

    try:
        import pandas as pd

        excel_path = app_status['excel_path']
        with EXCEL_LOCK:
            df = pd.read_excel(excel_path)

            # 确保修改状态列存在
            if 'projection_modified' not in df.columns:
                df['projection_modified'] = False
            if 'orientation_modified' not in df.columns:
                df['orientation_modified'] = False
            if 'modified_at' not in df.columns:
                df['modified_at'] = ''
            if 'reviewed' not in df.columns:
                df['reviewed'] = False
            if 'reviewed_at' not in df.columns:
                df['reviewed_at'] = ''

            mask = (df['image_id'].astype(str) == str(data['image_id'])) & (df['filename'] == data['filename'])
            if not mask.any():
                return jsonify({'success': False, 'message': '未找到对应记录'}), 404

            now_str = time.strftime('%Y-%m-%d %H:%M:%S')
            if 'projection' in data:
                df.loc[mask, 'final_projection'] = data['projection']
                df.loc[mask, 'projection_modified'] = True
                df.loc[mask, 'modified_at'] = now_str
            elif 'orientation' in data:
                df.loc[mask, 'final_orientation'] = data['orientation']
                df.loc[mask, 'orientation_modified'] = True
                df.loc[mask, 'modified_at'] = now_str

            # 使用原子写入保存（防止写入中断导致文件损坏）
            safe_save_excel(df, excel_path, create_backup=True)
            
            # 记录修改日志（用于灾难恢复）
            log_modification(data)

        # 更新内存样本，保证刷新后显示一致
        for sample in app_status['samples']:
            if str(sample['image_id']) != str(data['image_id']):
                continue
            for img in sample['images']:
                if img['filename'] != data['filename']:
                    continue
                if 'projection' in data:
                    img['final_projection'] = data['projection']
                    img['projection_modified'] = True
                    img['modified_at'] = now_str
                elif 'orientation' in data:
                    img['final_orientation'] = data['orientation']
                    img['orientation_modified'] = True
                    img['modified_at'] = now_str
                break

        return jsonify({'success': True, 'message': '已保存'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500


@app.route('/api/review', methods=['POST'])
def save_review_status():
    """保存复核状态"""
    data = request.json or {}
    if not app_status.get('excel_path'):
        return jsonify({'success': False, 'message': '未找到Excel数据文件'}), 500

    image_id = data.get('image_id')
    filename = data.get('filename')
    reviewed = bool(data.get('reviewed', True))

    if not image_id or not filename:
        return jsonify({'success': False, 'message': '参数缺失'}), 400

    try:
        import pandas as pd

        excel_path = app_status['excel_path']
        with EXCEL_LOCK:
            df = pd.read_excel(excel_path)
            if 'reviewed' not in df.columns:
                df['reviewed'] = False
            if 'reviewed_at' not in df.columns:
                df['reviewed_at'] = ''

            mask = (df['image_id'].astype(str) == str(image_id)) & (df['filename'] == filename)
            if not mask.any():
                return jsonify({'success': False, 'message': '未找到对应记录'}), 404

            now_str = time.strftime('%Y-%m-%d %H:%M:%S') if reviewed else ''
            df.loc[mask, 'reviewed'] = reviewed
            df.loc[mask, 'reviewed_at'] = now_str

            # 使用原子写入保存
            safe_save_excel(df, excel_path, create_backup=True)
            
            # 记录批量修改日志
            log_modification({'type': 'bulk_review', 'items': items})

        # 更新内存样本
        for sample in app_status['samples']:
            if str(sample['image_id']) != str(image_id):
                continue
            for img in sample['images']:
                if img['filename'] != filename:
                    continue
                img['reviewed'] = reviewed
                img['reviewed_at'] = now_str
                break

        return jsonify({'success': True, 'message': '已保存'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500


@app.route('/api/review/bulk', methods=['POST'])
def save_review_bulk():
    """批量保存复核状态"""
    data = request.json or {}
    items = data.get('items', [])

    if not app_status.get('excel_path'):
        return jsonify({'success': False, 'message': '未找到Excel数据文件'}), 500

    if not items:
        return jsonify({'success': False, 'message': '没有需要更新的项目'}), 400

    try:
        import pandas as pd

        excel_path = app_status['excel_path']
        with EXCEL_LOCK:
            df = pd.read_excel(excel_path)
            if 'reviewed' not in df.columns:
                df['reviewed'] = False
            if 'reviewed_at' not in df.columns:
                df['reviewed_at'] = ''

            now_str = time.strftime('%Y-%m-%d %H:%M:%S')

            for item in items:
                image_id = item.get('image_id')
                filename = item.get('filename')
                reviewed = bool(item.get('reviewed', True))
                if not image_id or not filename:
                    continue

                mask = (df['image_id'].astype(str) == str(image_id)) & (df['filename'] == filename)
                if not mask.any():
                    continue

                df.loc[mask, 'reviewed'] = reviewed
                df.loc[mask, 'reviewed_at'] = now_str if reviewed else ''

            # 使用原子写入保存
            safe_save_excel(df, excel_path, create_backup=True)
            
            # 记录批量修改日志
            log_modification({'type': 'bulk_review', 'count': len(items)})

        # 更新内存样本
        item_set = {(str(i.get('image_id')), i.get('filename')) for i in items}
        for sample in app_status['samples']:
            for img in sample['images']:
                key = (str(sample['image_id']), img['filename'])
                if key in item_set:
                    img['reviewed'] = True
                    img['reviewed_at'] = now_str

        return jsonify({'success': True, 'message': '已保存'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500


@app.route('/api/invalidate', methods=['POST'])
def invalidate_sample_image():
    """标记图片为无效样本（软删除）"""
    data = request.json or {}
    if not app_status.get('excel_path'):
        return jsonify({'success': False, 'message': '未找到Excel数据文件'}), 500

    image_id = data.get('image_id')
    filename = data.get('filename')
    invalid_label = data.get('invalid_label', 'invalid_sample')

    if not image_id or not filename:
        return jsonify({'success': False, 'message': '参数缺失'}), 400

    try:
        import pandas as pd

        excel_path = app_status['excel_path']
        with EXCEL_LOCK:
            df = pd.read_excel(excel_path)
            if 'is_invalid' not in df.columns:
                df['is_invalid'] = False
            if 'invalid_label' not in df.columns:
                df['invalid_label'] = ''
            if 'invalid_at' not in df.columns:
                df['invalid_at'] = ''

            mask = (df['image_id'].astype(str) == str(image_id)) & (df['filename'] == filename)
            if not mask.any():
                return jsonify({'success': False, 'message': '未找到对应记录'}), 404

            now_str = time.strftime('%Y-%m-%d %H:%M:%S')
            df.loc[mask, 'is_invalid'] = True
            df.loc[mask, 'invalid_label'] = str(invalid_label)
            df.loc[mask, 'invalid_at'] = now_str
            
            # 使用原子写入保存
            safe_save_excel(df, excel_path, create_backup=True)
            
            # 记录无效标记日志
            log_modification({'type': 'invalidate', 'image_id': image_id, 'filename': filename})

        # 更新内存样本：移除该图片；若样本无图则移除样本
        for sample in list(app_status['samples']):
            if str(sample['image_id']) != str(image_id):
                continue
            sample['images'] = [img for img in sample['images'] if img['filename'] != filename]
            sample['needs_review'] = any(img.get('needs_review', False) for img in sample['images'])
            if not sample['images']:
                app_status['samples'].remove(sample)
            break

        return jsonify({'success': True, 'message': '已标记为无效样本并写入Excel标签'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'标记失败: {str(e)}'}), 500


@app.route('/api/export')
def export_results():
    """导出结果"""
    result = []
    
    # 如果有样本数据，使用样本数据
    if app_status['samples']:
        for sample in app_status['samples']:
            sample_data = {
                'image_id': sample['image_id'],
                'original_part': sample['original_part'],
                'images': []
            }
            for img in sample['images']:
                sample_data['images'].append({
                    'filename': img['filename'],
                    'final_projection': img['final_projection'],
                    'final_orientation': img['final_orientation'],
                    'projection_modified': img.get('projection_modified', False),
                    'orientation_modified': img.get('orientation_modified', False),
                    'modified_at': img.get('modified_at', ''),
                    'needs_review': img['needs_review'],
                    'is_invalid': img.get('is_invalid', False),
                    'invalid_label': img.get('invalid_label', ''),
                    'invalid_at': img.get('invalid_at', '')
                })
            result.append(sample_data)
    else:
        # 如果没有样本数据，只返回修改记录
        for key, modified in app_status['modified_data'].items():
            parts = key.rsplit('_', 1)  # 简化分割
            if len(parts) == 2:
                image_id, filename = parts
                result.append({
                    'image_id': image_id,
                    'filename': filename,
                    'modified_projection': modified.get('projection'),
                    'modified_orientation': modified.get('orientation')
                })
    
    return jsonify(result)


@app.route('/api/image/<path:image_path>')
def get_image(image_path):
    """获取图片"""
    # 从多个可能的数据目录提供图片
    data_roots = [
        Path(__file__).parent.parent / 'data',
        Path(__file__).parent / 'data'
    ]
    rel_path = Path(image_path)
    if rel_path.parts and rel_path.parts[0] == 'data':
        rel_path = Path(*rel_path.parts[1:])

    for root in data_roots:
        try:
            full_path = (root / rel_path).resolve()
            if root.resolve() not in full_path.parents and full_path != root.resolve():
                continue
            if full_path.exists():
                return send_from_directory(str(full_path.parent), full_path.name)
        except Exception:
            continue

    return "Image not found", 404


@app.route('/static/<path:filename>')
def static_files(filename):
    """静态文件"""
    static_dir = app.static_folder or str(Path(__file__).parent / 'static')
    return send_from_directory(static_dir, filename)


def init_data(excel_path=None):
    """初始化数据"""
    # 查找最新的结果文件
    if excel_path is None:
        outputs_dir = Path(__file__).parent.parent
        excel_files = list(outputs_dir.glob('*.xlsx'))
        if excel_files:
            excel_path = max(excel_files, key=lambda p: p.stat().st_mtime)
    
    if excel_path and os.path.exists(excel_path):
        app_status['samples'] = load_data_from_excel(excel_path)
        app_status['excel_path'] = str(excel_path)
        print(f"✓ 已加载数据: {excel_path} ({len(app_status['samples'])} 个样本)")
    else:
        print(f"⚠ 未找到数据文件: {excel_path}")


def open_browser(port):
    """自动打开浏览器"""
    time.sleep(1.5)
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    print(f"✓ 已在浏览器中打开: {url}")


def run_server(port=5000, excel_path=None, auto_open=True):
    """运行服务器"""
    # 确保目录存在
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # 初始化数据
    init_data(excel_path)
    
    if auto_open:
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    print(f"\n{'='*60}")
    print(f"🚀 Flask服务器已启动!")
    print(f"{'='*60}")
    print(f"访问地址: http://localhost:{port}")
    print(f"按 Ctrl+C 停止服务器\n")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\n\n正在停止服务器...")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Flask复核服务器')
    parser.add_argument('--port', '-p', type=int, default=5000, help='端口号')
    parser.add_argument('--input', '-i', help='输入Excel文件路径')
    parser.add_argument('--no-open', action='store_true', help='不自动打开浏览器')
    
    args = parser.parse_args()
    
    run_server(port=args.port, excel_path=args.input, auto_open=not args.no_open)
