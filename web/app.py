#!/usr/bin/env python3
"""
Flask Web服务器 - 用于预览和人工复核X光标注结果
支持离线运行，所有静态资源已本地化
"""

import os
import sys
import json
import logging
import logging.handlers
import webbrowser
import threading
import time
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from flask import Flask, render_template, jsonify, request, send_from_directory

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 日志配置
# ============================================================================
LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 创建日志格式化器
LOG_FORMATTER = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 创建根日志记录器
logger = logging.getLogger('medgemma_label')
logger.setLevel(logging.DEBUG)

# 文件日志 - 按大小轮转，保留10个备份
file_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / 'web_server.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=10,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(LOG_FORMATTER)
logger.addHandler(file_handler)

# 控制台日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(LOG_FORMATTER)
logger.addHandler(console_handler)

logger.info(f"日志系统初始化完成，日志目录: {LOG_DIR}")

# ============================================================================
# Flask应用初始化
# ============================================================================
app = Flask(__name__)

EXCEL_LOCK = threading.Lock()

# 延迟批量保存机制
class DelayedBatchSaver:
    """
    延迟批量保存器 - 将频繁的小改动先缓存，定时批量写入Excel
    避免频繁读写大Excel文件导致的性能问题
    """
    def __init__(self, flush_interval=5.0):
        self.pending_changes = {}  # {(image_id, filename): change_data}
        self.lock = threading.Lock()
        self.flush_interval = flush_interval
        self.last_flush_time = time.time()
        self._running = True
        self._start_flush_timer()
        logger.info(f"DelayedBatchSaver初始化完成，flush_interval={flush_interval}s")
    
    def add_change(self, image_id, filename, change_data):
        """添加一个待保存的修改"""
        key = (str(image_id), str(filename))
        with self.lock:
            if key in self.pending_changes:
                # 合并修改
                old_data = self.pending_changes[key].copy()
                self.pending_changes[key].update(change_data)
                logger.debug(f"合并修改: {key}, 旧数据={old_data}, 新数据={change_data}")
            else:
                self.pending_changes[key] = change_data.copy()
                logger.debug(f"添加新修改: {key}, 数据={change_data}")
        
        queue_size = len(self.pending_changes)
        if queue_size >= 50:
            logger.info(f"待保存队列达到阈值({queue_size}条)，触发立即保存")
            self.flush()
    
    def flush(self):
        """立即执行批量保存"""
        with self.lock:
            if not self.pending_changes:
                return
            
            changes_to_save = dict(self.pending_changes)
            self.pending_changes.clear()
        
        logger.info(f"开始批量保存: {len(changes_to_save)}条记录")
        # 在后台线程执行保存
        threading.Thread(target=self._save_changes, args=(changes_to_save,), daemon=True).start()
        self.last_flush_time = time.time()
    
    def _save_changes(self, changes):
        """实际执行保存到Excel"""
        try:
            import pandas as pd
            
            excel_path = app_status.get('excel_path')
            if not excel_path:
                logger.error("延迟保存失败: 未找到Excel路径")
                return
            
            logger.debug(f"读取Excel: {excel_path}")
            with EXCEL_LOCK:
                df = pd.read_excel(excel_path)
                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                for (image_id, filename), change_data in changes.items():
                    mask = (df['image_id'].astype(str) == image_id) & (df['filename'] == filename)
                    if not mask.any():
                        continue
                    
                    # 应用所有修改
                    for key, value in change_data.items():
                        if key in ['final_projection', 'final_orientation', 'reviewed']:
                            df.loc[mask, key] = value
                        if key == 'projection_modified' and value:
                            df.loc[mask, 'projection_modified'] = True
                            df.loc[mask, 'modified_at'] = now_str
                        if key == 'orientation_modified' and value:
                            df.loc[mask, 'orientation_modified'] = True
                            df.loc[mask, 'modified_at'] = now_str
                        if key == 'reviewed' and value:
                            df.loc[mask, 'reviewed_at'] = now_str
                        # 处理删除标记字段
                        if key == 'is_invalid' and value:
                            df.loc[mask, 'is_invalid'] = True
                        if key == 'invalid_label' and value:
                            df.loc[mask, 'invalid_label'] = value
                        if key == 'invalid_at' and value:
                            df.loc[mask, 'invalid_at'] = value
                
                # 原子写入
                safe_save_excel(df, excel_path, create_backup=True)
            
            logger.info(f"✅ 延迟批量保存完成: {len(changes)} 条修改")
            
        except Exception as e:
            logger.error(f"❌ 延迟批量保存失败: {e}", exc_info=True)
            # 失败时将修改重新加入队列
            with self.lock:
                for key, data in changes.items():
                    if key not in self.pending_changes:
                        self.pending_changes[key] = data
                        logger.warning(f"修改已重新加入队列: {key}")
    
    def _start_flush_timer(self):
        """启动定时保存线程"""
        def timer_loop():
            logger.info("定时保存线程已启动")
            while self._running:
                time.sleep(1.0)  # 每秒检查一次
                if time.time() - self.last_flush_time >= self.flush_interval:
                    if self.pending_changes:
                        logger.info(f"定时触发保存，队列大小: {len(self.pending_changes)}")
                        self.flush()
            logger.info("定时保存线程已停止")
        
        threading.Thread(target=timer_loop, daemon=True).start()
    
    def shutdown(self):
        """关闭前强制保存所有待处理修改"""
        logger.info("开始关闭DelayedBatchSaver...")
        self._running = False
        self.flush()
        # 等待保存完成
        time.sleep(0.5)
        logger.info("DelayedBatchSaver已关闭")

# 全局延迟保存器实例
batch_saver = DelayedBatchSaver(flush_interval=5.0)


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
        logger.debug(f"开始保存Excel: {excel_path}, 行数={len(df)}")
        
        # 1. 写入临时文件
        df.to_excel(temp_path, index=False)
        logger.debug(f"临时文件已写入: {temp_path}, 大小={temp_path.stat().st_size if temp_path.exists() else 0} bytes")
        
        # 验证临时文件写入成功
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise IOError("临时文件写入失败或为空")
        
        # 2. 创建备份（如果原文件存在）
        if create_backup and excel_path.exists():
            shutil.copy2(excel_path, backup_path)
            logger.debug(f"已创建备份: {backup_path}")
        
        # 3. 原子重命名（操作系统保证这是原子操作）
        temp_path.replace(excel_path)
        logger.debug(f"原子重命名完成: {temp_path} -> {excel_path}")
        
        # 4. 成功后保留最近10个备份，删除旧的
        if create_backup:
            cleanup_old_backups(excel_path.parent, keep_count=10)
        
        logger.debug(f"Excel保存成功: {excel_path}")
            
    except Exception as e:
        logger.error(f"保存Excel失败: {e}, 文件={excel_path}", exc_info=True)
        
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
                logger.warning(f"已从备份恢复: {excel_path}")
            except Exception as restore_error:
                logger.error(f"恢复备份失败: {restore_error}")
        
        raise Exception(f"保存Excel失败: {str(e)}")


def cleanup_old_backups(directory, keep_count=10):
    """清理旧备份文件，只保留最近的指定数量"""
    try:
        backup_files = sorted(
            Path(directory).glob('*.xlsx.bak'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        deleted_count = 0
        for old_backup in backup_files[keep_count:]:
            try:
                old_backup.unlink()
                deleted_count += 1
            except:
                pass
        if deleted_count > 0:
            logger.debug(f"已清理 {deleted_count} 个旧备份文件")
    except Exception as e:
        logger.warning(f"清理旧备份失败: {e}")  # 清理失败不影响主流程


def log_modification(data):
    """
    记录修改操作到JSONL日志，用于灾难恢复
    即使Excel文件损坏，也能从日志重建数据
    """
    try:
        log_path = OUTPUTS_DIR / 'modifications.jsonl'
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        logger.debug(f"修改日志已记录: {data.get('type', 'unknown')}")
    except Exception as e:
        logger.error(f"记录修改日志失败: {e}")  # 日志失败不影响主流程

# 配置
DATA_DIR = Path(__file__).parent / 'data'
OUTPUTS_DIR = Path(__file__).parent / 'outputs'

logger.info(f"配置加载: DATA_DIR={DATA_DIR}, OUTPUTS_DIR={OUTPUTS_DIR}")

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
    
    logger.info(f"开始从Excel加载数据: {excel_path}")
    
    if not os.path.exists(excel_path):
        logger.warning(f"Excel文件不存在: {excel_path}")
        return []
    
    try:
        df = pd.read_excel(excel_path)
        logger.info(f"Excel读取成功: {len(df)} 行, 列={list(df.columns)}")
    except Exception as e:
        logger.error(f"读取Excel失败: {e}", exc_info=True)
        return []
    
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
    if 'part' not in df.columns:
        df['part'] = df['original_part']  # 如果没有part列，使用original_part作为默认值
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
            'part': safe_str(valid_group['part'].iloc[0], 'unknown'),  # 原始检查部位名称
            'needs_review': safe_bool(valid_group['needs_review'].any()),
            'images': []
        }
        
        for _, row in valid_group.iterrows():
            # URL 编码文件名，处理特殊字符
            encoded_filename = quote(str(row['filename']), safe='')
            img_path = f"/api/image/{image_id}/{encoded_filename}"
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
    
    logger.info(f"数据加载完成: {len(samples)} 个样本")
    return samples


@app.route('/')
def index():
    """主页面"""
    logger.debug("访问主页")
    return render_template('review.html')


@app.route('/api/samples')
def get_samples():
    """获取所有样本数据"""
    logger.debug(f"API: get_samples, 返回 {len(app_status['samples'])} 个样本")
    return jsonify(app_status['samples'])


@app.route('/api/stats')
def get_stats():
    """获取统计信息"""
    stats = {
        'total_samples': len(app_status['samples']),
        'total_images': sum(len(s['images']) for s in app_status['samples']),
        'review_count': sum(sum(1 for img in s['images'] if img['needs_review']) 
                             for s in app_status['samples']),
        'modified_count': len(app_status['modified_data'])
    }
    logger.info(f"API: get_stats, 样本={stats['total_samples']}, 图像={stats['total_images']}, 需复核={stats['review_count']}")
    return jsonify(stats)


@app.route('/api/save', methods=['POST'])
def save_modification():
    """保存修改 - 智能合并版本（支持多用户并发）"""
    import pandas as pd
    
    data = request.json or {}
    logger.debug(f"API: save_modification, data={data}")
    
    if not app_status.get('excel_path'):
        logger.error("保存失败: 未找到Excel数据文件")
        return jsonify({'success': False, 'message': '未找到Excel数据文件'}), 500

    if not data.get('image_id') or not data.get('filename'):
        logger.warning("保存失败: 参数缺失")
        return jsonify({'success': False, 'message': '参数缺失'}), 400

    try:
        image_id = str(data['image_id'])
        filename = str(data['filename'])
        client_modified_at = data.get('client_modified_at', '')  # 客户端最后修改时间
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"保存修改: image_id={image_id}, filename={filename}, type={'projection' if 'projection' in data else 'orientation'}")
        
        excel_path = app_status['excel_path']
        
        # 读取当前Excel数据（带锁）
        with EXCEL_LOCK:
            df = pd.read_excel(excel_path)
            
            mask = (df['image_id'].astype(str) == image_id) & (df['filename'] == filename)
            if not mask.any():
                return jsonify({'success': False, 'message': '未找到对应记录'}), 404
            
            # 获取服务器端当前值
            server_row = df.loc[mask].iloc[0]
            server_modified_at = str(server_row.get('modified_at', ''))
            
            # 检查是否被其他用户修改过（且不是当前用户刚改的）
            conflict_fields = []
            auto_merged_fields = []
            
            if client_modified_at and server_modified_at != client_modified_at:
                # 有并发修改，检查字段冲突
                if 'projection' in data:
                    server_projection = str(server_row.get('final_projection', ''))
                    if server_projection != data['projection']:
                        # 服务器值与客户端期望值不同，说明被他人修改
                        conflict_fields.append({
                            'field': 'final_projection',
                            'server_value': server_projection,
                            'client_value': data['projection']
                        })
                
                if 'orientation' in data:
                    server_orientation = str(server_row.get('final_orientation', ''))
                    if server_orientation != data['orientation']:
                        conflict_fields.append({
                            'field': 'final_orientation',
                            'server_value': server_orientation,
                            'client_value': data['orientation']
                        })
            
            # 如果有冲突，返回冲突信息让前端处理
            if conflict_fields:
                return jsonify({
                    'success': False,
                    'conflict': True,
                    'message': '该记录已被其他用户修改',
                    'server_modified_at': server_modified_at,
                    'conflict_fields': conflict_fields
                }), 409  # HTTP 409 Conflict
            
            # 无冲突，执行保存
            if 'projection' in data:
                df.loc[mask, 'final_projection'] = data['projection']
                df.loc[mask, 'projection_modified'] = True
                df.loc[mask, 'modified_at'] = now_str
            elif 'orientation' in data:
                df.loc[mask, 'final_orientation'] = data['orientation']
                df.loc[mask, 'orientation_modified'] = True
                df.loc[mask, 'modified_at'] = now_str
            
            # 原子写入
            safe_save_excel(df, excel_path, create_backup=True)
        
        # 记录日志
        log_modification(data)
        
        # 更新内存
        for sample in app_status['samples']:
            if str(sample['image_id']) != image_id:
                continue
            for img in sample['images']:
                if img['filename'] != filename:
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
        
        return jsonify({
            'success': True,
            'message': '已保存',
            'modified_at': now_str
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500


@app.route('/api/review', methods=['POST'])
def save_review_status():
    """保存复核状态 - 使用延迟批量保存"""
    data = request.json or {}
    logger.debug(f"API: save_review_status, data={data}")
    
    if not app_status.get('excel_path'):
        logger.error("复核状态保存失败: 未找到Excel数据文件")
        return jsonify({'success': False, 'message': '未找到Excel数据文件'}), 500

    image_id = data.get('image_id')
    filename = data.get('filename')
    reviewed = bool(data.get('reviewed', True))

    if not image_id or not filename:
        logger.warning(f"复核状态保存失败: 参数缺失, image_id={image_id}, filename={filename}")
        return jsonify({'success': False, 'message': '参数缺失'}), 400

    try:
        logger.info(f"保存复核状态: {image_id}/{filename}, reviewed={reviewed}")
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') if reviewed else ''
        
        # 添加到延迟保存队列
        change_data = {'reviewed': reviewed, 'reviewed_at': now_str}
        batch_saver.add_change(image_id, filename, change_data)
        
        # 记录日志
        log_modification({'type': 'review', 'image_id': image_id, 'filename': filename, 'reviewed': reviewed})

        # 立即更新内存
        for sample in app_status['samples']:
            if str(sample['image_id']) != str(image_id):
                continue
            for img in sample['images']:
                if img['filename'] != filename:
                    continue
                img['reviewed'] = reviewed
                img['reviewed_at'] = now_str
                break

        logger.info(f"✅ 复核状态已加入保存队列: {image_id}/{filename}")
        return jsonify({'success': True, 'message': '已加入保存队列'})
    except Exception as e:
        logger.error(f"复核状态保存失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500


@app.route('/api/review/bulk', methods=['POST'])
def save_review_bulk():
    """批量保存复核状态 - 使用延迟批量保存"""
    data = request.json or {}
    items = data.get('items', [])
    logger.info(f"API: save_review_bulk, 批量复核 {len(items)} 项")

    if not app_status.get('excel_path'):
        logger.error("批量复核保存失败: 未找到Excel数据文件")
        return jsonify({'success': False, 'message': '未找到Excel数据文件'}), 500

    if not items:
        logger.warning("批量复核保存失败: 没有需要更新的项目")
        return jsonify({'success': False, 'message': '没有需要更新的项目'}), 400

    try:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 批量添加到延迟保存队列
        for item in items:
            image_id = item.get('image_id')
            filename = item.get('filename')
            reviewed = bool(item.get('reviewed', True))
            if not image_id or not filename:
                continue
            
            change_data = {'reviewed': reviewed, 'reviewed_at': now_str if reviewed else ''}
            batch_saver.add_change(image_id, filename, change_data)
        
        # 记录日志
        log_modification({'type': 'bulk_review', 'count': len(items)})

        # 立即更新内存
        item_set = {(str(i.get('image_id')), i.get('filename')) for i in items}
        for sample in app_status['samples']:
            for img in sample['images']:
                key = (str(sample['image_id']), img['filename'])
                if key in item_set:
                    img['reviewed'] = True
                    img['reviewed_at'] = now_str

        return jsonify({'success': True, 'message': f'已加入保存队列: {len(items)} 项'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500


@app.route('/api/invalidate', methods=['POST'])
def invalidate_sample_image():
    """标记图片为无效样本（软删除）- 使用轻量级JSONL日志+延迟合并"""
    data = request.json or {}
    logger.info(f"API: invalidate_sample_image, data={data}")
    
    if not app_status.get('excel_path'):
        logger.error("删除失败: 未找到Excel数据文件")
        return jsonify({'success': False, 'message': '未找到Excel数据文件'}), 500

    image_id = data.get('image_id')
    filename = data.get('filename')
    invalid_label = data.get('invalid_label', 'invalid_sample')

    if not image_id or not filename:
        logger.warning(f"删除失败: 参数缺失, image_id={image_id}, filename={filename}")
        return jsonify({'success': False, 'message': '参数缺失'}), 400

    try:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 确保目录存在
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. 立即写入轻量级JSONL日志（非常快，不阻塞）
        invalidation_record = {
            'type': 'invalidate',
            'image_id': image_id,
            'filename': filename,
            'invalid_label': invalid_label,
            'invalid_at': now_str,
            'timestamp': datetime.now().isoformat()
        }
        
        # 写入专门的删除日志文件
        invalid_log_path = OUTPUTS_DIR / 'invalidations.jsonl'
        with open(invalid_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(invalidation_record, ensure_ascii=False) + '\n')
        
        logger.info(f"📝 删除记录已写入日志: {image_id}/{filename}")
        
        # 2. 添加到批量保存队列（延迟合并到Excel）
        change_data = {
            'is_invalid': True,
            'invalid_label': str(invalid_label),
            'invalid_at': now_str
        }
        batch_saver.add_change(image_id, filename, change_data)
        
        logger.info(f"⏳ 删除操作已加入延迟保存队列: {image_id}/{filename}")
        
        # 3. 立即更新内存（前端立即可见）
        for sample in list(app_status['samples']):
            if str(sample['image_id']) != str(image_id):
                continue
            original_count = len(sample['images'])
            sample['images'] = [img for img in sample['images'] if img['filename'] != filename]
            removed_count = original_count - len(sample['images'])
            sample['needs_review'] = any(img.get('needs_review', False) for img in sample['images'])
            if not sample['images']:
                app_status['samples'].remove(sample)
                logger.info(f"样本 {image_id} 已为空，从列表中移除")
            break
        
        logger.info(f"✅ 删除成功: {image_id}/{filename}")
        return jsonify({'success': True, 'message': '已标记为无效样本'})
    except Exception as e:
        logger.error(f"删除失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'标记失败: {str(e)}'}), 500


@app.route('/api/export')
def export_results():
    """导出结果"""
    logger.info(f"API: export_results, 导出 {len(app_status['samples'])} 个样本")
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
    logger.debug(f"获取图片: {image_path}")
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
                logger.debug(f"图片找到: {full_path}")
                return send_from_directory(str(full_path.parent), full_path.name)
        except Exception as e:
            logger.debug(f"搜索图片路径失败: {root} - {e}")
            continue

    logger.warning(f"图片未找到: {image_path}")
    return "Image not found", 404


@app.route('/static/<path:filename>')
def static_files(filename):
    """静态文件 - 对 .map 文件返回空响应避免 404 污染日志"""
    # Source map 文件在离线环境中不需要，返回空响应避免 404
    if filename.endswith('.map'):
        logger.debug(f"Source map 请求: {filename}")
        return '', 200, {'Content-Type': 'application/json'}
    
    static_dir = app.static_folder or str(Path(__file__).parent / 'static')
    logger.debug(f"静态文件请求: {filename}")
    return send_from_directory(static_dir, filename)


def init_data(excel_path=None):
    """初始化数据"""
    logger.info("开始初始化数据...")
    # 查找最新的结果文件
    if excel_path is None:
        outputs_dir = Path(__file__).parent.parent
        excel_files = list(outputs_dir.glob('*.xlsx'))
        if excel_files:
            excel_path = max(excel_files, key=lambda p: p.stat().st_mtime)
    
    if excel_path and os.path.exists(excel_path):
        logger.info(f"正在加载: {excel_path}")
        app_status['samples'] = load_data_from_excel(excel_path)
        app_status['excel_path'] = str(excel_path)
        logger.info(f"✓ 已加载数据: {excel_path} ({len(app_status['samples'])} 个样本)")
    else:
        logger.error(f"⚠ 未找到数据文件: {excel_path}")


def open_browser(port):
    """自动打开浏览器"""
    time.sleep(1.5)
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    logger.info(f"✓ 已在浏览器中打开: {url}")


def run_server(port=5000, excel_path=None, auto_open=True):
    """运行服务器"""
    # 确保目录存在
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # 初始化数据
    init_data(excel_path)
    
    if auto_open:
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    logger.info(f"{'='*60}")
    logger.info(f"🚀 Flask服务器已启动!")
    logger.info(f"{'='*60}")
    logger.info(f"访问地址: http://localhost:{port}")
    logger.info(f"按 Ctrl+C 停止服务器")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        logger.info("\n\n正在停止服务器...")
        # 强制保存所有待处理的修改
        logger.info("正在保存待处理的修改...")
        batch_saver.shutdown()
        logger.info("✅ 所有修改已保存")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Flask复核服务器')
    parser.add_argument('--port', '-p', type=int, default=5000, help='端口号')
    parser.add_argument('--input', '-i', help='输入Excel文件路径')
    parser.add_argument('--no-open', action='store_true', help='不自动打开浏览器')
    
    args = parser.parse_args()
    
    run_server(port=args.port, excel_path=args.input, auto_open=not args.no_open)
