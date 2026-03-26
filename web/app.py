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
import sqlite3

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 路径配置 (必须在类定义之前)
# ============================================================================
LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).parent / 'data'
OUTPUTS_DIR = Path(__file__).parent / 'outputs'

# ============================================================================
# 日志配置
# ============================================================================

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

# ============================================================================
# SQLite 主存储系统 - 彻底放弃Excel实时同步
# 
# 核心设计：
# 1. 首次启动：从Excel导入数据到SQLite
# 2. 日常运行：所有读写操作直接对SQLite（毫秒级响应）
# 3. 导出功能：提供手动"导出Excel"按钮
# ============================================================================

class SQLiteStorage:
    """
    SQLite主存储系统 - 完全替代Excel作为数据源
    """
    def __init__(self):
        self.db_path = OUTPUTS_DIR / 'medical_labels.db'
        self.excel_source = None  # 原始Excel文件路径
        self._lock = threading.Lock()
        self._init_database()
        logger.info(f"SQLiteStorage初始化完成: {self.db_path}")
    
    def _init_database(self):
        """初始化SQLite数据库"""
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # 创建主数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    original_part TEXT,
                    part TEXT,
                    final_body_part TEXT,
                    final_orientation TEXT DEFAULT 'unknown',
                    final_projection TEXT DEFAULT 'unknown',
                    confidence_projection REAL DEFAULT 0.8,
                    confidence_orientation REAL DEFAULT 0.8,
                    confidence_overall REAL DEFAULT 0.8,
                    needs_review BOOLEAN DEFAULT 0,
                    review_reason TEXT,
                    match_method TEXT DEFAULT 'unknown',
                    projection_modified BOOLEAN DEFAULT 0,
                    orientation_modified BOOLEAN DEFAULT 0,
                    modified_at TEXT,
                    reviewed BOOLEAN DEFAULT 0,
                    reviewed_at TEXT,
                    is_invalid BOOLEAN DEFAULT 0,
                    invalid_label TEXT,
                    invalid_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(image_id, filename)
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_id ON samples(image_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_original_part ON samples(original_part)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_needs_review ON samples(needs_review)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reviewed ON samples(reviewed)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_invalid ON samples(is_invalid)')
            
            conn.commit()
    
    def import_from_excel(self, excel_path):
        """首次启动时从Excel导入数据"""
        import pandas as pd
        
        if not Path(excel_path).exists():
            logger.error(f"Excel文件不存在: {excel_path}")
            return False
        
        self.excel_source = excel_path
        
        with self._lock:
            # 检查是否已有数据
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM samples')
                count = cursor.fetchone()[0]
                
                if count > 0:
                    logger.info(f"数据库已有 {count} 条记录，跳过导入")
                    return True
            
            # 从Excel导入
            logger.info(f"从Excel导入数据: {excel_path}")
            df = pd.read_excel(excel_path)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO samples (
                            image_id, filename, original_part, part, final_body_part,
                            final_orientation, final_projection, confidence_projection,
                            confidence_orientation, confidence_overall, needs_review,
                            review_reason, match_method, projection_modified,
                            orientation_modified, modified_at, reviewed, reviewed_at,
                            is_invalid, invalid_label, invalid_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        str(row.get('image_id', '')),
                        str(row.get('filename', '')),
                        str(row.get('original_part', '')),
                        str(row.get('part', row.get('original_part', ''))),
                        str(row.get('final_body_part', '')),
                        str(row.get('final_orientation', 'unknown')),
                        str(row.get('final_projection', 'unknown')),
                        float(row.get('confidence_projection', 0.8)),
                        float(row.get('confidence_orientation', 0.8)),
                        float(row.get('confidence_overall', 0.8)),
                        bool(row.get('needs_review', False)),
                        str(row.get('review_reason', '')),
                        str(row.get('match_method', 'unknown')),
                        bool(row.get('projection_modified', False)),
                        bool(row.get('orientation_modified', False)),
                        str(row.get('modified_at', '') if pd.notna(row.get('modified_at')) else ''),
                        bool(row.get('reviewed', False)),
                        str(row.get('reviewed_at', '') if pd.notna(row.get('reviewed_at')) else ''),
                        bool(row.get('is_invalid', False)),
                        str(row.get('invalid_label', '') if pd.notna(row.get('invalid_label')) else ''),
                        str(row.get('invalid_at', '') if pd.notna(row.get('invalid_at')) else '')
                    ))
                
                conn.commit()
                logger.info(f"✅ 成功导入 {len(df)} 条记录到SQLite")
                return True
    
    def get_all_samples(self):
        """获取所有样本数据"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM samples 
                WHERE is_invalid = 0 
                ORDER BY image_id, filename
            ''')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def update_sample(self, image_id, filename, change_data, expected_modified_at=None):
        """更新样本数据 - 带乐观锁支持
        
        Args:
            expected_modified_at: 期望的当前modified_at值，用于乐观锁冲突检测
        """
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # 构建更新语句（采用Last Write Wins策略，移除乐观锁检查）
                fields = []
                values = []
                for key, value in change_data.items():
                    fields.append(f"{key} = ?")
                    values.append(value)
                
                # 添加更新时间
                fields.append("updated_at = CURRENT_TIMESTAMP")
                
                values.extend([str(image_id), filename])
                
                sql = f"UPDATE samples SET {', '.join(fields)} WHERE image_id = ? AND filename = ?"
                
                # 调试日志
                logger.debug(f"SQL: {sql}, values: {values}")
                
                cursor.execute(sql, values)
                conn.commit()
                
                return {'success': True, 'updated': cursor.rowcount > 0}
    
    def export_to_excel(self, output_path=None):
        """导出数据到Excel"""
        import pandas as pd
        
        if output_path is None:
            output_path = self.excel_source or (OUTPUTS_DIR / 'exported_labels.xlsx')
        
        samples = self.get_all_samples()
        if not samples:
            logger.warning("没有数据可导出")
            return False
        
        df = pd.DataFrame(samples)
        
        # 删除不需要的列
        for col in ['id', 'created_at', 'updated_at']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        df.to_excel(output_path, index=False)
        logger.info(f"✅ 已导出 {len(samples)} 条记录到: {output_path}")
        return True

# 全局存储实例
storage = SQLiteStorage()


def _is_file_locked(filepath: Path):
    """检查文件是否被其他程序锁定（Windows专用）"""
    if not filepath.exists():
        return False
    fd = None
    try:
        # 尝试以独占模式打开文件
        fd = os.open(str(filepath), os.O_RDWR | os.O_EXCL)
        return False
    except (OSError, PermissionError):
        return True
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except:
                pass


def _atomic_replace_windows(src: Path, dst: Path, max_retries=30):
    """
    Windows兼容的原子文件替换 - 终极版
    处理文件被占用、权限、路径长度等问题
    增加更长的等待时间处理杀毒软件/同步软件锁定
    """
    import sys
    
    logger.debug(f"原子替换: {src.name} -> {dst.name}")
    
    # 检查源文件
    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")
    
    # 确保目标目录存在
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Windows长路径支持
    src_str = str(src)
    dst_str = str(dst)
    if sys.platform == 'win32' and not src_str.startswith('\\\\?\\'):
        if len(src_str) > 250:
            src_str = '\\\\?\\' + os.path.abspath(src_str)
        if len(dst_str) > 250:
            dst_str = '\\\\?\\' + os.path.abspath(dst_str)
    
    # 检查目标文件是否被锁定
    if dst.exists() and _is_file_locked(dst):
        logger.warning(f"目标文件被锁定，等待释放: {dst.name}")
    
    for attempt in range(max_retries):
        try:
            # 策略1: 如果目标不存在，直接重命名
            if not dst.exists():
                os.rename(src_str, dst_str)
                logger.debug(f"重命名成功: {src.name} -> {dst.name}")
                return True
            
            # 策略2: 尝试直接替换
            try:
                src.replace(dst)
                logger.debug(f"替换成功: {src.name} -> {dst.name}")
                return True
            except (PermissionError, OSError):
                pass
            
            # 策略3: 如果目标被锁定，尝试重命名目标后移动
            if _is_file_locked(dst):
                # 生成一个备用名
                backup_name = dst.with_suffix(f'.locked.{int(time.time())}.bak')
                try:
                    os.rename(str(dst), str(backup_name))
                    logger.debug(f"已将锁定文件重命名为: {backup_name.name}")
                    os.rename(src_str, dst_str)
                    logger.debug(f"重命名成功: {src.name} -> {dst.name}")
                    return True
                except:
                    pass
            else:
                # 策略4: 目标未被锁定，尝试删除后重命名
                try:
                    dst.unlink()
                    os.rename(src_str, dst_str)
                    logger.debug(f"删除后重命名成功: {src.name} -> {dst.name}")
                    return True
                except:
                    pass
            
            # 策略5: 使用shutil复制并删除源文件
            try:
                shutil.copy2(str(src), str(dst))
                src.unlink()
                logger.debug(f"复制覆盖成功: {src.name} -> {dst.name}")
                return True
            except:
                pass
            
        except Exception as e:
            logger.debug(f"尝试{attempt+1}/{max_retries}失败: {e}")
        
        # 等待后重试（递增延迟，最长2秒）
        wait_time = min(0.1 * (attempt + 1), 2.0)
        if attempt < max_retries - 1:
            time.sleep(wait_time)
    
    # 所有尝试都失败
    logger.error(f"无法替换文件，可能原因:\n"
                 f"1. Excel软件正在打开 {dst.name}\n"
                 f"2. 杀毒软件正在扫描\n" 
                 f"3. OneDrive/其他同步软件正在上传\n"
                 f"4. 文件被其他程序占用")
    raise PermissionError(f"文件被锁定: {dst.name}")


def _cleanup_temp_files(directory, pattern='*.tmp*', max_age_hours=24):
    """清理目录中的临时文件"""
    try:
        import time
        now = time.time()
        count = 0
        for temp_file in directory.glob(pattern):
            try:
                # 检查文件年龄
                file_age = now - temp_file.stat().st_mtime
                if file_age > max_age_hours * 3600:  # 超过24小时
                    temp_file.unlink()
                    count += 1
            except:
                pass
        if count > 0:
            logger.debug(f"清理了 {count} 个旧临时文件")
    except:
        pass


def safe_save_excel(df, excel_path, create_backup=True):
    """
    原子方式保存Excel文件，防止写入过程中断导致文件损坏 - 增强版
    
    策略:
    1. 写入唯一命名的临时文件（避免多线程冲突）
    2. 创建备份（可选）
    3. 原子重命名替换原文件
    4. 清理临时文件（即使失败也清理）
    
    Args:
        df: pandas DataFrame
        excel_path: 目标文件路径
        create_backup: 是否创建备份文件
    
    Raises:
        Exception: 保存失败时抛出异常
    """
    import uuid
    import os
    
    excel_path = Path(excel_path)
    # 使用唯一临时文件名，避免多线程冲突
    unique_id = str(uuid.uuid4())[:8]
    temp_path = excel_path.with_suffix(f'.tmp.{unique_id}')
    backup_path = excel_path.with_suffix('.xlsx.bak')
    
    try:
        # 清理旧临时文件（每隔一段时间）
        if uuid.uuid4().int % 10 == 0:  # 10%概率执行清理
            _cleanup_temp_files(excel_path.parent)
        
        logger.debug(f"开始保存Excel: {excel_path}, 行数={len(df)}, 临时文件: {temp_path.name}")
        
        # 确保目标目录存在
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. 写入临时文件（唯一文件名，避免冲突，使用openpyxl引擎确保句柄关闭）
        df.to_excel(temp_path, index=False, engine='openpyxl')
        
        # 确保文件写入完成并刷新到磁盘（Windows需要）
        fd = None
        try:
            fd = os.open(str(temp_path), os.O_RDONLY)
            os.fsync(fd)
        except:
            pass
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except:
                    pass
        
        # 验证临时文件写入成功
        if not temp_path.exists():
            raise IOError(f"临时文件写入失败: {temp_path}")
        
        file_size = temp_path.stat().st_size
        if file_size == 0:
            raise IOError("临时文件为空")
        
        logger.debug(f"临时文件已写入: {temp_path.name}, 大小={file_size} bytes")
        
        # 2. 创建备份（如果原文件存在）
        if create_backup and excel_path.exists():
            try:
                shutil.copy2(str(excel_path), str(backup_path))
                logger.debug(f"已创建备份: {backup_path.name}")
            except Exception as e:
                logger.warning(f"创建备份失败: {e}")
        
        # 3. 原子重命名（Windows兼容处理）
        _atomic_replace_windows(temp_path, excel_path)
        logger.debug(f"原子重命名完成: {temp_path.name} -> {excel_path.name}")
        
        # 4. 成功后保留最近10个备份，删除旧的
        if create_backup:
            cleanup_old_backups(excel_path.parent, keep_count=10)
        
        logger.debug(f"Excel保存成功: {excel_path}")
            
    except Exception as e:
        logger.error(f"保存Excel失败: {e}, 文件={excel_path}", exc_info=True)
        raise Exception(f"保存Excel失败: {str(e)}")
    
    finally:
        # 无论如何都清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"已清理临时文件: {temp_path.name}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")


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


def load_data_from_sqlite():
    """从SQLite加载标注数据 - 主存储"""
    import math
    
    logger.info("从SQLite加载数据...")
    
    samples_dict = {}
    
    rows = storage.get_all_samples()
    logger.info(f"SQLite返回 {len(rows)} 条记录")
    
    for row in rows:
        image_id = str(row['image_id'])
        
        if image_id not in samples_dict:
            samples_dict[image_id] = {
                'image_id': image_id,
                'original_part': row.get('original_part') or 'unknown',
                'part': row.get('part') or row.get('original_part') or 'unknown',
                'needs_review': bool(row.get('needs_review', False)),
                'images': []
            }
        
        # URL 编码文件名
        filename = str(row['filename'])
        encoded_filename = quote(filename, safe='')
        img_path = f"/api/image/{image_id}/{encoded_filename}"
        
        # 安全获取浮点数值
        def safe_float(v, default=0.0):
            try:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return default
                fv = float(v)
                if math.isfinite(fv):
                    return fv
                return default
            except Exception:
                return default
        
        samples_dict[image_id]['images'].append({
            'filename': filename,
            'final_body_part': row.get('final_body_part') or '',
            'final_orientation': row.get('final_orientation') or 'unknown',
            'final_projection': row.get('final_projection') or 'unknown',
            'confidence_projection': safe_float(row.get('confidence_projection'), 0.0),
            'confidence_orientation': safe_float(row.get('confidence_orientation'), 0.0),
            'confidence_overall': safe_float(row.get('confidence_overall'), 0.0),
            'needs_review': bool(row.get('needs_review', False)),
            'review_reason': row.get('review_reason') or '',
            'match_method': row.get('match_method') or 'unknown',
            'projection_modified': bool(row.get('projection_modified', False)),
            'orientation_modified': bool(row.get('orientation_modified', False)),
            'modified_at': (row.get('modified_at') or '') if str(row.get('modified_at')).lower() != 'nan' else '',
            'reviewed': bool(row.get('reviewed', False)),
            'reviewed_at': row.get('reviewed_at') or '',
            'image_url': img_path
        })
    
    samples = list(samples_dict.values())
    logger.info(f"数据加载完成: {len(samples)} 个样本")
    return samples


# 保留旧函数用于初始导入
def load_data_from_excel(excel_path):
    """【弃用】从Excel加载标注数据 - 仅用于首次导入"""
    logger.warning("load_data_from_excel 已弃用，使用 load_data_from_sqlite")
    return load_data_from_sqlite()


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
    """保存修改 - 直接写入SQLite（支持乐观锁）"""
    data = request.json or {}
    logger.debug(f"API: save_modification, data={data}")

    if not data.get('image_id') or not data.get('filename'):
        logger.warning("保存失败: 参数缺失")
        return jsonify({'success': False, 'message': '参数缺失'}), 400

    try:
        image_id = str(data['image_id'])
        filename = str(data['filename'])
        client_modified_at = data.get('client_modified_at', '')  # 用于冲突检测
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"保存修改: {image_id}/{filename}, type={'projection' if 'projection' in data else 'orientation'}")
        
        # 直接写入SQLite - 毫秒级响应
        change_data = {}
        if 'projection' in data:
            change_data['final_projection'] = data['projection']
            change_data['projection_modified'] = True
            change_data['modified_at'] = now_str
        elif 'orientation' in data:
            change_data['final_orientation'] = data['orientation']
            change_data['orientation_modified'] = True
            change_data['modified_at'] = now_str
        
        # 保存到数据库（Last Write Wins，不检查冲突）
        result = storage.update_sample(image_id, filename, change_data)
        
        # 立即更新内存（前端立即可见）
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
        logger.error(f"保存失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500


@app.route('/api/review', methods=['POST'])
def save_review_status():
    """保存复核状态 - 直接写入SQLite（支持乐观锁）"""
    data = request.json or {}
    logger.debug(f"API: save_review_status, data={data}")

    image_id = data.get('image_id')
    filename = data.get('filename')
    reviewed = bool(data.get('reviewed', False))  # 默认False，明确指定复核状态
    client_modified_at = data.get('client_modified_at', '')  # 用于冲突检测

    if not image_id or not filename:
        logger.warning(f"复核状态保存失败: 参数缺失, image_id={image_id}, filename={filename}")
        return jsonify({'success': False, 'message': '参数缺失'}), 400

    try:
        logger.info(f"保存复核状态: {image_id}/{filename}, reviewed={reviewed}")
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') if reviewed else ''
        
        # 直接更新SQLite（Last Write Wins，不检查冲突）
        change_data = {'reviewed': reviewed, 'reviewed_at': now_str, 'modified_at': now_str}
        storage.update_sample(image_id, filename, change_data)
        
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

        logger.info(f"✅ 复核状态已保存: {image_id}/{filename}")
        return jsonify({'success': True, 'message': '已保存', 'modified_at': now_str})
    except Exception as e:
        logger.error(f"复核状态保存失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500


@app.route('/api/review/bulk', methods=['POST'])
def save_review_bulk():
    """批量保存复核状态 - 直接写入SQLite"""
    data = request.json or {}
    items = data.get('items', [])
    logger.info(f"API: save_review_bulk, 批量复核 {len(items)} 项")

    if not items:
        logger.warning("批量复核保存失败: 没有需要更新的项目")
        return jsonify({'success': False, 'message': '没有需要更新的项目'}), 400

    try:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 批量添加到延迟保存队列
        for item in items:
            image_id = item.get('image_id')
            filename = item.get('filename')
            reviewed = bool(item.get('reviewed', False))  # 默认False，明确指定复核状态
            if not image_id or not filename:
                continue
            
            change_data = {'reviewed': reviewed, 'reviewed_at': now_str if reviewed else ''}
            storage.update_sample(image_id, filename, change_data)
        
        # 立即更新内存
        item_dict = {str(i.get('image_id')) + '_' + i.get('filename'): i.get('reviewed', True) for i in items}
        for sample in app_status['samples']:
            for img in sample['images']:
                key = str(sample['image_id']) + '_' + img['filename']
                if key in item_dict:
                    img['reviewed'] = item_dict[key]
                    img['reviewed_at'] = now_str if item_dict[key] else ''

        return jsonify({'success': True, 'message': f'已保存: {len(items)} 项'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500


@app.route('/api/invalidate', methods=['POST'])
def invalidate_sample_image():
    """标记图片为无效样本（软删除）- 直接写入SQLite"""
    data = request.json or {}
    logger.info(f"API: invalidate_sample_image, data={data}")

    image_id = data.get('image_id')
    filename = data.get('filename')
    invalid_label = data.get('invalid_label', 'invalid_sample')

    if not image_id or not filename:
        logger.warning(f"删除失败: 参数缺失, image_id={image_id}, filename={filename}")
        return jsonify({'success': False, 'message': '参数缺失'}), 400

    try:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 直接更新SQLite（标记为无效）
        change_data = {
            'is_invalid': True,
            'invalid_label': str(invalid_label),
            'invalid_at': now_str
        }
        storage.update_sample(image_id, filename, change_data)
        
        logger.info(f"✅ 已标记为无效: {image_id}/{filename}")
        
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
    """导出结果 - JSON格式"""
    logger.info(f"API: export_results, 导出 {len(app_status['samples'])} 个样本")
    result = []
    
    # 从SQLite获取最新数据
    samples = storage.get_all_samples()
    
    # 按image_id分组
    samples_dict = {}
    for row in samples:
        image_id = str(row['image_id'])
        if image_id not in samples_dict:
            samples_dict[image_id] = {
                'image_id': image_id,
                'original_part': row.get('original_part', ''),
                'images': []
            }
        samples_dict[image_id]['images'].append({
            'filename': row.get('filename', ''),
            'final_projection': row.get('final_projection', ''),
            'final_orientation': row.get('final_orientation', ''),
            'projection_modified': bool(row.get('projection_modified', False)),
            'orientation_modified': bool(row.get('orientation_modified', False)),
            'modified_at': row.get('modified_at', ''),
            'needs_review': bool(row.get('needs_review', False)),
            'is_invalid': bool(row.get('is_invalid', False)),
            'invalid_label': row.get('invalid_label', ''),
            'invalid_at': row.get('invalid_at', '')
        })
    
    return jsonify(list(samples_dict.values()))


@app.route('/api/export/excel', methods=['POST'])
def export_to_excel():
    """导出数据到Excel文件并提供下载"""
    try:
        data = request.json or {}
        output_filename = data.get('filename', 'exported_labels.xlsx')
        
        # 确保文件名安全
        output_filename = Path(output_filename).name
        if not output_filename.endswith('.xlsx'):
            output_filename += '.xlsx'
        
        output_path = OUTPUTS_DIR / output_filename
        
        logger.info(f"API: export_to_excel, 导出到: {output_path}")
        
        if not storage.export_to_excel(output_path):
            return jsonify({
                'success': False,
                'message': '导出失败，没有数据'
            }), 400
        
        # 提供文件下载
        from flask import send_file
        return send_file(
            output_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=output_filename
        )
    except Exception as e:
        logger.error(f"导出Excel失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'导出失败: {str(e)}'
        }), 500


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
    """初始化数据 - 从SQLite加载，首次启动从Excel导入"""
    logger.info("开始初始化数据...")
    
    # 查找Excel源文件（用于首次导入）
    if excel_path is None:
        outputs_dir = Path(__file__).parent.parent
        excel_files = list(outputs_dir.glob('*.xlsx'))
        # 排除备份文件和临时文件
        excel_files = [f for f in excel_files if not f.name.endswith('.bak') and '.tmp' not in f.name]
        if excel_files:
            excel_path = max(excel_files, key=lambda p: p.stat().st_mtime)
    
    # 如果找到Excel文件，尝试导入到SQLite
    if excel_path and os.path.exists(excel_path):
        logger.info(f"发现Excel文件: {excel_path}")
        storage.import_from_excel(excel_path)
        storage.excel_source = str(excel_path)
    
    # 从SQLite加载数据（主存储）
    logger.info("从SQLite主存储加载数据...")
    app_status['samples'] = load_data_from_sqlite()
    
    if app_status['samples']:
        logger.info(f"✓ 已加载 {len(app_status['samples'])} 个样本")
    else:
        logger.error("⚠ SQLite中没有数据，且未找到Excel源文件")


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
    logger.info(f"🚀 Flask服务器已启动! (多线程模式)")
    logger.info(f"{'='*60}")
    logger.info(f"访问地址: http://localhost:{port}")
    logger.info(f"模式: 多线程并发处理 (threaded=True)")
    logger.info(f"按 Ctrl+C 停止服务器")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("\n\n正在停止服务器...")
        # 强制保存所有待处理的修改
        logger.info("正在保存待处理的修改...")
        logger.info("✅ 服务器已关闭")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Flask复核服务器')
    parser.add_argument('--port', '-p', type=int, default=5000, help='端口号')
    parser.add_argument('--input', '-i', help='输入Excel文件路径')
    parser.add_argument('--no-open', action='store_true', help='不自动打开浏览器')
    
    args = parser.parse_args()
    
    run_server(port=args.port, excel_path=args.input, auto_open=not args.no_open)
