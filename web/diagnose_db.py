#!/usr/bin/env python3
"""诊断数据库字段问题"""

import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / 'outputs' / 'medical_labels.db'

if not db_path.exists():
    print(f"❌ 数据库不存在: {db_path}")
    exit(1)

print(f"✓ 找到数据库: {db_path}")
print()

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# 1. 检查表结构
print("=== 表结构 ===")
cursor.execute("PRAGMA table_info(samples)")
for row in cursor.fetchall():
    print(f"  {row[1]}: {row[2]}")
print()

# 2. 检查记录总数
print("=== 记录统计 ===")
cursor.execute("SELECT COUNT(*) FROM samples")
total = cursor.fetchone()[0]
print(f"总记录数: {total}")

# 3. 检查 modified_at 字段分布
print("\n=== modified_at 字段分布 ===")
cursor.execute("SELECT modified_at, COUNT(*) FROM samples GROUP BY modified_at")
rows = cursor.fetchall()
for val, count in rows:
    display = f"'{val}'" if val else "NULL/空字符串"
    print(f"  {display}: {count} 条")

# 4. 检查是否有 'nan' 字符串
print("\n=== 检查 'nan' 字符串 ===")
cursor.execute("SELECT COUNT(*) FROM samples WHERE modified_at = 'nan' OR modified_at = 'NaN'")
nan_count = cursor.fetchone()[0]
print(f"modified_at = 'nan' 的记录数: {nan_count}")

# 5. 查看几条样例数据
print("\n=== 样例数据（前3条）===")
cursor.execute("SELECT image_id, filename, modified_at, reviewed FROM samples LIMIT 3")
for row in cursor.fetchall():
    print(f"  image_id={row[0]}, filename={row[1]}, modified_at='{row[2]}', reviewed={row[3]}")

conn.close()
print("\n诊断完成!")
