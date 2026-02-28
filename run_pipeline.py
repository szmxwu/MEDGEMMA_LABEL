#!/usr/bin/env python3
"""
完整流水线 - LLM标注 + 生成复核页面 + 启动Web服务
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_llm_labeling(workers=4, limit=None):
    """运行LLM标注"""
    print("=" * 60)
    print("步骤1: 运行LLM标注")
    print("=" * 60)
    
    cmd = [sys.executable, 'LLM_lable.py', '--workers', str(workers)]
    if limit:
        cmd.extend(['--limit', str(limit)])
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def start_review_server(port=5000, auto_open=True):
    """启动复核Web服务器"""
    print("\n" + "=" * 60)
    print("步骤2: 启动复核Web服务器")
    print("=" * 60)
    
    cmd = [
        sys.executable,
        'web/app.py',
        '--port', str(port)
    ]
    
    if not auto_open:
        cmd.append('--no-open')
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n服务器已停止")

def main():
    parser = argparse.ArgumentParser(description='X光标注完整流水线')
    parser.add_argument('--workers', '-w', type=int, default=4, help='标注线程数')
    parser.add_argument('--limit', '-l', type=int, help='限制标注样本数')
    parser.add_argument('--port', '-p', type=int, default=5000, help='Web服务器端口')
    parser.add_argument('--no-open', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--review-only', action='store_true', help='仅启动复核服务器')
    
    args = parser.parse_args()
    
    if args.review_only:
        # 仅启动复核服务器
        start_review_server(args.port, not args.no_open)
    else:
        # 完整流水线
        success = run_llm_labeling(args.workers, args.limit)
        if success:
            start_review_server(args.port, not args.no_open)

if __name__ == '__main__':
    main()
