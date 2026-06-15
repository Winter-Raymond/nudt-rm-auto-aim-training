#!/bin/bash

# 目录切换到当前脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}" || exit 1

# 清理旧临时文件、旧输出
rm -f detector.tmp
rm -rf output/*

echo "===== 启动 Python 检测端 ====="
python3 detector.py &
PY_PID=$!
echo "Python PID: ${PY_PID}"

# 短暂等待 Python 初始化
sleep 0.8

echo "===== 启动 C++ 跟踪端 ====="
./build/tracker &
CPP_PID=$!
echo "C++ PID: ${CPP_PID}"

# 等待程序退出，结束后杀死两个进程
wait ${CPP_PID}
echo "程序结束，清理进程..."

kill ${PY_PID} 2>/dev/null
rm -f detector.tmp

echo "全部退出完成"
