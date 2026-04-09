#!/usr/bin/env bash
echo "=========================================="
echo "  训练监控脚本"
echo "=========================================="
echo ""

echo "[GPU状态]"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        IFS=',' read -r idx name temp util mem_used mem_total <<< "$line"
        echo "  GPU $idx: $name"
        echo "    温度: ${temp}°C | 利用率: ${util}% | 显存: ${mem_used}/${mem_total}MB"
    done
else
    echo "  nvidia-smi 不可用"
fi

echo ""
echo "[训练进程]"
if pgrep -f "train.py" > /dev/null; then
    echo "  训练进程运行中 (PID: $(pgrep -f 'train.py'))"
else
    echo "  无训练进程"
fi

echo ""
echo "[最新日志]"
log_file=$(ls -t train_*.log 2>/dev/null | head -1)
if [ -n "$log_file" ]; then
    echo "  日志文件: $log_file"
    echo "  最后10行:"
    tail -10 "$log_file" | sed 's/^/    /'
else
    echo "  无日志文件"
fi

echo ""
echo "[模型检查点]"
if [ -d "checkpoints_hr" ]; then
    latest_model=$(ls -t checkpoints_hr/*.pth 2>/dev/null | head -1)
    if [ -n "$latest_model" ]; then
        echo "  最新模型: $latest_model"
        echo "  文件大小: $(du -h "$latest_model" | cut -f1)"
        echo "  修改时间: $(stat -c %y "$latest_model" 2>/dev/null | cut -d. -f1)"
    else
        echo "  无模型文件"
    fi
else
    echo "  checkpoints_hr 目录不存在"
fi