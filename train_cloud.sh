#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "  云端GPU训练启动脚本"
echo "=========================================="

echo "[1/4] 检查CUDA可用性..."
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

echo ""
echo "[2/4] 检查数据文件..."
if [ -f "data/hr/train_hr.csv" ] && [ -f "data/hr/val_hr.csv" ]; then
    echo "训练数据: $(wc -l < data/hr/train_hr.csv) 行"
    echo "验证数据: $(wc -l < data/hr/val_hr.csv) 行"
else
    echo "错误: 数据文件不存在!"
    exit 1
fi

echo ""
echo "[3/4] 创建输出目录..."
mkdir -p checkpoints_hr

echo ""
echo "[4/4] 启动训练..."
echo "训练参数:"
echo "  NUM_EPOCHS=${NUM_EPOCHS:-100}"
echo "  BATCH_SIZE=${BATCH_SIZE:-16}"
echo "  LEARNING_RATE=${LEARNING_RATE:-5e-4}"
echo "  MODEL_ARCH=${MODEL_ARCH:-causal_gru}"
echo "  FEATURE_MODE=${FEATURE_MODE:-enhanced}"
echo ""

python3 code/train.py 2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "  训练完成!"
echo "=========================================="