#!/usr/bin/env bash
# Chạy toàn bộ thí nghiệm imbalance trên 1 baseline (mặc định fold 0) rồi in bảng so sánh.
#   bash scripts/run_imbalance.sh configs/baselines/cnn_efficientnet_b0.yaml [override thêm...]
set -e
cd "$(dirname "$0")/.."
BASE=${1:?Cần config baseline}; shift
for ov in configs/imbalance/*.yaml; do
  python scripts/train.py --config "$BASE" "$ov" --set "$@"
done
python scripts/compare_runs.py
