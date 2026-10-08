#!/usr/bin/env bash
# Ablation nhánh ảnh: chạy 1 config với [clin], [derm], [clin,derm] rồi in bảng so sánh.
#   bash scripts/run_ablation_views.sh configs/baselines/cnn_efficientnet_b0.yaml [override thêm...]
set -e
cd "$(dirname "$0")/.."
CFG=${1:?Cần đường dẫn config}; shift
for v in "[clin]" "[derm]" "[clin,derm]"; do
  python scripts/train.py --config "$CFG" --set "model.views=$v" "$@"
done
python scripts/compare_runs.py
