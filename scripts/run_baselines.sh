#!/usr/bin/env bash
# Chạy toàn bộ baseline rồi in bảng xếp hạng. Ghi đè chung qua biến EXTRA, ví dụ:
#   EXTRA="train.folds=[0] train.epochs=10" bash scripts/run_baselines.sh
set -e
cd "$(dirname "$0")/.."
python scripts/train_tabular.py --model logreg
python scripts/train_tabular.py --model hgb
for cfg in configs/baselines/*.yaml; do
  python scripts/train.py --config "$cfg" --set $EXTRA
done
python scripts/compare_runs.py
