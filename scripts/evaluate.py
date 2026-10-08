"""Phân tích dự đoán OOF của một run: bảng leaderboard theo lớp + confusion matrix.

  python scripts/evaluate.py --run outputs/<run_name>
  python scripts/evaluate.py --run outputs/<run_name> --postprocess softmax   # xem ảnh hưởng của hậu xử lý
"""
import argparse

import _bootstrap  # noqa: F401
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from milk10k import CLASSES  # noqa: E402
from milk10k.config import load_config  # noqa: E402
from milk10k.metrics import compute_metrics, confusion, leaderboard_table, summary_line  # noqa: E402
from milk10k.utils import resolve_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--postprocess", choices=["top1", "softmax"], default=None)
    args = ap.parse_args()
    run_dir = resolve_path(args.run)
    mode = args.postprocess
    if mode is None:
        cfg_path = run_dir / "config.yaml"
        mode = load_config(cfg_path)["predict"].get("postprocess", "top1") if cfg_path.exists() else "top1"

    oof = pd.read_csv(run_dir / "oof.csv")
    y, p = oof.label.to_numpy(), oof[CLASSES].to_numpy()
    m = compute_metrics(y, p, mode)
    print(f"Hậu xử lý: {mode}\n{summary_line(m)} | balanced_acc={m['balanced_acc']:.4f} top1_acc={m['top1_acc']:.4f}\n")
    print(leaderboard_table(m))
    pd.DataFrame(m["per_class"]).T.round(4).to_csv(run_dir / f"per_class_{mode}.csv")

    cm = confusion(y, p)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.6 else "#0b0b0b")
    ax.set_xlabel("Dự đoán (argmax)")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion matrix OOF — {run_dir.name}\n(màu: tỉ lệ theo hàng, số: số lượng)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    print(f"\nĐã lưu per_class_{mode}.csv và confusion_matrix.png vào {run_dir}")


if __name__ == "__main__":
    main()
