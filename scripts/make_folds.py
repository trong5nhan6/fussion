"""Chia train thành K fold phân tầng theo nhãn. Chạy: python scripts/make_folds.py"""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from milk10k import CLASSES
from milk10k.config import load_config
from milk10k.utils import resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))

    df = pd.read_csv(resolve_path(cfg["data"]["root"]) / cfg["data"]["train_csv"])
    df["label"] = df[CLASSES].to_numpy().argmax(axis=1)
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=cfg["seed"])
    for k, (_, va) in enumerate(skf.split(df, df.label)):
        df.loc[va, "fold"] = k
    df["label_name"] = [CLASSES[i] for i in df.label]

    out = resolve_path(cfg["data"]["folds_csv"])
    df[["lesion_id", "label_name", "fold"]].to_csv(out, index=False)
    print(f"Đã ghi {out}")
    print(pd.crosstab(df.label_name, df.fold, margins=True).to_string())


if __name__ == "__main__":
    main()
