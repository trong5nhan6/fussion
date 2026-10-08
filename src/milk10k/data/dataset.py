from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .. import CLASSES
from ..utils import resolve_path

VIEW_COLUMNS = {"clin": "clinical_path", "derm": "derm_path"}


def load_train_df(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    df = pd.read_csv(resolve_path(d["root"]) / d["train_csv"])
    df["label"] = df[CLASSES].to_numpy().argmax(axis=1)
    folds_path = resolve_path(d["folds_csv"])
    if not folds_path.exists():
        raise FileNotFoundError(f"Không thấy {folds_path}. Hãy chạy: python scripts/make_folds.py")
    folds = pd.read_csv(folds_path)[["lesion_id", "fold"]]
    df = df.merge(folds, on="lesion_id", how="left", validate="1:1")
    if df["fold"].isna().any():
        raise ValueError("Một số lesion không có fold — file folds không khớp với train_combined.csv")
    df["fold"] = df["fold"].astype(int)
    return df


def load_test_df(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    return pd.read_csv(resolve_path(d["root"]) / d["test_csv"])


def image_dir(cfg: dict, split: str) -> Path:
    d = cfg["data"]
    return resolve_path(d["root"]) / d[f"{split}_img_dir"]


class MilkDataset(Dataset):
    """Mỗi mẫu là một lesion: dict gồm ảnh theo từng view, vector metadata và nhãn (nếu có)."""

    def __init__(self, df: pd.DataFrame, img_dir: Path, views, transform, meta: np.ndarray | None = None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.views = list(views)
        self.transform = transform
        self.meta = meta
        self.has_label = "label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = {}
        for v in self.views:
            with Image.open(self.img_dir / row[VIEW_COLUMNS[v]]) as im:
                images[v] = self.transform(im.convert("RGB"))
        item = {"lesion_id": row["lesion_id"], "images": images}
        if self.meta is not None:
            item["meta"] = torch.from_numpy(self.meta[idx])
        if self.has_label:
            item["label"] = torch.tensor(int(row["label"]))
        return item
