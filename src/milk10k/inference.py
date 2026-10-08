"""Suy luận tập test và ghi file nộp theo định dạng ISIC MILK10k."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import CLASSES
from .data import MetadataEncoder, MilkDataset, build_transforms, image_dir, load_test_df
from .metrics import THRESHOLD, postprocess
from .models import build_model

ID_COL = "lesion_id"  # code chấm điểm (isic-challenge-scoring) chấp nhận "image" hoặc "lesion_id"


def fold_checkpoints(run_dir: Path) -> list[Path]:
    """Mỗi fold: ưu tiên checkpoint sau cRT nếu có."""
    return [d / "best_crt.pt" if (d / "best_crt.pt").exists() else d / "best.pt"
            for d in sorted(run_dir.glob("fold*")) if (d / "best.pt").exists()]


def validate_submission(sub: pd.DataFrame, expected_ids) -> None:
    """Các kiểm tra giống load_csv.py của ISIC: cột, trùng, thiếu, kiểu số, khoảng [0, 1]."""
    if list(sub.columns) != [ID_COL] + CLASSES:
        raise ValueError(f"Cột sai: {list(sub.columns)}")
    if sub[ID_COL].duplicated().any():
        raise ValueError("Có lesion trùng lặp")
    if set(sub[ID_COL]) != set(expected_ids):
        raise ValueError("Danh sách lesion không khớp tập test")
    vals = sub[CLASSES]
    if vals.isna().any().any() or not all(np.issubdtype(t, np.floating) for t in vals.dtypes):
        raise ValueError("Có giá trị thiếu hoặc không phải số thực")
    if ((vals < 0) | (vals > 1)).any().any():
        raise ValueError("Có giá trị ngoài [0, 1]")


def write_submission(ids, probs: np.ndarray, mode: str, out_dir: Path, expected_ids, logger=print) -> Path:
    """Ghi submission.csv (điểm sau hậu xử lý, để nộp) và test_probs.csv (softmax thô, để ensemble sau)."""
    raw = pd.DataFrame(probs, columns=CLASSES)
    raw.insert(0, ID_COL, list(ids))
    raw.to_csv(out_dir / "test_probs.csv", index=False)

    sub = pd.DataFrame(postprocess(probs.astype(np.float64), mode), columns=CLASSES)
    sub.insert(0, ID_COL, list(ids))
    sub = sub.sort_values(ID_COL).reset_index(drop=True)
    validate_submission(sub, expected_ids)
    path = out_dir / "submission.csv"
    sub.to_csv(path, index=False, float_format="%.6f")

    positives = (sub[CLASSES] > THRESHOLD)
    dist = positives.sum().to_dict()
    logger(f"File nộp: {path} | {len(sub)} lesion | hậu xử lý={mode} | "
           f"số lesion không có lớp dương: {int((positives.sum(axis=1) == 0).sum())}")
    logger("Số dự đoán dương theo lớp (test): " + ", ".join(f"{c}={n}" for c, n in dist.items()))
    return path


def predict_test(cfg: dict, run_dir: Path, device, tta: bool, logger=print) -> Path:
    from .engine import make_loader, predict  # tránh import vòng

    cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}  # trọng số lấy từ checkpoint
    test_df = load_test_df(cfg)
    amp = cfg["train"].get("amp", False) and device.type == "cuda"
    ckpts = fold_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"Không có checkpoint trong {run_dir}")

    all_probs, ids = [], None
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location=device)
        meta_enc = MetadataEncoder().load_state_dict(ckpt["meta_encoder"])
        model = build_model(cfg, meta_enc.dim).to(device)
        model.load_state_dict(ckpt["model"])
        mean, std = model.normalization()
        ds = MilkDataset(test_df, image_dir(cfg, "test"), cfg["model"]["views"],
                         build_transforms(cfg["data"]["img_size"], False, mean, std), meta_enc.transform(test_df))
        fold_ids, probs, _ = predict(model, make_loader(ds, cfg, False), device, tta=tta, amp=amp)
        if ids is not None and list(fold_ids) != list(ids):
            raise RuntimeError("Thứ tự lesion giữa các fold không khớp")
        ids = fold_ids
        all_probs.append(probs)
        logger(f"Test: {ckpt_path.parent.name}/{ckpt_path.name} xong (TTA={tta})")

    return write_submission(ids, np.mean(all_probs, axis=0), cfg["predict"].get("postprocess", "top1"),
                            run_dir, test_df.lesion_id, logger)
