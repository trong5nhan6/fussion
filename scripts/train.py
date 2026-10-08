"""Train một cấu hình qua các fold -> checkpoint, OOF, metrics, train.log và file nộp test.

Ví dụ:
  python scripts/train.py --config configs/baselines/cnn_efficientnet_b0.yaml
  python scripts/train.py --config configs/baselines/vit_small.yaml --set model.views=[derm]
  python scripts/train.py --config configs/baselines/vit_small.yaml configs/imbalance/logit_adjusted.yaml
  python scripts/train.py --config configs/baselines/vit_small.yaml --set train.folds=[0,1,2,3,4]
"""
import argparse
import time
from datetime import datetime

import _bootstrap  # noqa: F401
import pandas as pd
import torch

from milk10k import CLASSES
from milk10k.config import load_config, save_config
from milk10k.data import load_train_df
from milk10k.engine import log_leaderboard, train_one_fold
from milk10k.inference import predict_test
from milk10k.metrics import SUMMARY_KEYS, compute_metrics
from milk10k.utils import get_device, get_logger, resolve_path, save_json, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, nargs="+",
                    help="1 hoặc nhiều file, gộp theo thứ tự: baseline rồi overlay (vd configs/imbalance/*.yaml)")
    ap.add_argument("--set", nargs="*", default=[], help="Ghi đè config, ví dụ train.epochs=5")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config([resolve_path(c) for c in args.config], args.set)
    views = "+".join(cfg["model"].get("views", []))
    tag = f"_{cfg['tag']}" if cfg.get("tag") else ""
    run_name = args.run_name or f"{cfg['name']}{tag}_{views}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = resolve_path(cfg["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    logger = get_logger(run_dir / "train.log")
    device = get_device()
    torch.backends.cudnn.benchmark = True  # kích thước ảnh cố định -> cuDNN chọn thuật toán nhanh nhất
    mode = cfg["predict"].get("postprocess", "top1")
    logger.info(f"Run          : {run_name}")
    logger.info(f"Thư mục      : {run_dir}")
    logger.info(f"Config       : {' + '.join(args.config)}" + (f" | --set {' '.join(args.set)}" if args.set else ""))
    logger.info(f"Folds        : {cfg['train']['folds']}" + (" | DEBUG" if cfg.get("debug") else ""))

    t0 = time.time()
    df = load_train_df(cfg)
    oofs, per_fold = [], {}
    for fold in cfg["train"]["folds"]:
        seed_everything(cfg["seed"] + fold)
        oof = train_one_fold(cfg, fold, df, run_dir, device, logger)
        m = compute_metrics(oof.label.to_numpy(), oof[CLASSES].to_numpy(), mode)
        per_fold[fold] = {k: m[k] for k in SUMMARY_KEYS}
        log_leaderboard(logger, f"[fold {fold}] VAL (checkpoint tốt nhất, TTA={cfg['predict']['tta']})", m)
        oofs.append(oof)
        pd.concat(oofs).to_csv(run_dir / "oof.csv", index=False)  # lưu dần, phòng khi bị dừng giữa chừng

    oof = pd.concat(oofs, ignore_index=True)
    overall = compute_metrics(oof.label.to_numpy(), oof[CLASSES].to_numpy(), mode)
    imbalance = {"loss": cfg["loss"], "sampler_q": cfg["train"].get("sampler_q"),
                 "crt": cfg.get("crt", {}).get("enabled", False)}
    save_json({"run": run_name, "name": cfg["name"], "tag": cfg.get("tag", ""), "model": cfg["model"], "imbalance": imbalance,
               "postprocess": mode, "folds": cfg["train"]["folds"],
               "overall": overall, "per_fold": per_fold}, run_dir / "metrics.json")
    logger.info("=" * 100)
    n = len(cfg["train"]["folds"])
    log_leaderboard(logger, "KẾT QUẢ VAL" + (" (fold %d)" % cfg["train"]["folds"][0] if n == 1 else f" (OOF {n} fold)"),
                    overall)

    if cfg["predict"].get("after_train", True):
        logger.info("=" * 100)
        predict_test(cfg, run_dir, device, tta=cfg["predict"]["tta"], logger=logger.info)
    if not cfg.get("keep_checkpoints", True):
        for ckpt in run_dir.glob("fold*/best*.pt"):
            ckpt.unlink()
        logger.info("Đã xoá checkpoint (keep_checkpoints=false); test_probs.csv vẫn giữ để ensemble")
    logger.info(f"Tổng thời gian: {(time.time() - t0) / 60:.1f} phút")


if __name__ == "__main__":
    main()
