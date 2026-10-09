"""Suy luận lại tập test từ một run đã train (train.py đã tự làm bước này khi xong).

  python scripts/predict.py --run outputs/<run_name>
  python scripts/predict.py --run outputs/<run_name> --postprocess softmax
"""
import argparse

import _bootstrap  # noqa: F401

from milk10k.config import load_config
from milk10k.inference import predict_test
from milk10k.utils import get_device, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--postprocess", choices=["auto", "top1", "softmax", "sigmoid"], default=None)
    args = ap.parse_args()

    run_dir = resolve_path(args.run)
    cfg = load_config(run_dir / "config.yaml")
    if args.postprocess:
        cfg["predict"]["postprocess"] = args.postprocess
    predict_test(cfg, run_dir, get_device(), tta=cfg["predict"]["tta"] and not args.no_tta)


if __name__ == "__main__":
    main()
