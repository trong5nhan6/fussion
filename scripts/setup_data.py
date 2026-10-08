"""Chuẩn bị dữ liệu MILK10k: tải file zip từ Google Drive (hoặc dùng thư mục có sẵn), giải nén, kiểm tra.

File zip trên Drive cần chứa thư mục MILK10k với cấu trúc:
  MILK10k/train/train_combined.csv, MILK10k/train/MILK10k_Training_Input/<lesion_id>/*.jpg
  MILK10k/test/test_combined.csv,   MILK10k/test/MILK10k_Test_Input/<lesion_id>/*.jpg
(Tạo bằng cách nén thư mục datasets/MILK10k.) Chia sẻ ở chế độ "Anyone with the link".

  python scripts/setup_data.py --drive-url "https://drive.google.com/file/d/<ID>/view?usp=sharing" --dest /tmp/MILK10k
  python scripts/setup_data.py --source /kaggle/input/milk10k --dest /tmp/MILK10k   # dữ liệu đã có (Kaggle Dataset)
  python scripts/setup_data.py --dest datasets/MILK10k --check-only
"""
import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from milk10k.utils import resolve_path

EXPECTED = {"train": 5240, "test": 479}


def find_root(base: Path) -> Path | None:
    """Tìm thư mục chứa train/train_combined.csv (zip có thể lồng thêm cấp thư mục)."""
    for csv in base.rglob("train_combined.csv"):
        if csv.parent.name == "train":
            return csv.parent.parent
    return None


def check(root: Path) -> list[str]:
    """Trả về danh sách lỗi (rỗng = hợp lệ)."""
    errors = []
    specs = [("train", "train/train_combined.csv", "train/MILK10k_Training_Input"),
             ("test", "test/test_combined.csv", "test/MILK10k_Test_Input")]
    for split, csv_rel, img_rel in specs:
        csv = root / csv_rel
        if not csv.exists():
            errors.append(f"thiếu {csv_rel}")
            continue
        df = pd.read_csv(csv)
        if len(df) != EXPECTED[split]:
            errors.append(f"{csv_rel}: {len(df)} dòng, cần {EXPECTED[split]}")
        paths = pd.concat([df.clinical_path, df.derm_path])
        missing = [p for p in paths if not (root / img_rel / p).exists()]
        if missing:
            errors.append(f"{split}: thiếu {len(missing)} ảnh, vd {missing[:3]}")
        else:
            print(f"  {split}: {len(df)} lesion, {len(paths)} ảnh — OK")
    return errors


def download(url: str, work: Path) -> Path:
    try:
        import gdown
    except ImportError:
        sys.exit("Cần gdown: pip install gdown")
    work.mkdir(parents=True, exist_ok=True)
    out = work / "milk10k.zip"
    # Lấy file ID từ link chia sẻ (.../file/d/<ID>/view hoặc ...?id=<ID>); tham số id= có ở gdown 4.x-6.x
    match = re.search(r"/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    if not match:
        sys.exit(f"Không đọc được file ID từ link Google Drive: {url}")
    if not out.exists():
        print(f"Tải dữ liệu từ Google Drive (id={match.group(1)}) -> {out}")
        if gdown.download(id=match.group(1), output=str(out), quiet=False) is None:
            sys.exit("Tải thất bại: kiểm tra link và quyền chia sẻ 'Anyone with the link'")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-url", default=None)
    ap.add_argument("--source", default=None, help="Thư mục/zip dữ liệu có sẵn (vd Kaggle Dataset)")
    ap.add_argument("--dest", default="datasets/MILK10k")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    dest = resolve_path(args.dest)

    if dest.exists() and not check(dest):
        print(f"Dữ liệu đã sẵn sàng tại {dest}")
        return
    if args.check_only:
        sys.exit(f"Dữ liệu tại {dest} chưa hợp lệ")

    src = Path(args.source) if args.source else download(args.drive_url, dest.parent / "_download") \
        if args.drive_url else sys.exit("Cần --drive-url hoặc --source")
    if src.is_file() and src.suffix == ".zip":
        extract_dir = dest.parent / "_extract"
        print(f"Giải nén {src} ...")
        with zipfile.ZipFile(src) as z:
            z.extractall(extract_dir)
        src = extract_dir
    root = find_root(src)
    if root is None:
        sys.exit(f"Không tìm thấy train/train_combined.csv trong {src}")

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if os.access(root, os.W_OK) and not args.source:
        shutil.move(str(root), dest)        # dữ liệu vừa giải nén -> chuyển, không chép
    else:
        shutil.copytree(root, dest)         # nguồn chỉ đọc (Kaggle input) -> chép
    for tmp in (dest.parent / "_extract", dest.parent / "_download"):
        shutil.rmtree(tmp, ignore_errors=True)

    errors = check(dest)
    if errors:
        sys.exit("Dữ liệu lỗi:\n  " + "\n  ".join(errors))
    print(f"Dữ liệu sẵn sàng tại {dest}")


if __name__ == "__main__":
    main()
