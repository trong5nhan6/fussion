"""Import trước mọi thứ trong các script: thêm src/ vào sys.path."""
import os
import sys
from pathlib import Path

# Anaconda trên Windows nạp 2 bản OpenMP (numpy MKL + torch) -> lỗi OMP #15 khi import torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Console Windows mặc định cp1252 -> lỗi khi in tiếng Việt.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
