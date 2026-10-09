"""So sánh cách resize ảnh: project này (giữ tỉ lệ + pad) vs MILK10K_SOLUTION (resize thẳng 224x224).

Chạy: python EDA/resize_compare.py   ->  EDA/figures/12_resize_comparison.png
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms.v2 import functional as TF  # noqa: E402

from milk10k import CLASSES  # noqa: E402
from milk10k.data.transforms import ResizePad  # noqa: E402

SIZE = 224
DATA = ROOT / "datasets" / "MILK10k"
OUT = ROOT / "EDA" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2 = "#0b0b0b", "#52514e"

df = pd.read_csv(DATA / "train" / "train_combined.csv")
df["label"] = df[CLASSES].to_numpy().argmax(1)
# 3 lesion minh hoạ: tổn thương nhỏ (NV), tổn thương lớn (BCC), lớp hiếm (DF)
picks = [df[df.label == CLASSES.index(c)].sample(1, random_state=7).iloc[0] for c in ("NV", "BCC", "DF")]

ours = ResizePad(SIZE)                                   # đúng transform đang dùng (trước normalize)
cols = ["Ảnh gốc 600×450", f"Project này: giữ tỉ lệ + pad\n{SIZE}×{SIZE} (ảnh thật {SIZE}×{SIZE * 3 // 4}, 25% viền đen)",
        f"Project kia: resize thẳng\n{SIZE}×{SIZE} (dùng 100% điểm ảnh, méo 4:3 → 1:1)"]
fig, axes = plt.subplots(len(picks) * 2, 3, figsize=(11, 3.3 * len(picks) * 2))
for i, row in enumerate(picks):
    for j, (view, col) in enumerate((("clinical", "clinical_path"), ("dermoscopy", "derm_path"))):
        img = Image.open(DATA / "train" / "MILK10k_Training_Input" / row[col]).convert("RGB")
        a = ours(TF.pil_to_tensor(img)).permute(1, 2, 0).numpy()
        b = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR))
        r = i * 2 + j
        for c, im in enumerate((np.asarray(img), a, b)):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#c3c2b7")
            if r == 0:
                ax.set_title(cols[c], fontsize=9.5, color=INK)
        axes[r, 0].set_ylabel(f"{CLASSES[row.label]} · {view}", fontsize=10, color=INK2)
fig.suptitle("Cách resize ảnh trước khi đưa vào mô hình (224px)", fontsize=13, fontweight="bold", color=INK, y=0.995)
fig.tight_layout()
fig.savefig(OUT / "12_resize_comparison.png", dpi=130, facecolor="white")
print("Đã lưu", OUT / "12_resize_comparison.png")
