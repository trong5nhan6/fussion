"""Gộp kết quả mọi run thành các bảng báo cáo (markdown + CSV).

  python scripts/summarize.py                                   # đọc outputs/
  python scripts/summarize.py --dirs outputs /kaggle/input/a/outputs /kaggle/input/b/outputs

Bảng sinh ra (outputs/summary/):
  1. stage1_backbone_x_view : Dice theo backbone x nhánh ảnh (loss mặc định ce_sqrtinv)
  2. stage2_<backbone>      : Dice theo overlay imbalance x nhánh ảnh
  3. ml_models              : các mô hình ML trên metadata
  4. all_runs.csv           : toàn bộ metric của mọi run
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from milk10k.metrics import LEADERBOARD_COLUMNS
from milk10k.utils import load_json, resolve_path

VIEW_ORDER = ["clin", "derm", "clin+derm"]
TAG_ORDER = ["ce_plain", "ce_sqrtinv", "ce_inv", "cb_b0999", "cb_focal", "la_t1", "samp_q05", "samp_q1", "crt"]
TAG_NAMES = {"ce_plain": "CE", "ce_sqrtinv": "CE + √inv weight", "ce_inv": "CE + inv weight",
             "cb_b0999": "Class-Balanced (β=0.999)", "cb_focal": "CB Focal (γ=2)", "la_t1": "Logit-adjusted (τ=1)",
             "samp_q05": "Sampler q=0.5", "samp_q1": "Sampler q=1", "crt": "cRT (decoupling)"}
METRICS = [name for _, name in LEADERBOARD_COLUMNS]


def collect(dirs) -> pd.DataFrame:
    rows = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*/metrics.json")):
            m = load_json(f)
            if "dice" not in m["overall"]:
                continue
            run = m.get("run", f.parent.name)
            folds = m.get("folds") or sorted(m.get("per_fold", {}))
            row = {"run": run, "folds": "+".join(str(x) for x in folds),
                   **{name: m["overall"][k] for k, name in LEADERBOARD_COLUMNS},
                   "Balanced Acc": m["overall"]["balanced_acc"]}
            if m.get("model", {}).get("inputs") == "metadata":
                row.update(kind="ml", backbone=m["model"]["name"], tag="", views="metadata")
            else:
                parts = run.split("__")  # <backbone>__<tag>__<views>[__<hậu tố>]
                backbone = parts[0] if len(parts) >= 3 else m["model"].get("backbone")
                if len(parts) >= 4:
                    backbone += f" [{parts[3]}]"  # run với tham số khác -> dòng riêng trong bảng
                row.update(kind="image", backbone=backbone,
                           tag=m.get("tag", ""), views="+".join(m["model"].get("views", [])))
            rows[run] = row  # cùng tên run ở nhiều thư mục -> lấy bản đọc sau
    return pd.DataFrame(rows.values())


def pivot(df: pd.DataFrame, index: str, index_order=None, rename=None) -> pd.DataFrame:
    t = df.pivot_table(index=index, columns="views", values="Dice Coefficient", aggfunc="first")
    t = t.reindex(columns=[v for v in VIEW_ORDER if v in t.columns])
    if index_order:
        t = t.reindex([i for i in index_order if i in t.index])
    if rename:
        t.index = [rename.get(i, i) for i in t.index]
    return t


def bold_max(t: pd.DataFrame) -> str:
    """Markdown, in đậm giá trị lớn nhất mỗi cột."""
    out = t.copy().astype(object)
    for c in t.columns:
        col = t[c]
        out[c] = [("**%.4f**" if v == col.max() else "%.4f") % v if pd.notna(v) else "—" for v in col]
    return out.to_markdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["outputs"])
    ap.add_argument("--out", default="outputs/summary")
    args = ap.parse_args()
    df = collect([resolve_path(d) for d in args.dirs])
    if df.empty:
        print("Chưa có run nào.")
        return
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values(["kind", "backbone", "tag", "views"]).to_csv(out / "all_runs.csv", index=False)

    md = ["# Kết quả MILK10k (val)", "",
          "Metric giống leaderboard ISIC: trung bình macro 11 lớp, ngưỡng > 0.5. Xếp hạng theo **Dice**.", ""]
    if df.folds.nunique() > 1:
        md += [f"> Lưu ý: các run dùng tập val khác nhau ({', '.join(sorted(df.folds.unique()))}).", ""]

    img = df[df.kind == "image"]
    s1 = img[img.tag == "ce_sqrtinv"]
    if not s1.empty:
        t = pivot(s1, "backbone")
        t.to_csv(out / "stage1_backbone_x_view.csv")
        md += ["## Stage 1 — Baseline (CE + √inv weight): Dice theo backbone × nhánh ảnh", "", bold_max(t), ""]

    for bb, g in img.groupby("backbone"):
        if g.tag.nunique() < 2:
            continue
        t = pivot(g, "tag", TAG_ORDER, TAG_NAMES)
        t.to_csv(out / f"stage2_{bb}.csv")
        md += [f"## Stage 2 — Xử lý mất cân bằng ({bb}): Dice theo phương pháp × nhánh ảnh", "", bold_max(t), ""]
        full = g.assign(method=g.tag.map(TAG_NAMES).fillna(g.tag)).set_index(["method", "views"])[METRICS]
        md += [f"<details><summary>Đầy đủ 6 metric ({bb})</summary>", "",
               full.sort_values("Dice Coefficient", ascending=False).to_markdown(floatfmt=".4f"), "", "</details>", ""]

    ml = df[df.kind == "ml"]
    if not ml.empty:
        t = ml.set_index("backbone")[METRICS + ["Balanced Acc"]].sort_values("Dice Coefficient", ascending=False)
        t.index.name = "model"
        t.to_csv(out / "ml_models.csv")
        md += ["## Mô hình ML trên metadata", "", t.to_markdown(floatfmt=".4f"), ""]

    if not img.empty:
        best = img.sort_values("Dice Coefficient", ascending=False).head(10)
        md += ["## Top 10 run ảnh theo Dice", "",
               best.set_index("run")[METRICS].to_markdown(floatfmt=".4f"), ""]

    text = "\n".join(md)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nĐã lưu bảng vào {out}")


if __name__ == "__main__":
    main()
