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
ARCH_NAMES = {"A": "A · gate theo nguồn", "B": "B · transformer MoE", "C": "C · B + long-tail",
              "D": "D · A + long-tail", "E": "E · nối 4 token + long-tail"}
TAG_ORDER = ["ce_plain", "ce_sqrtinv", "ce_inv", "cb_b0999", "cb_focal", "la_t1", "samp_q05", "samp_q1", "crt"]
TAG_NAMES = {"ce_plain": "CE", "ce_sqrtinv": "CE + √inv weight", "ce_inv": "CE + inv weight",
             "cb_b0999": "Class-Balanced (β=0.999)", "cb_focal": "CB Focal (γ=2)", "la_t1": "Logit-adjusted (τ=1)",
             "samp_q05": "Sampler q=0.5", "samp_q1": "Sampler q=1", "crt": "cRT (decoupling)",
             "lt": "3 head long-tail (loss riêng)", "bce": "BCE + pos_weight (sigmoid)",
             "r1_bce": "R1: BCE + sigmoid", "r2_squash": "R2: resize thẳng", "r3_aug": "R3: augmentation mạnh",
             "r4_lr": "R4: lr backbone ×1 + drop path", "r5_all": "R5: gộp tất cả (BCE)",
             "r5_nobce": "R5 không BCE (sampler q=0.5)"}
TAG_ORDER = TAG_ORDER + ["bce"]
ABLATION_ORDER = ["samp_q05", "r1_bce", "r2_squash", "r3_aug", "r4_lr", "r5_all"]   # R0 = samp_q05 (stage 2)
TAG_ORDER_MOE = TAG_ORDER + ["lt", "r5_all", "r5_nobce"]
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
            row = {"run": run, "dir": str(f.parent), "folds": "+".join(str(x) for x in folds),
                   **{name: m["overall"][k] for k, name in LEADERBOARD_COLUMNS},
                   "Balanced Acc": m["overall"]["balanced_acc"]}
            if m.get("model", {}).get("inputs") == "metadata":
                row.update(kind="ml", backbone=m["model"]["name"], tag="", views="metadata")
            elif m.get("model", {}).get("name") == "moe":
                parts = run.split("__")  # <backbone>__<hướng>__<tag>__<views>[__<hậu tố>]
                backbone = parts[0] + (f" [{parts[4]}]" if len(parts) >= 5 else "")
                row.update(kind="moe", backbone=backbone, arch=m["model"]["arch"], tag=m.get("tag") or "lt",
                           views="+".join(m["model"].get("views", [])))
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


def explain_section(g: pd.DataFrame) -> list[str]:
    """Giải thích từ oof.csv của run MoE tốt nhất (clin+derm) mỗi loại fusion:
    A/D: trọng số gate trung bình theo lớp thật; B/C: tỉ lệ expert top-1 của token CLS theo lớp thật."""
    from milk10k import CLASSES

    md = []
    for kind, archs in (("gate", "AD"), ("expert", "BC")):
        cand = g[g.arch.isin(list(archs)) & (g.views == "clin+derm")]
        if cand.empty:
            continue
        best = cand.sort_values("Dice Coefficient", ascending=False).iloc[0]
        oof_path = Path(best["dir"]) / "oof.csv"
        if not oof_path.exists():
            continue
        oof = pd.read_csv(oof_path)
        oof["class"] = [CLASSES[i] for i in oof.label]
        if kind == "gate":
            cols = [c for c in oof.columns if c.startswith("gate_")]
            if not cols:
                continue
            t = oof.groupby("class")[cols].mean().reindex(CLASSES)
            t.columns = [c.removeprefix("gate_") for c in cols]
            title = f"Trọng số gate trung bình theo lớp — {best.run} (mô hình dựa vào nguồn nào)"
        else:
            if "expert_cls" not in oof.columns:
                continue
            t = pd.crosstab(oof["class"], oof["expert_cls"].astype(int), normalize="index").reindex(CLASSES) * 100
            t.columns = [f"expert {c}" for c in t.columns]
            title = f"Expert top-1 của token CLS theo lớp (%) — {best.run} (expert chuyên môn hoá thế nào)"
        md += [f"### {title}", "", t.to_markdown(floatfmt=".2f" if kind == "gate" else ".1f"), ""]
    return md


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
    df.drop(columns="dir").sort_values(["kind", "backbone", "tag", "views"]).to_csv(out / "all_runs.csv", index=False)

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

    abl = img[img.tag.isin(ABLATION_ORDER[1:])]
    for bb in sorted(abl.backbone.unique()):
        g = img[(img.backbone == bb) & img.tag.isin(ABLATION_ORDER)]
        t = pivot(g, "tag", ABLATION_ORDER, {**TAG_NAMES, "samp_q05": "R0: mốc (sampler q=0.5, softmax)"})
        t.to_csv(out / f"stage4_ablation_{bb}.csv")
        md += [f"## Stage 4 — Ablation ({bb}): Dice theo yếu tố × nhánh ảnh", "", bold_max(t), ""]
        full = g.assign(method=g.tag.map(TAG_NAMES).fillna(g.tag)).set_index(["method", "views"])[METRICS]
        md += [f"<details><summary>Đầy đủ 6 metric — ablation ({bb})</summary>", "",
               full.sort_values("Dice Coefficient", ascending=False).to_markdown(floatfmt=".4f"), "", "</details>", ""]

    ml = df[df.kind == "ml"]
    if not ml.empty:
        t = ml.set_index("backbone")[METRICS + ["Balanced Acc"]].sort_values("Dice Coefficient", ascending=False)
        t.index.name = "model"
        t.to_csv(out / "ml_models.csv")
        md += ["## Mô hình ML trên metadata", "", t.to_markdown(floatfmt=".4f"), ""]

    moe = df[df.kind == "moe"]
    for bb, g in moe.groupby("backbone"):
        g = g.assign(row=g.arch.map(ARCH_NAMES) + " — " + g.tag.map(TAG_NAMES).fillna(g.tag))
        order = [f"{ARCH_NAMES[a]} — {TAG_NAMES.get(t, t)}" for a in ARCH_NAMES for t in TAG_ORDER_MOE]
        t = pivot(g, "row", order)
        t.to_csv(out / f"stage3_moe_{bb}.csv")
        md += [f"## Stage 3 — Multimodal MoE ({bb}): Dice theo hướng × loss × nhánh ảnh", "", bold_max(t), ""]
        ref = img[img.backbone == bb]
        if not ref.empty:  # cùng backbone, cùng loss: chỉ ảnh (stage 2) vs từng hướng MoE
            cmp = {}
            for tag in sorted(set(g.tag) & set(ref.tag), key=TAG_ORDER_MOE.index):
                for v in VIEW_ORDER:
                    r = ref[(ref.tag == tag) & (ref.views == v)]
                    if r.empty:
                        continue
                    row = {"Chỉ ảnh (stage 2)": r["Dice Coefficient"].iloc[0]}
                    for a in ARCH_NAMES:
                        x = g[(g.arch == a) & (g.tag == tag) & (g.views == v)]
                        row[a] = x["Dice Coefficient"].iloc[0] if not x.empty else float("nan")
                    if all(pd.isna(row[a]) for a in ARCH_NAMES):
                        continue  # không có run MoE nào cho (loss, nhánh ảnh) này
                    cmp[f"{TAG_NAMES.get(tag, tag)} — {v}"] = row
            if cmp:
                c = pd.DataFrame(cmp).T.dropna(axis=1, how="all")
                c.to_csv(out / f"stage3_vs_image_{bb}.csv")
                md += [f"### MoE so với chỉ ảnh ({bb}, cùng loss)", "", c.to_markdown(floatfmt=".4f"), ""]
        md += explain_section(g)
    if not moe.empty:
        best = moe.sort_values("Dice Coefficient", ascending=False).head(10)
        md += ["## Top 10 run MoE theo Dice", "", best.set_index("run")[METRICS].to_markdown(floatfmt=".4f"), ""]

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
