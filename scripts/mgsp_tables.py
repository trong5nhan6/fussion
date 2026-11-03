"""Bảng cho bài TAIMMA (stage 8, MGSP) từ các run vit_base__abl_<R?>__f<k> và run ML trên metadata.

  python scripts/mgsp_tables.py --dirs outputs [--test-scores outputs/test_scores.yaml]

Bảng 2  so sánh chính (5 fold): chỉ metadata · late fusion · R1 · R2 · R4 — Dice mean ± std theo fold, OOF gộp, Δ vs R4
Bảng 3  ablation fold 0: R2..R8, Δ vs R4 + KTC 95% bootstrap ghép cặp (cùng lesion)
Bảng 4  Dice theo lớp (OOF gộp) của R1 / R2 / R4
Bảng 5  chi phí: tham số, GFLOPs, phút train
Bảng 6  số liệu diễn giải từ oof.csv (entropy / chồng lấn / khối lượng attention) theo variant và theo lớp (R4)
Late fusion: p = (1 − w)·p_R1 + w·p_metadata, w chọn trên các fold KHÁC (không dùng nhãn fold đang đánh giá).
Ensemble test: trung bình test_probs các fold -> submissions/<run>__cv<n>/submission.csv (nộp leaderboard).
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from ablation_tables import aggregate, collect, ensemble_test, md_table, read_run, write_excel
from milk10k import CLASSES
from milk10k.inference import write_submission
from milk10k.metrics import compute_metrics
from milk10k.utils import resolve_path

DESC = {
    "R1": "Chỉ ảnh: ConcatMLP [CLS_c ‖ CLS_d] -> MLP",
    "R2": "Nối metadata: ConcatMLP [CLS_c ‖ CLS_d ‖ meta] -> MLP",
    "R3": "MGSP chỉ query tĩnh (k_meta=0, k_static=4)",
    "R4": "MGSP đầy đủ (k_meta=2, k_static=2, attention riêng từng ảnh)",
    "R5": "MGSP attention chỉ trên derm (clin chỉ dùng CLS)",
    "R6": "MGSP bỏ CLS khỏi z",
    "R7": "MGSP gộp 2 ảnh (392 token + view embedding, 1 softmax)",
    "R8": "MGSP 4 query metadata + 4 query tĩnh",
}
MAIN, ABLATION, REF = ["R1", "R2", "R4"], ["R2", "R3", "R4", "R5", "R6", "R7", "R8"], "R4"
W_GRID = np.round(np.arange(0.0, 0.55, 0.05), 2)


def fast_dice(y: np.ndarray, p: np.ndarray) -> float:
    """Dice macro với hậu xử lý top1 (= F1 của argmax từng lớp, freebie 1.0 khi lớp vắng và không bị dự đoán)."""
    pred = p.argmax(1)
    out = []
    for k in range(len(CLASSES)):
        t, q = y == k, pred == k
        den = t.sum() + q.sum()
        out.append(2 * (t & q).sum() / den if den else 1.0)
    return float(np.mean(out))


def f3(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def fold_pm(a: dict) -> str:
    n = len(a["folds"])
    return f"{a['dice_mean']:.3f} ± {a['dice_std']:.3f}" if n > 1 else f"{a['dice_mean']:.3f}"


def paired_delta(a: dict, ref: dict) -> str:
    common = sorted(set(a["dice"]) & set(ref["dice"]))
    if not common or a is ref:
        return "—"
    d = np.array([a["dice"][f] - ref["dice"][f] for f in common])
    return f"{d.mean():+.3f}" + (f" ± {d.std(ddof=1):.3f} ({int((d > 0).sum())}/{len(d)} fold tăng)" if len(d) > 1 else "")


def bootstrap_delta(a_oof: pd.DataFrame, r_oof: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple:
    """Δ Dice (a − ref) trên cùng lesion, KTC 95% bootstrap theo lesion, P(Δ > 0)."""
    m = a_oof.merge(r_oof, on="lesion_id", suffixes=("_a", "_r"))
    y = m["label_a"].to_numpy()
    pa, pr = m[[c + "_a" for c in CLASSES]].to_numpy(), m[[c + "_r" for c in CLASSES]].to_numpy()
    rng = np.random.default_rng(seed)
    ds = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        ds[i] = fast_dice(y[idx], pa[idx]) - fast_dice(y[idx], pr[idx])
    return fast_dice(y, pa) - fast_dice(y, pr), np.percentile(ds, 2.5), np.percentile(ds, 97.5), (ds > 0).mean()


def pooled_row(name: str, desc: str, oof: pd.DataFrame, folds_dice: dict | None = None) -> dict:
    m = compute_metrics(oof.label.to_numpy(), oof[CLASSES].to_numpy(), "top1")
    d = np.array(list((folds_dice or {}).values()))
    return {"Mô hình": name, "Mô tả": desc, "Số fold": oof.fold.nunique(),
            "Dice (mean ± std fold)": (f"{d.mean():.3f} ± {d.std(ddof=1):.3f}" if len(d) > 1 else f3(d.mean()))
            if len(d) else "—",
            "Dice OOF gộp": m["dice"], "AUC": m["auc"], "AP": m["ap"], "BMA": m["balanced_acc"],
            "Sens": m["sensitivity"]}


def find_tabular(dirs) -> dict:
    """{tên mô hình ML: run} — chỉ lấy run ML có đủ 5 fold (ml_<tên>__5fold)."""
    out = {}
    for root in dirs:
        root = resolve_path(root)
        for d in sorted(root.glob("ml_*__5fold")) if root.exists() else []:
            r = read_run(d)
            if r:
                out[d.name.removeprefix("ml_").removesuffix("__5fold")] = r
    return out


def late_fusion(img_oof: pd.DataFrame, tab_oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    m = img_oof.merge(tab_oof[["lesion_id"] + CLASSES], on="lesion_id", suffixes=("", "_t"))
    pi, pt = m[CLASSES].to_numpy(), m[[c + "_t" for c in CLASSES]].to_numpy()
    y, folds = m.label.to_numpy(), m.fold.to_numpy()
    fused, ws = np.empty_like(pi), {}
    for f in np.unique(folds):
        other = folds != f
        scores = [fast_dice(y[other], (1 - w) * pi[other] + w * pt[other]) for w in W_GRID]
        w = float(W_GRID[int(np.argmax(scores))])
        ws[int(f)] = w
        fused[folds == f] = (1 - w) * pi[folds == f] + w * pt[folds == f]
    out = m[["lesion_id", "fold", "label"]].copy()
    out[CLASSES] = fused
    return out, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["outputs"])
    ap.add_argument("--out", default="outputs/mgsp_tables")
    ap.add_argument("--backbone", default="vit_base")
    ap.add_argument("--test-scores", default=None, help="yaml {tên thư mục submission hoặc run: Dice test}")
    args = ap.parse_args()
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tests = yaml.safe_load(open(args.test_scores, encoding="utf-8")) if args.test_scores else {}
    tests = tests or {}

    abl, _ = collect(args.dirs)
    V = {v: aggregate(runs) for (bb, v), runs in abl.items() if bb == args.backbone and v in DESC}
    if not V:
        raise SystemExit(f"Không thấy run {args.backbone}__abl_R?__f? trong {args.dirs}")
    print("Variant: " + ", ".join(f"{v}({len(a['folds'])} fold)" for v, a in sorted(V.items())))
    ref = V.get(REF)
    md, sheets = ["# Bảng kết quả MGSP (stage 8)", ""], {}

    # ---- ensemble test của variant có >= 2 fold
    subs = {}
    for v, a in sorted(V.items()):
        if len(a["folds"]) >= 2:
            d = ensemble_test(a, out, f"{args.backbone}__abl_{v}__cv{len(a['folds'])}")
            if d:
                subs[v] = d

    # ---- Bảng 2
    rows = []
    tab = find_tabular(args.dirs)
    if tab:
        best = max(tab, key=lambda k: tab[k]["metrics"]["overall"]["dice"])
        t_oof = pd.read_csv(tab[best]["dir"] / "oof.csv")
        rows.append(pooled_row(f"Chỉ metadata ({best})", "ML trên 34 trường metadata", t_oof,
                               {f: m["dice"] for f, m in tab[best]["metrics"]["per_fold"].items()}))
        if "R1" in V and len(V["R1"]["folds"]) == 5:
            lf_oof, ws = late_fusion(V["R1"]["oof"], t_oof)
            lf_dice = {f: fast_dice(g.label.to_numpy(), g[CLASSES].to_numpy()) for f, g in lf_oof.groupby("fold")}
            rows.append(pooled_row("Late fusion", f"(1−w)·R1 + w·{best}, w theo fold {ws}", lf_oof, lf_dice))
            if "R1" in subs:                                   # file nộp late fusion: w = trung bình các fold
                w = float(np.mean(list(ws.values())))
                pi = pd.read_csv(subs["R1"] / "test_probs.csv").set_index("lesion_id").sort_index()[CLASSES]
                pt = pd.read_csv(tab[best]["dir"] / "test_probs.csv").set_index("lesion_id").loc[pi.index, CLASSES]
                d = out / "submissions" / f"late_fusion_R1_{best}_w{w:.2f}"
                d.mkdir(parents=True, exist_ok=True)
                write_submission(pi.index, ((1 - w) * pi + w * pt).to_numpy(), "top1", d, pi.index, lambda *_: None)
    for v in MAIN:
        if v in V:
            a = V[v]
            r = pooled_row(v, DESC[v], a["oof"], a["dice"])
            r["Δ vs R4 (ghép cặp fold)"] = paired_delta(a, ref) if ref else "—"
            key = f"{args.backbone}__abl_{v}__cv{len(a['folds'])}"
            r["Dice test"] = tests.get(key)
            rows.append(r)
    t2 = pd.DataFrame(rows)
    sheets["Bang2_chinh"] = t2
    md += ["## Bảng 2 — So sánh chính (val, 5 fold)", "", md_table(t2.map(lambda x: f3(x) if isinstance(x, float) else x)), ""]

    # ---- Bảng 3: ablation fold 0
    rows = []
    r0 = ref["runs"].get(0) if ref else None
    r0_oof = pd.read_csv(r0["dir"] / "oof.csv") if r0 else None
    for v in ABLATION:
        if v not in V or 0 not in V[v]["runs"]:
            continue
        o = pd.read_csv(V[v]["runs"][0]["dir"] / "oof.csv")
        m = compute_metrics(o.label.to_numpy(), o[CLASSES].to_numpy(), "top1")
        row = {"Variant": v, "Mô tả": DESC[v], "Dice": m["dice"], "AUC": m["auc"], "AP": m["ap"],
               "BMA": m["balanced_acc"]}
        if r0_oof is not None and v != REF:
            dl, lo, hi, p = bootstrap_delta(o, r0_oof)
            row.update({"Δ Dice vs R4": dl, "KTC 95%": f"[{lo:+.3f}, {hi:+.3f}]", "P(Δ>0)": p})
        rows.append(row)
    t3 = pd.DataFrame(rows)
    sheets["Bang3_ablation_f0"] = t3
    md += ["## Bảng 3 — Ablation (fold 0, 1.048 lesion)", "",
           "Δ = Dice(variant) − Dice(R4) trên cùng lesion; KTC bootstrap 2.000 lần. Δ âm: thành phần bị đổi có ích.", "",
           md_table(t3.map(lambda x: f3(x) if isinstance(x, float) else x)), ""]

    # ---- Bảng 4: theo lớp
    per = {}
    for v in MAIN:
        if v in V:
            pc = V[v]["pooled"]["per_class"]
            per[v] = {c: pc[c]["dice"] for c in CLASSES}
            per["n"] = {c: int(pc[c]["n"]) for c in CLASSES}
    if per:
        t4 = pd.DataFrame(per).rename_axis("Lớp").reset_index()
        if {"R4", "R1"} <= set(per):
            t4["R4 − R1"] = t4["R4"] - t4["R1"]
        if {"R4", "R2"} <= set(per):
            t4["R4 − R2"] = t4["R4"] - t4["R2"]
        sheets["Bang4_theo_lop"] = t4
        md += ["## Bảng 4 — Dice theo lớp (OOF gộp)", "", md_table(t4.map(lambda x: f3(x) if isinstance(x, float) else x)), ""]

    # ---- Bảng 5: chi phí
    rows = []
    for v in sorted(V):
        r = V[v]["runs"][min(V[v]["runs"])]
        mins = [x["minutes"] for x in V[v]["runs"].values() if x["minutes"]]
        rows.append({"Variant": v, "Tham số": r["params"], "GFLOPs/lesion": r["gflops"],
                     "Phút train / fold": float(np.mean(mins)) if mins else None})
    t5 = pd.DataFrame(rows)
    sheets["Bang5_chi_phi"] = t5
    md += ["## Bảng 5 — Chi phí", "", md_table(t5.map(lambda x: f3(x) if isinstance(x, float) else x)), ""]

    # ---- Bảng 6: diễn giải attention
    rows = []
    for v in sorted(V):
        o = V[v]["oof"]
        cols = [c for c in o.columns if c.split("_")[0] in ("ent", "ovl", "mass")]
        if cols:
            rows.append({"Variant": v, **o[cols].mean().round(3).to_dict()})
    if rows:
        t6 = pd.DataFrame(rows)
        sheets["Bang6_attention"] = t6
        md += ["## Bảng 6 — Attention (trung bình trên OOF)", "",
               "ent = entropy chuẩn hoá (1 trải đều, 0 dồn 1 ô) · ovl = chồng lấn bản đồ query metadata vs query tĩnh · "
               "mass = (R7) tỉ lệ attention rơi vào từng ảnh", "",
               md_table(t6.map(lambda x: f3(x) if isinstance(x, float) else x)), ""]
        if REF in V:
            o = V[REF]["oof"]
            cols = [c for c in o.columns if c.split("_")[0] in ("ent", "ovl")]
            t6b = o.assign(Lớp=[CLASSES[i] for i in o.label]).groupby("Lớp")[cols].mean().round(3).reset_index()
            sheets["Bang6b_attention_lop_R4"] = t6b
            md += ["### Attention của R4 theo lớp", "", md_table(t6b.map(lambda x: f3(x) if isinstance(x, float) else x)), ""]

    if subs:
        md += ["## File nộp leaderboard", ""] + [f"- `{d.relative_to(out)}/submission.csv`" for d in sorted(
            (out / "submissions").iterdir())]
    (out / "mgsp_tables.md").write_text("\n".join(md), encoding="utf-8")
    write_excel(out / "mgsp_results.xlsx", sheets)
    print("\n".join(md))
    print(f"\n-> {out / 'mgsp_tables.md'} | {out / 'mgsp_results.xlsx'}")


if __name__ == "__main__":
    main()
