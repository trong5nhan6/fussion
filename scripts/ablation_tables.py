"""Tổng hợp 5 bảng của paper từ các run ablation (experiments/stage6_ablation.yaml) và các run backbone (notebook 07).

  python scripts/ablation_tables.py --dirs outputs [thư mục outputs khác ...] [--test-scores test_scores.yaml]

Đọc mọi run có metrics.json + oof.csv trong --dirs:
  - run ablation   <backbone>__abl_<variant>__f<fold>  -> gộp theo variant (mean ± std qua fold, Dice OOF gộp)
  - run khác       <backbone>__D__r5_nobce__clin+derm   -> Bảng 4 (backbone / độ phân giải)
Ghi vào --out (mặc định outputs/ablation_tables):
  ablation_tables.md, table1..5_*.csv
  submissions/<backbone>__abl_<variant>__cv<n>/  submission.csv + test_probs.csv (trung bình xác suất các fold)
  submissions/<backbone>__abl_M0__cv<n>_thr/     submission.csv với ngưỡng theo lớp fit trên toàn bộ OOF của M0
--test-scores: yaml {tên run hoặc tên thư mục submission: Dice test leaderboard} để điền cột "Dice test".
"""
import argparse
import re
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from milk10k import CLASSES
from milk10k.inference import write_submission
from milk10k.metrics import compute_metrics, postprocess
from milk10k.thresholds import apply_thresholds, fit_thresholds, nested_threshold_scores
from milk10k.utils import load_json, resolve_path

ABL = re.compile(r"^(?P<bb>.+?)__abl_(?P<var>[^_]+)__f(?P<fold>\d+)(?:__(?P<suffix>.+))?$")
MODALITY = [("C", "C (clinical)", 0.392), ("D", "D (dermoscopy)", 0.420), ("CD", "C + D", 0.456),
            ("CM", "C + M", 0.376), ("DM", "D + M", 0.413), ("M0", "C + D + M (M0)", 0.505)]
COMPONENTS = [("M0", "đầy đủ"), ("E", "bỏ gate: nối 4 token + MLP"), ("t0", "1 head CE (τ = 0) thay 3 head"),
              ("A", "gate + 1 head (loss theo overlay)"), ("B", "transformer + sparse MoE thay gate"),
              ("t1", "1 head logit-adjusted (τ = 1)"), ("nosamp", "bỏ sampler q = 0.5"),
              ("aux0", "bỏ 4 head phụ"), ("md0", "bỏ modality dropout"), ("d1024", "không chiếu, token 1024 chiều")]
# Bảng 1: kết quả đã công bố trên test ẩn 479 lesion (Dice = macro F1 ngưỡng 0.5)
PUBLISHED = [("CAFFM", "bảng của JI-ADF", 0.232, 0.787), ("SkinM2Former", "bảng của JI-ADF", 0.405, 0.841),
             ("Fine-tuned PanDerm", "bảng của JI-ADF", 0.434, 0.840), ("CoscatNet-UFS", "bảng của JI-ADF", 0.435, 0.800),
             ("DualRefNet", "bảng của JI-ADF", 0.441, 0.848), ("ViT–GAT hybrid", "luận văn UOC", 0.478, None),
             ("Coding agent baseline", "arXiv 2608.23336", 0.490, None), ("JI-ADF", "ICML 2026", 0.505, 0.866)]


# ------------------------------------------------------------------ đọc run
def read_run(d: Path) -> dict | None:
    if not ((d / "metrics.json").exists() and (d / "oof.csv").exists()):
        return None
    m = load_json(d / "metrics.json")
    if not m.get("overall"):
        return None                                   # full_data: không có val
    log = (d / "train.log").read_text(encoding="utf-8", errors="replace") if (d / "train.log").exists() else ""
    params = re.search(r"Tham số\s*: tổng ([\d.]+[MBK])", log)
    flops = re.search(r"Tính toán\s*: ([\d.]+) GFLOPs", log)
    hist = sorted(d.glob("fold*/history.csv"))
    minutes = sum(pd.read_csv(h)["time_s"].sum() for h in hist if "time_s" in pd.read_csv(h, nrows=0).columns) / 60
    return {"dir": d, "name": d.name, "metrics": m, "mode": m.get("postprocess", "top1"),
            "params": params.group(1) if params else None, "gflops": float(flops.group(1)) if flops else None,
            "minutes": minutes or None, "img": m.get("model", {}).get("backbone")}


def collect(dirs) -> tuple[dict, list]:
    """-> ({(backbone, variant): {fold: run}}, [run khác])"""
    abl, other = {}, []
    for root in dirs:
        root = resolve_path(root)
        if not root.exists():
            print(f"[bỏ qua] không thấy {root}")
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            r = read_run(d)
            if r is None:
                continue
            m = ABL.match(d.name)
            if m:
                key = (m["bb"] + (f"__{m['suffix']}" if m["suffix"] else ""), m["var"])
                abl.setdefault(key, {})[int(m["fold"])] = r
            else:
                other.append(r)
    return abl, other


def aggregate(runs: dict) -> dict:
    """Gộp các fold của một variant."""
    folds = sorted(runs)
    dice = np.array([runs[f]["metrics"]["overall"]["dice"] for f in folds])
    auc = np.array([runs[f]["metrics"]["overall"]["auc"] for f in folds])
    oof = pd.concat([pd.read_csv(runs[f]["dir"] / "oof.csv") for f in folds], ignore_index=True)
    mode = runs[folds[0]]["mode"]
    pooled = compute_metrics(oof.label.to_numpy(), oof[CLASSES].to_numpy(), mode)
    return {"folds": folds, "dice": dict(zip(folds, dice)), "dice_mean": dice.mean(),
            "dice_std": dice.std(ddof=1) if len(dice) > 1 else np.nan, "auc_mean": auc.mean(),
            "oof": oof, "pooled": pooled, "mode": mode, "runs": runs}


def ensemble_test(agg: dict, out: Path, name: str) -> Path | None:
    """Trung bình test_probs các fold -> submission (cùng hậu xử lý với run)."""
    probs = [pd.read_csv(r["dir"] / "test_probs.csv").set_index("lesion_id").sort_index()
             for r in agg["runs"].values() if (r["dir"] / "test_probs.csv").exists()]
    if not probs:
        return None
    mean = sum(p[CLASSES] for p in probs) / len(probs)
    d = out / "submissions" / name
    d.mkdir(parents=True, exist_ok=True)
    write_submission(mean.index, mean.to_numpy(), agg["mode"], d, mean.index, logger=lambda *_: None)
    return d


# ------------------------------------------------------------------ định dạng
def f3(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def pm(agg: dict | None) -> str:
    if agg is None:
        return "chưa chạy"
    n = len(agg["folds"])
    if n == 1:
        return f"{agg['dice_mean']:.3f} (fold {agg['folds'][0]})"
    return f"{agg['dice_mean']:.3f} ± {agg['dice_std']:.3f} ({n} fold)"


def delta(agg: dict | None, ref: dict | None) -> str:
    """Δ ghép cặp trên các fold chung với M0."""
    if agg is None or ref is None:
        return "—"
    common = sorted(set(agg["dice"]) & set(ref["dice"]))
    if not common:
        return "—"
    d = np.array([agg["dice"][f] - ref["dice"][f] for f in common])
    if len(d) == 1:
        return f"{d[0]:+.3f} (1 fold)"
    return f"{d.mean():+.3f} ± {d.std(ddof=1):.3f} ({len(d)} fold, {int((d < 0).sum())}/{len(d)} fold giảm)"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["outputs"])
    ap.add_argument("--out", default="outputs/ablation_tables")
    ap.add_argument("--test-scores", default=None, help="yaml {tên run / thư mục submission: Dice test}")
    ap.add_argument("--backbone", default=None, help="Backbone của các run ablation (mặc định: tự nhận nếu chỉ có 1)")
    args = ap.parse_args()
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tests = {}
    if args.test_scores and resolve_path(args.test_scores).exists():
        tests = yaml.safe_load(resolve_path(args.test_scores).read_text(encoding="utf-8")) or {}

    abl, other = collect(args.dirs)
    backbones = sorted({bb for bb, _ in abl})
    if not args.backbone and len(backbones) > 1:
        raise SystemExit(f"Có nhiều backbone ablation {backbones}: chọn bằng --backbone")
    bb = args.backbone or (backbones[0] if backbones else "swin_base_384")
    V = {v: aggregate(r) for (b, v), r in abl.items() if b == bb}
    m0 = V.get("M0")
    print(f"Backbone ablation: {bb} | variant: " + ", ".join(f"{v}({len(a['folds'])} fold)" for v, a in V.items()))

    # Ensemble test của mọi variant đủ >= 2 fold
    sub_dirs = {}
    for v, a in V.items():
        if len(a["folds"]) > 1:
            d = ensemble_test(a, out, f"{bb}__abl_{v}__cv{len(a['folds'])}")
            if d:
                sub_dirs[v] = d.name

    md = ["# Bảng ablation", "",
          f"Backbone: `{bb}`. Dice = macro F1 ngưỡng 0.5 (như leaderboard). Val = OOF trên fold đã chạy; "
          "test = leaderboard (điền qua --test-scores).", ""]

    # ---- Bảng 1
    rows = [{"Phương pháp": n, "Nguồn": s, "Dice test": f3(d), "AUC test": f3(a), "Dice val": "—"} for n, s, d, a in PUBLISHED]
    ref_run = f"{bb}__D__r5_nobce__clin+derm"
    rows.append({"Phương pháp": "M0 · fold 0 (notebook 07)", "Nguồn": "của bạn", "Dice test": f3(tests.get(ref_run)),
                 "AUC test": "—", "Dice val": f3(next((r["metrics"]["overall"]["dice"] for r in other if r["name"] == ref_run), None))})
    if m0:
        name = sub_dirs.get("M0")
        rows.append({"Phương pháp": f"M0 · ensemble {len(m0['folds'])} fold", "Nguồn": "của bạn",
                     "Dice test": f3(tests.get(name)) if name else "—", "AUC test": "—",
                     "Dice val": f3(m0["pooled"]["dice"]) + " (OOF gộp)"})
        rows.append({"Phương pháp": "M0 · ensemble + ngưỡng theo lớp", "Nguồn": "của bạn",
                     "Dice test": f3(tests.get(f"{name}_thr")) if name else "—", "AUC test": "—", "Dice val": "xem Bảng 5"})
    t1 = pd.DataFrame(rows)
    md += ["## Bảng 1 · So sánh với các phương pháp đã công bố (test ẩn 479 lesion)", "", md_table(t1), ""]

    # ---- Bảng 2
    t2 = pd.DataFrame([{"Đầu vào": label, "Variant": v, "Dice val": pm(V.get(v)),
                        "Dice OOF gộp": f3(V[v]["pooled"]["dice"]) if v in V else "—",
                        "Δ so với M0": delta(V.get(v), m0) if v != "M0" else "0",
                        "JI-ADF (test)": f3(ref)} for v, label, ref in MODALITY])
    md += ["## Bảng 2 · Ablation modality (C = clinical, D = dermoscopy, M = metadata)", "",
           "Δ = Dice(variant) − Dice(M0) ghép cặp theo fold. Δ âm: nguồn bị bỏ có ích.", "", md_table(t2), ""]

    # ---- Bảng 3
    t3 = pd.DataFrame([{"Variant": v, "Thay đổi": desc, "Dice val": pm(V.get(v)),
                        "Δ so với M0": delta(V.get(v), m0) if v != "M0" else "0",
                        "AUC val": f3(V[v]["auc_mean"]) if v in V else "—"} for v, desc in COMPONENTS])
    md += ["## Bảng 3 · Ablation thành phần", "",
           "Chỉ kết luận được với dòng có ≥ 3 fold; dòng 1 fold nhiễu khoảng ±0.035.", "", md_table(t3), ""]

    # ---- Bảng 4
    rows = []
    for r in other:
        if not r["name"].endswith("__D__r5_nobce__clin+derm"):
            continue
        o = r["metrics"]["overall"]
        rows.append({"Run": r["name"], "Backbone (timm)": r["img"], "Tham số": r["params"] or "—",
                     "GFLOPs/lesion": f"{r['gflops']:.0f}" if r["gflops"] else "—",
                     "Phút": f"{r['minutes']:.0f}" if r["minutes"] else "—",
                     "Dice val": f3(o["dice"]), "AUC val": f3(o["auc"]), "Dice test": f3(tests.get(r["name"]))})
    if "sw224" in V:
        r = V["sw224"]["runs"][V["sw224"]["folds"][0]]
        rows.append({"Run": r["name"], "Backbone (timm)": r["img"], "Tham số": r["params"] or "—",
                     "GFLOPs/lesion": f"{r['gflops']:.0f}" if r["gflops"] else "—",
                     "Phút": f"{r['minutes']:.0f}" if r["minutes"] else "—", "Dice val": f3(V["sw224"]["dice_mean"]),
                     "AUC val": f3(V["sw224"]["auc_mean"]), "Dice test": f3(tests.get(r["name"]))})
    t4 = pd.DataFrame(rows)
    md += ["## Bảng 4 · Backbone và độ phân giải (cùng công thức D · r5_nobce · clin+derm, fold 0)", "",
           md_table(t4) if len(t4) else "_Chưa có run: thêm thư mục outputs của notebook 07 vào --dirs._", ""]

    # ---- Bảng 5
    rows = []
    if m0:
        oof = m0["oof"]
        y, p, folds = oof.label.to_numpy(), oof[CLASSES].to_numpy(), oof.fold.to_numpy()
        top1 = compute_metrics(y, p, "top1")
        raw = compute_metrics(y, p, "softmax")
        thr_scores, per_fold, nested = nested_threshold_scores(y, p, folds)
        thr = compute_metrics(y, thr_scores, "softmax")
        name = sub_dirs.get("M0")
        for label, m, key in [("top1, ngưỡng 0.5 (đang dùng)", top1, name),
                              ("xác suất softmax thô, ngưỡng 0.5", raw, None),
                              ("ngưỡng riêng từng lớp" + ("" if nested else " (1 fold: lạc quan)"), thr,
                               f"{name}_thr" if name else None)]:
            rows.append({"Cách quyết định": label, "Dice OOF": f3(m["dice"]), "Sens": f3(m["sensitivity"]),
                         "Spec": f3(m["specificity"]), "Dice test": f3(tests.get(key)) if key else "—"})
        # Ngưỡng fit trên toàn bộ OOF -> áp lên test ensemble
        t_all = fit_thresholds(y, p)
        pd.DataFrame([dict(zip(CLASSES, t_all))]).to_csv(out / "thresholds_M0.csv", index=False)
        pd.DataFrame(per_fold).T.to_csv(out / "thresholds_M0_per_fold.csv")
        if name:
            probs = pd.read_csv(out / "submissions" / name / "test_probs.csv").set_index("lesion_id")[CLASSES]
            d = out / "submissions" / f"{name}_thr"
            d.mkdir(parents=True, exist_ok=True)
            write_submission(probs.index, apply_thresholds(probs.to_numpy(), t_all), "softmax", d, probs.index,
                             logger=lambda *_: None)
    t5 = pd.DataFrame(rows)
    md += ["## Bảng 5 · Tầng quyết định (không train lại, trên OOF của M0)", "",
           "Ngưỡng của mỗi fold fit trên các fold còn lại (lồng nhau), nên Dice OOF không bị lạc quan.", "",
           md_table(t5) if len(t5) else "_Chưa có OOF của M0._", ""]

    if sub_dirs:
        md += ["## File nộp đã tạo", ""] + [f"- `submissions/{n}/submission.csv` ({v})" for v, n in sub_dirs.items()]
        if "M0" in sub_dirs:
            md.append(f"- `submissions/{sub_dirs['M0']}_thr/submission.csv` (M0, ngưỡng theo lớp)")
    for i, (t, n) in enumerate([(t1, "sota"), (t2, "modality"), (t3, "components"), (t4, "backbone"), (t5, "decision")], 1):
        t.to_csv(out / f"table{i}_{n}.csv", index=False)
    (out / "ablation_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
