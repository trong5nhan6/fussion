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
GROUPS = {"M0": "Gốc", **{v: "G1 Modality" for v in ["C", "D", "CD", "CM", "DM"]},
          **{v: "G2 Fusion" for v in ["E", "A", "B"]}, **{v: "G3 Long-tail" for v in ["t0", "t1", "nosamp"]},
          **{v: "G4 Thành phần" for v in ["aux0", "md0", "d1024"]}, "sw224": "G5 Độ phân giải"}
DESC = {**{v: label for v, label, _ in MODALITY}, **dict(COMPONENTS), "sw224": "Swin-B 224 (checkpoint + ảnh 224)"}
DESC["M0"] = "đầy đủ: Swin-B-384 · D · d256 · r5_nobce · clin+derm+meta"
ORDER = ["M0", "C", "D", "CD", "CM", "DM", "E", "A", "B", "t0", "t1", "nosamp", "aux0", "md0", "d1024", "sw224"]
# Bảng 4 (experiments/stage7_backbones.yaml): tên backbone trong tên run -> (tên hiển thị, họ, pretrain, ảnh)
BACKBONES = {
    "resnet50_384": ("ResNet-50", "CNN cổ điển", "IN-1k", 384),
    "resnet152_384": ("ResNet-152", "CNN cổ điển", "IN-1k", 384),
    "efficientnetv2_m": ("EfficientNetV2-M", "CNN hiện đại", "IN-21k → 1k", 384),
    "convnext_base_384": ("ConvNeXt-B", "CNN hiện đại", "IN-22k → 1k", 384),
    "vit_base_384": ("ViT-B/16", "Transformer thuần", "IN-21k → 1k", 384),
    "dinov2_base_392": ("DINOv2-B/14", "Tự giám sát", "LVD-142M", 392),
    "swin_base_384": ("Swin-B", "Transformer phân cấp", "IN-22k → 1k", 384),
    "swin_large_384": ("Swin-L", "Transformer phân cấp", "IN-22k → 1k", 384),
}
EXCEL_README = [
    ("Đọc trước", "Tóm tắt các sheet và cách đọc"),
    ("1_SOTA", "So sánh với các phương pháp đã công bố trên test ẩn 479 lesion"),
    ("2_Modality", "Bỏ từng nguồn dữ liệu: C = clinical, D = dermoscopy, M = metadata"),
    ("3_Thanh_phan", "Bỏ / thay từng thành phần thiết kế (gate, 3 head, sampler, head phụ, modality dropout, chiếu 256)"),
    ("4_Backbone", "8 backbone x (chỉ backbone | full phương pháp), fold 0, cùng công thức; Δ = full − chỉ backbone"),
    ("4b_Backbone_Kaggle", "Run backbone cũ của notebook 07 (Kaggle T4) — chỉ để tham khảo"),
    ("5_Quyet_dinh", "Đổi cách biến xác suất thành nhãn (không train lại) trên OOF của M0"),
    ("Tom_tat_variant", "Số liệu dạng số của mọi variant: Dice mean/std, OOF gộp, Δ ghép cặp với M0, test"),
    ("Chi_tiet_fold", "Từng run (variant × fold): Dice, AUC, AP, Sens, Spec, BMA, top-1, số phút"),
    ("Theo_lop", "Dice / Sens / Spec / AUC từng lớp, tính trên OOF gộp của mỗi variant"),
    ("Nguong_M0", "Ngưỡng tối ưu từng lớp của M0 (fit trên toàn bộ OOF và theo từng fold)"),
    ("Ghi chú", "Dice = macro F1 ngưỡng 0.5 như leaderboard. Δ = Dice(variant) − Dice(M0) ghép cặp theo fold; "
                "Δ âm nghĩa là phần bị bỏ có ích. Dòng chỉ có 1 fold nhiễu khoảng ±0.035."),
]


def write_excel(path: Path, sheets: dict) -> None:
    """Mỗi DataFrame một sheet; tiêu đề in đậm, cố định dòng đầu, độ rộng cột theo nội dung."""
    from openpyxl.styles import Alignment, Font
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 70)
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, float):
                        c.number_format = "0.000"


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
    ap.add_argument("--no-excel", action="store_true", help="Không ghi ablation_results.xlsx")
    ap.add_argument("--backbone", default=None, help="Backbone của các run ablation (mặc định: tự nhận nếu chỉ có 1)")
    args = ap.parse_args()
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tests = {}
    if args.test_scores and resolve_path(args.test_scores).exists():
        tests = yaml.safe_load(resolve_path(args.test_scores).read_text(encoding="utf-8")) or {}

    abl, other = collect(args.dirs)
    backbones = sorted({bb for bb, _ in abl})
    # Bảng 1-3, 5: backbone của ablation stage 6 (Swin-B-384). Stage 7 thêm run M0/img của backbone khác -> Bảng 4.
    bb = args.backbone or ("swin_base_384" if "swin_base_384" in backbones or not backbones else backbones[0])
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

    # ---- Bảng 4: mỗi backbone 2 nhánh ở fold 0 (stage 7) + Swin-B 224 (độ phân giải, stage 6)
    def dice0(key, var):
        r = abl.get((key, var), {}).get(0)
        return (r, r["metrics"]["overall"]) if r else (None, None)

    rows = []
    for key in sorted({b for b, v in abl if v in ("M0", "img") and b in BACKBONES}, key=list(BACKBONES).index):
        label, family, pre, size = BACKBONES[key]
        (rf, of), (ri, oi) = dice0(key, "M0"), dice0(key, "img")
        r = rf or ri
        rows.append({"Backbone": label, "Họ": family, "Pretrain": pre, "Ảnh": size, "Tham số": r["params"] or "—",
                     "GFLOPs/lesion": f"{r['gflops']:.0f}" if r["gflops"] else "—",
                     "Phút (full)": f"{rf['minutes']:.0f}" if rf and rf["minutes"] else "—",
                     "Dice chỉ backbone": f3(oi["dice"]) if oi else "chưa chạy",
                     "Dice full": f3(of["dice"]) if of else "chưa chạy",
                     "Δ (full − chỉ backbone)": f"{of['dice'] - oi['dice']:+.3f}" if of and oi else "—",
                     "AUC full": f3(of["auc"]) if of else "—",
                     "Dice test (full)": f3(tests.get(rf["name"])) if rf else "—"})
    if "sw224" in V:
        r = V["sw224"]["runs"][V["sw224"]["folds"][0]]
        rows.append({"Backbone": "Swin-B (ảnh 224)", "Họ": "Transformer phân cấp", "Pretrain": "IN-22k → 1k",
                     "Ảnh": 224, "Tham số": r["params"] or "—",
                     "GFLOPs/lesion": f"{r['gflops']:.0f}" if r["gflops"] else "—",
                     "Phút (full)": f"{r['minutes']:.0f}" if r["minutes"] else "—", "Dice chỉ backbone": "—",
                     "Dice full": f3(V["sw224"]["dice_mean"]), "Δ (full − chỉ backbone)": "—",
                     "AUC full": f3(V["sw224"]["auc_mean"]), "Dice test (full)": f3(tests.get(r["name"]))})
    t4 = pd.DataFrame(rows)
    md += ["## Bảng 4 · Backbone (fold 0, cùng công thức; chỉ backbone thay đổi)", "",
           "Chỉ backbone = backbone dùng chung 2 ảnh -> nối đặc trưng -> MLP, CE + sampler q0.5, không metadata. "
           "Full = M0 (MoE D · 3 head long-tail · metadata). Δ > 0: phương pháp có ích trên backbone đó.", "",
           md_table(t4) if len(t4) else "_Chưa có run: chạy experiments/stage7_backbones.yaml (notebook 09)._", ""]

    # Run backbone cũ của notebook 07 (Kaggle T4, batch / môi trường khác) -> sheet tham khảo
    rows = []
    for r in other:
        if not r["name"].endswith("__D__r5_nobce__clin+derm"):
            continue
        o = r["metrics"]["overall"]
        rows.append({"Run": r["name"], "Backbone (timm)": r["img"], "Tham số": r["params"] or "—",
                     "GFLOPs/lesion": f"{r['gflops']:.0f}" if r["gflops"] else "—",
                     "Phút": f"{r['minutes']:.0f}" if r["minutes"] else "—",
                     "Dice val": f3(o["dice"]), "AUC val": f3(o["auc"]), "Dice test": f3(tests.get(r["name"]))})
    t4k = pd.DataFrame(rows)

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
    if not args.no_excel:
        summary, detail, per_class = [], [], []
        for v in sorted(V, key=lambda x: ORDER.index(x) if x in ORDER else 99):
            a = V[v]
            common = sorted(set(a["dice"]) & set(m0["dice"])) if m0 else []
            d = np.array([a["dice"][f] - m0["dice"][f] for f in common]) if common and v != "M0" else np.array([])
            sub = sub_dirs.get(v)
            single = a["runs"][a["folds"][0]]["name"] if len(a["folds"]) == 1 else None
            summary.append({"Variant": v, "Nhóm": GROUPS.get(v, ""), "Mô tả": DESC.get(v, ""), "Số fold": len(a["folds"]),
                            "Fold": ",".join(map(str, a["folds"])), "Dice val TB": a["dice_mean"],
                            "Dice val std": a["dice_std"], "Dice OOF gộp": a["pooled"]["dice"], "AUC val TB": a["auc_mean"],
                            "Δ TB so với M0": d.mean() if d.size else np.nan,
                            "Δ std": d.std(ddof=1) if d.size > 1 else np.nan,
                            "Số fold giảm": int((d < 0).sum()) if d.size else np.nan,
                            "File nộp ensemble": sub or "", "Dice test": tests.get(sub) if sub else tests.get(single)})
            for f in a["folds"]:
                r = a["runs"][f]
                o = r["metrics"]["overall"]
                detail.append({"Variant": v, "Fold": f, "Run": r["name"], **{k: o.get(k) for k in
                               ["dice", "auc", "ap", "sensitivity", "specificity", "balanced_acc", "top1_acc"]},
                               "Phút": r["minutes"], "Dice test": tests.get(r["name"])})
            for c, row in a["pooled"]["per_class"].items():
                per_class.append({"Variant": v, "Lớp": c, "n": int(row["n"]), "Dice": row["dice"],
                                  "Sens": row["sensitivity"], "Spec": row["specificity"], "AUC": row["auc"]})
        thr_sheet = pd.DataFrame()
        if (out / "thresholds_M0.csv").exists():
            all_t = pd.read_csv(out / "thresholds_M0.csv").assign(Fit="toàn bộ OOF")
            per = pd.read_csv(out / "thresholds_M0_per_fold.csv").rename(columns={"Unnamed: 0": "Fold"})
            thr_sheet = pd.concat([all_t, per.assign(Fit=lambda x: "fold " + x["Fold"].astype(str) + " (fit trên fold khác)")
                                   .drop(columns="Fold")], ignore_index=True)[["Fit"] + CLASSES]
        sheets = {"Đọc trước": pd.DataFrame(EXCEL_README, columns=["Sheet", "Nội dung"]),
                  "1_SOTA": t1, "2_Modality": t2, "3_Thanh_phan": t3, "4_Backbone": t4,
                  "4b_Backbone_Kaggle": t4k, "5_Quyet_dinh": t5,
                  "Tom_tat_variant": pd.DataFrame(summary), "Chi_tiet_fold": pd.DataFrame(detail),
                  "Theo_lop": pd.DataFrame(per_class), "Nguong_M0": thr_sheet}
        write_excel(out / "ablation_results.xlsx", sheets)
        print(f"Excel: {out / 'ablation_results.xlsx'}")
    for i, (t, n) in enumerate([(t1, "sota"), (t2, "modality"), (t3, "components"), (t4, "backbone"), (t5, "decision")], 1):
        t.to_csv(out / f"table{i}_{n}.csv", index=False)
    (out / "ablation_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
