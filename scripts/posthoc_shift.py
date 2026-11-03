"""Hậu xử lý label shift (không train) cho mọi nhóm run 5 fold + bảng so sánh stage 9 (bài Q1).

Bối cảnh: test MILK10k lệch phân bố lớp mạnh so với train (vd BCC 48% -> ~13%), còn p(ảnh, metadata | lớp) gần như
giữ nguyên -> sửa tối ưu Bayes là nhân lại prior: p'(y|x) ∝ p(y|x) · [π_test(y) / π_train(y)]^α.
π_test ước lượng bằng BBSE (Lipton et al., 2018) — CHỈ dùng xác suất dự đoán trên ảnh test (không nhãn) + OOF có nhãn:
  C[i, j] = E[p_i(x) | y = j] trên OOF     μ = mean p(x_test)     giải  C·π = μ,  π ≥ 0  (NNLS), chuẩn hoá tổng 1.

Mỗi nhóm <backbone>__abl_<variant> (đủ 5 fold) -> 3 cách giải mã: raw (top1 như cũ) · bbse (α = 1) · bbse½ (α = 0.5).
Chọn cách giải mã bằng MÔ PHỎNG CROSS-FIT trên OOF, không nhìn nhãn hay điểm test:
  với mỗi fold k: ma trận C học từ 4 fold còn lại; từ fold k rút (có hoàn lại) 479 lesion theo phân bố π̂_test
  (trung vị các ước lượng BBSE của mọi nhóm trên ảnh test thật, không nhãn — chung cho mọi nhóm để Δ so sánh được)
  -> ước lượng lại π trên mẫu rút -> sửa prior (trọng số chặn trong [1/10, 10]) -> Dice. Lặp --reps lần mỗi fold.
  Kèm mô phỏng "không lệch" (rút theo prior train) để chắc chắn BBSE không làm hại khi không có shift.
Cách được chọn = Dice mô phỏng trung bình cao nhất (lệch theo π̂_test); hoà trong 0.002 -> giữ raw.

Đầu ra (--out, mặc định outputs/posthoc_shift):
  posthoc_shift.md / .xlsx   bảng tổng hợp + Δ so với nhóm mốc (--ref, mặc định swin_base_384__abl_M0)
  submissions/<nhóm>__cv5__<cách>/submission.csv  (file NỘP: 1 file / nhóm, đúng cách đã chọn)
  submissions/_khac/...                            các cách còn lại (tham khảo, không nộp)

  python scripts/posthoc_shift.py --dirs outputs /content/drive/MyDrive/milk10k_ablation/outputs
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ablation_tables import aggregate, collect, md_table, write_excel
from milk10k import CLASSES
from milk10k.inference import write_submission
from milk10k.utils import resolve_path

K = len(CLASSES)
N_TEST = 479
DECODERS = {"raw": None, "bbse": 1.0, "bbse½": 0.5}
MIN_CASES = 5      # sàn số ca mỗi lớp trong mẫu giả-test
MAX_RATIO = 10.0   # chặn trọng số π_test/π_train trong [1/10, 10] (điều chuẩn kiểu RLLS: 479 ca test -> π̂ rất nhiễu)
DESC = {
    "vit_base__abl_D": "MoE-D · ViT-B/224 ImageNet (đối chứng backbone)",
    "panderm_base__abl_D": "MoE-D · PanDerm-Base/224",
    "dermlip_vit_b16__abl_D": "MoE-D · DermLIP ViT-B/224",
    "panderm_base__abl_HMF": "HMF · PanDerm-Base/224",
    "dermlip_vit_b16__abl_HMF": "HMF · DermLIP ViT-B/224",
    "swin_base_384__abl_HMF": "HMF · Swin-B/384",
    "swin_base_384__abl_q1": "MoE-D · Swin-B/384 · sampler q=1 (prior đều)",
    "swin_base_384__abl_M0": "M0: MoE-D · Swin-B/384 (stage 6, mốc)",
}


def fast_dice(y: np.ndarray, p: np.ndarray) -> float:
    """Dice macro top1 (giống compute_metrics của project; lớp vắng và không bị dự đoán = 1)."""
    pred = p.argmax(1)
    out = []
    for k in range(K):
        t, q = y == k, pred == k
        den = t.sum() + q.sum()
        out.append(2 * (t & q).sum() / den if den else 1.0)
    return float(np.mean(out))


def soft_confusion(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """C[:, j] = xác suất dự đoán trung bình trên các mẫu có nhãn j (cột của lớp vắng = one-hot, tránh suy biến)."""
    C = np.eye(K)
    for j in range(K):
        if (y == j).any():
            C[:, j] = p[y == j].mean(0)
    return C


def bbse_prior(C: np.ndarray, p_target: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    pi, _ = nnls(C, p_target.mean(0))
    pi = np.clip(pi, floor, None)
    return pi / pi.sum()


def adjust(p: np.ndarray, pi_src: np.ndarray, pi_tgt: np.ndarray, alpha: float | None) -> np.ndarray:
    if alpha is None:
        return p
    w = np.clip(pi_tgt / np.clip(pi_src, 1e-12, None), 1 / MAX_RATIO, MAX_RATIO)
    q = p * w ** alpha
    return q / q.sum(1, keepdims=True)


def simulate(oof: pd.DataFrame, target_prior: np.ndarray, reps: int, seed: int = 0) -> dict:
    """Cross-fit: C học trên 4 fold, mẫu giả-test rút từ fold còn lại theo target_prior. -> {cách: mảng Dice}."""
    rng = np.random.default_rng(seed)
    y_all, p_all, f_all = oof.label.to_numpy(), oof[CLASSES].to_numpy(), oof.fold.to_numpy()
    res = {d: [] for d in DECODERS}
    for k in np.unique(f_all):
        cal, te = f_all != k, f_all == k
        C, pi_src = soft_confusion(p_all[cal], y_all[cal]), np.bincount(y_all[cal], minlength=K) / cal.sum()
        y_te, p_te = y_all[te], p_all[te]
        pools = [np.where(y_te == c)[0] for c in range(K)]
        # mỗi lớp ít nhất MIN_CASES ca (test thật có đủ 11 lớp; lớp vắng sẽ cho Dice "miễn phí" = 1 -> lạc quan giả)
        tp = np.maximum(target_prior, MIN_CASES / N_TEST) * np.array([len(pl) > 0 for pl in pools])
        tp = tp / tp.sum()
        for _ in range(reps):
            n = rng.multinomial(N_TEST, tp)
            idx = np.concatenate([rng.choice(pools[c], n[c], replace=True) for c in range(K) if n[c]])
            yy, pp = y_te[idx], p_te[idx]
            pi_hat = bbse_prior(C, pp)
            for d, a in DECODERS.items():
                res[d].append(fast_dice(yy, adjust(pp, pi_src, pi_hat, a)))
    return {d: np.array(v) for d, v in res.items()}


def load_group(runs: dict) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    agg = aggregate(runs)
    probs = [pd.read_csv(r["dir"] / "test_probs.csv").set_index("lesion_id").sort_index()[CLASSES]
             for r in runs.values() if (r["dir"] / "test_probs.csv").exists()]
    test = sum(probs) / len(probs) if len(probs) == len(runs) else None
    return agg["oof"], test, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["outputs"])
    ap.add_argument("--out", default="outputs/posthoc_shift")
    ap.add_argument("--ref", default="swin_base_384__abl_M0", help="nhóm mốc để tính Δ")
    ap.add_argument("--groups", nargs="*", default=None, help="chỉ các nhóm này (mặc định: mọi nhóm đủ 5 fold)")
    ap.add_argument("--reps", type=int, default=200, help="số lần rút mỗi fold trong mô phỏng")
    args = ap.parse_args()
    out = resolve_path(args.out)
    (out / "submissions" / "_khac").mkdir(parents=True, exist_ok=True)

    abl, _ = collect(args.dirs)
    groups = {f"{bb}__abl_{var}": runs for (bb, var), runs in abl.items() if len(runs) == 5}
    if args.groups:
        groups = {g: r for g, r in groups.items() if g in args.groups}
    if not groups:
        raise SystemExit(f"Không có nhóm nào đủ 5 fold trong {args.dirs}")
    print("Nhóm:", ", ".join(sorted(groups)))

    loaded, pis = {}, []
    for g in groups:
        oof, test, agg = load_group(groups[g])
        y, p = oof.label.to_numpy(), oof[CLASSES].to_numpy()
        pi_hat = bbse_prior(soft_confusion(p, y), test.to_numpy()) if test is not None else None
        loaded[g] = (oof, test, agg, pi_hat)
        if pi_hat is not None:
            pis.append(pi_hat)
    # Prior mục tiêu CHUNG của mô phỏng: trung vị các π̂ (BBSE từng nhóm, không nhãn test) -> mọi nhóm cùng phép rút
    target = np.median(pis, axis=0) if pis else None
    if target is not None:
        target = target / target.sum()
        print("π̂_test chung (x479):", dict(zip(CLASSES, np.round(target * N_TEST).astype(int))))

    rows, sims, counts = [], {}, {}
    for g in sorted(groups, key=lambda g: (g != args.ref, g)):
        oof, test, agg, pi_hat = loaded[g]
        y = oof.label.to_numpy()
        pi_train = np.bincount(y, minlength=K) / len(y)
        sim_shift = simulate(oof, target if target is not None else pi_train, args.reps)
        sim_noshift = simulate(oof, pi_train, max(20, args.reps // 4), seed=1)
        means = {d: v.mean() for d, v in sim_shift.items()}
        best = max(means, key=means.get)
        if means[best] - means["raw"] < 0.002:
            best = "raw"
        sims[g] = sim_shift[best]
        row = {"Nhóm": g, "Mô tả": DESC.get(g, ""), "Dice fold (mean ± sd)": f"{agg['dice_mean']:.3f} ± {agg['dice_std']:.3f}",
               "Dice OOF gộp": agg["pooled"]["dice"], "AUC OOF": agg["pooled"]["auc"]}
        for d in DECODERS:
            row[f"mô phỏng {d}"] = f"{sim_shift[d].mean():.3f} ± {sim_shift[d].std():.3f}"
        row["không lệch: raw → bbse"] = f"{sim_noshift['raw'].mean():.3f} → {sim_noshift['bbse'].mean():.3f}"
        row["Cách chọn"] = best
        if test is not None:
            ids = test.index
            for d, a in DECODERS.items():
                q = adjust(test.to_numpy(), pi_train, pi_hat, a)
                sub = out / "submissions" / (f"{g}__cv5__{d}" if d == best else f"_khac/{g}__cv5__{d}")
                sub.mkdir(parents=True, exist_ok=True)
                write_submission(ids, q, "top1", sub, ids, logger=lambda *_: None)
                if d == best:
                    counts[g] = np.bincount(q.argmax(1), minlength=K)
            counts[g + " (π̂ BBSE x479)"] = np.round(pi_hat * N_TEST).astype(int)
            row["File nộp"] = f"submissions/{g}__cv5__{best}/submission.csv"
        else:
            row["File nộp"] = "thiếu test_probs.csv"
        rows.append(row)
        print(f"{g}: chọn {best} | " + " ".join(f"{d}={means[d]:.3f}" for d in DECODERS))

    table = pd.DataFrame(rows)
    if args.ref in sims:                                  # Δ so với mốc trên CÙNG cách rút (cùng seed, cùng số mẫu)
        ref = sims[args.ref]
        table["Δ mô phỏng vs mốc"] = [
            "—" if r["Nhóm"] == args.ref or len(sims[r["Nhóm"]]) != len(ref) else
            f"{(sims[r['Nhóm']] - ref).mean():+.3f} (P>0 = {np.mean(sims[r['Nhóm']] > ref):.2f})" for r in rows]
    cnt = pd.DataFrame(counts, index=CLASSES).T.reset_index().rename(columns={"index": "Nhóm"})

    md = ["# Hậu xử lý label shift (BBSE) — stage 9", "",
          "Mô phỏng: cross-fit trên OOF, mẫu giả-test 479 lesion rút theo π̂_test (BBSE trên ảnh test, không nhãn). "
          "Chọn cách giải mã theo mô phỏng, chốt TRƯỚC khi nộp. Mỗi nhóm nộp đúng 1 file (cột 'File nộp').", "",
          md_table(table.map(lambda v: f"{v:.3f}" if isinstance(v, float) else v)), "",
          "## Số lesion dự đoán / ước lượng theo lớp trên test (không có nhãn test)", "", md_table(cnt), ""]
    (out / "posthoc_shift.md").write_text("\n".join(md), encoding="utf-8")
    write_excel(out / "posthoc_shift.xlsx", {"Tong_hop": table, "So_luong_lop": cnt})
    print("\n".join(md))


if __name__ == "__main__":
    main()
