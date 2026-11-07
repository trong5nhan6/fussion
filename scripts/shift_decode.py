"""Shift-aware set-valued decoding (không train): tận dụng cách chấm one-vs-rest của MILK10k để tăng macro Dice.

Cách chấm: mỗi lớp là một bài toán nhị phân riêng ở ngưỡng 0.5 -> một lesion ĐƯỢC PHÉP dương ở nhiều lớp. Giải mã top-1
hiện tại bỏ phí điều đó. Phân tích leaderboard: AUC của M0 (0.925) ngang/hơn các đội 0.61 nhưng Dice thấp hơn ~0.09
-> mất điểm ở bước quyết định (vd BCC bị dự đoán ~2 lần số ca thật do label shift; DF AUC 0.99 nhưng Dice 0.56).

Bộ giải mã (mọi bước chỉ dùng OOF có nhãn + xác suất trên ảnh test KHÔNG nhãn):
  1. p      = trung bình xác suất test của 5 fold (như file nộp cv5 hiện tại).
  2. calib  = vector scaling (nhiệt độ + bias theo lớp) fit trên OOF -> xác suất khớp prior train.
  3. π̂      = BBSE (Lipton et al., 2018) trên xác suất đã calib; π½ ∝ sqrt(π̂ · π_train) (đi nửa bước, giảm nhiễu ước lượng).
  4. q      = calib(p) · w / Σ, w = π_target / π_train chặn trong [1/10, 10].
  5. Với từng lớp c: tập dương = k lesion có q_c cao nhất, k cực đại F1 kỳ vọng  2·Σ_{top k} q_c / (k + Σ q_c).
  6. Nhãn cuối = argmax(p) ∪ tập dương của từng lớp (mỗi lesion luôn có ≥ 1 lớp dương).
  Điểm nộp: lớp dương 0.5 + 0.5·q, lớp âm 0.5·q (thứ tự trong lớp gần như giữ nguyên -> AUC gần như không đổi).

Các cách giải mã so sánh: raw (top-1 như cũ) · prior_bbse(_half) (nhân lại prior rồi top-1, không đa nhãn) · ef1_cal (bước 5 trên calib, không ước lượng prior) · ef1_bbse_half · ef1_bbse.

Chọn cách giải mã bằng MÔ PHỎNG CROSS-FIT trên OOF (chốt trước khi nộp, không nhìn nhãn hay điểm test):
  fold k giữ lại; calib + ma trận nhầm lẫn + π_train fit trên 4 fold còn lại; rút 479 lesion từ fold k theo prior mục tiêu
  (mỗi lớp ≥ 5 ca như test thật): "không lệch" (prior train) · "π̂ test" (BBSE trên ảnh test thật, không nhãn) ·
  --random prior ngẫu nhiên Dirichlet(2) (seed cố định). Δ = Dice(cách giải mã) − Dice(raw), ghép cặp trên cùng mẫu.
  LUẬT CHỌN: trong các cách có  Δ TB trên các kịch bản lệch ≥ +0.01  và  Δ không lệch ≥ −0.005,  chọn cách có Δ lệch TB
  cao nhất; không cách nào đạt -> giữ raw. Mỗi nhóm đúng 1 file nộp.
Lưu ý: mẫu giả-test dùng xác suất OOF của 1 mô hình, còn test thật là trung bình 5 mô hình (sắc hơn một chút).

  python scripts/shift_decode.py --dirs /content/drive/MyDrive/milk10k_ablation/outputs --groups swin_base_384__abl_M0
Đầu ra (--out): shift_decode.md, shift_decode_sim.csv, submissions/<nhóm>__cv5__<cách>/ (nộp) + submissions/_khac/.
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ablation_tables import collect, md_table
from milk10k import CLASSES
from milk10k.inference import write_submission
from milk10k.utils import resolve_path
from posthoc_shift import bbse_prior, load_group, soft_confusion

K = len(CLASSES)
N_TEST = 479
MIN_CASES = 5
MAX_RATIO = 10.0
DECODERS = ["raw", "prior_bbse_half", "prior_bbse", "ef1_cal", "ef1_bbse_half", "ef1_bbse"]
GAIN_SHIFT, LOSS_NOSHIFT = 0.01, -0.005          # luật chọn (chốt trước)


# ------------------------------------------------------------------ các bước giải mã
class VectorScaling:
    """softmax(log p / T + b): T, b fit bằng NLL trên dữ liệu có nhãn (calib theo prior của dữ liệu đó)."""

    def fit(self, p, y):
        lp = np.log(np.clip(p, 1e-8, 1))

        def nll(w):
            z = lp / np.exp(w[0]) + w[1:]
            z = z - z.max(1, keepdims=True)
            return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()

        self.w = minimize(nll, np.zeros(K + 1), method="L-BFGS-B").x
        return self

    def __call__(self, p):
        z = np.log(np.clip(p, 1e-8, 1)) / np.exp(self.w[0]) + self.w[1:]
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def reweight(q, pi_src, pi_tgt):
    w = np.clip(pi_tgt / np.clip(pi_src, 1e-12, None), 1 / MAX_RATIO, MAX_RATIO)
    q = q * w
    return q / q.sum(1, keepdims=True)


def top1(p):
    Y = np.zeros(p.shape, bool)
    Y[np.arange(len(p)), p.argmax(1)] = True
    return Y


def ef1_sets(q):
    """Từng lớp: chọn k lesion có q_c cao nhất, k cực đại F1 kỳ vọng (plug-in) 2·Σtop-k q / (k + Σ q)."""
    Y = np.zeros(q.shape, bool)
    for c in range(K):
        order = np.argsort(-q[:, c])
        cs = np.cumsum(q[order, c])
        k = int(np.argmax(2 * cs / (np.arange(1, len(cs) + 1) + q[:, c].sum()))) + 1
        Y[order[:k], c] = True
    return Y


def decode(p, cal, M, pi_src):
    """-> {cách: (nhãn bool [n, K], q dùng làm điểm [n, K])}. p: xác suất thô; cal/M/pi_src: fit trên dữ liệu có nhãn."""
    pc = cal(p)
    pi_hat = bbse_prior(M, pc)
    pi_half = np.sqrt(pi_hat * pi_src)
    pi_half /= pi_half.sum()
    A = top1(p)
    q_half, q_full = reweight(pc, pi_src, pi_half), reweight(pc, pi_src, pi_hat)
    return {"raw": (A, p), "prior_bbse_half": (top1(q_half), q_half), "prior_bbse": (top1(q_full), q_full),
            "ef1_cal": (A | ef1_sets(pc), pc),
            "ef1_bbse_half": (A | ef1_sets(q_half), q_half), "ef1_bbse": (A | ef1_sets(q_full), q_full)}, pi_hat


def dice_per_class(y, Y):
    out = np.empty(K)
    for c in range(K):
        t, p = y == c, Y[:, c]
        den = t.sum() + p.sum()
        out[c] = 2 * (t & p).sum() / den if den else 1.0
    return out


def scores_from(Y, q):
    return np.where(Y, 0.5 + 0.5 * q, 0.5 * np.minimum(q, 0.999))


# ------------------------------------------------------------------ mô phỏng cross-fit
def simulate(oof, targets: dict, reps: int, seed: int = 0) -> pd.DataFrame:
    """-> DataFrame dài: kịch bản, rep, cách, macro Dice, Dice từng lớp."""
    rng = np.random.default_rng(seed)
    y_all, p_all, f_all = oof.label.to_numpy(), oof[CLASSES].to_numpy(), oof.fold.to_numpy()
    rows = []
    for k in np.unique(f_all):
        cal_m, te = f_all != k, f_all == k
        cal = VectorScaling().fit(p_all[cal_m], y_all[cal_m])
        M = soft_confusion(cal(p_all[cal_m]), y_all[cal_m])
        pi_src = np.bincount(y_all[cal_m], minlength=K) / cal_m.sum()
        y_te, p_te = y_all[te], p_all[te]
        pools = [np.where(y_te == c)[0] for c in range(K)]
        avail = np.array([len(pl) > 0 for pl in pools])
        for scen, tgt in targets.items():
            tp = np.maximum(tgt, MIN_CASES / N_TEST) * avail
            tp /= tp.sum()
            for r in range(reps):
                n = rng.multinomial(N_TEST, tp)
                idx = np.concatenate([rng.choice(pools[c], n[c]) for c in range(K) if n[c]])
                dec, _ = decode(p_te[idx], cal, M, pi_src)
                for name, (Y, _) in dec.items():
                    d = dice_per_class(y_te[idx], Y)
                    rows.append({"scenario": scen, "fold": int(k), "rep": r, "decoder": name, "dice": d.mean(),
                                 **dict(zip(CLASSES, d))})
    return pd.DataFrame(rows)


def choose(sim: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    piv = sim.pivot_table(index=["scenario", "fold", "rep"], columns="decoder", values="dice")
    delta = piv.sub(piv["raw"], axis=0).groupby(level="scenario").mean()
    shift = delta.drop(index="không lệch").mean()
    noshift = delta.loc["không lệch"]
    ok = [d for d in DECODERS[1:] if shift[d] >= GAIN_SHIFT and noshift[d] >= LOSS_NOSHIFT]
    best = max(ok, key=lambda d: shift[d]) if ok else "raw"
    summary = delta.T.copy()
    summary.insert(0, "Δ TB các kịch bản lệch", shift)
    summary["Dice TB (π̂ test)"] = piv.groupby(level="scenario").mean().loc["π̂ test"]
    summary["đạt luật"] = ["—" if d == "raw" else ("có" if d in ok else "không") for d in summary.index]
    return best, summary.reindex(DECODERS)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["outputs"])
    ap.add_argument("--groups", nargs="*", default=None, help="vd swin_base_384__abl_M0 (mặc định: mọi nhóm đủ 5 fold)")
    ap.add_argument("--out", default="outputs/shift_decode")
    ap.add_argument("--reps", type=int, default=100, help="số mẫu giả-test mỗi fold mỗi kịch bản")
    ap.add_argument("--random", type=int, default=4, help="số prior ngẫu nhiên Dirichlet(2)")
    args = ap.parse_args()
    out = resolve_path(args.out)
    (out / "submissions" / "_khac").mkdir(parents=True, exist_ok=True)

    abl, _ = collect(args.dirs)
    groups = {f"{bb}__abl_{var}": runs for (bb, var), runs in abl.items() if len(runs) == 5}
    if args.groups:
        missing = set(args.groups) - set(groups)
        if missing:
            raise SystemExit(f"Không đủ 5 fold cho: {sorted(missing)} trong {args.dirs}")
        groups = {g: groups[g] for g in args.groups}
    if not groups:
        raise SystemExit(f"Không có nhóm nào đủ 5 fold trong {args.dirs}")

    md = ["# Shift-aware set-valued decoding", "",
          f"Luật chọn (chốt trước khi nộp): Δ TB các kịch bản lệch ≥ +{GAIN_SHIFT} và Δ không lệch ≥ {LOSS_NOSHIFT}; "
          "không đạt -> raw. Δ = so với top-1 hiện tại, ghép cặp trên cùng mẫu giả-test.", ""]
    sims, rows_cnt = [], {}
    for g, runs in groups.items():
        oof, test, agg = load_group(runs)
        if test is None:
            print(f"[bỏ qua] {g}: thiếu test_probs.csv ở một số fold")
            continue
        y, p = oof.label.to_numpy(), oof[CLASSES].to_numpy()
        cal = VectorScaling().fit(p, y)
        M = soft_confusion(cal(p), y)
        pi_train = np.bincount(y, minlength=K) / len(y)
        p_test = test.to_numpy()
        dec, pi_hat = decode(p_test, cal, M, pi_train)

        rng = np.random.default_rng(123)
        targets = {"không lệch": pi_train, "π̂ test": pi_hat}
        targets.update({f"ngẫu nhiên {i}": rng.dirichlet(np.full(K, 2.0)) for i in range(args.random)})
        print(f"{g}: mô phỏng {len(targets)} kịch bản x 5 fold x {args.reps} mẫu ...")
        sim = simulate(oof, targets, args.reps)
        sim.insert(0, "group", g)
        sims.append(sim)
        best, summary = choose(sim)

        ids = test.index
        for name, (Y, q) in dec.items():
            sub = out / "submissions" / (f"{g}__cv5__{name}" if name == best else f"_khac/{g}__cv5__{name}")
            sub.mkdir(parents=True, exist_ok=True)
            write_submission(ids, scores_from(Y, q), "softmax", sub, ids, logger=lambda *_: None)
            rows_cnt[f"{name}{' (CHỌN)' if name == best else ''}"] = Y.sum(0)
        rows_cnt["π̂ BBSE x479"] = np.round(pi_hat * N_TEST).astype(int)
        lab = {n: Y.sum(1).mean() for n, (Y, _) in dec.items()}

        per = sim[sim.scenario == "π̂ test"].groupby("decoder")[CLASSES].mean().reindex(DECODERS)
        md += [f"## {g}", "",
               f"Dice OOF gộp (top-1): {agg['pooled']['dice']:.3f} · AUC OOF {agg['pooled']['auc']:.3f}", "",
               f"**Cách được chọn: `{best}`** -> nộp `submissions/{g}__cv5__{best}/submission.csv`", "",
               "### Δ Dice mô phỏng so với raw (TB trên 5 fold x mẫu)", "",
               md_table(summary.reset_index().rename(columns={"decoder": "cách"}).map(
                   lambda v: f"{v:+.3f}" if isinstance(v, float) else v)), "",
               "### Dice từng lớp, kịch bản π̂ test", "",
               md_table(per.reset_index().rename(columns={"decoder": "cách"}).map(
                   lambda v: f"{v:.3f}" if isinstance(v, float) else v)), "",
               "### Số lesion dương theo lớp trên test thật (không nhãn)", "",
               md_table(pd.DataFrame(rows_cnt, index=CLASSES).T.reset_index().rename(columns={"index": "cách"})), "",
               "Số nhãn dương TB / lesion: " + ", ".join(f"{n} {v:.2f}" for n, v in lab.items()), ""]
        rows_cnt = {}
        print(f"{g}: chọn {best}\n{summary.round(3).to_string()}")

    pd.concat(sims).to_csv(out / "shift_decode_sim.csv", index=False)
    (out / "shift_decode.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nBáo cáo: {out / 'shift_decode.md'}")


if __name__ == "__main__":
    main()
