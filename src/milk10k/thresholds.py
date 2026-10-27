"""Tầng quyết định: ngưỡng riêng cho từng lớp (thay cho ngưỡng 0.5 chung của leaderboard).

Leaderboard luôn so điểm với 0.5, nên ngưỡng t_c được "dời" về 0.5 bằng một phép biến đổi đơn điệu theo từng lớp:
    s = 0.5 · p / t                 nếu p <= t
    s = 0.5 + 0.5 · (p − t)/(1 − t)  nếu p > t
=> s > 0.5  <=>  p > t_c. Thứ tự điểm trong từng lớp giữ nguyên nên AUC / AP không đổi, chỉ Dice / Sens / Spec đổi.

Để không "học trên tập đánh giá", Dice OOF được tính lồng nhau: ngưỡng của fold k chỉ fit trên các fold còn lại.
"""
import numpy as np

from . import CLASSES

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def _dice(truth: np.ndarray, pred: np.ndarray) -> float:
    tp = np.sum(truth & pred)
    denom = 2 * tp + np.sum(~truth & pred) + np.sum(truth & ~pred)
    return 2 * tp / denom if denom else 1.0      # quy ước freebie của ISIC


def fit_thresholds(y: np.ndarray, probs: np.ndarray, grid=GRID) -> np.ndarray:
    """Ngưỡng tối đa hoá Dice của từng lớp; hoà thì chọn ngưỡng gần 0.5 nhất (ổn định hơn)."""
    y = np.asarray(y)
    out = np.full(probs.shape[1], 0.5)
    for k in range(probs.shape[1]):
        truth = y == k
        scores = np.array([_dice(truth, probs[:, k] > t) for t in grid])
        best = np.flatnonzero(scores >= scores.max() - 1e-12)
        out[k] = grid[best[np.argmin(np.abs(grid[best] - 0.5))]]
    return out


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Biến xác suất thành điểm nộp bài sao cho điểm > 0.5 đúng khi p > t_c."""
    p = np.asarray(probs, dtype=np.float64)
    t = np.clip(np.asarray(thresholds, dtype=np.float64), 1e-6, 1 - 1e-6)[None, :]
    return np.where(p <= t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t))


def nested_threshold_scores(y: np.ndarray, probs: np.ndarray, folds: np.ndarray):
    """Điểm OOF với ngưỡng fit lồng nhau theo fold.

    Trả về (scores, thresholds theo fold, lồng_nhau: bool). Chỉ có 1 fold -> fit và áp trên cùng fold
    (lạc quan, chỉ để tham khảo) và lồng_nhau = False.
    """
    y, folds = np.asarray(y), np.asarray(folds)
    uniq = np.unique(folds)
    scores = np.zeros_like(probs, dtype=np.float64)
    per_fold = {}
    for f in uniq:
        test = folds == f
        fit = ~test if len(uniq) > 1 else test
        t = fit_thresholds(y[fit], probs[fit])
        per_fold[int(f)] = dict(zip(CLASSES, t.tolist()))
        scores[test] = apply_thresholds(probs[test], t)
    return scores, per_fold, len(uniq) > 1
