"""Metric theo đúng cách chấm của ISIC MILK10k (isic-challenge-scoring 5.8.0, classification.py / metrics.py).

- Mỗi lớp được chấm như một bài toán nhị phân độc lập: dự đoán dương khi điểm > 0.5 (`.gt(0.5)`).
- Accuracy / Sensitivity / Specificity / Dice / PPV / NPV tính từ ma trận nhầm lẫn nhị phân của từng lớp,
  cùng quy ước "freebie" = 1.0 khi mẫu số bằng 0. AUC và AP dùng điểm liên tục.
- Leaderboard = trung bình macro trên 11 lớp. Xếp hạng theo Dice (= macro F1).
- balanced_acc: metric tổng hợp đa lớp của ISIC (argmax), top1_acc: accuracy đa lớp thông thường.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from . import CLASSES, NUM_CLASSES

THRESHOLD = 0.5
# (khoá, tên cột như trên leaderboard)
LEADERBOARD_COLUMNS = [("auc", "AUC"), ("ap", "Average Precision"), ("accuracy", "Accuracy"),
                       ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("dice", "Dice Coefficient")]
PER_CLASS_KEYS = ["auc", "ap", "accuracy", "sensitivity", "specificity", "dice", "ppv", "npv"]
SUMMARY_KEYS = ["dice", "auc", "ap", "accuracy", "sensitivity", "specificity", "balanced_acc", "top1_acc"]


def postprocess(probs: np.ndarray, mode: str = "top1") -> np.ndarray:
    """Biến xác suất softmax thành điểm nộp bài.

    softmax: giữ nguyên. Nhiều lesion có max < 0.5 -> không lớp nào được tính dương.
    top1:    lớp argmax -> 0.5 + 0.5p (luôn > 0.5), các lớp khác -> 0.5p (luôn < 0.5).
             Mỗi lesion có đúng 1 lớp dương, khớp với nhãn đơn lớp; thứ tự điểm trong từng nhóm được giữ.
    """
    if mode == "softmax":
        return probs
    if mode == "top1":
        out = 0.5 * probs
        idx = probs.argmax(axis=1)
        rows = np.arange(len(probs))
        out[rows, idx] = 0.5 + 0.5 * probs[rows, idx]
        return out
    raise KeyError(f"postprocess không hỗ trợ: {mode}")


def _binary_scores(truth: np.ndarray, pred: np.ndarray) -> dict:
    tp = float(np.sum(truth & pred))
    tn = float(np.sum(~truth & ~pred))
    fp = float(np.sum(~truth & pred))
    fn = float(np.sum(truth & ~pred))
    div = lambda a, b: a / b if b else 1.0  # noqa: E731  quy ước freebie của ISIC
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "dice": div(2 * tp, 2 * tp + fp + fn),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
    }


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, mode: str = "top1") -> dict:
    """y_true: nhãn int [N]; probs: xác suất softmax [N, C] (chưa hậu xử lý)."""
    y_true = np.asarray(y_true)
    scores = postprocess(np.asarray(probs, dtype=np.float64), mode)
    rows = {}
    for k, c in enumerate(CLASSES):
        truth = y_true == k
        r = _binary_scores(truth, scores[:, k] > THRESHOLD)
        has_both = 0 < truth.sum() < len(truth)  # AUC/AP không xác định nếu fold thiếu lớp
        r["auc"] = float(roc_auc_score(truth, scores[:, k])) if has_both else np.nan
        r["ap"] = float(average_precision_score(truth, scores[:, k])) if has_both else np.nan
        r["n"] = int(truth.sum())
        rows[c] = r
    per_class = pd.DataFrame(rows).T[["n"] + PER_CLASS_KEYS]
    macro = per_class[PER_CLASS_KEYS].mean(skipna=True)

    pred = scores.argmax(axis=1)
    cm = confusion_matrix(y_true, pred, labels=list(range(NUM_CLASSES)))
    support = cm.sum(axis=1)
    present = support > 0
    return {
        **{k: float(macro[k]) for k in PER_CLASS_KEYS},
        "balanced_acc": float((cm.diagonal()[present] / support[present]).mean()),
        "top1_acc": float((pred == y_true).mean()),
        "per_class": per_class.to_dict(orient="index"),
    }


def confusion(y_true, probs) -> np.ndarray:
    return confusion_matrix(y_true, np.asarray(probs).argmax(axis=1), labels=list(range(NUM_CLASSES)))


def leaderboard_table(m: dict) -> str:
    """Bảng giống leaderboard ISIC: từng lớp + dòng trung bình macro."""
    per_class = pd.DataFrame(m["per_class"]).T
    cols = [k for k, _ in LEADERBOARD_COLUMNS]
    table = per_class[["n"] + cols].copy()
    table.loc["Macro average"] = [int(per_class["n"].sum())] + [m[k] for k in cols]
    table = table.rename(columns=dict(LEADERBOARD_COLUMNS))
    table["n"] = table["n"].astype(int)
    return table.to_string(float_format=lambda v: f"{v:.4f}")


def summary_line(m: dict) -> str:
    return " ".join(f"{name}={m[k]:.4f}" for k, name in
                    [("auc", "AUC"), ("ap", "AP"), ("accuracy", "Acc"), ("sensitivity", "Sens"),
                     ("specificity", "Spec"), ("dice", "Dice")])
