"""Loss cho dữ liệu mất cân bằng.

class_weight:
  none          không trọng số
  inv           N / (K * n_c)
  sqrt_inv      sqrt(inv) — giảm nhẹ hơn inv
  effective_num Class-Balanced (Cui et al., 2019): w_c = (1 - beta) / (1 - beta^n_c), beta -> 1 tiến về inv
loss:
  ce | focal | logit_adjusted (Menon et al., 2021): CE(logits + tau * log prior, y). Chỉ cộng bias khi train;
  lúc suy luận dùng logit gốc -> mô hình tự "bù" prior, tối ưu trực tiếp balanced error.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import NUM_CLASSES


def class_counts(labels) -> np.ndarray:
    return np.bincount(np.asarray(labels), minlength=NUM_CLASSES).astype(np.float64)


def class_weights(labels, mode: str, beta: float = 0.999) -> torch.Tensor | None:
    """Trọng số lớp từ tần suất trong tập train của fold, chuẩn hoá về trung bình 1."""
    if mode in (None, "none"):
        return None
    counts = np.maximum(class_counts(labels), 1)
    inv = counts.sum() / (NUM_CLASSES * counts)
    if mode == "inv":
        w = inv
    elif mode == "sqrt_inv":
        w = np.sqrt(inv)
    elif mode == "effective_num":
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise KeyError(f"class_weight không hỗ trợ: {mode}")
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        logits = logits.float()
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = F.softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = (1 - pt) ** self.gamma * ce
        denom = self.weight[target].sum() if self.weight is not None else target.numel()
        return loss.sum() / denom


class LogitAdjustedLoss(nn.Module):
    def __init__(self, prior: torch.Tensor, tau=1.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("log_prior", tau * torch.log(prior))
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits.float() + self.log_prior, target, weight=self.weight,
                               label_smoothing=self.label_smoothing)


def build_loss(cfg: dict, train_labels) -> nn.Module:
    lc = cfg["loss"]
    weight = class_weights(train_labels, lc.get("class_weight"), lc.get("cb_beta", 0.999))
    ls = lc.get("label_smoothing", 0.0)
    if lc["name"] == "ce":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)
    if lc["name"] == "focal":
        return FocalLoss(weight, lc.get("focal_gamma", 2.0), ls)
    if lc["name"] == "logit_adjusted":
        counts = np.maximum(class_counts(train_labels), 1)
        prior = torch.tensor(counts / counts.sum(), dtype=torch.float32)
        return LogitAdjustedLoss(prior, lc.get("la_tau", 1.0), weight, ls)
    raise KeyError(f"Loss không hỗ trợ: {lc['name']}")
