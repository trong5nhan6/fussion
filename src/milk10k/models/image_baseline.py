"""Baseline chỉ dùng ảnh: một backbone timm (CNN hoặc Transformer) dùng chung cho các view,
nối đặc trưng các view rồi đưa qua một lớp tuyến tính."""
import timm
import torch
from torch import nn

from .. import NUM_CLASSES


VALID_VIEWS = ("clin", "derm")


def check_views(views) -> list:
    views = list(views)
    if not views or not set(views) <= set(VALID_VIEWS) or len(set(views)) != len(views):
        raise ValueError(f"model.views phải là tập con không rỗng, không lặp của {VALID_VIEWS}, nhận {views}")
    return views


def create_encoder(backbone: str, pretrained: bool, img_size: int, grad_checkpointing=False) -> nn.Module:
    """Tạo backbone timm không có head (num_classes=0 -> trả về vector đặc trưng)."""
    try:
        # ViT/Swin cần biết img_size để nội suy position embedding / window
        enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=img_size)
    except TypeError:
        enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    if grad_checkpointing:
        enc.set_grad_checkpointing(True)
    return enc


class ImageBaseline(nn.Module):
    def __init__(self, backbone: str, views, pretrained=True, dropout=0.3, img_size=224,
                 grad_checkpointing=False, num_classes=NUM_CLASSES):
        super().__init__()
        self.views = check_views(views)
        self.encoder = create_encoder(backbone, pretrained, img_size, grad_checkpointing)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.num_features * len(self.views), num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        feats = [self.encoder(batch["images"][v]) for v in self.views]
        return self.head(torch.cat(feats, dim=1))

    def head_module(self) -> nn.Module:
        """Phần classifier — được train lại ở giai đoạn cRT."""
        return self.head

    def param_groups(self, lr: float, backbone_lr_mult: float):
        return [
            {"params": self.encoder.parameters(), "lr": lr * backbone_lr_mult},
            {"params": self.head.parameters(), "lr": lr},
        ]

    def normalization(self):
        cfg = self.encoder.pretrained_cfg
        return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))
