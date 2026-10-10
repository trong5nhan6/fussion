"""Nối thẳng đặc trưng ảnh (CLS) rồi qua MLP 2 lớp — không chiếu token, không MoE.

  clin -> backbone -> CLS [B, F] ┐
  derm -> backbone -> CLS [B, F] ┼-> nối [B, F*n_views (+34 nếu concat_meta)] -> Linear -> GELU -> Dropout -> Linear -> [B, 11]
  (tuỳ chọn) metadata  [B, 34]   ┘
Với DINOv2-B: F = 768 (token CLS, global_pool="token").
"""
import torch
from torch import nn

from .. import NUM_CLASSES
from .image_baseline import check_views, create_encoder


class ConcatMLP(nn.Module):
    def __init__(self, backbone: str, views, pretrained=True, img_size=224, grad_checkpointing=False,
                 drop_path_rate=0.0, hidden=512, dropout=0.3, concat_meta=False, meta_dim=0,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.views = check_views(views)
        self.encoder = create_encoder(backbone, pretrained, img_size, grad_checkpointing, drop_path_rate)
        self.uses_metadata = bool(concat_meta)
        if self.uses_metadata and meta_dim <= 0:
            raise ValueError("model.concat_meta=true cần vector metadata (meta_dim > 0)")
        in_dim = self.encoder.num_features * len(self.views) + (meta_dim if self.uses_metadata else 0)
        self.head = nn.Sequential(                      # MLP 2 lớp
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        feats = [self.encoder(batch["images"][v]) for v in self.views]   # mỗi ảnh: CLS [B, F]
        if self.uses_metadata:
            feats.append(batch["meta"].to(feats[0].dtype))
        return self.head(torch.cat(feats, dim=1))

    def head_module(self) -> nn.Module:
        return self.head

    def param_groups(self, lr: float, backbone_lr_mult: float):
        return [
            {"params": self.encoder.parameters(), "lr": lr * backbone_lr_mult},
            {"params": self.head.parameters(), "lr": lr},
        ]

    def normalization(self):
        cfg = self.encoder.pretrained_cfg
        return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))
