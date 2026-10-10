"""Baseline chỉ dùng ảnh: một backbone timm (CNN hoặc Transformer) dùng chung cho các view,
nối đặc trưng các view rồi đưa qua MLP 2 lớp (Linear -> GELU -> Dropout -> Linear)."""
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


def backbone_blocks(enc: nn.Module) -> list:
    """Các khối nối tiếp của backbone: ViT/DINOv2 `blocks` (12), Swin `layers` (4 stage),
    ConvNeXt `stages` (4), ResNet `layer1..4` (4)."""
    for attr in ("blocks", "layers", "stages"):
        seq = getattr(enc, attr, None)
        if isinstance(seq, (nn.Sequential, nn.ModuleList)):
            return list(seq)
    if all(hasattr(enc, f"layer{i}") for i in range(1, 5)):
        return [getattr(enc, f"layer{i}") for i in range(1, 5)]
    raise ValueError(f"Không xác định được các block của backbone {type(enc).__name__} để đóng băng")


def freeze_except_last_blocks(enc: nn.Module, n: int | None) -> None:
    """n = None: train toàn bộ backbone. n >= 0: đóng băng tất cả (patch embed, pos embed, CLS/register, các block
    đầu), chỉ train n block cuối + lớp norm cuối. n = 0: chỉ train norm cuối (gần như linear probe)."""
    if n is None:
        return
    blocks = backbone_blocks(enc)
    enc.requires_grad_(False)
    for b in blocks[max(0, len(blocks) - int(n)):] if n > 0 else []:
        b.requires_grad_(True)
    for name in ("norm", "fc_norm", "head"):        # norm cuối (ConvNeXt: head.norm khi num_classes=0)
        mod = getattr(enc, name, None)
        if isinstance(mod, nn.Module):
            mod.requires_grad_(True)


def create_encoder(backbone: str, pretrained: bool, img_size: int, grad_checkpointing=False,
                   drop_path_rate: float = 0.0, trainable_blocks: int | None = None) -> nn.Module:
    """Tạo backbone timm không có head (num_classes=0 -> trả về vector đặc trưng)."""
    kw = {"drop_path_rate": drop_path_rate} if drop_path_rate else {}  # stochastic depth (regularization)
    try:
        # ViT/Swin cần biết img_size để nội suy position embedding / window
        enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=img_size, **kw)
    except TypeError:
        enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
    if grad_checkpointing:
        enc.set_grad_checkpointing(True)
    freeze_except_last_blocks(enc, trainable_blocks)
    return enc


class ImageBaseline(nn.Module):
    def __init__(self, backbone: str, views, pretrained=True, dropout=0.3, img_size=224,
                 grad_checkpointing=False, drop_path_rate=0.0, mlp_hidden=512, trainable_blocks=None,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.views = check_views(views)
        self.encoder = create_encoder(backbone, pretrained, img_size, grad_checkpointing, drop_path_rate,
                                      trainable_blocks)
        self.head = nn.Sequential(                       # MLP 2 lớp trên đặc trưng đã nối
            nn.Linear(self.encoder.num_features * len(self.views), mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
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
