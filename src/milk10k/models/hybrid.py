"""Hybrid CNN + ViT trên nền MoE hướng D: mỗi ảnh đi qua CẢ hai backbone, mỗi backbone cho một token riêng.

  ảnh v ─┬─ ViT/Swin (model.backbone, dùng chung 2 ảnh) ─> token v        (như MoE-D, d = moe.d_model)
         └─ CNN (hybrid.cnn_backbone, dùng chung 2 ảnh) ──> Linear+LN ──> token v_cnn
  6 token {clin, derm, demo, monet, clin_cnn, derm_cnn} -> modality dropout + type emb -> gate theo nguồn
  -> 3 head long-tail (τ = moe.lt_taus) + head phụ cho từng token. Gate (cột gate_* trong oof.csv) cho biết mỗi lesion
  dựa vào CNN hay ViT bao nhiêu, cho từng ảnh.

Hai backbone có thể chuẩn hoá ảnh khác nhau (vd ViT augreg mean/std 0.5, CNN ImageNet): pipeline dữ liệu chuẩn hoá theo
backbone ViT (normalization()), nhánh CNN đổi lại sang mean/std của nó ngay trong forward.
Backbone CNN nhận cùng lr với backbone ViT (train.lr * backbone_lr_mult), cùng drop path, cùng grad checkpointing.
"""
import torch
from torch import nn

from .image_baseline import create_encoder, encoder_param_groups
from .moe import MultimodalMoE


def _mean_std(enc):
    cfg = enc.pretrained_cfg
    return (torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1),
            torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))


class HybridMoE(MultimodalMoE):
    uses_metadata = True

    def __init__(self, arch, backbone, views, *, cnn_backbone, **moe_kw):
        if not cnn_backbone:
            raise ValueError("HybridMoE cần hybrid.cnn_backbone (tên timm của backbone CNN)")
        super().__init__(arch, backbone, views, extra_modalities=[f"{v}_cnn" for v in views], **moe_kw)
        self.cnn = create_encoder(cnn_backbone, moe_kw.get("pretrained", True), moe_kw.get("img_size", 224),
                                  moe_kw.get("grad_checkpointing", False), moe_kw.get("drop_path_rate", 0.0),
                                  moe_kw.get("trainable_blocks"))
        self.cnn_proj = nn.Sequential(nn.Linear(self.cnn.num_features, self.tokens.d), nn.LayerNorm(self.tokens.d))
        (mv, sv), (mc, sc) = _mean_std(self.tokens.encoder), _mean_std(self.cnn)
        # x_cnn = (x * s_v + m_v - m_c) / s_c  ==  x * scale + shift
        self.register_buffer("cnn_scale", sv / sc, persistent=False)
        self.register_buffer("cnn_shift", (mv - mc) / sc, persistent=False)

    def forward(self, batch):
        extra = [self.cnn_proj(self.cnn(batch["images"][v] * self.cnn_scale + self.cnn_shift))
                 for v in self.tokens.views]
        return self.decode_tokens(self.tokens(batch, extra=extra))

    def param_groups(self, lr, backbone_lr_mult):
        cnn_ids = {id(p) for p in self.cnn.parameters()}
        enc_ids = {id(p) for p in self.tokens.encoder.parameters()}
        rest = [p for p in self.parameters() if id(p) not in enc_ids and id(p) not in cnn_ids]
        return [*encoder_param_groups(self.tokens.encoder, lr * backbone_lr_mult, self.layer_decay),
                {"params": list(self.cnn.parameters()), "lr": lr * backbone_lr_mult},
                {"params": rest, "lr": lr * self.fusion_lr_mult}]
