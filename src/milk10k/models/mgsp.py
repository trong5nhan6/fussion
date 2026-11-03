"""MGSP — Metadata-Guided Spatial Pooling (bài TAIMMA, tách khỏi hướng MoE/gate của bài Q1).

Metadata sinh query, query chạy cross-attention lên lưới patch token của TỪNG ảnh (clin, derm) riêng biệt,
dùng chung trọng số attention và chung bộ query cho 2 ảnh:

  clin -> backbone (dùng chung) -> CLS_c [B, D] + patch P_c [B, N, D] ┐
  derm -> backbone (dùng chung) -> CLS_d [B, D] + patch P_d [B, N, D] ┤
  meta [B, 34] -> MLP -> k_meta query  ┐                              │
  k_static query học được (tĩnh)      ┴-> Q [B, k, D] ── cross-attn ──┴-> o_c, o_d [B, D] (trung bình theo k)
  z = [o_c ‖ o_d ‖ CLS_c ‖ CLS_d ‖ meta] -> Linear -> GELU -> Dropout -> Linear -> [B, 11]

Biến thể (khối `mgsp:` trong config):
  k_meta=0            : chỉ query tĩnh (attention pooling không có metadata dẫn hướng)
  attn_views=[derm]   : chỉ derm qua attention, clin chỉ dùng CLS
  use_cls=false       : bỏ CLS của các ảnh đã qua attention khỏi z
  joint=true          : gộp patch các ảnh thành 1 chuỗi (+ view embedding), 1 softmax chung cho mọi ảnh
"""
import math

import torch
from torch import nn

from .. import NUM_CLASSES
from .image_baseline import check_views, create_encoder


def encode_tokens(enc: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Một lần chạy backbone -> (CLS / vector pool [B, D], patch token [B, N, D]).

    ViT/DINOv2: [B, prefix + N, D] -> bỏ CLS/register. Swin (NHWC) [B, H, W, D] và CNN (NCHW) [B, D, H, W] -> [B, H*W, D].
    Vector pool lấy bằng forward_head(pre_logits=True): giống hệt đầu ra của enc(x) mà ConcatMLP dùng.
    """
    feats = enc.forward_features(x)
    pooled = enc.forward_head(feats, pre_logits=True)
    d = enc.num_features
    if feats.ndim == 3:
        patches = feats[:, getattr(enc, "num_prefix_tokens", 0):]
    elif feats.ndim == 4 and feats.shape[-1] == d:          # NHWC (Swin)
        patches = feats.flatten(1, 2)
    elif feats.ndim == 4 and feats.shape[1] == d:           # NCHW (ConvNeXt, ResNet, EfficientNet)
        patches = feats.flatten(2).transpose(1, 2)
    else:
        raise ValueError(f"Không nhận dạng được shape đặc trưng {tuple(feats.shape)} (num_features={d})")
    return pooled, patches


class QueryAttnPool(nn.Module):
    """Pre-norm cross-attention của k query lên N token + FFN, cả hai có residual (khối decoder chuẩn)."""

    def __init__(self, d: int, n_heads: int, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_ffn = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * ffn_mult, d))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.ln_kv(kv)
        h, w = self.attn(self.ln_q(q), kv, kv, need_weights=True, average_attn_weights=True)  # w: [B, k, N]
        o = q + h
        return o + self.ffn(self.ln_ffn(o)), w


class MGSP(nn.Module):
    uses_metadata = True

    def __init__(self, backbone: str, views, meta_dim: int, pretrained=True, img_size=224, grad_checkpointing=False,
                 drop_path_rate=0.0, trainable_blocks=None, k_meta=2, k_static=2, n_heads=8, attn_views=None,
                 joint=False, use_cls=True, meta_in_z=True, meta_hidden=256, ffn_mult=2, attn_dropout=0.1,
                 hidden=512, dropout=0.3, num_classes=NUM_CLASSES):
        super().__init__()
        if meta_dim <= 0:
            raise ValueError("MGSP cần vector metadata (meta_dim > 0)")
        if k_meta + k_static <= 0:
            raise ValueError("mgsp.k_meta + mgsp.k_static phải > 0")
        self.views = check_views(views)
        self.attn_views = check_views(attn_views) if attn_views else list(self.views)
        if not set(self.attn_views) <= set(self.views):
            raise ValueError(f"mgsp.attn_views {self.attn_views} phải là tập con của model.views {self.views}")
        self.joint, self.use_cls, self.meta_in_z = bool(joint), bool(use_cls), bool(meta_in_z)
        self.k_meta, self.k_static = int(k_meta), int(k_static)

        self.encoder = create_encoder(backbone, pretrained, img_size, grad_checkpointing, drop_path_rate,
                                      trainable_blocks)
        d = self.encoder.num_features
        if d % n_heads:
            raise ValueError(f"num_features {d} không chia hết cho mgsp.n_heads {n_heads}")
        self.d = d
        self.meta_query = (nn.Sequential(nn.Linear(meta_dim, meta_hidden), nn.GELU(), nn.Dropout(attn_dropout),
                                         nn.Linear(meta_hidden, self.k_meta * d)) if self.k_meta else None)
        self.static_query = nn.Parameter(torch.randn(self.k_static, d) * 0.02) if self.k_static else None
        self.pool = QueryAttnPool(d, n_heads, ffn_mult, attn_dropout)
        self.view_embed = nn.Parameter(torch.zeros(len(self.attn_views), d)) if self.joint else None

        n_pooled = 1 if self.joint else len(self.attn_views)
        n_cls = sum(1 for v in self.views if self.use_cls or v not in self.attn_views)
        in_dim = d * (n_pooled + n_cls) + (meta_dim if self.meta_in_z else 0)
        self.head = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, num_classes))
        self.last_explain: dict = {}
        self.last_attn: dict = {}   # {view hoặc "joint": [B, k, N]} của lần forward gần nhất (vẽ heatmap)

    # ------------------------------------------------------------------ query
    def queries(self, meta: torch.Tensor) -> torch.Tensor:
        b = meta.shape[0]
        parts = []
        if self.meta_query is not None:
            parts.append(self.meta_query(meta).view(b, self.k_meta, self.d))
        if self.static_query is not None:
            parts.append(self.static_query.unsqueeze(0).expand(b, -1, -1))
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict) -> torch.Tensor:
        cls, patches = {}, {}
        for v in self.views:
            cls[v], patches[v] = encode_tokens(self.encoder, batch["images"][v])
        meta = batch["meta"].to(cls[self.views[0]].dtype)
        q = self.queries(meta)

        pooled, attn = [], {}
        if self.joint:
            kv = torch.cat([patches[v] + self.view_embed[i] for i, v in enumerate(self.attn_views)], dim=1)
            o, attn["joint"] = self.pool(q, kv)
            pooled.append(o.mean(1))
        else:
            for v in self.attn_views:
                o, attn[v] = self.pool(q, patches[v])
                pooled.append(o.mean(1))

        z = pooled + [cls[v] for v in self.views if self.use_cls or v not in self.attn_views]
        if self.meta_in_z:
            z.append(meta)
        self._record(attn, {v: patches[v].shape[1] for v in self.attn_views})
        return self.head(torch.cat(z, dim=1))

    @torch.no_grad()
    def _record(self, attn: dict, n_tokens: dict) -> None:
        """Ghi số liệu diễn giải (mỗi mẫu 1 số -> cột trong oof.csv) và bản đồ attention của lần forward gần nhất.

        ent_<view> : entropy attention chuẩn hoá về [0, 1], trung bình theo query (1 = trải đều, 0 = dồn vào 1 ô)
        ovl_<view> : độ chồng lấn sum(min) giữa bản đồ trung bình của query metadata và của query tĩnh (1 = giống hệt)
        mass_<view>: (joint) tổng trọng số attention rơi vào ảnh đó, trung bình theo query
        """
        self.last_attn = {k: w.detach() for k, w in attn.items()}
        ex = {}
        for key, w in attn.items():
            w = w.float()
            ent = -(w * (w + 1e-12).log()).sum(-1) / math.log(w.shape[-1])
            ex[f"ent_{key}"] = ent.mean(1)
            if key != "joint" and self.k_meta and self.k_static:
                a_meta, a_stat = w[:, :self.k_meta].mean(1), w[:, self.k_meta:].mean(1)
                ex[f"ovl_{key}"] = torch.minimum(a_meta, a_stat).sum(-1)
            if key == "joint":
                start = 0
                for v in self.attn_views:
                    ex[f"mass_{v}"] = w[:, :, start:start + n_tokens[v]].sum(-1).mean(1)
                    start += n_tokens[v]
        self.last_explain = ex

    # ------------------------------------------------------------------ giao diện engine
    def param_groups(self, lr: float, backbone_lr_mult: float):
        enc = set(id(p) for p in self.encoder.parameters())
        return [
            {"params": list(self.encoder.parameters()), "lr": lr * backbone_lr_mult},
            {"params": [p for p in self.parameters() if id(p) not in enc], "lr": lr},
        ]

    def normalization(self):
        cfg = self.encoder.pretrained_cfg
        return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))

    def head_module(self) -> nn.Module:
        return self.head
