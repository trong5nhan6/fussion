"""Multimodal Mixture-of-Experts: ảnh clinical / dermoscopy + metadata (demographics, MONET).

Mỗi lesion -> tối đa 4 token d chiều: [clin], [derm] (theo model.views), demo (20 chiều), monet (14 chiều).
Bốn hướng (model.arch):
  A  gate theo nguồn: expert riêng mỗi nguồn -> h_m; gate(mọi token) -> g; z = Σ g_m h_m      -> 1 head
  B  transformer + sparse MoE: [CLS, token...] -> (attention -> MoE-FFN top-k) x n_blocks; z = CLS -> 1 head
  C  như B, 3 head long-tail (logit-adjusted với τ khác nhau), suy luận = trung bình xác suất 3 head
  D  như A, 3 head long-tail
Chung: head phụ cho từng token (deep supervision), modality dropout, load-balancing loss (B, C).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .. import NUM_CLASSES
from .image_baseline import check_views, create_encoder

DEMO_DIM = 20   # tuổi(2) + giới(3) + skin tone(7) + vị trí(8), xem data/metadata.py
MONET_DIM = 14  # 7 khái niệm x 2 ảnh
ARCHS = {"A": ("gated", False), "B": ("transformer", False), "C": ("transformer", True), "D": ("gated", True)}


def mlp(d_in, d_out, dropout=0.1):
    return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_out, d_out))


class TokenEncoder(nn.Module):
    """Ảnh (backbone timm dùng chung cho các view) + 2 MLP metadata -> tokens [B, M, d]."""

    def __init__(self, backbone, views, pretrained, img_size, d, grad_checkpointing, modality_dropout, dropout):
        super().__init__()
        self.views = check_views(views)
        self.modalities = self.views + ["demo", "monet"]
        self.encoder = create_encoder(backbone, pretrained, img_size, grad_checkpointing)
        self.img_proj = nn.Sequential(nn.Linear(self.encoder.num_features, d), nn.LayerNorm(d))
        self.demo = nn.Sequential(mlp(DEMO_DIM, d, dropout), nn.LayerNorm(d))
        self.monet = nn.Sequential(mlp(MONET_DIM, d, dropout), nn.LayerNorm(d))
        M = len(self.modalities)
        self.type_emb = nn.Parameter(torch.zeros(M, d))   # "nhãn nguồn" của từng token
        self.missing = nn.Parameter(torch.zeros(M, d))    # token thay thế khi nguồn bị che
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        nn.init.trunc_normal_(self.missing, std=0.02)
        self.p_drop = modality_dropout

    def forward(self, batch):
        meta = batch["meta"]
        toks = [self.img_proj(self.encoder(batch["images"][v])) for v in self.views]
        toks += [self.demo(meta[:, :DEMO_DIM]), self.monet(meta[:, DEMO_DIM:DEMO_DIM + MONET_DIM])]
        x = torch.stack(toks, dim=1)                      # [B, M, d]
        if self.training and self.p_drop > 0:
            B, M, _ = x.shape
            drop = torch.rand(B, M, device=x.device) < self.p_drop
            all_dropped = drop.all(dim=1)                 # luôn giữ lại ít nhất 1 nguồn
            if all_dropped.any():
                keep = torch.randint(0, M, (int(all_dropped.sum()),), device=x.device)
                drop[all_dropped.nonzero(as_tuple=True)[0], keep] = False
            x = torch.where(drop[..., None], self.missing.to(x.dtype).expand(B, -1, -1), x)
        return x + self.type_emb.to(x.dtype)


class GatedFusion(nn.Module):
    """Hướng A/D: mỗi nguồn một expert, gate theo mẫu trộn đặc trưng các nguồn."""

    def __init__(self, n_tokens, d, dropout):
        super().__init__()
        self.experts = nn.ModuleList([mlp(d, d, dropout) for _ in range(n_tokens)])
        self.gate = nn.Sequential(nn.Linear(n_tokens * d, d), nn.GELU(), nn.Linear(d, n_tokens))
        self.norm = nn.LayerNorm(d)
        self.aux_loss = None
        self.last_gate = None

    def forward(self, x):                                 # x: [B, M, d]
        h = torch.stack([e(x[:, m]) for m, e in enumerate(self.experts)], dim=1)
        g = self.gate(x.flatten(1)).float().softmax(dim=-1)   # [B, M]
        self.last_gate = g.detach()
        return self.norm((g.to(h.dtype)[..., None] * h).sum(dim=1))


class MoEFFN(nn.Module):
    """FFN thay bằng E expert; router chọn top-k expert cho TỪNG token (Switch/V-MoE)."""

    def __init__(self, d, n_experts, top_k, dropout):
        super().__init__()
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                                    nn.Linear(2 * d, d)) for _ in range(n_experts)])
        self.E, self.k = n_experts, top_k
        self.aux_loss = None
        self.last_top1 = None

    def forward(self, x):                                 # x: [B, T, d]
        probs = self.router(x).float().softmax(dim=-1)    # [B, T, E]
        top_p, top_i = probs.topk(self.k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            w = (top_p * (top_i == e)).sum(dim=-1)        # [B, T]: trọng số của expert e (0 nếu không được chọn)
            if w.any():
                out = out + w.to(x.dtype)[..., None] * expert(x)
        # Load-balancing (Fedus et al., 2021): E * Σ_e f_e * P_e, f_e = tỉ lệ lượt gán, P_e = xác suất router TB
        f = torch.stack([(top_i == e).float().sum(-1).mean() for e in range(self.E)]) / self.k
        self.aux_loss = self.E * (f * probs.mean(dim=(0, 1))).sum()
        self.last_top1 = top_i[..., 0].detach()
        return out


class TransformerMoEFusion(nn.Module):
    """Hướng B/C: [CLS, token...] -> (self-attention -> MoE-FFN) x n_blocks; z = CLS."""

    def __init__(self, d, n_blocks, n_heads, n_experts, top_k, dropout):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d), "attn": nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(d), "moe": MoEFFN(d, n_experts, top_k, dropout), "drop": nn.Dropout(dropout)}))
        self.norm = nn.LayerNorm(d)
        self.aux_loss = None
        self.last_top1 = None

    def forward(self, x):
        x = torch.cat([self.cls.to(x.dtype).expand(x.shape[0], -1, -1), x], dim=1)
        aux = 0.0
        for blk in self.blocks:
            h = blk["ln1"](x)
            x = x + blk["drop"](blk["attn"](h, h, h, need_weights=False)[0])
            x = x + blk["drop"](blk["moe"](blk["ln2"](x)))
            aux = aux + blk["moe"].aux_loss
        self.aux_loss = aux / len(self.blocks)
        self.last_top1 = self.blocks[-1]["moe"].last_top1    # [B, 1+M]: expert top-1 của từng token ở khối cuối
        return self.norm(x[:, 0])


class LongTailHeads(nn.Module):
    """3 head cùng nhìn z, train bằng logit-adjusted loss với τ khác nhau; suy luận = trung bình xác suất."""

    def __init__(self, d, taus, dropout, num_classes=NUM_CLASSES):
        super().__init__()
        self.taus = [float(t) for t in taus]
        self.heads = nn.ModuleList([nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes)) for _ in self.taus])
        self.register_buffer("log_prior", torch.full((num_classes,), -math.log(num_classes)))

    def forward(self, z):
        return [h(z) for h in self.heads]


class MultimodalMoE(nn.Module):
    uses_metadata = True

    def __init__(self, arch, backbone, views, pretrained=True, img_size=224, grad_checkpointing=False,
                 d_model=256, n_experts=4, top_k=2, n_blocks=2, n_heads=4, balance_alpha=0.01, aux_weight=0.25,
                 modality_dropout=0.15, lt_taus=(0.0, 0.5, 1.0), dropout=0.1, head_dropout=0.3,
                 fusion_lr_mult=1.0, num_classes=NUM_CLASSES):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"model.arch phải là một trong {list(ARCHS)}, nhận {arch!r}")
        self.arch = arch
        fusion, self.longtail = ARCHS[arch]
        self.tokens = TokenEncoder(backbone, views, pretrained, img_size, d_model, grad_checkpointing,
                                   modality_dropout, dropout)
        M = len(self.tokens.modalities)
        self.fusion = (GatedFusion(M, d_model, dropout) if fusion == "gated" else
                       TransformerMoEFusion(d_model, n_blocks, n_heads, n_experts, top_k, dropout))
        self.head = (LongTailHeads(d_model, lt_taus, head_dropout, num_classes) if self.longtail else
                     nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d_model, num_classes)))
        self.aux_heads = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in range(M)])
        self.balance_alpha, self.aux_weight = balance_alpha, aux_weight
        self.fusion_lr_mult = fusion_lr_mult
        self._aux_logits, self._lt_logits = None, None
        self.last_explain = {}

    # ------------------------------------------------------------------ forward / loss
    def forward(self, batch):
        x = self.tokens(batch)
        z = self.fusion(x)
        self._aux_logits = [h(x[:, m]) for m, h in enumerate(self.aux_heads)]
        self._record_explain()
        if self.longtail:
            self._lt_logits = self.head(z)
            probs = torch.stack([l.float().softmax(dim=-1) for l in self._lt_logits]).mean(dim=0)
            return probs.clamp_min(1e-8).log()            # softmax(log p) = p -> engine dùng như logits
        return self.head(z)

    def set_class_prior(self, prior: torch.Tensor):
        """Prior lớp hiệu dụng khi train (đã tính sampler) cho logit-adjusted loss của các head long-tail."""
        if self.longtail:
            self.head.log_prior.copy_(prior.clamp_min(1e-12).log().to(self.head.log_prior))

    def compute_loss(self, logits, y, criterion):
        if self.longtail:   # mỗi head: CE(logit + τ·log π) — τ = 0 là CE thường
            lp = self.head.log_prior
            main = sum(F.cross_entropy(l.float() + t * lp, y) for l, t in zip(self._lt_logits, self.head.taus))
            main = main / len(self._lt_logits)
            aux = sum(F.cross_entropy(a.float(), y) for a in self._aux_logits) / len(self._aux_logits)
        else:
            main = criterion(logits, y)
            aux = sum(criterion(a, y) for a in self._aux_logits) / len(self._aux_logits)
        loss = main + self.aux_weight * aux
        if self.fusion.aux_loss is not None:
            loss = loss + self.balance_alpha * self.fusion.aux_loss
        return loss

    def _record_explain(self):
        """Lưu thông tin giải thích của lần forward gần nhất (engine ghi vào oof.csv)."""
        mods = self.tokens.modalities
        if isinstance(self.fusion, GatedFusion):
            g = self.fusion.last_gate
            self.last_explain = {f"gate_{m}": g[:, i] for i, m in enumerate(mods)}
        else:
            t = self.fusion.last_top1
            self.last_explain = {f"expert_{m}": t[:, i] for i, m in enumerate(["cls"] + mods)}

    # ------------------------------------------------------------------ giao diện chung của project
    def head_module(self):
        return self.head

    def param_groups(self, lr, backbone_lr_mult):
        enc = list(self.tokens.encoder.parameters())
        enc_ids = {id(p) for p in enc}
        rest = [p for p in self.parameters() if id(p) not in enc_ids]
        # phần mới khởi tạo (token MLP, fusion, head): lr * fusion_lr_mult; backbone pretrained: lr * backbone_lr_mult
        return [{"params": enc, "lr": lr * backbone_lr_mult}, {"params": rest, "lr": lr * self.fusion_lr_mult}]

    def normalization(self):
        cfg = self.tokens.encoder.pretrained_cfg
        return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))
