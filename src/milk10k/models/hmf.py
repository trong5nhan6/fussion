"""HMF — Hybrid Metadata Fusion: metadata đi vào mô hình ở 3 độ sâu, trên nền MoE hướng D (bài Q1).

Động lực (stage 8, ViT-B/224): nối metadata thẳng vào head (R2) và dùng metadata làm query cho attention pooling
trên patch (MGSP, R4) mạnh ở các nhóm lớp khác nhau (R2: MEL/BCC; R4: INF/NV/DF), dự đoán khác nhau ~12% lesion.
HMF gom cả ba cách vào MỘT mô hình (một backbone, một lần forward mỗi ảnh):

  ảnh v ─ backbone (dùng chung) ─┬─ f_v  [B, D]   (đặc trưng gộp, như MoE-D)
                                 └─ P_v  [B, N, D] (lưới patch)
  (1) SỚM   : [f_clin ‖ f_derm ‖ meta] -> MLP                                   -> logits_E
  (2) GIỮA  : meta -> k_meta query (+ k_static query học được) -> cross-attn lên P_v (chung trọng số cho 2 ảnh)
              -> o_v [B, D] -> Linear+LN -> token q_v                           (MGSP)
  (3) TOKEN : {f_clin, f_derm, demo, monet, q_clin, q_derm} -> gate theo nguồn -> 3 head long-tail -> p_T
  đầu ra : log p = ½ (log softmax(logits_E) + log p_T)     (trung bình hình học của 2 nhánh, cố định — không gate)
  loss   : CE(đầu ra) + w·[CE(logits_E) + loss long-tail(p_T)] + head phụ từng token + load-balancing
           (w = hmf.branch_weight, mặc định 0.25: mỗi nhánh tự học được, không nhánh nào lấn át)

Mọi tham số MoE (d_model, modality dropout, lt_taus, ...) giữ như MoE-D; khối `hmf:` trong config cho phần mới.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .. import NUM_CLASSES
from .image_baseline import check_views
from .mgsp import QueryAttnPool, encode_tokens
from .moe import MultimodalMoE


class HybridMetaFusion(MultimodalMoE):
    uses_metadata = True

    def __init__(self, backbone, views, meta_dim, *, arch="D", k_meta=2, k_static=2, attn_heads=8, attn_dropout=0.1,
                 ffn_mult=2, meta_hidden=256, early_hidden=512, early_dropout=0.4, branch_weight=0.25,
                 num_classes=NUM_CLASSES, **moe_kw):
        views = check_views(views)
        if meta_dim <= 0:
            raise ValueError("HMF cần vector metadata (meta_dim > 0)")
        if moe_kw.get("use_metadata") is False:
            raise ValueError("HMF luôn dùng metadata (moe.use_metadata phải là true)")
        if k_meta + k_static <= 0:
            raise ValueError("hmf.k_meta + hmf.k_static phải > 0")
        super().__init__(arch, backbone, views, extra_modalities=[f"q_{v}" for v in views],
                         num_classes=num_classes, **moe_kw)
        D, d = self.tokens.encoder.num_features, self.tokens.d
        if D % attn_heads:
            raise ValueError(f"num_features {D} không chia hết cho hmf.attn_heads {attn_heads}")
        self.D, self.k_meta, self.k_static = D, int(k_meta), int(k_static)
        self.meta_query = (nn.Sequential(nn.Linear(meta_dim, meta_hidden), nn.GELU(), nn.Dropout(attn_dropout),
                                         nn.Linear(meta_hidden, self.k_meta * D)) if self.k_meta else None)
        self.static_query = nn.Parameter(torch.randn(self.k_static, D) * 0.02) if self.k_static else None
        self.pool = QueryAttnPool(D, attn_heads, ffn_mult, attn_dropout)
        self.q_proj = nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))
        self.early = nn.Sequential(nn.Linear(len(views) * D + meta_dim, early_hidden), nn.GELU(),
                                   nn.Dropout(early_dropout), nn.Linear(early_hidden, num_classes))
        self.branch_weight = float(branch_weight)
        self._logits_E = self._logp_T = None
        self.last_attn: dict = {}

    def queries(self, meta):
        b, parts = meta.shape[0], []
        if self.meta_query is not None:
            parts.append(self.meta_query(meta).view(b, self.k_meta, self.D))
        if self.static_query is not None:
            parts.append(self.static_query.unsqueeze(0).expand(b, -1, -1))
        return torch.cat(parts, dim=1)

    def forward(self, batch):
        views = self.tokens.views
        feats, patches = {}, {}
        for v in views:                                   # backbone chạy 1 lần mỗi ảnh
            feats[v], patches[v] = encode_tokens(self.tokens.encoder, batch["images"][v])
        meta = batch["meta"].to(feats[views[0]].dtype)
        q = self.queries(meta)
        qtok, attn = [], {}
        for v in views:                                   # (2) MGSP: chung query + chung trọng số cho 2 ảnh
            o, attn[v] = self.pool(q, patches[v])
            qtok.append(self.q_proj(o.mean(1)))
        x = self.tokens(batch, feats=feats, extra=qtok)   # (3) token -> gate -> head long-tail
        z = self.fusion(x)
        self._aux_logits = [h(x[:, m]) for m, h in enumerate(self.aux_heads)]
        self._record_explain()
        for v, w in attn.items():                         # entropy attention chuẩn hoá (1 = trải đều)
            w = w.detach().float()
            self.last_explain[f"ent_{v}"] = (-(w * (w + 1e-12).log()).sum(-1) / torch.log(
                torch.tensor(float(w.shape[-1])))).mean(1)
        self.last_attn = {k: w.detach() for k, w in attn.items()}
        if self.longtail:
            self._lt_logits = self.head(z)
            logp_T = torch.stack([l.float().softmax(-1) for l in self._lt_logits]).mean(0).clamp_min(1e-8).log()
        else:
            logp_T = self.head(z).float().log_softmax(-1)
        self._logp_T = logp_T
        self._logits_E = self.early(torch.cat([feats[v] for v in views] + [meta], dim=1))   # (1) sớm
        return 0.5 * (self._logits_E.float().log_softmax(-1) + logp_T)

    def compute_loss(self, logits, y, criterion):
        final = F.cross_entropy(logits.float(), y)
        branches = F.cross_entropy(self._logits_E.float(), y) + self.main_loss(self._logp_T, y, criterion)
        return final + self.branch_weight * branches + self.side_loss(y, criterion)
