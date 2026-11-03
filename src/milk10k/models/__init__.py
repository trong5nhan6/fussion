"""Registry mô hình: thêm mô hình mới (ví dụ multimodal fusion) bằng cách đăng ký vào MODELS.

Mỗi mô hình nhận `batch` (dict: images, meta, label) và cần có:
  - forward(batch) -> logits [B, NUM_CLASSES]
  - param_groups(lr, backbone_lr_mult) -> list nhóm tham số cho optimizer
  - normalization() -> (mean, std) dùng cho transform ảnh
  - head_module() -> nn.Module classifier (dùng cho cRT)
  - thuộc tính `uses_metadata` (mặc định False)
Tuỳ chọn (engine tự gọi nếu có):
  - compute_loss(logits, y, criterion) -> loss (thêm loss phụ, load-balancing, ...)
  - set_class_prior(prior) -> nhận prior lớp hiệu dụng khi train
  - last_explain: dict tensor [B] (gate, expert) -> ghi vào oof.csv
"""
from .concat_mlp import ConcatMLP
from .hmf import HybridMetaFusion
from .image_baseline import ImageBaseline
from .mgsp import MGSP
from .moe import MultimodalMoE


def _build_image(cfg: dict, meta_dim: int):
    m = cfg["model"]
    return ImageBaseline(m["backbone"], m["views"], pretrained=m.get("pretrained", True),
                         dropout=m.get("dropout", 0.3), img_size=cfg["data"]["img_size"],
                         grad_checkpointing=m.get("grad_checkpointing", False),
                         drop_path_rate=m.get("drop_path_rate", 0.0), mlp_hidden=m.get("mlp_hidden", 512),
                         trainable_blocks=m.get("trainable_blocks"))


def _moe_kwargs(cfg: dict) -> dict:
    m = cfg["model"]
    return dict(pretrained=m.get("pretrained", True), img_size=cfg["data"]["img_size"],
                grad_checkpointing=m.get("grad_checkpointing", False), head_dropout=m.get("dropout", 0.3),
                drop_path_rate=m.get("drop_path_rate", 0.0), trainable_blocks=m.get("trainable_blocks"),
                layer_decay=m.get("layer_decay"), **cfg.get("moe", {}))


def _build_moe(cfg: dict, meta_dim: int):
    m = cfg["model"]
    return MultimodalMoE(m["arch"], m["backbone"], m["views"], **_moe_kwargs(cfg))


def _build_hmf(cfg: dict, meta_dim: int):
    m = cfg["model"]
    return HybridMetaFusion(m["backbone"], m["views"], meta_dim, arch=m.get("arch", "D"),
                            **cfg.get("hmf", {}), **_moe_kwargs(cfg))


def _build_concat_mlp(cfg: dict, meta_dim: int):
    m = cfg["model"]
    return ConcatMLP(m["backbone"], m["views"], pretrained=m.get("pretrained", True), img_size=cfg["data"]["img_size"],
                     grad_checkpointing=m.get("grad_checkpointing", False), drop_path_rate=m.get("drop_path_rate", 0.0),
                     hidden=m.get("mlp_hidden", 512), dropout=m.get("dropout", 0.3),
                     concat_meta=m.get("concat_meta", False), meta_dim=meta_dim,
                     trainable_blocks=m.get("trainable_blocks"))


def _build_mgsp(cfg: dict, meta_dim: int):
    m = cfg["model"]
    return MGSP(m["backbone"], m["views"], meta_dim, pretrained=m.get("pretrained", True),
                img_size=cfg["data"]["img_size"], grad_checkpointing=m.get("grad_checkpointing", False),
                drop_path_rate=m.get("drop_path_rate", 0.0), trainable_blocks=m.get("trainable_blocks"),
                hidden=m.get("mlp_hidden", 512), dropout=m.get("dropout", 0.3), **cfg.get("mgsp", {}))


MODELS = {
    "image": _build_image,
    "moe": _build_moe,      # multimodal MoE, hướng A/B/C/D/E (configs/moe/)
    "concat_mlp": _build_concat_mlp,  # nối CLS các ảnh (+ metadata tuỳ chọn) -> MLP 2 lớp
    "mgsp": _build_mgsp,    # metadata-guided spatial pooling (bài TAIMMA, configs: khối mgsp)
    "hmf": _build_hmf,      # hybrid metadata fusion = MoE-D + nhánh nối sớm + token MGSP (bài Q1, khối hmf)
}


def build_model(cfg: dict, meta_dim: int = 0):
    name = cfg["model"]["name"]
    if name not in MODELS:
        raise KeyError(f"Model '{name}' chưa đăng ký. Có: {list(MODELS)}")
    return MODELS[name](cfg, meta_dim)


def uses_metadata(model) -> bool:
    return getattr(model, "uses_metadata", False)
