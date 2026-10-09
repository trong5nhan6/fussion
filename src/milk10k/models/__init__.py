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
from .image_baseline import ImageBaseline
from .moe import MultimodalMoE


def _build_image(cfg: dict, meta_dim: int):
    m = cfg["model"]
    return ImageBaseline(m["backbone"], m["views"], pretrained=m.get("pretrained", True),
                         dropout=m.get("dropout", 0.3), img_size=cfg["data"]["img_size"],
                         grad_checkpointing=m.get("grad_checkpointing", False),
                         drop_path_rate=m.get("drop_path_rate", 0.0))


def _build_moe(cfg: dict, meta_dim: int):
    m, e = cfg["model"], cfg.get("moe", {})
    return MultimodalMoE(m["arch"], m["backbone"], m["views"], pretrained=m.get("pretrained", True),
                         img_size=cfg["data"]["img_size"], grad_checkpointing=m.get("grad_checkpointing", False),
                         head_dropout=m.get("dropout", 0.3), drop_path_rate=m.get("drop_path_rate", 0.0), **e)


MODELS = {
    "image": _build_image,
    "moe": _build_moe,      # multimodal MoE, hướng A/B/C/D (configs/moe/)
}


def build_model(cfg: dict, meta_dim: int = 0):
    name = cfg["model"]["name"]
    if name not in MODELS:
        raise KeyError(f"Model '{name}' chưa đăng ký. Có: {list(MODELS)}")
    return MODELS[name](cfg, meta_dim)


def uses_metadata(model) -> bool:
    return getattr(model, "uses_metadata", False)
