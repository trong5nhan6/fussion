"""Registry mô hình: thêm mô hình mới (ví dụ multimodal fusion) bằng cách đăng ký vào MODELS.

Mỗi mô hình nhận `batch` (dict: images, meta, label) và cần có:
  - forward(batch) -> logits [B, NUM_CLASSES]
  - param_groups(lr, backbone_lr_mult) -> list nhóm tham số cho optimizer
  - normalization() -> (mean, std) dùng cho transform ảnh
  - head_module() -> nn.Module classifier (dùng cho cRT)
  - thuộc tính `uses_metadata` (mặc định False)
"""
from .image_baseline import ImageBaseline


def _build_image(cfg: dict, meta_dim: int):
    m = cfg["model"]
    return ImageBaseline(m["backbone"], m["views"], pretrained=m.get("pretrained", True),
                         dropout=m.get("dropout", 0.3), img_size=cfg["data"]["img_size"],
                         grad_checkpointing=m.get("grad_checkpointing", False))


MODELS = {
    "image": _build_image,
}


def build_model(cfg: dict, meta_dim: int = 0):
    name = cfg["model"]["name"]
    if name not in MODELS:
        raise KeyError(f"Model '{name}' chưa đăng ký. Có: {list(MODELS)}")
    return MODELS[name](cfg, meta_dim)


def uses_metadata(model) -> bool:
    return getattr(model, "uses_metadata", False)
