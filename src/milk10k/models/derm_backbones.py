"""Backbone pretrain trên ảnh da liễu, nạp vào kiến trúc ViT-B/16 của timm -> dùng chung mọi mô hình của project
(MoE, HMF, MGSP...) như một backbone timm bình thường (forward_features, forward_head, grad checkpointing, drop path).

  panderm_base    PanDerm-Base (Yan et al., Nature Medicine 2025): ViT-B/16, MAE + CLIP-distillation trên ~2 triệu ảnh
                  da liễu (dermoscopy, clinical, TBP, mô bệnh học). Checkpoint 1 file .pth trên Google Drive của tác giả
                  (tự tải bằng gdown). Kiến trúc kiểu BEiT/CAE: LayerScale (gamma_1/2), qkv có q_bias/v_bias riêng,
                  pos embed sin-cos -> đổi tên khoá sang timm. Chuẩn hoá ảnh: ImageNet mean/std (như script fine-tune gốc).
  dermlip_vit_b16 DermLIP ViT-B/16 (Derm1M, ICCV 2025): CLIP ảnh-văn bản trên ~400k cặp ảnh da liễu. Lấy nhánh ảnh từ
                  checkpoint OpenCLIP trên HuggingFace (repo gated: cần chấp nhận điều khoản + HF_TOKEN). Chuẩn hoá: CLIP.

Vị trí lưu trọng số: biến môi trường MILK10K_WEIGHTS (mặc định <repo>/weights). Giấy phép: cả hai chỉ cho nghiên cứu
phi thương mại; DermLIP cấm phân phối lại trọng số -> không đưa checkpoint đã fine-tune lên nơi công khai.
pretrained=False: chỉ dựng kiến trúc (smoke test, không tải gì).
"""
import json
import os
from pathlib import Path

import timm
import torch

PANDERM_GDRIVE_ID = "17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB"     # panderm_bb_data6_checkpoint-499.pth (README PanDerm)
DERMLIP_REPO = "redlessone/DermLIP_ViT-B-16"
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

DERM_BACKBONES = ("panderm_base", "dermlip_vit_b16")


def weights_dir() -> Path:
    from ..utils import PROJECT_ROOT
    d = Path(os.environ.get("MILK10K_WEIGHTS", PROJECT_ROOT / "weights"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _set_norm(enc, mean_std) -> None:
    cfg = dict(getattr(enc, "pretrained_cfg", {}) or {})
    cfg.update(mean=tuple(mean_std[0]), std=tuple(mean_std[1]))
    enc.pretrained_cfg = cfg


def _load_checked(enc, sd: dict, name: str) -> None:
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}       # num_classes=0: bỏ head/proj CLIP
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{name}: nạp trọng số không khớp — thiếu {missing[:8]}, thừa {unexpected[:8]}")


# ---------------------------------------------------------------------------------------------- PanDerm
def convert_panderm(sd: dict) -> dict:
    """Khoá PanDerm (BEiT/CAE) -> timm VisionTransformer(init_values>0)."""
    sd = sd.get("model", sd)
    out = {}
    for k, v in sd.items():
        if k.startswith(("decoder.", "teacher.", "head.", "fc_norm.")):
            continue
        k = k[len("encoder."):] if k.startswith("encoder.") else k
        k = k.replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma")
        out[k] = v
    for k in [k for k in out if k.endswith("attn.q_bias")]:           # qkv.bias = [q_bias, 0, v_bias]
        q, v = out.pop(k), out.pop(k.replace("q_bias", "v_bias"))
        out[k.replace("q_bias", "qkv.bias")] = torch.cat([q, torch.zeros_like(v), v])
    return out


def create_panderm(img_size=224, pretrained=True, **kw):
    enc = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, img_size=img_size,
                            init_values=0.1, **kw)
    _set_norm(enc, IMAGENET)
    if not pretrained:
        return enc
    path = weights_dir() / "panderm_bb_data6_checkpoint-499.pth"
    if not path.exists():
        import gdown
        if gdown.download(id=PANDERM_GDRIVE_ID, output=str(path), quiet=False) is None:
            raise RuntimeError("Không tải được PanDerm-Base từ Google Drive (đặt file vào MILK10K_WEIGHTS)")
    sd = convert_panderm(torch.load(path, map_location="cpu", weights_only=True))
    if sd["pos_embed"].shape != enc.pos_embed.shape:                   # ảnh khác 224 -> nội suy pos embed
        from timm.layers import resample_abs_pos_embed
        gs = enc.patch_embed.grid_size
        sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], new_size=gs, num_prefix_tokens=1)
    _load_checked(enc, sd, "PanDerm-Base")
    return enc


# ---------------------------------------------------------------------------------------------- DermLIP
def _dermlip_files():
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN") or None
    try:
        cfg = json.loads(Path(hf_hub_download(DERMLIP_REPO, "open_clip_config.json", token=tok,
                                              cache_dir=weights_dir() / "hf")).read_text())
        ckpt = hf_hub_download(DERMLIP_REPO, "open_clip_model.safetensors", token=tok, cache_dir=weights_dir() / "hf")
    except Exception as e:  # repo gated
        raise RuntimeError(f"Không tải được {DERMLIP_REPO}: vào https://huggingface.co/{DERMLIP_REPO} bấm "
                           "'Request access' (tự duyệt), rồi đặt HF_TOKEN. Lỗi gốc: " + repr(e)) from e
    return cfg, ckpt


def create_dermlip(img_size=224, pretrained=True, **kw):
    quick_gelu, mean_std = False, CLIP
    if pretrained:
        cfg, ckpt = _dermlip_files()
        mc, pc = cfg.get("model_cfg", {}), cfg.get("preprocess_cfg", {})
        quick_gelu = bool(mc.get("quick_gelu", False))
        vc = mc.get("vision_cfg", {})
        if vc and (vc.get("width", 768), vc.get("layers", 12), vc.get("patch_size", 16)) != (768, 12, 16):
            raise RuntimeError(f"DermLIP vision_cfg không phải ViT-B/16: {vc}")
        if pc.get("mean") and pc.get("std"):
            mean_std = (pc["mean"], pc["std"])
    name = "vit_base_patch16_clip_quickgelu_224" if quick_gelu else "vit_base_patch16_clip_224"
    enc = timm.create_model(name, pretrained=False, num_classes=0, img_size=img_size, **kw)
    _set_norm(enc, mean_std)
    if not pretrained:
        return enc
    from safetensors.torch import load_file
    from timm.models.vision_transformer import checkpoint_filter_fn
    sd = checkpoint_filter_fn(load_file(ckpt), enc)                    # OpenCLIP visual.* -> timm (+ nội suy pos)
    _load_checked(enc, sd, "DermLIP")
    return enc


def create_derm_backbone(name: str, img_size: int, pretrained: bool, **kw):
    if name == "panderm_base":
        return create_panderm(img_size, pretrained, **kw)
    if name == "dermlip_vit_b16":
        return create_dermlip(img_size, pretrained, **kw)
    raise KeyError(name)
