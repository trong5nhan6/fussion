"""Tiền xử lý và augmentation ảnh.

data.resize: squash — resize thẳng về vuông (mặc định trong configs/default.yaml; dùng 100% điểm ảnh, kéo dọc nhẹ)
             pad    — resize giữ tỉ lệ 4:3 rồi pad thành vuông (ảnh thật chiếm 75% khung)
data.aug:    strong — RandomResizedCrop, xoay, lật, màu, blur/nhiễu, affine, xoá vùng ngẫu nhiên (mặc định)
             basic  — lật, affine ±30°, ColorJitter nhẹ
"""
import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizePad(nn.Module):
    """Resize cạnh dài về `size` (giữ tỉ lệ) rồi pad đều hai bên thành ảnh vuông."""

    def __init__(self, size: int, fill: int = 0):
        super().__init__()
        self.size = size
        self.fill = fill

    def forward(self, img):
        h, w = F.get_size(img)
        scale = self.size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        img = F.resize(img, [nh, nw], antialias=True)
        pad_h, pad_w = self.size - nh, self.size - nw
        return F.pad(img, [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)


def _resize(img_size: int, resize: str):
    if resize == "pad":
        return ResizePad(img_size)
    if resize == "squash":
        return v2.Resize((img_size, img_size), antialias=True)
    raise ValueError(f"data.resize phải là pad | squash, nhận {resize!r}")


def _strong_aug(img_size: int) -> tuple[list, list]:
    """(phép biến đổi trên ảnh uint8, phép biến đổi trên tensor float trước khi normalize)."""
    on_image = [
        v2.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.333), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomApply([v2.RandomRotation(30)], p=0.5),
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        v2.RandomApply([v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1))], p=0.4),
    ]
    on_tensor = []
    if hasattr(v2, "GaussianNoise"):  # torchvision >= 0.20
        on_tensor.append(v2.RandomApply([v2.GaussianNoise(sigma=0.04)], p=0.15))
    on_tensor.append(v2.RandomErasing(p=0.3, scale=(0.004, 0.06), value=0))  # xoá 1 vùng nhỏ (CoarseDropout)
    return on_image, on_tensor


def build_transforms(img_size: int, train: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                     resize: str = "pad", aug: str = "basic"):
    ops = [v2.ToImage(), _resize(img_size, resize)]
    on_tensor = []
    if train and aug == "basic":
        ops += [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        ]
    elif train and aug == "strong":
        on_image, on_tensor = _strong_aug(img_size)
        ops += on_image
    elif train:
        raise ValueError(f"data.aug phải là basic | strong, nhận {aug!r}")
    ops += [v2.ToDtype(torch.float32, scale=True), *on_tensor, v2.Normalize(mean, std)]
    return v2.Compose(ops)


def transforms_from_cfg(cfg: dict, train: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    d = cfg["data"]
    return build_transforms(d["img_size"], train, mean, std, resize=d.get("resize", "pad"), aug=d.get("aug", "basic"))
