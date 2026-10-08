"""Augmentation cơ bản cho baseline. Pipeline nâng cao sẽ thiết kế riêng cho mô hình fusion."""
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


def build_transforms(img_size: int, train: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    ops = [v2.ToImage(), ResizePad(img_size)]
    if train:
        ops += [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        ]
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(ops)
