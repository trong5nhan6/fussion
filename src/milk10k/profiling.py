"""Thông tin mô hình cho file log: số tham số, GFLOPs, bộ nhớ GPU."""
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def count_flops(model: nn.Module, views, img_size: int, meta_dim: int, device) -> int:
    """FLOPs của 1 lần forward cho 1 lesion (tất cả view). FLOPs = 2 x MACs."""
    was_training = model.training
    model.eval()
    batch = {"images": {v: torch.zeros(1, 3, img_size, img_size, device=device) for v in views},
             "meta": torch.zeros(1, meta_dim, device=device)}
    counter = FlopCounterMode(display=False)
    with counter:
        model(batch)
    model.train(was_training)
    return int(counter.get_total_flops())


def fmt_num(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def model_report(model: nn.Module, cfg: dict, meta_dim: int, device) -> list[str]:
    m, views, size = cfg["model"], cfg["model"]["views"], cfg["data"]["img_size"]
    total, trainable = count_params(model)
    lines = [
        f"Model        : {cfg['name']}" + (f" [{cfg['tag']}]" if cfg.get("tag") else ""),
        f"Kiến trúc    : {m['name']} | backbone={m.get('backbone')} | views={views} | "
        f"pretrained={m.get('pretrained')}",
        f"Tham số      : tổng {fmt_num(total)} ({total:,}) | train được {fmt_num(trainable)}",
    ]
    for name, child in model.named_children():
        lines.append(f"  - {name:<10}: {fmt_num(count_params(child)[0])}")
    try:
        flops = count_flops(model, views, size, meta_dim, device)
        lines.append(f"Tính toán    : {flops / 1e9:.2f} GFLOPs / lesion ({flops / 2e9:.2f} GMACs), "
                     f"đầu vào {len(views)} x 3x{size}x{size}")
    except Exception as e:  # một số op không đếm được -> không chặn việc train
        lines.append(f"Tính toán    : không đếm được FLOPs ({type(e).__name__}: {e})")
    return lines


def device_report(device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"Thiết bị     : {p.name} ({p.total_memory / 2**30:.1f} GB) | torch {torch.__version__}"
    return f"Thiết bị     : CPU | torch {torch.__version__}"


def peak_memory_gb(device) -> float | None:
    if device.type != "cuda":
        return None
    v = torch.cuda.max_memory_allocated(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)
    return v
