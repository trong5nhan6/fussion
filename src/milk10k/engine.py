"""Vòng train / validate / predict dùng chung cho mọi mô hình ảnh.

Một fold gồm:
  Giai đoạn 1 — train toàn bộ mô hình (loss/sampler theo config), chọn best.pt theo `train.monitor`.
  Giai đoạn 2 (tuỳ chọn, `crt.enabled`) — classifier re-training (Kang et al., 2020): đóng băng phần
    trích đặc trưng, (khởi tạo lại và) train riêng head với sampler cân bằng -> best_crt.pt.
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from . import CLASSES
from .data import MetadataEncoder, MilkDataset, build_transforms, image_dir
from .losses import build_loss, class_counts, class_weights
from .metrics import compute_metrics, leaderboard_table, summary_line
from .models import build_model
from .models.moe import LONGTAIL_ARCHS
from .profiling import device_report, model_report, peak_memory_gb

DEBUG_BATCHES = 3


def to_device(batch: dict, device) -> dict:
    out = dict(batch)
    out["images"] = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
    for k in ("meta", "label"):
        if k in batch:
            out[k] = batch[k].to(device, non_blocking=True)
    return out


def make_sampler(labels: np.ndarray, q) -> WeightedRandomSampler | None:
    """Xác suất lấy mẫu lớp c tỉ lệ với n_c^(1-q): q=0 tự nhiên, q=0.5 căn bậc hai, q=1 cân bằng hoàn toàn.
    q=None -> không dùng sampler (shuffle thường)."""
    if q is None:
        return None
    counts = np.maximum(class_counts(labels), 1)
    w = counts[labels] ** (-float(q))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels))


def make_loader(ds, cfg, train: bool, sampler=None):
    return DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=train and sampler is None, sampler=sampler,
        num_workers=0 if cfg.get("debug") else cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(), drop_last=train, persistent_workers=False,
    )


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def predict(model, loader, device, tta=False, amp=False, max_batches=None, explain=None):
    """Trả về (lesion_ids, probs[N, C], labels hoặc None).

    explain: dict rỗng -> được điền {tên: mảng [N]} từ model.last_explain (gate/expert của ảnh gốc, không lật).
    """
    model.eval()
    ids, probs, labels = [], [], []
    parts = {}
    for i, batch in enumerate(tqdm(loader, leave=False, desc="predict")):
        if max_batches and i >= max_batches:
            break
        b = to_device(batch, device)
        variants = [b]
        if tta:
            for dims in ([3], [2]):  # lật ngang, lật dọc
                variants.append({**b, "images": {k: v.flip(dims) for k, v in b["images"].items()}})
        with torch.autocast(device_type=device.type, enabled=amp):
            outs = []
            for j, v in enumerate(variants):
                outs.append(model(v).float().softmax(dim=1))
                if j == 0 and explain is not None:
                    for k, t in getattr(model, "last_explain", {}).items():
                        parts.setdefault(k, []).append(t.float().cpu().numpy())
            p = torch.stack(outs).mean(0)
        probs.append(p.cpu().numpy())
        ids.extend(batch["lesion_id"])
        if "label" in batch:
            labels.append(batch["label"].numpy())
    if explain is not None:
        explain.update({k: np.concatenate(v) for k, v in parts.items()})
    return ids, np.concatenate(probs), (np.concatenate(labels) if labels else None)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy (không trọng số) từ xác suất — loss val so sánh được giữa mọi cấu hình loss."""
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def _run_stage(*, model, tr_dl, va_dl, criterion, optimizer, epochs, warmup_epochs, accum, patience,
               monitor, mode, ckpt_path, ckpt_extra, set_train_mode, device, use_amp, max_batches, logger, prefix):
    """Vòng train + validate mỗi epoch, lưu checkpoint tốt nhất theo `monitor`. Trả về lịch sử."""
    steps_per_epoch = min(len(tr_dl), max_batches or len(tr_dl))
    opt_steps = math.ceil(steps_per_epoch / accum)
    scheduler = warmup_cosine(optimizer, warmup_epochs * opt_steps, epochs * opt_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    params = [p for g in optimizer.param_groups for p in g["params"]]

    best, best_epoch, history = -np.inf, -1, []
    for epoch in range(epochs):
        t0 = time.time()
        set_train_mode()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, correct, seen = 0.0, 0, 0
        pbar = tqdm(tr_dl, total=steps_per_epoch, leave=False, desc=f"{prefix} ep{epoch}")
        for i, batch in enumerate(pbar):
            if i >= steps_per_epoch:
                break
            b = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(b)
                loss = (model.compute_loss(logits, b["label"], criterion) if hasattr(model, "compute_loss")
                        else criterion(logits, b["label"]))
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            n = b["label"].numel()
            loss_sum += loss.item() * n
            correct += (logits.argmax(1) == b["label"]).sum().item()
            seen += n
            pbar.set_postfix(loss=f"{loss_sum / seen:.4f}", acc=f"{correct / seen:.3f}")
        train_time = time.time() - t0

        _, va_probs, va_labels = predict(model, va_dl, device, amp=use_amp, max_batches=max_batches)
        m = compute_metrics(va_labels, va_probs, mode)
        row = {"epoch": epoch, "lr": optimizer.param_groups[-1]["lr"],
               "train_loss": loss_sum / seen, "train_acc": correct / seen,
               "val_loss": nll(va_probs, va_labels), "val_acc": m["top1_acc"],
               **{k: m[k] for k in ("auc", "ap", "accuracy", "sensitivity", "specificity", "dice", "balanced_acc")},
               "time_s": time.time() - t0, "train_time_s": train_time}
        mem = peak_memory_gb(device)
        if mem is not None:
            row["gpu_mem_gb"] = mem
        history.append(row)

        improved = m[monitor] > best
        logger.info(f"[{prefix}] ep {epoch:02d} | lr {row['lr']:.2e} | "
                    f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
                    f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} | {summary_line(m)} | "
                    f"{row['time_s']:.0f}s" + (f" {mem:.2f}GB" if mem is not None else "")
                    + (" *" if improved else ""))
        if improved:
            best, best_epoch = m[monitor], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": best, **ckpt_extra}, ckpt_path)
        elif epoch - best_epoch >= patience:
            logger.info(f"[{prefix}] early stop tại epoch {epoch}")
            break
    logger.info(f"[{prefix}] best epoch {best_epoch} | {monitor}={best:.4f}")
    return history


def _oof_frame(ids, probs, labels, fold, explain=None) -> pd.DataFrame:
    oof = pd.DataFrame(probs, columns=CLASSES)
    oof.insert(0, "lesion_id", ids)
    oof.insert(1, "fold", fold)
    oof.insert(2, "label", labels)
    for k, v in (explain or {}).items():  # vd gate_clin, gate_derm (A/D) hoặc expert_cls, expert_derm (B/C)
        oof[k] = v
    return oof


def _final_oof(model, ckpt_path, va_dl, cfg, device, use_amp, max_batches, fold):
    """Nạp checkpoint tốt nhất và dự đoán val (+TTA nếu bật)."""
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    explain = {}
    ids, probs, labels = predict(model, va_dl, device, tta=cfg["predict"]["tta"], amp=use_amp,
                                 max_batches=max_batches, explain=explain)
    return _oof_frame(ids, probs, labels, fold, explain)


def _log_setup(cfg, fold, model, meta_dim, tr_df, va_df, device, logger):
    tc, lc = cfg["train"], cfg["loss"]
    logger.info("=" * 100)
    logger.info(f"FOLD {fold}")
    for line in model_report(model, cfg, meta_dim, device):
        logger.info(line)
    if cfg["model"]["name"] == "moe":
        e = cfg.get("moe", {})
        arch = cfg["model"]["arch"]
        desc = {"A": "gate theo nguồn", "B": "transformer + sparse MoE", "C": "B + 3 head long-tail",
                "D": "A + 3 head long-tail", "E": "nối 4 token + 3 head long-tail"}[arch]
        logger.info(f"MoE          : hướng {arch} ({desc}) | token: {getattr(model.tokens, 'modalities', '?')} | "
                    f"d={e.get('d_model')} | aux_weight={e.get('aux_weight')} | modality_dropout={e.get('modality_dropout')}"
                    + (f" | experts={e.get('n_experts')} top_k={e.get('top_k')} blocks={e.get('n_blocks')} "
                       f"balance_alpha={e.get('balance_alpha')}" if arch in "BC" else "")
                    + (f" | lt_taus={e.get('lt_taus')} (bỏ qua loss overlay)" if arch in LONGTAIL_ARCHS else ""))
    logger.info(device_report(device))
    logger.info(f"Dữ liệu      : train {len(tr_df)} | val {len(va_df)} | img_size {cfg['data']['img_size']}")
    counts = pd.DataFrame({"train": tr_df.label.value_counts(), "val": va_df.label.value_counts()}) \
        .reindex(range(len(CLASSES))).fillna(0).astype(int)
    logger.info("Phân bố lớp  : " + ", ".join(f"{c}={t}/{v}" for c, (t, v) in zip(CLASSES, counts.values))
                + "  (train/val)")
    w = class_weights(tr_df.label.to_numpy(), lc.get("class_weight"), lc.get("cb_beta", 0.999))
    if cfg["model"].get("name") == "moe" and cfg["model"].get("arch") in LONGTAIL_ARCHS:
        logger.info(f"Loss         : 3 head long-tail, logit-adjusted τ={cfg.get('moe', {}).get('lt_taus')} "
                    f"(prior theo sampler_q={tc.get('sampler_q')}); head phụ: CE — cấu hình loss overlay không dùng")
    else:
        logger.info(f"Loss         : {lc['name']} | class_weight={lc.get('class_weight')}"
                    + (f" [{', '.join(f'{x:.2f}' for x in w.tolist())}]" if w is not None else "")
                    + (f" | tau={lc.get('la_tau')}" if lc["name"] == "logit_adjusted" else "")
                    + (f" | gamma={lc.get('focal_gamma')}" if lc["name"] == "focal" else ""))
    logger.info(f"Huấn luyện   : epochs {tc['epochs']} | batch {tc['batch_size']} x accum {tc.get('accum_steps', 1)} "
                f"| AdamW lr {tc['lr']} (backbone x{tc['backbone_lr_mult']}) wd {tc['weight_decay']} "
                f"| warmup {tc['warmup_epochs']} | sampler_q {tc.get('sampler_q')} | amp {tc.get('amp')} "
                f"| monitor {tc['monitor']} | patience {tc['patience']}")
    logger.info(f"Đánh giá     : ngưỡng > 0.5 mỗi lớp, hậu xử lý={cfg['predict'].get('postprocess', 'top1')} "
                f"| val loss = CE không trọng số | acc = top-1 đa lớp")
    lc_, q = cfg["loss"], tc.get("sampler_q")
    if q and (lc_.get("class_weight") not in (None, "none") or lc_["name"] == "logit_adjusted"):
        logger.info("CẢNH BÁO: vừa dùng sampler (q>0) vừa reweight/logit-adjust -> bù mất cân bằng hai lần")
    logger.info("-" * 100)


def train_one_fold(cfg: dict, fold: int, df: pd.DataFrame, run_dir: Path, device, logger) -> pd.DataFrame:
    tc = cfg["train"]
    mode = cfg["predict"].get("postprocess", "top1")
    fold_dir = run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    tr_df, va_df = df[df.fold != fold].reset_index(drop=True), df[df.fold == fold].reset_index(drop=True)
    tr_labels = tr_df.label.to_numpy()

    meta_enc = MetadataEncoder().fit(tr_df)
    model = build_model(cfg, meta_enc.dim).to(device)
    if hasattr(model, "set_class_prior"):  # prior lớp mà mô hình thực sự thấy khi train: P(c) ∝ n_c^(1-q)
        q = tc.get("sampler_q") or 0.0
        eff = np.maximum(class_counts(tr_labels), 1) ** (1.0 - float(q))
        model.set_class_prior(torch.tensor(eff / eff.sum(), dtype=torch.float32, device=device))
    _log_setup(cfg, fold, model, meta_enc.dim, tr_df, va_df, device, logger)
    mean, std = model.normalization()
    img_dir = image_dir(cfg, "train")
    size = cfg["data"]["img_size"]
    views = cfg["model"]["views"]
    tr_ds = MilkDataset(tr_df, img_dir, views, build_transforms(size, True, mean, std), meta_enc.transform(tr_df))
    va_ds = MilkDataset(va_df, img_dir, views, build_transforms(size, False, mean, std), meta_enc.transform(va_df))
    va_dl = make_loader(va_ds, cfg, False)

    max_batches = DEBUG_BATCHES if cfg.get("debug") else None
    use_amp = tc.get("amp", False) and device.type == "cuda"
    common = dict(model=model, va_dl=va_dl, monitor=tc["monitor"], mode=mode, device=device, use_amp=use_amp,
                  max_batches=max_batches, logger=logger)

    # ---- Giai đoạn 1: train toàn bộ
    best_path = fold_dir / "best.pt"
    history = _run_stage(
        **common, tr_dl=make_loader(tr_ds, cfg, True, make_sampler(tr_labels, tc.get("sampler_q"))),
        criterion=build_loss(cfg, tr_labels).to(device),
        optimizer=torch.optim.AdamW(model.param_groups(tc["lr"], tc["backbone_lr_mult"]),
                                    weight_decay=tc["weight_decay"]),
        epochs=tc["epochs"], warmup_epochs=tc["warmup_epochs"], accum=tc.get("accum_steps", 1),
        patience=tc["patience"], ckpt_path=best_path, ckpt_extra={"meta_encoder": meta_enc.state_dict()},
        set_train_mode=model.train, prefix=f"fold{fold}")
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    oof = _final_oof(model, best_path, va_dl, cfg, device, use_amp, max_batches, fold)

    crt = cfg.get("crt", {})
    if not crt.get("enabled"):
        return oof

    # ---- Giai đoạn 2: cRT — đóng băng đặc trưng, train lại head với sampler cân bằng
    oof.to_csv(fold_dir / "oof_stage1.csv", index=False)
    m1 = compute_metrics(oof.label.to_numpy(), oof[CLASSES].to_numpy(), mode)
    logger.info(f"[fold {fold}] giai đoạn 1 (trước cRT, TTA): {summary_line(m1)}")

    head = model.head_module()
    model.requires_grad_(False)
    head.requires_grad_(True)
    if crt.get("reinit_head", True):
        for mod in head.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()
    logger.info(f"cRT          : epochs {crt.get('epochs', 10)} | lr {crt.get('lr', 1e-3)} | "
                f"sampler_q {crt.get('sampler_q', 1.0)} | reinit_head {crt.get('reinit_head', True)} | "
                f"tham số train được {sum(p.numel() for p in head.parameters()):,}")

    def crt_train_mode():
        model.eval()   # giữ BatchNorm/Dropout của backbone ở chế độ suy luận
        head.train()

    crt_path = fold_dir / "best_crt.pt"
    crt_history = _run_stage(
        **common, tr_dl=make_loader(tr_ds, cfg, True, make_sampler(tr_labels, crt.get("sampler_q", 1.0))),
        criterion=nn.CrossEntropyLoss().to(device),
        optimizer=torch.optim.AdamW(head.parameters(), lr=crt.get("lr", 1e-3), weight_decay=tc["weight_decay"]),
        epochs=crt.get("epochs", 10), warmup_epochs=0, accum=tc.get("accum_steps", 1),
        patience=crt.get("patience", crt.get("epochs", 10)), ckpt_path=crt_path,
        ckpt_extra={"meta_encoder": meta_enc.state_dict()}, set_train_mode=crt_train_mode,
        prefix=f"fold{fold}-cRT")
    pd.DataFrame(crt_history).to_csv(fold_dir / "history_crt.csv", index=False)
    return _final_oof(model, crt_path, va_dl, cfg, device, use_amp, max_batches, fold)


def log_leaderboard(logger, title: str, m: dict) -> None:
    logger.info(f"{title}: {summary_line(m)} | balanced_acc={m['balanced_acc']:.4f} top1_acc={m['top1_acc']:.4f}")
    for line in leaderboard_table(m).splitlines():
        logger.info("  " + line)
