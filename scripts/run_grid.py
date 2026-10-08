"""Chạy một lưới thí nghiệm (experiments/*.yaml): backbone x overlay x nhánh ảnh.

- Song song: mỗi GPU một worker, mỗi worker chạy lần lượt các run (Kaggle T4 x2 -> 2 run cùng lúc).
- Resume: run đã có metrics.json + submission.csv thì bỏ qua; run dở dang thì xoá và chạy lại.
  Kết quả phiên trước (vd output của notebook Kaggle được gắn làm input) khôi phục bằng --restore.
- Giới hạn thời gian: không bắt đầu run mới nếu không kịp xong trước --time-budget-h.

  python scripts/run_grid.py experiments/stage1_baselines.yaml --env configs/env/kaggle_t4.yaml --dry-run
  python scripts/run_grid.py experiments/stage2_imbalance.yaml --env configs/env/kaggle_t4.yaml \
      --backbones convnext_base --restore /kaggle/input/<notebook-stage1>/outputs --time-budget-h 11
"""
import argparse
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from milk10k.config import load_config
from milk10k.utils import PROJECT_ROOT, load_json, resolve_path

# Tham số in ra khi --dry-run để kiểm tra cấu hình thực tế (sau khi gộp config + overlay + env + --set)
SHOW_KEYS = ["data.img_size", "train.batch_size", "train.accum_steps", "train.epochs", "train.patience", "train.lr",
             "train.backbone_lr_mult", "train.weight_decay", "train.folds", "model.grad_checkpointing",
             "model.dropout", "loss.name", "loss.class_weight", "train.sampler_q", "crt.enabled",
             "predict.tta", "predict.postprocess"]


@dataclass
class Job:
    name: str
    configs: list
    sets: list
    status: str = "pending"   # pending | done_before | done | failed | not_started
    minutes: float = 0.0
    dice: float | None = None
    gpu: str | None = None
    extra: dict = field(default_factory=dict)


def build_jobs(exp: dict, env: str | None, only_backbones=None, extra_set=(), suffix: str = "") -> list[Job]:
    unknown = set(only_backbones or []) - set(exp["backbones"])
    if unknown:
        raise SystemExit(f"Backbone không có trong {exp['name']}: {sorted(unknown)}. Có: {list(exp['backbones'])}")
    jobs = []
    for key, entry in exp["backbones"].items():
        if only_backbones and key not in only_backbones:
            continue
        entry = {"config": entry} if isinstance(entry, str) else entry
        for ov in exp.get("overlays") or [None]:
            tag = yaml.safe_load(resolve_path(ov).read_text(encoding="utf-8"))["tag"] if ov else "default"
            for views in exp["views"]:
                configs = [entry["config"]] + ([ov] if ov else []) + ([env] if env else [])
                # thứ tự ưu tiên tăng dần: chung của lưới < riêng backbone < view < CLI
                sets = [*(exp.get("set") or []), *(entry.get("set") or []),
                        f"model.views=[{','.join(views)}]", *extra_set]
                name = f"{key}__{tag}__{'+'.join(views)}" + (f"__{suffix}" if suffix else "")
                jobs.append(Job(name, configs, sets))
    return jobs


def effective_table(jobs: list[Job]) -> str:
    """Bảng cấu hình thực tế: mỗi backbone một cột (lấy run đầu tiên của backbone đó)."""
    import pandas as pd

    cols = {}
    for j in jobs:
        key = j.name.split("__")[0]
        if key in cols:
            continue
        cfg = load_config([resolve_path(c) for c in j.configs], j.sets)
        row = {}
        for k in SHOW_KEYS:
            node = cfg
            for part in k.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            row[k] = node
        row["model.backbone"] = cfg["model"]["backbone"]
        cols[key] = row
    return pd.DataFrame(cols).to_string()


def is_done(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists() and (run_dir / "submission.csv").exists()


def restore(sources, out_dir: Path) -> int:
    """Chép các run đã hoàn tất từ phiên trước vào out_dir (bỏ qua checkpoint .pt)."""
    n = 0
    for src in sources:
        src = Path(src)
        if not src.exists():
            print(f"[restore] KHÔNG thấy {src} — bỏ qua")
            continue
        for run in sorted(p for p in src.iterdir() if p.is_dir() and is_done(p)):
            dest = out_dir / run.name
            if is_done(dest):
                continue
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(run, dest, ignore=shutil.ignore_patterns("*.pt"))
            n += 1
    return n


def detect_gpus(arg: str) -> list:
    if arg == "cpu":
        return [None]
    if arg != "auto":
        return [g.strip() for g in arg.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.CalledProcessError):
        n = 0
    return [str(i) for i in range(n)] or [None]


class Grid:
    def __init__(self, jobs, out_dir: Path, deadline: float, est_run_h: float, poll_s: int):
        self.q = queue.Queue()
        for j in jobs:
            self.q.put(j)
        self.out_dir, self.deadline, self.est_run_h, self.poll_s = out_dir, deadline, est_run_h, poll_s
        self.lock = threading.Lock()
        self.durations = []
        self.log_dir = out_dir / "_grid_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def say(self, msg: str):
        with self.lock:
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def estimate_s(self) -> float:
        """Thời gian dự kiến của 1 run = trung vị các run đã xong trong phiên (chưa có thì dùng --est-run-h)."""
        with self.lock:
            return statistics.median(self.durations) if self.durations else self.est_run_h * 3600

    def worker(self, gpu):
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                return
            left = self.deadline - time.time()
            if left < self.estimate_s():
                job.status = "not_started"
                self.say(f"[GPU {gpu}] bỏ qua {job.name}: còn {left / 60:.0f} phút, không kịp xong")
                continue
            self.run(job, gpu)

    def run(self, job: Job, gpu):
        run_dir = self.out_dir / job.name
        if run_dir.exists():
            shutil.rmtree(run_dir)  # run dở dang từ phiên trước
        cmd = [sys.executable, "scripts/train.py", "--config", *job.configs, "--run-name", job.name,
               "--set", *job.sets]
        env = {**os.environ, "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        job.gpu = gpu
        self.say(f"[GPU {gpu}] BẮT ĐẦU {job.name}")
        t0 = time.time()
        log_path = self.log_dir / f"{job.name}.out"
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            offset = 0
            while proc.poll() is None:
                time.sleep(self.poll_s)
                offset = self.echo_epochs(run_dir / "train.log", offset, gpu, job.name)
            self.echo_epochs(run_dir / "train.log", offset, gpu, job.name)
        job.minutes = (time.time() - t0) / 60
        if proc.returncode == 0 and is_done(run_dir):
            job.status = "done"
            job.dice = load_json(run_dir / "metrics.json")["overall"]["dice"]
            with self.lock:
                self.durations.append(time.time() - t0)
            self.say(f"[GPU {gpu}] XONG {job.name} | Dice={job.dice:.4f} | {job.minutes:.0f} phút")
        else:
            job.status = "failed"
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            self.say(f"[GPU {gpu}] LỖI {job.name} (exit {proc.returncode}) — log: {log_path}\n    "
                     + "\n    ".join(tail))

    def echo_epochs(self, log_file: Path, offset: int, gpu, name: str) -> int:
        """In các dòng epoch mới trong train.log để theo dõi tiến độ trên notebook."""
        if not log_file.exists():
            return offset
        with open(log_file, encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            new = f.read()
            offset = f.tell()
        for line in new.splitlines():
            if "] ep " in line or "early stop" in line:
                self.say(f"[GPU {gpu}] {name} {line.split(' | ', 1)[-1][:170]}")
        return offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--env", default=None, help="Profile môi trường, vd configs/env/kaggle_t4.yaml")
    ap.add_argument("--backbones", default=None, help="Chỉ chạy các backbone này, phân cách bằng dấu phẩy")
    ap.add_argument("--gpus", default="auto", help="auto | cpu | 0,1")
    ap.add_argument("--restore", nargs="*", default=[], help="Thư mục outputs của phiên trước")
    ap.add_argument("--time-budget-h", type=float, default=11.0, help="Kaggle giới hạn 12 giờ/phiên")
    ap.add_argument("--est-run-h", type=float, default=1.5, help="Ước lượng thời gian 1 run khi chưa có số liệu")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--set", nargs="*", default=[], help="Override thêm cho mọi run")
    ap.add_argument("--suffix", default="", help="Hậu tố tên run, vd ep30 -> resnet152__ce_sqrtinv__clin__ep30")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    start = time.time()

    exp = yaml.safe_load(resolve_path(args.experiment).read_text(encoding="utf-8"))
    only = [b.strip() for b in args.backbones.split(",")] if args.backbones else None
    jobs = build_jobs(exp, args.env, only, args.set, args.suffix)
    out_dir = resolve_path("outputs")
    out_dir.mkdir(exist_ok=True)
    if args.restore:
        print(f"Khôi phục {restore(args.restore, out_dir)} run từ phiên trước")

    for j in jobs:
        if is_done(out_dir / j.name):
            j.status = "done_before"
    pending = [j for j in jobs if j.status == "pending"]
    gpus = detect_gpus(args.gpus)
    print(f"Lưới {exp['name']}: {len(jobs)} run | đã xong {len(jobs) - len(pending)} | cần chạy {len(pending)} "
          f"| GPU {gpus} | ngân sách {args.time_budget_h} giờ")
    for j in jobs:
        print(f"  [{'x' if j.status == 'done_before' else ' '}] {j.name}")
    if args.set:
        print("Override chung: " + " ".join(args.set))
    if args.dry_run:
        print()
        print("Cấu hình thực tế theo backbone (loss/sampler/cRT thay đổi theo overlay ở stage 2):")
        print(effective_table(jobs))
    if args.dry_run or not pending:
        return

    grid = Grid(pending, out_dir, start + args.time_budget_h * 3600, args.est_run_h, args.poll_s)
    threads = [threading.Thread(target=grid.worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n" + "=" * 80)
    labels = {"done_before": "đã xong trước", "done": "xong", "failed": "LỖI", "not_started": "chưa chạy (hết giờ)"}
    for j in jobs:
        extra = f" | Dice={j.dice:.4f} | {j.minutes:.0f} phút | GPU {j.gpu}" if j.status == "done" else ""
        print(f"  {labels.get(j.status, j.status):<20} {j.name}{extra}")
    remaining = [j for j in jobs if j.status in ("failed", "not_started")]
    print(f"Tổng thời gian {(time.time() - start) / 3600:.2f} giờ. "
          + (f"Còn {len(remaining)} run — chạy lại notebook với --restore để tiếp tục." if remaining
             else "Đã hoàn tất toàn bộ lưới."))


if __name__ == "__main__":
    main()
