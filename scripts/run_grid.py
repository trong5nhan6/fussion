"""Chạy một lưới thí nghiệm (experiments/*.yaml): backbone x overlay x nhánh ảnh.

- Song song: mỗi GPU một worker, mỗi worker chạy lần lượt các run (Kaggle T4 x2 -> 2 run cùng lúc).
- Resume: run đã có metrics.json + submission.csv thì bỏ qua; run dở dang thì xoá và chạy lại.
  Kết quả phiên trước (vd output của notebook Kaggle được gắn làm input) khôi phục bằng --restore.
- Giới hạn thời gian: không bắt đầu run mới nếu không kịp xong trước --time-budget-h.
- Nhiều máy cùng lưới (vd 2 notebook Colab): --shared-dir <thư mục chung, vd Google Drive>. Trước mỗi run, máy kiểm tra
  run đã xong ở thư mục chung chưa, rồi ghi file đánh dấu _claims/<run> (cập nhật mỗi phút khi đang chạy). Máy khác thấy
  đánh dấu còn "sống" thì bỏ qua run đó. --reverse: chạy danh sách theo thứ tự ngược (máy thứ 2) để hai máy ít chạm nhau.

  python scripts/run_grid.py experiments/stage1_baselines.yaml --env configs/env/kaggle_t4.yaml --dry-run
  python scripts/run_grid.py experiments/stage2_imbalance.yaml --env configs/env/kaggle_t4.yaml \
      --backbones convnext_base --restore /kaggle/input/<notebook-stage1>/outputs --time-budget-h 11
"""
import argparse
import os
import queue
import socket
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

from milk10k.config import _load_yaml, load_config
from milk10k.utils import PROJECT_ROOT, load_json, resolve_path

# Tham số in ra khi --dry-run để kiểm tra cấu hình thực tế (sau khi gộp config + overlay + env + --set)
SHOW_KEYS = ["model.name", "model.arch", "model.views", "moe.use_metadata", "moe.lt_taus", "moe.aux_weight",
             "moe.modality_dropout", "moe.d_model", "data.img_size", "data.resize", "data.aug", "model.drop_path_rate", "train.batch_size", "train.accum_steps", "train.epochs", "train.patience", "train.lr",
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


def _yaml(path) -> dict:
    """Đọc 1 file config, giải quyết kế thừa `base:` (overlay ablation kế thừa overlay imbalance)."""
    return _load_yaml(resolve_path(path))


def _is_sampler_overlay(path) -> bool:
    """Overlay chỉ đổi cách lấy mẫu (sampler_q), không đổi loss/cRT -> vẫn dùng được với head long-tail.
    Overlay ghi `lt_compatible: true` (vd r5_nosampler) cũng được giữ."""
    d = _yaml(path)
    if d.get("lt_compatible"):
        return True
    return bool((d.get("train") or {}).get("sampler_q")) and not (d.get("crt") or {}).get("enabled")


def _check(kind: str, chosen, available, exp_name: str):
    unknown = set(chosen or []) - set(available)
    if unknown:
        raise SystemExit(f"{kind} không có trong {exp_name}: {sorted(unknown)}. Có: {list(available)}")


def build_jobs(exp: dict, env: str | None, only_backbones=None, extra_set=(), suffix: str = "",
               only_archs=None, only_overlays=None, only_views=None, lt_base: bool = True) -> list[Job]:
    """Sinh danh sách run: backbone x [hướng MoE] x overlay x nhánh ảnh.

    Tên run: <backbone>[__<hướng>]__<tag>__<views>[__<hậu tố>]. Hướng long-tail (C, D) tự có loss riêng
    -> chỉ giữ overlay sampler, cộng 1 run không overlay (tag "lt"; tắt bằng lt_base=False).
    """
    archs = exp.get("archs") or {None: None}
    overlays = exp.get("overlays") or [None]
    _check("Backbone", only_backbones, exp["backbones"], exp["name"])
    _check("Hướng", only_archs, [a for a in archs if a], exp["name"])
    _check("Overlay", only_overlays, [Path(o).stem for o in overlays if o], exp["name"])
    _check("Nhánh ảnh", only_views, ["+".join(v) for v in exp["views"]], exp["name"])
    if any(archs):
        from milk10k.models.moe import ARCHS  # (fusion, longtail) theo hướng
    jobs = []
    for key, entry in exp["backbones"].items():
        if only_backbones and key not in only_backbones:
            continue
        entry = {"config": entry} if isinstance(entry, str) else entry
        for arch, arch_cfg in archs.items():
            if only_archs and arch not in only_archs:
                continue
            longtail = bool(arch) and ARCHS[_yaml(arch_cfg)["model"]["arch"]][1]
            ovs = [o for o in overlays if o is None or not only_overlays or Path(o).stem in only_overlays]
            if longtail:
                ovs = [None] * lt_base + [o for o in ovs if o and _is_sampler_overlay(o)]
            for ov in ovs:
                tag = _yaml(ov)["tag"] if ov else ("lt" if longtail else "default")
                for views in exp["views"]:
                    vname = "+".join(views)
                    if only_views and vname not in only_views:
                        continue
                    configs = [entry["config"], *([arch_cfg] if arch_cfg else []), *([ov] if ov else []),
                               *([env] if env else [])]
                    # thứ tự ưu tiên tăng dần: chung của lưới < riêng backbone < view < CLI
                    sets = [*(exp.get("set") or []), *(entry.get("set") or []), f"model.views=[{','.join(views)}]",
                            *extra_set]
                    name = "__".join([key, *([arch] if arch else []), tag, vname]) + (f"__{suffix}" if suffix else "")
                    jobs.append(Job(name, configs, sets, extra={"col": key + (f" {arch}" if arch else "")}))
    return jobs


OVERLAY_DIRS = ("configs/imbalance", "configs/ablation")


def _overlay_path(name: str) -> str:
    for d in OVERLAY_DIRS:
        if (PROJECT_ROOT / d / f"{name}.yaml").exists():
            return f"{d}/{name}.yaml"
    raise SystemExit(f"Không tìm thấy overlay '{name}' trong {OVERLAY_DIRS}")


def build_variant_jobs(exp: dict, env: str | None, only_backbones=None, extra_set=(), suffix: str = "",
                       only_variants=None, folds=None) -> list[Job]:
    """Lưới dạng `variants` (ablation): mỗi variant là một cấu hình đặt tên, mỗi fold là một job riêng.

    variant: {arch: D, overlay: r5_nobce, views: [clin, derm], set: [...], folds: [0, ...], config: <baseline>}
    Tên run: <backbone>__abl_<variant>__f<fold>[__<hậu tố>]. `folds` (CLI) ghi đè folds của từng variant.
    Thứ tự job: theo fold trước (mọi variant fold 0, rồi fold 1, ...) để có kết quả fold 0 sớm nhất.
    """
    variants = exp["variants"]
    _check("Backbone", only_backbones, exp["backbones"], exp["name"])
    _check("Variant", only_variants, variants, exp["name"])
    jobs = []
    for key, entry in exp["backbones"].items():
        if only_backbones and key not in only_backbones:
            continue
        entry = {"config": entry} if isinstance(entry, str) else entry
        plan = []
        for vname, v in variants.items():
            if only_variants and vname not in only_variants:
                continue
            for fold in (folds if folds is not None else v.get("folds", [0])):
                plan.append((int(fold), vname, v))
        plan.sort(key=lambda t: t[0])
        for fold, vname, v in plan:
            configs = [v.get("config", entry["config"]), f"configs/moe/{v.get('arch', 'D')}.yaml",
                       *([_overlay_path(v["overlay"])] if v.get("overlay") else []), *([env] if env else [])]
            views = v.get("views", ["clin", "derm"])
            sets = [*(exp.get("set") or []), *(entry.get("set") or []), *(v.get("set") or []),
                    f"model.views=[{','.join(views)}]", f"train.folds=[{fold}]", *extra_set]
            name = f"{key}__abl_{vname}__f{fold}" + (f"__{suffix}" if suffix else "")
            jobs.append(Job(name, configs, sets, extra={"col": f"{key} {vname}", "variant": vname, "fold": fold}))
    return jobs


def effective_table(jobs: list[Job]) -> str:
    """Bảng cấu hình thực tế: mỗi backbone một cột (lấy run đầu tiên của backbone đó)."""
    import pandas as pd

    cols = {}
    for j in jobs:
        key = j.extra.get("col", j.name.split("__")[0])
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
    table = pd.DataFrame(cols)
    if any("variant" in j.extra for j in jobs) and table.shape[1] > 1:
        # Lưới ablation: mỗi variant 1 dòng, chỉ hiện tham số khác nhau; phần chung in một dòng
        same = table.astype(str).nunique(axis=1) == 1
        common = ", ".join(f"{k}={table.loc[k].iloc[0]}" for k in table.index[same])
        return "Chung: " + common + "\n\n" + table[~same].T.to_string()
    return table.to_string()


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


CLAIM_STALE_S = 20 * 60   # đánh dấu không được cập nhật quá 20 phút -> coi như máy kia đã dừng


def claim_path(shared: Path, name: str) -> Path:
    return shared / "_claims" / name


def try_claim(shared: Path, name: str, owner: str, stale_s: float = CLAIM_STALE_S) -> str:
    """-> "done" (đã xong ở thư mục chung) | "busy" (máy khác đang chạy) | "claimed" (máy này nhận run)."""
    if is_done(shared / name):
        return "done"
    path = claim_path(shared, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        holder = path.read_text(encoding="utf-8", errors="replace").strip()
        if age < stale_s and holder != owner:
            return "busy"
    path.write_text(owner, encoding="utf-8")
    return "claimed"


def touch_claim(shared: Path, name: str) -> None:
    try:
        os.utime(claim_path(shared, name))
    except OSError:
        pass


def release_claim(shared: Path, name: str) -> None:
    try:
        claim_path(shared, name).unlink()
    except OSError:
        pass


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
    def __init__(self, jobs, out_dir: Path, deadline: float, est_run_h: float, poll_s: int,
                 shared: Path | None = None):
        self.q = queue.Queue()
        for j in jobs:
            self.q.put(j)
        self.out_dir, self.deadline, self.est_run_h, self.poll_s = out_dir, deadline, est_run_h, poll_s
        self.shared = shared
        self.owner = f"{socket.gethostname()}:{os.getpid()}"
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
            if self.shared is not None:
                with self.lock:
                    state = try_claim(self.shared, job.name, self.owner)
                if state != "claimed":
                    job.status = "done_elsewhere" if state == "done" else "busy_elsewhere"
                    self.say(f"[GPU {gpu}] bỏ qua {job.name}: " + ("đã xong ở máy khác" if state == "done"
                                                                    else "máy khác đang chạy"))
                    continue
            self.run(job, gpu)
            if self.shared is not None and job.status == "failed":
                release_claim(self.shared, job.name)   # để máy khác (hoặc lần chạy sau) thử lại

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
                if self.shared is not None:
                    touch_claim(self.shared, job.name)   # nhịp tim: máy khác biết run vẫn đang chạy
            self.echo_epochs(run_dir / "train.log", offset, gpu, job.name)
        job.minutes = (time.time() - t0) / 60
        if proc.returncode == 0 and is_done(run_dir):
            job.status = "done"
            job.dice = load_json(run_dir / "metrics.json")["overall"].get("dice")  # full_data: không có
            with self.lock:
                self.durations.append(time.time() - t0)
            dice = f"Dice={job.dice:.4f}" if job.dice is not None else "full_data (không val)"
            self.say(f"[GPU {gpu}] XONG {job.name} | {dice} | {job.minutes:.0f} phút")
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
    ap.add_argument("--archs", default=None, help="Chỉ chạy các hướng MoE này, vd A,B,C,D")
    ap.add_argument("--overlays", default=None, help="Chỉ dùng các overlay này (tên file), vd ce_sqrt_inv,sampler_q05")
    ap.add_argument("--views", default=None, help="Chỉ chạy các nhánh ảnh này, vd clin,derm,clin+derm")
    ap.add_argument("--variants", default=None, help="Lưới dạng variants (ablation): chỉ chạy các variant này")
    ap.add_argument("--folds", default=None, help="Lưới variants: chạy các fold này cho mọi variant, vd 0,1,2,3,4")
    ap.add_argument("--skip-lt-base", action="store_true",
                    help="Hướng long-tail (C, D, E): không thêm run không overlay (tag lt)")
    ap.add_argument("--gpus", default="auto", help="auto | cpu | 0,1")
    ap.add_argument("--restore", nargs="*", default=[], help="Thư mục outputs của phiên trước")
    ap.add_argument("--time-budget-h", type=float, default=11.0, help="Kaggle giới hạn 12 giờ/phiên")
    ap.add_argument("--est-run-h", type=float, default=1.5, help="Ước lượng thời gian 1 run khi chưa có số liệu")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--set", nargs="*", default=[], help="Override thêm cho mọi run")
    ap.add_argument("--suffix", default="", help="Hậu tố tên run, vd ep30 -> resnet152__ce_sqrtinv__clin__ep30")
    ap.add_argument("--shared-dir", default=None,
                    help="Thư mục chung giữa nhiều máy (vd Google Drive): bỏ qua run đã xong / máy khác đang chạy")
    ap.add_argument("--reverse", action="store_true", help="Chạy danh sách theo thứ tự ngược (máy thứ 2)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    start = time.time()

    exp = yaml.safe_load(resolve_path(args.experiment).read_text(encoding="utf-8"))
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None  # noqa: E731
    if "variants" in exp:
        folds = [int(f) for f in split(args.folds)] if args.folds else None
        jobs = build_variant_jobs(exp, args.env, split(args.backbones), args.set, args.suffix,
                                  split(args.variants), folds)
    else:
        jobs = build_jobs(exp, args.env, split(args.backbones), args.set, args.suffix,
                          split(args.archs), split(args.overlays), split(args.views), lt_base=not args.skip_lt_base)
    out_dir = resolve_path("outputs")
    out_dir.mkdir(exist_ok=True)
    if args.restore:
        print(f"Khôi phục {restore(args.restore, out_dir)} run từ phiên trước")

    for j in jobs:
        if is_done(out_dir / j.name):
            j.status = "done_before"
    pending = [j for j in jobs if j.status == "pending"]
    if args.reverse:
        pending.reverse()
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

    shared = resolve_path(args.shared_dir) if args.shared_dir else None
    if shared is not None:
        shared.mkdir(parents=True, exist_ok=True)
        print(f"Thư mục chung: {shared} | máy: {socket.gethostname()}" + (" | thứ tự ngược" if args.reverse else ""))
    grid = Grid(pending, out_dir, start + args.time_budget_h * 3600, args.est_run_h, args.poll_s, shared)
    threads = [threading.Thread(target=grid.worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n" + "=" * 80)
    labels = {"done_before": "đã xong trước", "done": "xong", "failed": "LỖI", "not_started": "chưa chạy (hết giờ)",
              "done_elsewhere": "xong ở máy khác", "busy_elsewhere": "máy khác đang chạy"}
    for j in jobs:
        dice = f"Dice={j.dice:.4f}" if j.dice is not None else "không val"
        extra = f" | {dice} | {j.minutes:.0f} phút | GPU {j.gpu}" if j.status == "done" else ""
        print(f"  {labels.get(j.status, j.status):<20} {j.name}{extra}")
    remaining = [j for j in jobs if j.status in ("failed", "not_started")]
    print(f"Tổng thời gian {(time.time() - start) / 3600:.2f} giờ. "
          + (f"Còn {len(remaining)} run — chạy lại notebook với --restore để tiếp tục." if remaining
             else "Đã hoàn tất toàn bộ lưới."))


if __name__ == "__main__":
    main()
