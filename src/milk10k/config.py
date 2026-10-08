"""Đọc config YAML có kế thừa (`base:`) và ghi đè từ CLI (`key.sub=value`)."""
import os
from pathlib import Path

import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        return _deep_update(_load_yaml((path.parent / base).resolve()), cfg)
    return cfg


def load_config(paths, overrides=None) -> dict:
    """`paths`: 1 file hoặc danh sách file, gộp theo thứ tự (file sau ghi đè file trước).

    File overlay (ví dụ configs/imbalance/*.yaml) có thể đặt `tag`; các tag được nối lại để đặt tên run.
    """
    paths = [Path(p) for p in (paths if isinstance(paths, (list, tuple)) else [paths])]
    cfg, tags = {}, []
    for p in paths:
        part = _load_yaml(p)
        if part.get("tag"):
            tags.append(str(part.pop("tag")))
        _deep_update(cfg, part)
    if tags:
        cfg["tag"] = "_".join(tags)
    path = paths[0]
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override sai định dạng (cần key=value): {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    cfg.setdefault("name", path.stem)
    # Dữ liệu đặt ở nơi khác (vd Kaggle: /tmp/MILK10k) -> chỉ cần đặt biến môi trường, không sửa config
    if os.environ.get("MILK10K_ROOT") and "data" in cfg:
        cfg["data"]["root"] = os.environ["MILK10K_ROOT"]
    return cfg


def save_config(cfg: dict, path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
