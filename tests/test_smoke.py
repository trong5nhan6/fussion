"""Kiểm tra nhanh (không tải trọng số): python -m pytest tests -q"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from milk10k import NUM_CLASSES  # noqa: E402
from milk10k.config import load_config  # noqa: E402
from milk10k.data import MetadataEncoder, load_test_df, load_train_df  # noqa: E402
from milk10k.data.transforms import build_transforms  # noqa: E402
from milk10k.losses import FocalLoss, class_weights  # noqa: E402
from milk10k.metrics import compute_metrics  # noqa: E402
from milk10k.models import build_model  # noqa: E402

BASELINES = sorted((ROOT / "configs" / "baselines").glob("*.yaml"))


@pytest.mark.parametrize("cfg_path", BASELINES, ids=lambda p: p.stem)
def test_baseline_forward(cfg_path):
    cfg = load_config(cfg_path, ["model.pretrained=false"])
    model = build_model(cfg, meta_dim=33).eval()
    s = cfg["data"]["img_size"]
    batch = {"images": {v: torch.randn(2, 3, s, s) for v in cfg["model"]["views"]},
             "meta": torch.randn(2, 33)}
    with torch.no_grad():
        assert model(batch).shape == (2, NUM_CLASSES)


def test_transforms_keep_aspect_and_square():
    img = Image.new("RGB", (600, 450), (255, 255, 255))
    x = build_transforms(320, train=False)(img)
    assert x.shape == (3, 320, 320)
    # 450/600*320 = 240 hàng ảnh thật, 40 hàng pad mỗi phía (ảnh trắng -> giá trị dương sau normalize)
    assert x[:, 40:280].min() > 0 and x[:, :40].max() < 0


def test_metadata_encoder():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    df, test = load_train_df(cfg), load_test_df(cfg)
    enc = MetadataEncoder().fit(df)
    X, Xt = enc.transform(df), enc.transform(test)
    assert X.shape == (len(df), enc.dim) and Xt.shape[1] == enc.dim
    assert np.isfinite(X).all() and np.isfinite(Xt).all()


def test_folds_no_leak():
    df = load_train_df(load_config(ROOT / "configs" / "default.yaml"))
    assert df.lesion_id.is_unique and set(df.fold) == {0, 1, 2, 3, 4}
    assert (df.groupby("label").fold.nunique() == 5).all()  # mọi lớp có mặt ở mọi fold


def test_focal_and_metrics():
    w = class_weights(np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), "sqrt_inv")
    loss = FocalLoss(w)(torch.randn(4, NUM_CLASSES, requires_grad=True), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    y = np.arange(NUM_CLASSES)
    m = compute_metrics(y, np.eye(NUM_CLASSES))
    assert m["dice"] == 1.0 and m["auc"] == 1.0 and m["ap"] == 1.0 and m["sensitivity"] == 1.0


@pytest.mark.parametrize("views", [[], ["clin", "clin"], ["dermoscopy"]])
def test_invalid_views_rejected(views):
    cfg = load_config(ROOT / "configs" / "default.yaml", ["model.pretrained=false", "model.backbone=resnet18"])
    cfg["model"]["views"] = views
    with pytest.raises(ValueError):
        build_model(cfg)


@pytest.mark.parametrize("views", [["clin"], ["derm"], ["clin", "derm"]])
def test_views_ablation_forward(views):
    cfg = load_config(ROOT / "configs" / "default.yaml",
                      ["model.pretrained=false", "model.backbone=resnet18", f"model.views=[{','.join(views)}]"])
    model = build_model(cfg).eval()
    batch = {"images": {v: torch.randn(2, 3, 64, 64) for v in ("clin", "derm")}}  # dư view vẫn chỉ dùng view đã chọn
    with torch.no_grad():
        assert model(batch).shape == (2, NUM_CLASSES)
    assert model.head[1].in_features == model.encoder.num_features * len(views)


# ---------------------------------------------------------------- imbalance
from milk10k.engine import make_sampler  # noqa: E402
from milk10k.losses import LogitAdjustedLoss, build_loss  # noqa: E402

LABELS = np.array([0] * 900 + [1] * 90 + [2] * 9 + list(range(3, NUM_CLASSES)))


def test_effective_num_between_none_and_inv():
    inv = class_weights(LABELS, "inv").numpy()
    cb = class_weights(LABELS, "effective_num", beta=0.999).numpy()
    cb_near_inv = class_weights(LABELS, "effective_num", beta=0.999999).numpy()
    ratio = lambda w: w[2] / w[0]  # noqa: E731  lớp hiếm / lớp lớn
    assert 1 < ratio(cb) < ratio(inv)
    assert np.allclose(cb_near_inv, inv, rtol=1e-2)


def test_logit_adjusted_tau0_equals_ce():
    logits, y = torch.randn(8, NUM_CLASSES), torch.randint(0, NUM_CLASSES, (8,))
    prior = torch.full((NUM_CLASSES,), 1 / NUM_CLASSES)
    la = LogitAdjustedLoss(torch.rand(NUM_CLASSES) + 0.01, tau=0.0)(logits, y)
    assert torch.allclose(la, torch.nn.functional.cross_entropy(logits, y))
    # prior đều -> dịch logit một hằng số -> loss không đổi
    assert torch.allclose(LogitAdjustedLoss(prior, tau=1.0)(logits, y), torch.nn.functional.cross_entropy(logits, y))


@pytest.mark.parametrize("name", ["ce", "focal", "logit_adjusted"])
def test_build_loss_all(name):
    cfg = {"loss": {"name": name, "class_weight": "effective_num", "la_tau": 1.0}}
    loss = build_loss(cfg, LABELS)(torch.randn(4, NUM_CLASSES), torch.tensor([0, 1, 2, 3]))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
def test_sampler_q_class_mass(q):
    s = make_sampler(LABELS, q)
    w = s.weights.numpy()
    mass = np.array([w[LABELS == c].sum() for c in range(NUM_CLASSES)])
    counts = np.bincount(LABELS, minlength=NUM_CLASSES)
    expected = counts ** (1 - q)
    assert np.allclose(mass / mass.sum(), expected / expected.sum())
    assert make_sampler(LABELS, None) is None


def test_overlay_config_merge_and_tag():
    cfg = load_config([ROOT / "configs/baselines/vit_small.yaml", ROOT / "configs/imbalance/crt.yaml"])
    assert cfg["name"] == "vit_small" and cfg["tag"] == "crt"
    assert cfg["crt"]["enabled"] and cfg["loss"]["class_weight"] == "none"
    assert cfg["data"]["img_size"] == 224  # giữ nguyên phần của baseline


@pytest.mark.parametrize("ov", sorted((ROOT / "configs" / "imbalance").glob("*.yaml")), ids=lambda p: p.stem)
def test_imbalance_overlays_valid(ov):
    cfg = load_config([ROOT / "configs/default.yaml", ov])
    assert torch.isfinite(build_loss(cfg, LABELS)(torch.randn(4, NUM_CLASSES), torch.tensor([0, 1, 2, 3])))
    make_sampler(LABELS, cfg["train"].get("sampler_q"))


# ---------------------------------------------------------------- metric ISIC + file nộp
import pandas as pd  # noqa: E402

from milk10k import CLASSES  # noqa: E402
from milk10k.inference import validate_submission  # noqa: E402
from milk10k.metrics import postprocess  # noqa: E402


def test_postprocess_top1_exactly_one_positive():
    p = torch.softmax(torch.randn(200, NUM_CLASSES), 1).numpy().astype(np.float64)
    s = postprocess(p, "top1")
    assert ((s > 0.5).sum(axis=1) == 1).all()
    assert (s.argmax(1) == p.argmax(1)).all() and s.min() >= 0 and s.max() <= 1


def test_isic_freebie_and_threshold():
    # Lớp không có mẫu dương và không dự đoán dương -> sensitivity/dice = 1.0 (quy ước ISIC)
    y = np.zeros(4, dtype=int)
    p = np.zeros((4, NUM_CLASSES))
    p[:, 0] = 0.9
    m = compute_metrics(y, p, "softmax")
    assert m["per_class"]["BCC"]["dice"] == 1.0 and m["per_class"]["AKIEC"]["dice"] == 1.0
    p[:, 0] = 0.5  # đúng 0.5 KHÔNG tính là dương (.gt(0.5))
    assert compute_metrics(y, p, "softmax")["per_class"]["AKIEC"]["sensitivity"] == 0.0


def test_validate_submission():
    ids = [f"IL_{i}" for i in range(3)]
    sub = pd.DataFrame(np.full((3, NUM_CLASSES), 0.1), columns=CLASSES)
    sub.insert(0, "lesion_id", ids)
    validate_submission(sub, ids)
    with pytest.raises(ValueError):
        validate_submission(sub.assign(BCC=1.5), ids)
    with pytest.raises(ValueError):
        validate_submission(sub.iloc[:2], ids)


# ---------------------------------------------------------------- run_grid (notebook Kaggle)
import yaml  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import run_grid  # noqa: E402


def _exp(name):
    return yaml.safe_load((ROOT / "experiments" / name).read_text(encoding="utf-8"))


def test_stage1_grid_order_and_names():
    jobs = run_grid.build_jobs(_exp("stage1_baselines.yaml"), "configs/env/kaggle_t4.yaml")
    assert len(jobs) == 12
    # mỗi backbone: clin -> derm -> clin+derm
    assert [j.name for j in jobs[:3]] == ["resnet152__ce_sqrtinv__clin", "resnet152__ce_sqrtinv__derm",
                                          "resnet152__ce_sqrtinv__clin+derm"]


def test_stage2_grid_filter_and_effective_config():
    jobs = run_grid.build_jobs(_exp("stage2_imbalance.yaml"), "configs/env/kaggle_t4.yaml", ["resnet152"])
    assert len(jobs) == 27 and len({j.name for j in jobs}) == 27
    j = next(j for j in jobs if j.name == "resnet152__crt__derm")
    cfg = load_config([ROOT / c for c in j.configs], j.sets)
    assert cfg["crt"]["enabled"] and cfg["model"]["views"] == ["derm"]
    assert cfg["train"]["batch_size"] == 8 and cfg["model"]["grad_checkpointing"] is False  # riêng backbone > env
    assert cfg["keep_checkpoints"] is False and cfg["tag"] == "crt"
    with pytest.raises(SystemExit):
        run_grid.build_jobs(_exp("stage2_imbalance.yaml"), None, ["resnet50"])


def test_restore_and_done(tmp_path):
    prev, out = tmp_path / "prev", tmp_path / "out"
    for name, files in [("a__t__clin", ["metrics.json", "submission.csv", "fold0/best.pt"]),
                        ("b__t__clin", ["metrics.json"])]:  # b dở dang -> không khôi phục
        for f in files:
            (prev / name / f).parent.mkdir(parents=True, exist_ok=True)
            (prev / name / f).write_text("{}")
    out.mkdir()
    assert run_grid.restore([prev], out) == 1
    assert run_grid.is_done(out / "a__t__clin") and not (out / "a__t__clin" / "fold0" / "best.pt").exists()
    assert not (out / "b__t__clin").exists()


def test_grid_suffix_and_cli_override_priority():
    jobs = run_grid.build_jobs(_exp("stage1_baselines.yaml"), "configs/env/kaggle_t4.yaml", ["resnet152"],
                               ["train.batch_size=4", "train.epochs=30"], suffix="ep30")
    assert jobs[0].name == "resnet152__ce_sqrtinv__clin__ep30"
    cfg = load_config([ROOT / c for c in jobs[0].configs], jobs[0].sets)
    assert cfg["train"]["batch_size"] == 4 and cfg["train"]["epochs"] == 30  # override notebook > riêng backbone
    assert "data.img_size" in run_grid.effective_table(jobs)
