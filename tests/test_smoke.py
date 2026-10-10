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
    assert model.head[0].in_features == model.encoder.num_features * len(views)   # MLP 2 lớp: Linear đầu tiên
    assert model.head[-1].out_features == NUM_CLASSES


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
    assert len(jobs) == 15   # 5 backbone (thêm dinov2_base) x 3 nhánh ảnh
    # mỗi backbone: clin -> derm -> clin+derm
    assert [j.name for j in jobs[:3]] == ["resnet152__ce_sqrtinv__clin", "resnet152__ce_sqrtinv__derm",
                                          "resnet152__ce_sqrtinv__clin+derm"]


def test_stage2_grid_filter_and_effective_config():
    jobs = run_grid.build_jobs(_exp("stage2_imbalance.yaml"), "configs/env/kaggle_t4.yaml", ["resnet152"])
    assert len(jobs) == 30 and len({j.name for j in jobs}) == 30   # 10 overlay (9 + bce) x 3 nhánh ảnh
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


# ---------------------------------------------------------------- multimodal MoE (A, B, C, D)
from milk10k.models.moe import TokenEncoder  # noqa: E402


def _moe(arch, views, extra=()):
    cfg = load_config([ROOT / "configs/default.yaml", ROOT / f"configs/moe/{arch}.yaml"],
                      ["model.pretrained=false", "model.backbone=resnet18", f"model.views=[{','.join(views)}]", *extra])
    return cfg, build_model(cfg, 34)


@pytest.mark.parametrize("arch", list("ABCDE"))
@pytest.mark.parametrize("views", [["clin"], ["derm"], ["clin", "derm"]], ids=["clin", "derm", "clin+derm"])
def test_moe_forward_loss_backward(arch, views):
    cfg, m = _moe(arch, views)
    m.set_class_prior(torch.full((NUM_CLASSES,), 1 / NUM_CLASSES))
    batch = {"images": {v: torch.randn(4, 3, 64, 64) for v in views}, "meta": torch.randn(4, 34)}
    y = torch.tensor([0, 1, 6, 10])
    m.train()
    logits = m(batch)
    assert logits.shape == (4, NUM_CLASSES)
    loss = m.compute_loss(logits, y, build_loss(cfg, LABELS))
    assert torch.isfinite(loss)
    loss.backward()
    assert m.tokens.encoder.conv1.weight.grad is not None          # gradient tới backbone
    m.eval()
    p = m(batch).float().softmax(1)
    assert torch.allclose(p.sum(1), torch.ones(4), atol=1e-5)
    keys = set(m.last_explain)
    n_tok = len(views) + 2
    expected = {"A": {f"gate_{t}" for t in [*views, "demo", "monet"]},
                "B": {f"expert_{t}" for t in ["cls", *views, "demo", "monet"]}, "E": set()}
    expected.update(C=expected["B"], D=expected["A"])
    assert keys == expected[arch]
    if arch in "AD":
        g = torch.stack([m.last_explain[f"gate_{t}"] for t in [*views, "demo", "monet"]], 1)
        assert g.shape == (4, n_tok) and torch.allclose(g.sum(1), torch.ones(4), atol=1e-5)


def test_moe_longtail_heads_and_prior():
    _, m = _moe("C", ["derm"], ["moe.lt_taus=[0.0,0.5,1.0]"])
    assert len(m.head.heads) == 3 and m.head.taus == [0.0, 0.5, 1.0]
    prior = torch.tensor([0.5] + [0.05] * 10)
    m.set_class_prior(prior)
    assert torch.allclose(m.head.log_prior.exp(), prior)
    assert m.head_module() is m.head


def test_modality_dropout_keeps_one_token():
    enc = TokenEncoder("resnet18", ["clin", "derm"], False, 64, 32, False, modality_dropout=1.0, dropout=0.0)
    enc.train()
    x = enc({"images": {v: torch.randn(6, 3, 64, 64) for v in ("clin", "derm")}, "meta": torch.randn(6, 34)})
    missing = enc.missing + enc.type_emb
    is_missing = torch.isclose(x, missing.expand_as(x), atol=1e-6).all(-1)   # [6, 4]
    assert (is_missing.sum(1) == 3).all()   # p=1: che hết, rồi giữ lại đúng 1 nguồn


def test_moe_param_groups_split_backbone():
    _, m = _moe("B", ["clin", "derm"], ["moe.fusion_lr_mult=2.0"])
    g_enc, g_rest = m.param_groups(1e-4, 0.1)
    assert g_enc["lr"] == pytest.approx(1e-5) and g_rest["lr"] == pytest.approx(2e-4)
    n_all = sum(1 for _ in m.parameters())
    assert len(list(g_enc["params"])) + len(g_rest["params"]) == n_all


def test_stage3_grid_longtail_uses_only_sampler_overlays():
    exp = _exp("stage3_moe.yaml")
    jobs = run_grid.build_jobs(exp, "configs/env/kaggle_t4.yaml", ["vit_base"], suffix="",
                               only_archs=["A", "C"], only_overlays=["ce_sqrt_inv", "sampler_q05"],
                               only_views=["clin+derm"])
    names = sorted(j.name for j in jobs)
    assert names == sorted(["vit_base__A__ce_sqrtinv__clin+derm", "vit_base__A__samp_q05__clin+derm",
                            "vit_base__C__lt__clin+derm", "vit_base__C__samp_q05__clin+derm"])
    j = next(j for j in jobs if j.name == "vit_base__C__samp_q05__clin+derm")
    cfg = load_config([ROOT / c for c in j.configs], j.sets)
    assert cfg["model"]["name"] == "moe" and cfg["model"]["arch"] == "C" and cfg["train"]["sampler_q"] == 0.5
    with pytest.raises(SystemExit):
        run_grid.build_jobs(exp, None, only_archs=["F"])


def test_moe_E_is_concat_with_longtail():
    from milk10k.models.moe import ConcatFusion
    _, m = _moe("E", ["clin", "derm"])
    assert isinstance(m.fusion, ConcatFusion) and m.longtail and len(m.head.heads) == 3
    assert m.fusion.aux_loss is None                                   # không có load-balancing
    jobs = run_grid.build_jobs(_exp("stage3_moe.yaml"), None, ["vit_base"], only_archs=["E"],
                               only_overlays=["ce_sqrt_inv", "sampler_q05"], only_views=["derm"])
    assert sorted(j.name for j in jobs) == ["vit_base__E__lt__derm", "vit_base__E__samp_q05__derm"]


# ---------------------------------------------------------------- BCE / sigmoid / resize / aug / drop path (stage 4)
from PIL import Image as _Image  # noqa: E402

from milk10k.data.transforms import build_transforms as _bt  # noqa: E402
from milk10k.engine import nll  # noqa: E402
from milk10k.losses import PosWeightedBCE, bce_pos_weight  # noqa: E402
from milk10k.metrics import output_activation, resolve_postprocess  # noqa: E402


def test_bce_pos_weight_and_loss():
    w = bce_pos_weight(LABELS, 10.0)
    assert w[0] < 1.0 + 1e-6 or w[0] == pytest.approx((len(LABELS) - 900) / 900)   # lớp lớn: trọng số nhỏ
    assert w.max() == pytest.approx(10.0)                                          # bị chặn ở 10
    loss = PosWeightedBCE(w)(torch.randn(4, NUM_CLASSES, requires_grad=True), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert build_loss({"loss": {"name": "bce", "pos_weight_clip": 0}}, LABELS).pos_weight is None


def test_postprocess_auto_and_sigmoid():
    assert resolve_postprocess({"loss": {"name": "bce"}, "predict": {"postprocess": "auto"}}) == "sigmoid"
    assert resolve_postprocess({"loss": {"name": "ce"}, "predict": {"postprocess": "auto"}}) == "top1"
    assert resolve_postprocess({"loss": {"name": "ce"}, "predict": {"postprocess": "top1"}}) == "top1"
    assert output_activation({"loss": {"name": "bce"}}) == "sigmoid"
    with pytest.raises(ValueError):
        resolve_postprocess({"loss": {"name": "ce"}, "predict": {"postprocess": "sigmoid"}})
    p = np.full((2, NUM_CLASSES), 0.1); p[0, [1, 9]] = 0.8        # sigmoid: 2 lớp dương ở lesion 0
    assert (postprocess(p, "sigmoid") == p).all()
    m = compute_metrics(np.array([1, 0]), p, "sigmoid")
    assert m["per_class"]["BCC"]["sensitivity"] == 1.0 and m["per_class"]["SCCKA"]["specificity"] < 1.0
    assert np.isfinite(nll(p, np.array([1, 0]), "sigmoid"))


@pytest.mark.parametrize("resize", ["pad", "squash"])
@pytest.mark.parametrize("aug", ["basic", "strong"])
def test_transforms_resize_aug(resize, aug):
    im = _Image.new("RGB", (600, 450), (255, 255, 255))
    for train in (True, False):
        x = _bt(224, train, resize=resize, aug=aug)(im)
        assert x.shape == (3, 224, 224) and torch.isfinite(x).all()
    val = _bt(224, False, resize=resize, aug=aug)(im)
    has_pad = (val[:, :20] < 0).all()            # pad đen ở mép trên (âm sau normalize), squash thì không
    assert bool(has_pad) == (resize == "pad")
    with pytest.raises(ValueError):
        _bt(224, False, resize="crop")


def test_drop_path_and_bce_longtail_guard(tmp_path):
    cfg = load_config(ROOT / "configs/default.yaml", ["model.pretrained=false", "model.backbone=vit_tiny_patch16_224",
                                                     "model.drop_path_rate=0.1", "data.img_size=64"])
    m = build_model(cfg)
    assert any(type(x).__name__ == "DropPath" for x in m.modules())
    cfg_lt = load_config([ROOT / "configs/default.yaml", ROOT / "configs/moe/D.yaml", ROOT / "configs/imbalance/bce.yaml"])
    from milk10k.metrics import output_activation as oa
    assert oa(cfg_lt) == "sigmoid"     # engine sẽ báo lỗi khi gặp tổ hợp này với hướng long-tail


@pytest.mark.parametrize("ov", sorted((ROOT / "configs" / "ablation").glob("*.yaml")), ids=lambda p: p.stem)
def test_ablation_overlays_valid(ov):
    cfg = load_config([ROOT / "configs/baselines/vit_base.yaml", ov])
    assert cfg["tag"] == yaml.safe_load(ov.read_text(encoding="utf-8"))["tag"]
    assert torch.isfinite(build_loss(cfg, LABELS)(torch.randn(4, NUM_CLASSES), torch.tensor([0, 1, 2, 3])))


def test_grid_skip_lt_base_and_inherited_sampler():
    exp = _exp("stage3_moe.yaml")
    jobs = run_grid.build_jobs(exp, None, ["vit_base"], only_archs=["B", "D"], only_overlays=["r5_all", "r5_nobce"],
                               only_views=["clin+derm"], lt_base=False)
    assert sorted(j.name for j in jobs) == ["vit_base__B__r5_all__clin+derm", "vit_base__B__r5_nobce__clin+derm",
                                            "vit_base__D__r5_nobce__clin+derm"]
    jobs4 = run_grid.build_jobs(_exp("stage4_ablation.yaml"), None, ["vit_base"], only_views=["clin+derm"])
    assert [j.name.split("__")[1] for j in jobs4] == ["r1_bce", "r2_squash", "r3_aug", "r4_lr", "r5_all"]


# ---------------------------------------------------------------- DINOv2 + concat MLP, full_data
@pytest.mark.parametrize("concat_meta", [False, True])
def test_concat_mlp_dinov2(concat_meta):
    cfg = load_config(ROOT / "configs/baselines/dinov2_base.yaml",
                      ["model.pretrained=false", f"model.concat_meta={str(concat_meta).lower()}"])
    m = build_model(cfg, 34).eval()
    assert m.encoder.num_features == 768 and m.uses_metadata == concat_meta
    lin1, lin2 = [x for x in m.head if isinstance(x, torch.nn.Linear)]
    assert lin1.in_features == 768 * 2 + (34 if concat_meta else 0) and lin2.out_features == NUM_CLASSES   # MLP 2 lớp
    batch = {"images": {v: torch.randn(2, 3, 224, 224) for v in ("clin", "derm")}, "meta": torch.randn(2, 34)}
    with torch.no_grad():
        assert m(batch).shape == (2, NUM_CLASSES)


def test_dinov2_img_size_multiple_of_14():
    for size in (224, 336):
        cfg = load_config(ROOT / "configs/baselines/dinov2_base.yaml",
                          ["model.pretrained=false", f"data.img_size={size}", "model.views=[derm]"])
        m = build_model(cfg).eval()
        with torch.no_grad():
            assert m({"images": {"derm": torch.randn(1, 3, size, size)}}).shape == (1, NUM_CLASSES)


@pytest.mark.parametrize("n", [None, 0, 4])
def test_dinov2_trainable_blocks(n):
    v = "null" if n is None else n
    cfg = load_config(ROOT / "configs/baselines/dinov2_base.yaml", ["model.pretrained=false", f"model.trainable_blocks={v}"])
    e = build_model(cfg).encoder
    trainable = [any(p.requires_grad for p in b.parameters()) for b in e.blocks]
    if n is None:
        assert all(trainable) and all(p.requires_grad for p in e.patch_embed.parameters())
    else:
        assert trainable == [False] * (12 - n) + [True] * n
        assert not any(p.requires_grad for p in e.patch_embed.parameters()) and e.norm.weight.requires_grad
    # mặc định của config DINOv2 là 4 block cuối
    assert load_config(ROOT / "configs/baselines/dinov2_base.yaml")["model"]["trainable_blocks"] == 4


@pytest.mark.parametrize("cfg_name", ["swin_base", "cnn_convnext_base", "cnn_resnet152"])
def test_trainable_stages_other_backbones(cfg_name):
    cfg = load_config(ROOT / f"configs/baselines/{cfg_name}.yaml", ["model.pretrained=false", "model.trainable_blocks=1"])
    e = build_model(cfg).encoder
    n_tr, n_all = (sum(p.numel() for p in e.parameters() if p.requires_grad), sum(p.numel() for p in e.parameters()))
    assert 0 < n_tr < n_all
