"""Mô hình machine learning cổ điển trên metadata (tuổi, giới, skin tone, vị trí, 14 điểm MONET).

  python scripts/train_tabular.py --model all                 # mọi mô hình có sẵn thư viện
  python scripts/train_tabular.py --model hgb lightgbm
  python scripts/train_tabular.py --model all --set train.folds=[0,1,2,3,4]

Mọi mô hình dùng cùng trọng số mẫu sqrt-balanced (trừ KNN không hỗ trợ), cùng fold, cùng metric ISIC.
"""
import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from milk10k import CLASSES
from milk10k.config import load_config
from milk10k.data import MetadataEncoder, load_test_df, load_train_df
from milk10k.engine import log_leaderboard, nll
from milk10k.inference import write_submission
from milk10k.metrics import SUMMARY_KEYS, compute_metrics, summary_line
from milk10k.utils import get_logger, resolve_path, save_json


def _lightgbm(seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1)


def _xgboost(seed):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=1.0, tree_method="hist", eval_metric="mlogloss", random_state=seed)


def _catboost(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, random_seed=seed, verbose=0,
                              allow_writing_files=False)


# tên -> (hàm tạo mô hình, có hỗ trợ sample_weight không, tên tham số sample_weight trong pipeline)
MODELS = {
    "logreg": (lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)), True),
    "svm": (lambda s: make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=s)), True),
    "knn": (lambda s: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance")),
            False),
    "random_forest": (lambda s: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                                       random_state=s), True),
    "extra_trees": (lambda s: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                                   random_state=s), True),
    "hgb": (lambda s: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                                                     l2_regularization=1.0, early_stopping=True, random_state=s),
            True),
    "lightgbm": (_lightgbm, True),
    "xgboost": (_xgboost, True),
    "catboost": (_catboost, True),
}


def available(name: str, seed: int) -> bool:
    try:
        MODELS[name][0](seed)
        return True
    except ImportError:
        return False


def fit(name: str, seed: int, X, y):
    make, weighted = MODELS[name]
    model = make(seed)
    if not weighted:
        return model.fit(X, y)
    w = np.sqrt(compute_sample_weight("balanced", y))
    if hasattr(model, "steps"):  # Pipeline: truyền trọng số cho bước cuối
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def full_proba(model, X) -> np.ndarray:
    """predict_proba đủ 11 cột kể cả khi fold train thiếu lớp nào đó."""
    p = np.zeros((len(X), len(CLASSES)))
    p[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(X)
    return p


def run_model(name: str, cfg: dict, df: pd.DataFrame, test_df: pd.DataFrame):
    folds = cfg["train"]["folds"]
    mode = cfg["predict"].get("postprocess", "top1")
    tag = f"fold{folds[0]}" if len(folds) == 1 else f"{len(folds)}fold"
    run_name = f"ml_{name}__{tag}"
    run_dir = resolve_path(cfg["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(run_dir / "train.log", name=run_name)
    t0 = time.time()
    seed = cfg["seed"]
    logger.info(f"Run          : {run_name}")
    logger.info(f"Model        : {MODELS[name][0](seed)}")
    logger.info(f"Đầu vào      : metadata {MetadataEncoder().fit(df).dim} chiều (tuổi, giới, skin tone, vị trí, 14 MONET)")
    logger.info(f"Imbalance    : {'sample_weight = sqrt(balanced)' if MODELS[name][1] else 'không (mô hình không hỗ trợ)'}")
    logger.info(f"Đánh giá     : ngưỡng > 0.5 mỗi lớp, hậu xử lý={mode} | loss = CE | acc = top-1 đa lớp")

    oofs, test_probs, per_fold = [], [], {}
    for fold in folds:
        tr, va = df[df.fold != fold], df[df.fold == fold]
        enc = MetadataEncoder().fit(tr)
        X_tr, y_tr, X_va, y_va = enc.transform(tr), tr.label.to_numpy(), enc.transform(va), va.label.to_numpy()
        tf = time.time()
        model = fit(name, seed, X_tr, y_tr)
        fit_s = time.time() - tf
        p_tr, p = full_proba(model, X_tr), full_proba(model, X_va)
        m = compute_metrics(y_va, p, mode)
        per_fold[int(fold)] = {k: m[k] for k in SUMMARY_KEYS}
        logger.info("=" * 100)
        logger.info(f"FOLD {fold} | train {len(tr)} | val {len(va)} | fit {fit_s:.1f}s")
        logger.info(f"[fold{fold}] train loss {nll(p_tr, y_tr):.4f} acc {(p_tr.argmax(1) == y_tr).mean():.4f} | "
                    f"val loss {nll(p, y_va):.4f} acc {m['top1_acc']:.4f} | {summary_line(m)}")
        log_leaderboard(logger, f"[fold {fold}] VAL", m)
        oofs.append(pd.DataFrame(p, columns=CLASSES).assign(lesion_id=va.lesion_id.values, fold=fold,
                                                            label=y_va)[["lesion_id", "fold", "label"] + CLASSES])
        test_probs.append(full_proba(model, enc.transform(test_df)))

    oof = pd.concat(oofs, ignore_index=True)
    oof.to_csv(run_dir / "oof.csv", index=False)
    overall = compute_metrics(oof.label.to_numpy(), oof[CLASSES].to_numpy(), mode)
    save_json({"run": run_name, "name": f"ml_{name}", "tag": "", "model": {"name": name, "inputs": "metadata"},
               "postprocess": mode, "folds": folds, "overall": overall, "per_fold": per_fold},
              run_dir / "metrics.json")
    logger.info("=" * 100)
    log_leaderboard(logger, "KẾT QUẢ VAL", overall)
    logger.info("=" * 100)
    write_submission(test_df.lesion_id.values, np.mean(test_probs, axis=0), mode, run_dir, test_df.lesion_id,
                     logger.info)
    logger.info(f"Tổng thời gian: {time.time() - t0:.1f} giây")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["all"], help=f"all hoặc một số trong: {list(MODELS)}")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="Ghi đè config, ví dụ train.folds=[0,1,2,3,4]")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    names = list(MODELS) if args.model == ["all"] else args.model
    bad = set(names) - set(MODELS)
    if bad:
        raise SystemExit(f"Mô hình không hỗ trợ: {sorted(bad)}. Có: {list(MODELS)}")

    df, test_df = load_train_df(cfg), load_test_df(cfg)
    results = {}
    for name in names:
        if not available(name, cfg["seed"]):
            print(f"Bỏ qua {name}: chưa cài thư viện (pip install {name})")
            continue
        print(f"--- {name}")
        results[name] = run_model(name, cfg, df, test_df)
    if results:
        print("\n" + pd.DataFrame({n: {k: r[k] for k in ("dice", "auc", "ap", "accuracy", "sensitivity",
                                                          "specificity")} for n, r in results.items()}).T
              .sort_values("dice", ascending=False).to_markdown(floatfmt=".4f"))


if __name__ == "__main__":
    main()
