"""Bảng xếp hạng các run trong outputs/ — cùng các cột như leaderboard ISIC MILK10k.

  python scripts/compare_runs.py
"""
import _bootstrap  # noqa: F401
import pandas as pd

from milk10k.metrics import LEADERBOARD_COLUMNS
from milk10k.utils import load_json, resolve_path


def _val_tag(per_fold: dict) -> str:
    folds = sorted(per_fold, key=int)
    return f"fold{folds[0]}" if len(folds) == 1 else f"{len(folds)}-fold OOF"


def main():
    out = resolve_path("outputs")
    rows = []
    for f in sorted(out.glob("*/metrics.json")):
        m = load_json(f)
        o = m["overall"]
        if "dice" not in o:  # run cũ, định dạng metric trước đây
            print(f"Bỏ qua {f.parent.name}: metrics.json định dạng cũ (hãy train lại)")
            continue
        model = m.get("model", {})
        rows.append({"run": f.parent.name, "backbone": model.get("backbone", model.get("name", "")),
                     "views": "+".join(model.get("views", [])) or model.get("inputs", ""),
                     "tag": m.get("tag", ""), "val": _val_tag(m.get("per_fold", {})),
                     **{name: o[k] for k, name in LEADERBOARD_COLUMNS},
                     "Balanced Acc": o["balanced_acc"]})
    if not rows:
        print("Chưa có run nào trong outputs/")
        return
    # Xếp theo Dice như leaderboard; chỉ so sánh các run cùng tập val -> nhóm theo cột val
    lb = pd.DataFrame(rows).sort_values(["val", "Dice Coefficient"], ascending=[True, False])
    lb.to_csv(out / "leaderboard.csv", index=False)
    print(lb.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
