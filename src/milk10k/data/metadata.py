"""Mã hoá metadata (tuổi, giới, skin tone, vị trí, MONET) thành vector số.

Thống kê chuẩn hoá chỉ được fit trên tập train của từng fold để tránh rò rỉ.
"""
import numpy as np
import pandas as pd

SEXES = ["male", "female"]
SKIN_TONES = [0, 1, 2, 3, 4, 5]
SITES = ["trunk", "head_neck_face", "lower_extremity", "upper_extremity", "hand", "foot", "genital"]
MONET_CONCEPTS = ["ulceration_crust", "hair", "vasculature_vessels", "erythema", "pigmented",
                  "gel_water_drop_fluid_dermoscopy_liquid", "skin_markings_pen_ink_purple_pen"]
MONET_COLS = [f"{v}_MONET_{c}" for v in ("clin", "derm") for c in MONET_CONCEPTS]


def _one_hot(values: pd.Series, categories) -> np.ndarray:
    """One-hot + 1 cột 'unknown' cho giá trị thiếu hoặc lạ."""
    out = np.zeros((len(values), len(categories) + 1), dtype=np.float32)
    lookup = {c: i for i, c in enumerate(categories)}
    for row, v in enumerate(values):
        out[row, lookup.get(v, len(categories))] = 1.0
    return out


class MetadataEncoder:
    def __init__(self):
        self.age_mean = None
        self.age_std = None

    def fit(self, df: pd.DataFrame) -> "MetadataEncoder":
        age = pd.to_numeric(df["age_approx"], errors="coerce")
        self.age_mean = float(age.mean())
        self.age_std = float(age.std()) or 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        age = pd.to_numeric(df["age_approx"], errors="coerce")
        age_z = ((age - self.age_mean) / self.age_std).fillna(0.0).to_numpy(np.float32)[:, None]
        age_missing = age.isna().to_numpy(np.float32)[:, None]
        tone = pd.to_numeric(df["skin_tone_class"], errors="coerce")
        parts = [
            age_z, age_missing,
            _one_hot(df["sex"], SEXES),
            _one_hot(tone, SKIN_TONES),
            _one_hot(df["site"], SITES),
            df[MONET_COLS].fillna(0.5).to_numpy(np.float32),
        ]
        return np.concatenate(parts, axis=1)

    @property
    def feature_names(self):
        return (["age_z", "age_missing"] + [f"sex_{s}" for s in SEXES + ["unknown"]]
                + [f"tone_{t}" for t in SKIN_TONES + ["unknown"]]
                + [f"site_{s}" for s in SITES + ["unknown"]] + MONET_COLS)

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def state_dict(self) -> dict:
        return {"age_mean": self.age_mean, "age_std": self.age_std}

    def load_state_dict(self, state: dict) -> "MetadataEncoder":
        self.age_mean, self.age_std = state["age_mean"], state["age_std"]
        return self
