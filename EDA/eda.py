"""EDA cho bộ dữ liệu MILK10k. Chạy: python EDA/eda.py (từ thư mục AIAT2026)."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "MILK10k"
OUT = ROOT / "EDA"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
MALIGNANT = {"AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"}
MONET = ["ulceration_crust", "hair", "vasculature_vessels", "erythema", "pigmented",
         "gel_water_drop_fluid_dermoscopy_liquid", "skin_markings_pen_ink_purple_pen"]

# Palette (validated reference palette, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

report = []  # markdown lines
facts = {}


def save(fig, name):
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


def md_table(df, index=True):
    return df.to_markdown(index=index, floatfmt="g")


# ---------------------------------------------------------------- load
train = pd.read_csv(DATA / "train" / "train_combined.csv")
test = pd.read_csv(DATA / "test" / "test_combined.csv")
gt = pd.read_csv(DATA / "train" / "MILK10k_Training_GroundTruth.csv")
meta_tr = pd.read_csv(DATA / "train" / "MILK10k_Training_Metadata.csv")
meta_te = pd.read_csv(DATA / "test" / "MILK10k_Test_Metadata.csv")
fold_tr = pd.read_csv(DATA / "splits" / "train_fold0.csv")
fold_va = pd.read_csv(DATA / "splits" / "val_fold0.csv")

train["label"] = train[CLASSES].idxmax(axis=1)
fold_tr["label"] = fold_tr[CLASSES].idxmax(axis=1)
fold_va["label"] = fold_va[CLASSES].idxmax(axis=1)
train["malignant"] = train["label"].isin(MALIGNANT)

# ---------------------------------------------------------------- 1. overview & integrity
overview = pd.DataFrame({
    "file": ["train_combined", "test_combined", "GroundTruth", "Train Metadata", "Test Metadata",
             "train_fold0", "val_fold0"],
    "rows": [len(train), len(test), len(gt), len(meta_tr), len(meta_te), len(fold_tr), len(fold_va)],
    "cols": [train.shape[1] - 2, test.shape[1], gt.shape[1], meta_tr.shape[1], meta_te.shape[1],
             fold_tr.shape[1] - 1, fold_va.shape[1] - 1],
    "unique lesion_id": [train.lesion_id.nunique(), test.lesion_id.nunique(), gt.lesion_id.nunique(),
                         meta_tr.lesion_id.nunique(), meta_te.lesion_id.nunique(),
                         fold_tr.lesion_id.nunique(), fold_va.lesion_id.nunique()],
})
overview.to_csv(TAB / "01_overview.csv", index=False)

checks = {}
label_sums = gt[CLASSES].sum(axis=1)
checks["Ground truth one-hot (mỗi lesion đúng 1 nhãn)"] = bool((label_sums == 1).all())
checks["Nhãn chỉ gồm 0/1"] = bool(np.isin(gt[CLASSES].values, [0, 1]).all())
checks["train_combined khớp GroundTruth"] = set(train.lesion_id) == set(gt.lesion_id)
checks["Train ∩ Test lesion_id rỗng"] = len(set(train.lesion_id) & set(test.lesion_id)) == 0
checks["fold0 train ∩ val rỗng"] = len(set(fold_tr.lesion_id) & set(fold_va.lesion_id)) == 0
checks["fold0 train ∪ val = toàn bộ train"] = set(fold_tr.lesion_id) | set(fold_va.lesion_id) == set(train.lesion_id)
checks["Metadata train: mỗi lesion có 2 ảnh"] = bool((meta_tr.groupby("lesion_id").size() == 2).all())
checks["Metadata test: mỗi lesion có 2 ảnh"] = bool((meta_te.groupby("lesion_id").size() == 2).all())
checks["Không trùng isic_id giữa train và test"] = len(set(meta_tr.isic_id) & set(meta_te.isic_id)) == 0
checks["Không có lesion trùng lặp trong train_combined"] = not train.duplicated("lesion_id").any()

for split, df, img_dir in [("train", train, DATA / "train" / "MILK10k_Training_Input"),
                           ("test", test, DATA / "test" / "MILK10k_Test_Input")]:
    miss = sum(not (img_dir / p).exists() for p in pd.concat([df.clinical_path, df.derm_path]))
    checks[f"Ảnh {split} tồn tại đầy đủ (thiếu {miss})"] = miss == 0
pd.Series(checks).to_csv(TAB / "02_integrity_checks.csv", header=["pass"])

# image_type / manipulation
img_type = pd.concat([meta_tr.assign(split="train"), meta_te.assign(split="test")]) \
    .groupby(["split", "image_type"]).size().unstack(0).fillna(0).astype(int)
manip = pd.concat([meta_tr.assign(split="train"), meta_te.assign(split="test")]) \
    .groupby(["split", "image_manipulation"]).size().unstack(0).fillna(0).astype(int)
img_type.to_csv(TAB / "03_image_type.csv")
manip.to_csv(TAB / "03_image_manipulation.csv")

# ---------------------------------------------------------------- 2. missing values
cols_meta = ["age_approx", "sex", "skin_tone_class", "site"]
miss = pd.DataFrame({
    "train_missing": train[cols_meta + [c for c in train.columns if "MONET" in c]].isna().sum(),
    "test_missing": test[cols_meta + [c for c in test.columns if "MONET" in c]].isna().sum(),
})
miss["train_%"] = (miss.train_missing / len(train) * 100).round(2)
miss["test_%"] = (miss.test_missing / len(test) * 100).round(2)
miss.to_csv(TAB / "04_missing_values.csv")

# ---------------------------------------------------------------- 3. class distribution
cls = train.label.value_counts().reindex(CLASSES)
cls_df = pd.DataFrame({"count": cls, "percent": cls / len(train) * 100,
                       "malignant": [c in MALIGNANT for c in CLASSES]}).sort_values("count", ascending=False)
cls_df.to_csv(TAB / "05_class_distribution.csv")
facts["imbalance_ratio"] = cls.max() / cls.min()

fig, ax = plt.subplots(figsize=(8, 4.2))
order = cls_df.index.tolist()
colors = [ORANGE if c in MALIGNANT else BLUE for c in order]
bars = ax.barh(order[::-1], cls_df["count"][::-1], color=colors[::-1], height=0.7)
for b, (n, p) in zip(bars, zip(cls_df["count"][::-1], cls_df["percent"][::-1])):
    ax.text(b.get_width() + 15, b.get_y() + b.get_height() / 2, f"{n} ({p:.1f}%)",
            va="center", fontsize=9, color=INK2)
ax.set_xlabel("Số lesion")
ax.set_title("Phân bố nhãn chẩn đoán — tập train (n=5.240)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, cls.max() * 1.22)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="Ác tính"), Patch(color=BLUE, label="Lành tính / khác")],
          loc="lower right", frameon=False)
save(fig, "01_class_distribution.png")

# fold balance
fold_cmp = pd.DataFrame({
    "full_%": train.label.value_counts(normalize=True) * 100,
    "train_fold0_%": fold_tr.label.value_counts(normalize=True) * 100,
    "val_fold0_%": fold_va.label.value_counts(normalize=True) * 100,
    "val_fold0_n": fold_va.label.value_counts(),
}).reindex(CLASSES)
fold_cmp.to_csv(TAB / "06_fold0_stratification.csv")

# ---------------------------------------------------------------- 4. demographics
def numeric_age(s):
    return pd.to_numeric(s, errors="coerce")

train["age"] = numeric_age(train.age_approx)
test["age"] = numeric_age(test.age_approx)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
bins = np.arange(0, 95, 5)
for ax_, (name, df, col) in zip(axes[:1], [("", None, None)]):
    pass
ax = axes[0]
ax.hist(train.age.dropna(), bins=bins, density=True, color=BLUE, alpha=0.85, label="Train", rwidth=0.9)
ax.hist(test.age.dropna(), bins=bins, density=True, histtype="step", color=ORANGE, lw=2, label="Test")
ax.set_title("Tuổi (xấp xỉ, bước 5 năm)")
ax.set_xlabel("Tuổi"); ax.set_ylabel("Mật độ"); ax.legend(frameon=False)

ax = axes[1]
sx = pd.DataFrame({"Train": train.sex.value_counts(normalize=True, dropna=False),
                   "Test": test.sex.value_counts(normalize=True, dropna=False)}) * 100
sx.index = sx.index.fillna("missing")
x = np.arange(len(sx)); w = 0.38
ax.bar(x - w / 2 - 0.01, sx.Train, w, color=BLUE, label="Train")
ax.bar(x + w / 2 + 0.01, sx.Test, w, color=ORANGE, label="Test")
ax.set_xticks(x, sx.index); ax.set_ylabel("%"); ax.set_title("Giới tính"); ax.legend(frameon=False)
ax.grid(axis="x", visible=False)

ax = axes[2]
st = pd.DataFrame({"Train": train.skin_tone_class.value_counts(normalize=True),
                   "Test": test.skin_tone_class.value_counts(normalize=True)}).sort_index() * 100
x = np.arange(len(st))
ax.bar(x - w / 2 - 0.01, st.Train, w, color=BLUE, label="Train")
ax.bar(x + w / 2 + 0.01, st.Test, w, color=ORANGE, label="Test")
ax.set_xticks(x, [str(int(i)) for i in st.index]); ax.set_ylabel("%")
ax.set_title("Skin tone class"); ax.legend(frameon=False)
ax.grid(axis="x", visible=False)
save(fig, "02_demographics_train_vs_test.png")

site_tab = pd.DataFrame({"Train_%": train.site.value_counts(normalize=True, dropna=False) * 100,
                         "Test_%": test.site.value_counts(normalize=True, dropna=False) * 100})
site_tab.index = site_tab.index.fillna("missing")
site_tab = site_tab.sort_values("Train_%", ascending=True)
site_tab.iloc[::-1].to_csv(TAB / "07_site_distribution.csv")
fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(site_tab)); h = 0.38
ax.barh(y + h / 2 + 0.01, site_tab["Train_%"], h, color=BLUE, label="Train")
ax.barh(y - h / 2 - 0.01, site_tab["Test_%"], h, color=ORANGE, label="Test")
ax.set_yticks(y, site_tab.index); ax.set_xlabel("%"); ax.grid(axis="y", visible=False)
ax.set_title("Vị trí giải phẫu của tổn thương"); ax.legend(frameon=False, loc="lower right")
save(fig, "03_site_distribution.png")

demo_tab = pd.DataFrame({
    "Train": [train.age.mean(), train.age.median(), train.age.min(), train.age.max()],
    "Test": [test.age.mean(), test.age.median(), test.age.min(), test.age.max()],
}, index=["age_mean", "age_median", "age_min", "age_max"])
demo_tab.to_csv(TAB / "08_age_summary.csv")

# ---------------------------------------------------------------- 5. features vs label
age_by_cls = train.groupby("label").age.describe()[["count", "mean", "50%", "std", "min", "max"]].reindex(order)
age_by_cls.to_csv(TAB / "09_age_by_class.csv")

fig, ax = plt.subplots(figsize=(9, 4.2))
data = [train.loc[train.label == c, "age"].dropna() for c in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.55,
                medianprops=dict(color=INK, lw=1.5), flierprops=dict(marker="o", ms=3, mfc=GRID, mec=INK2))
for patch, c in zip(bp["boxes"], order):
    patch.set_facecolor(ORANGE if c in MALIGNANT else BLUE); patch.set_alpha(0.8); patch.set_edgecolor("white")
ax.set_ylabel("Tuổi"); ax.set_title("Phân bố tuổi theo nhãn (cam = ác tính)")
ax.grid(axis="x", visible=False)
save(fig, "04_age_by_class.png")


def heat(ct, title, fname, fmt="{:.0f}", cmap="Blues", figsize=(10, 5), xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ct.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(ct.shape[1]), ct.columns, rotation=40, ha="right")
    ax.set_yticks(range(ct.shape[0]), ct.index)
    ax.grid(False)
    vmax = np.nanmax(ct.values)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if v > vmax * 0.6 else INK)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, fname)


ct_site = pd.crosstab(train.site.fillna("missing"), train.label, normalize="columns").reindex(columns=order) * 100
ct_site.to_csv(TAB / "10_site_by_class_pct.csv")
heat(ct_site, "Vị trí theo nhãn (% trong mỗi nhãn)", "05_site_by_class.png", ylabel="site")

ct_sex = pd.crosstab(train.label, train.sex.fillna("missing"), normalize="index").reindex(order) * 100
ct_sex.to_csv(TAB / "11_sex_by_class_pct.csv")
ct_tone = pd.crosstab(train.label, train.skin_tone_class, normalize="index").reindex(order) * 100
ct_tone.to_csv(TAB / "12_skintone_by_class_pct.csv")
heat(ct_tone, "Skin tone theo nhãn (% trong mỗi nhãn)", "06_skintone_by_class.png",
     figsize=(7, 5), xlabel="skin_tone_class")

# ---------------------------------------------------------------- 6. MONET concept scores
mon_cols = [f"{m}_{c}" for m in ("clin_MONET", "derm_MONET") for c in MONET]
mon_desc = train[mon_cols].describe().T
mon_desc.to_csv(TAB / "13_monet_describe.csv")

short = {"ulceration_crust": "ulceration", "hair": "hair", "vasculature_vessels": "vessels",
         "erythema": "erythema", "pigmented": "pigmented",
         "gel_water_drop_fluid_dermoscopy_liquid": "gel/fluid", "skin_markings_pen_ink_purple_pen": "pen marks"}
for mod, nice in (("clin", "ảnh lâm sàng"), ("derm", "ảnh dermoscopy")):
    m = train.groupby("label")[[f"{mod}_MONET_{c}" for c in MONET]].mean().reindex(order)
    m.columns = [short[c] for c in MONET]
    m.to_csv(TAB / f"14_monet_mean_by_class_{mod}.csv")
    heat(m, f"Điểm MONET trung bình theo nhãn — {nice}", f"07_monet_by_class_{mod}.png",
         fmt="{:.2f}", figsize=(8, 5))

corr = train[mon_cols].corr()
corr.index = corr.columns = [c.replace("_MONET_", ":").replace("gel_water_drop_fluid_dermoscopy_liquid", "gel")
                             .replace("skin_markings_pen_ink_purple_pen", "pen").replace("vasculature_vessels", "vessels")
                             .replace("ulceration_crust", "ulcer") for c in corr.columns]
corr.to_csv(TAB / "15_monet_correlation.csv")
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right"); ax.set_yticks(range(len(corr)), corr.index)
ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr.values[i, j]) > 0.6 else INK)
ax.set_title("Tương quan Pearson giữa các điểm MONET (clin vs derm)")
fig.colorbar(im, ax=ax, shrink=0.8)
save(fig, "08_monet_correlation.png")

# train vs test MONET shift
shift = pd.DataFrame({"train_mean": train[mon_cols].mean(), "test_mean": test[mon_cols].mean()})
shift["diff"] = shift.test_mean - shift.train_mean
shift.to_csv(TAB / "16_monet_train_vs_test.csv")

# ---------------------------------------------------------------- 7. image properties
def img_stats(df, img_dir, split):
    rows = []
    for _, r in df.iterrows():
        for mod, col in (("clinical", "clinical_path"), ("dermoscopic", "derm_path")):
            p = img_dir / r[col]
            with Image.open(p) as im:
                w_, h_ = im.size
                small = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32)
            rows.append({"split": split, "lesion_id": r.lesion_id, "modality": mod, "width": w_, "height": h_,
                         "file_kb": p.stat().st_size / 1024, "mode": im.mode,
                         "mean_R": small[..., 0].mean(), "mean_G": small[..., 1].mean(),
                         "mean_B": small[..., 2].mean(), "brightness": small.mean(),
                         "contrast": small.mean(axis=2).std()})
    return pd.DataFrame(rows)


imgs = pd.concat([img_stats(train, DATA / "train" / "MILK10k_Training_Input", "train"),
                  img_stats(test, DATA / "test" / "MILK10k_Test_Input", "test")], ignore_index=True)
imgs = imgs.merge(train[["lesion_id", "label"]], on="lesion_id", how="left")
imgs.to_csv(TAB / "17_image_stats_per_image.csv", index=False)
img_sum = imgs.groupby(["split", "modality"])[["file_kb", "brightness", "contrast"]] \
    .agg(["mean", "min", "max"]).round(1)
img_sum.columns = [f"{a}_{b}" for a, b in img_sum.columns]
img_sum.to_csv(TAB / "18_image_stats_summary.csv")
size_counts = imgs.assign(size=imgs.width.astype(str) + "x" + imgs.height.astype(str)) \
    .groupby(["modality", "size"]).size().sort_values(ascending=False).groupby(level=0).head(5)
size_counts.to_csv(TAB / "19_top_image_sizes.csv")
modes = imgs["mode"].value_counts()

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, col, title in zip(axes, ["width", "brightness", "contrast"],
                          ["Chiều rộng ảnh (px)", "Độ sáng trung bình (0–255)", "Độ tương phản (std)"]):
    for mod, c in (("clinical", BLUE), ("dermoscopic", ORANGE)):
        v = imgs.loc[(imgs.split == "train") & (imgs.modality == mod), col]
        ax.hist(v, bins=40, color=c, alpha=0.55, label=mod)
    ax.set_title(title); ax.legend(frameon=False)
axes[0].set_ylabel("Số ảnh (train)")
save(fig, "09_image_properties.png")

fig, ax = plt.subplots(figsize=(7, 5))
for mod, c in (("clinical", BLUE), ("dermoscopic", ORANGE)):
    v = imgs[(imgs.split == "train") & (imgs.modality == mod)]
    ax.scatter(v.width, v.height, s=10, color=c, alpha=0.4, label=mod, edgecolors="none")
ax.set_xlabel("width"); ax.set_ylabel("height"); ax.set_title("Kích thước ảnh (train)")
ax.legend(frameon=False)
save(fig, "10_image_sizes_scatter.png")

br = imgs[imgs.split == "train"].groupby(["label", "modality"]).brightness.mean().unstack().reindex(order)
br.to_csv(TAB / "20_brightness_by_class.csv")

# ---------------------------------------------------------------- 8. sample grid
rng = np.random.default_rng(42)
fig, axes = plt.subplots(len(CLASSES), 6, figsize=(12, 2.1 * len(CLASSES)))
for i, c in enumerate(order):
    ids = train[train.label == c].sample(min(3, (train.label == c).sum()), random_state=42)
    for j in range(3):
        for k, col in enumerate(("clinical_path", "derm_path")):
            ax = axes[i, j * 2 + k]; ax.axis("off")
            if j < len(ids):
                ax.imshow(Image.open(DATA / "train" / "MILK10k_Training_Input" / ids.iloc[j][col]).convert("RGB"))
                if i == 0:
                    ax.set_title("clinical" if k == 0 else "dermoscopic", fontsize=9, color=INK2, fontweight="normal")
    axes[i, 0].text(-0.08, 0.5, c, transform=axes[i, 0].transAxes, ha="right", va="center",
                    fontsize=11, fontweight="bold", color=ORANGE if c in MALIGNANT else INK)
plt.subplots_adjust(wspace=0.04, hspace=0.06)
save(fig, "11_sample_images.png")

# ---------------------------------------------------------------- report
P = lambda x: f"{x:.1f}%"
lines = []
add = lines.append
add("# Báo cáo EDA — MILK10k\n")
add("_Sinh tự động bởi `EDA/eda.py`. Hình trong `figures/`, bảng chi tiết trong `tables/`._\n")
add("## 1. Tổng quan\n")
add("MILK10k là bộ dữ liệu tổn thương da: **mỗi lesion có đúng 2 ảnh** — 1 ảnh lâm sàng (clinical close-up) "
    "và 1 ảnh dermoscopy — kèm metadata (tuổi, giới, skin tone, vị trí) và 7 điểm khái niệm MONET cho mỗi ảnh. "
    "Bài toán: phân loại 11 lớp chẩn đoán (one-hot).\n")
add(md_table(overview, index=False) + "\n")
add(f"- Tổng ảnh: **{len(meta_tr)} train + {len(meta_te)} test**, định dạng JPG, mode ảnh: "
    + ", ".join(f"{k}={v}" for k, v in modes.items()) + ".")
add(f"- `*_combined.csv` là bản 'wide' gộp 2 ảnh/lesion vào 1 hàng (tiền tố `clin_` / `derm_` cho MONET).")
add(f"- `splits/` chứa fold 0 của một cách chia train/val: {len(fold_tr)} / {len(fold_va)} lesion "
    f"({len(fold_va) / len(train) * 100:.0f}% val).\n")
add("## 2. Kiểm tra tính toàn vẹn\n")
add("| Kiểm tra | Kết quả |\n|---|---|")
for k, v in checks.items():
    add(f"| {k} | {'✅ Đạt' if v else '❌ Không đạt'} |")
add("")
add("Loại ảnh và mức chỉnh sửa ảnh:\n")
add(md_table(img_type) + "\n")
add(md_table(manip) + "\n")
add("## 3. Giá trị thiếu\n")
mv = miss[(miss.train_missing > 0) | (miss.test_missing > 0)]
add(md_table(mv) + "\n" if len(mv) else "Không có giá trị thiếu ở metadata và điểm MONET.\n")
add("## 4. Phân bố nhãn\n")
add("![class](figures/01_class_distribution.png)\n")
add(md_table(cls_df.assign(percent=cls_df.percent.round(2))) + "\n")
add(f"- **Mất cân bằng nặng**: lớp lớn nhất `{cls.idxmax()}` ({cls.max()}) gấp **{facts['imbalance_ratio']:.0f}×** "
    f"lớp nhỏ nhất `{cls.idxmin()}` ({cls.min()}).")
add(f"- Tỉ lệ ác tính (AKIEC, BCC, MAL_OTH, MEL, SCCKA): **{P(train.malignant.mean() * 100)}**.")
add("- Fold 0 được chia phân tầng — tỉ lệ từng lớp gần như giống hệt giữa train/val:\n")
add(md_table(fold_cmp.round(2)) + "\n")
add("## 5. Nhân khẩu học & vị trí (train vs test)\n")
add("![demo](figures/02_demographics_train_vs_test.png)\n")
add(md_table(demo_tab.round(1)) + "\n")
add("![site](figures/03_site_distribution.png)\n")
add(md_table(site_tab.iloc[::-1].round(1)) + "\n")
add("## 6. Đặc trưng theo nhãn\n")
add("### Tuổi\n")
add("![age](figures/04_age_by_class.png)\n")
add(md_table(age_by_cls.round(1)) + "\n")
add("### Vị trí\n")
add("![site_cls](figures/05_site_by_class.png)\n")
add("### Skin tone\n")
add("![tone](figures/06_skintone_by_class.png)\n")
add("### Giới tính (% trong mỗi nhãn)\n")
add(md_table(ct_sex.round(1)) + "\n")
add("## 7. Điểm khái niệm MONET\n")
add("MONET là điểm xác suất (0–1) do mô hình MONET ước lượng cho 7 khái niệm trên mỗi ảnh.\n")
add("![monet_c](figures/07_monet_by_class_clin.png)\n")
add("![monet_d](figures/07_monet_by_class_derm.png)\n")
add("![monet_corr](figures/08_monet_correlation.png)\n")
pairs = [(f"clin:{s}", f"derm:{s}") for s in ["ulcer", "hair", "vessels", "erythema", "pigmented", "gel", "pen"]]
add("Tương quan giữa cùng một khái niệm trên ảnh clinical và dermoscopy:\n")
add("| Khái niệm | r(clin, derm) |\n|---|---|")
for a, b in pairs:
    add(f"| {a.split(':')[1]} | {corr.loc[a, b]:.2f} |")
add("")
add("Chênh lệch trung bình MONET test − train (lớn nhất theo trị tuyệt đối):\n")
add(md_table(shift.reindex(shift["diff"].abs().sort_values(ascending=False).index).head(5).round(3)) + "\n")
add("## 8. Thuộc tính ảnh\n")
add("![img](figures/09_image_properties.png)\n")
add("![size](figures/10_image_sizes_scatter.png)\n")
add(md_table(img_sum) + "\n")
add("Kích thước phổ biến nhất:\n")
add(md_table(size_counts.to_frame("count")) + "\n")
add("Độ sáng trung bình theo nhãn:\n")
add(md_table(br.round(1)) + "\n")
add("## 9. Ảnh mẫu\n")
add("![samples](figures/11_sample_images.png)\n")
add("## 10. Nhận xét & khuyến nghị\n")
add("""### Chất lượng dữ liệu
- Dữ liệu **sạch**: nhãn one-hot hợp lệ, không rò rỉ lesion/ảnh giữa train–test và giữa train–val fold 0, đủ 100% ảnh.
- Thiếu rất ít: `age_approx` (20 lesion train, 0.4%) và `site` (31 train / 6 test). Có thể điền median tuổi và thêm hạng mục `unknown` cho site.
- Mọi ảnh đều **600×450 RGB** (đã chuẩn hoá sẵn). Khi resize về ảnh vuông cần xử lý tỉ lệ 4:3 (pad hoặc random-resized-crop).
- ~3% ảnh có `image_manipulation = altered` — có thể dùng làm feature hoặc kiểm tra riêng.

### Phân bố nhãn — thách thức chính
- **BCC chiếm ~48%**; 5 lớp hiếm (DF, INF, VASC, BEN_OTH, MAL_OTH) cộng lại chỉ ~3.9%. `MAL_OTH` chỉ có **9 mẫu** (1 mẫu ở val fold 0) → điểm số val cho các lớp này rất nhiễu.
- Khuyến nghị: dùng **5-fold stratified CV** thay vì chỉ fold 0; weighted / focal loss hoặc logit-adjustment, oversampling lớp hiếm; đánh giá bằng **balanced accuracy / macro-F1 / macro-AUC** thay vì accuracy.

### Metadata mang tín hiệu mạnh
- **Tuổi**: NV trẻ hơn rõ rệt (median 40) so với BCC/SCCKA/AKIEC (65–70). DF, VASC, INF cũng trẻ hơn (~50–55).
- **Vị trí**: AKIEC tập trung ở đầu-mặt-cổ (52%); SCCKA/AKIEC xuất hiện ở bàn tay nhiều hơn hẳn các lớp khác; DF ở chi dưới (65%); NV, MEL, VASC chủ yếu ở thân mình.
- **Giới**: SCCKA, AKIEC, MAL_OTH thiên về nam (≥68%); NV cân bằng.
- **Skin tone**: đa số ở lớp 3–4, lớp 0 gần như không có → mô hình có thể tổng quát hoá kém cho nhóm skin tone hiếm.
- → Nên xây **mô hình đa phương thức** (ảnh + metadata), ví dụ nối embedding ảnh với vector metadata đã mã hoá.

### MONET
- `pigmented` (dermoscopy) cao ở **NV và MEL (~0.62)**, rất thấp ở SCCKA (0.07) → tách tốt nhóm sắc tố / không sắc tố.
- `ulceration_crust` cao nhất ở **SCCKA**; `erythema` cao ở BCC/AKIEC/INF; `vessels` (clinical) cao nhất ở VASC.
- MONET clinical và dermoscopy chỉ tương quan vừa phải (r 0.35–0.81) → hai ảnh mang thông tin bổ sung, nên giữ cả 14 điểm.
- `pen marks` cao ở NV/MEL — đây là **artifact** (vết bút đánh dấu) có nguy cơ gây shortcut learning.

### Hai modality ảnh
- Ảnh dermoscopy sáng và tương phản hơn ảnh clinical, dung lượng file nhỏ hơn → nên normalize/augment riêng cho từng modality, hoặc dùng backbone 2 nhánh rồi fusion.

### Train vs Test
- Tuổi, giới, skin tone tương đồng. Lệch nhẹ: test ít `trunk` hơn (27.8% vs 36.7%), nhiều `hand` hơn (4.6% vs 1.4%); MONET `ulceration` cao hơn và `pigmented` thấp hơn ở test → test **có thể** có nhiều SCCKA/AKIEC hơn và ít NV/MEL hơn train. Cẩn thận khi hiệu chỉnh ngưỡng theo prior của train.
""")
(OUT / "EDA_report.md").write_text("\n".join(lines), encoding="utf-8")
print("Done.")
