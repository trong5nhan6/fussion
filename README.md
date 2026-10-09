# MILK10k — Phân loại tổn thương da đa phương thức

Phân loại 11 lớp chẩn đoán cho mỗi lesion, dựa trên **ảnh lâm sàng + ảnh dermoscopy + metadata**
(tuổi, giới, skin tone, vị trí, 14 điểm khái niệm MONET). Phân tích dữ liệu xem trong [`EDA/EDA_report.md`](EDA/EDA_report.md).

> **Trạng thái:** baseline chỉ dùng ảnh (CNN/Transformer), 9 phương pháp xử lý mất cân bằng, mô hình ML trên metadata,
> và **multimodal MoE (ảnh + metadata)** với 4 hướng A/B/C/D.

## Chạy trên Kaggle (khuyến nghị)

Notebook tự clone repo, tải dữ liệu từ Google Drive và train trên **2 GPU T4 song song** (mỗi GPU một run).

| Notebook | Nội dung | Số run | GPU |
|---|---|---|---|
| [`01_baselines`](notebooks/01_baselines.ipynb) | ResNet-152, ConvNeXt-B, ViT-B/16, Swin-B × {clin, derm, clin+derm} | 12 | T4 x2 |
| [`02_imbalance`](notebooks/02_imbalance.ipynb) | Backbone tốt nhất × 9 phương pháp imbalance × 3 nhánh ảnh | 27 / backbone | T4 x2 |
| [`03_ml_models`](notebooks/03_ml_models.ipynb) | 9 mô hình ML trên metadata (LogReg, SVM, KNN, RF, ExtraTrees, HGB, LightGBM, XGBoost, CatBoost) | 9 | Không |
| [`04_results`](notebooks/04_results.ipynb) | Gộp output các notebook thành bảng báo cáo, file nộp tốt nhất, learning curve | — | Không |
| [`05_multimodal_moe`](notebooks/05_multimodal_moe.ipynb) | Multimodal MoE: chọn hướng (A/B/C/D) × loss × backbone × nhánh ảnh | tuỳ chọn (mặc định 24) | T4 x2 |

**Chuẩn bị một lần:**
1. Nén thư mục dữ liệu: `cd datasets && zip -r ../MILK10k.zip MILK10k` (khoảng 360MB). Tải lên Google Drive,
   chia sẻ **Anyone with the link**. Dữ liệu thuộc MILK study team, giấy phép CC-BY-NC 4.0 (chỉ dùng phi thương mại).
2. Push repo này lên GitHub (public).
3. Trên Kaggle: Import notebook → Settings: **Accelerator GPU T4 x2**, **Internet On** → sửa `REPO_URL`, `DRIVE_URL`
   trong ô *Cấu hình* → **Save Version → Save & Run All (Commit)**.

**Chỉnh tham số:** mỗi notebook có ô **THAM SỐ TRAIN** (`PARAMS`) gom các tham số quan trọng của
`configs/default.yaml`: epochs, patience, lr, batch, img_size, folds, TTA, hậu xử lý; notebook 02 có thêm τ, β, γ
và tham số cRT. `None` = giữ giá trị trong config (vài tham số khác nhau theo backbone). Giá trị đặt ở đây có ưu
tiên cao nhất và áp dụng cho mọi run. Ô *Danh sách run* in bảng **cấu hình thực tế** từng backbone để kiểm tra trước
khi train. Khi đổi tham số so với lần chạy trước, đặt `RUN_SUFFIX` (vd `"ep30"`) để run mới không bị bỏ qua do trùng tên.

**Chạy tiếp khi hết 12 giờ:** Add Input → output của version trước → điền thư mục `outputs` của nó vào `RESTORE`
→ Save Version. `run_grid.py` bỏ qua run đã xong, chạy lại run dở dang, và không nhận run mới nếu không kịp
xong trước `TIME_BUDGET_H`. Checkpoint bị xoá sau khi sinh file nộp (giới hạn output 20GB của Kaggle).

Chạy lưới tương tự trên máy riêng:
```bash
python scripts/setup_data.py --drive-url "<link>"                    # hoặc đặt sẵn dữ liệu ở datasets/MILK10k
python scripts/run_grid.py experiments/stage1_baselines.yaml --dry-run   # xem danh sách run
python scripts/run_grid.py experiments/stage1_baselines.yaml --gpus auto
python scripts/summarize.py                                          # -> outputs/summary/summary.md
```

## Cấu trúc

```
notebooks/                  # 01_baselines, 02_imbalance, 03_ml_models, 04_results (Kaggle)
experiments/                # lưới thí nghiệm: stage1_baselines, stage2_imbalance, stage3_moe
splits/folds_5.csv          # 5-fold phân tầng (seed 42), cố định cho mọi máy
configs/
  default.yaml              # cấu hình gốc, các file khác kế thừa bằng `base:`
  moe/                      # A, B, C, D: overlay chọn hướng multimodal MoE
  baselines/                # CNN: resnet50, resnet152, efficientnet_b0, convnext_tiny, convnext_base
                            # Transformer: vit_small, vit_base, swin_tiny, swin_base
  imbalance/                # overlay xử lý mất cân bằng (chồng lên 1 baseline bất kỳ)
  env/kaggle_t4.yaml        # profile Kaggle T4: batch 16, 2 worker/job, xoá checkpoint
src/milk10k/
  config.py                 # YAML có kế thừa + ghi đè --set key=value
  data/
    dataset.py              # MilkDataset: 1 mẫu = 1 lesion {images: {clin, derm}, meta, label}
    metadata.py             # MetadataEncoder -> vector 34 chiều (fit theo fold)
    transforms.py           # resize giữ tỉ lệ + pad vuông, augmentation cơ bản
  models/
    __init__.py             # registry MODELS + build_model()
    image_baseline.py       # backbone timm dùng chung cho các view, nối đặc trưng -> linear
    moe.py                  # multimodal MoE: TokenEncoder, gate theo nguồn, transformer + sparse MoE, head long-tail
  losses.py                 # CE / Focal có trọng số lớp
  metrics.py                # metric giống hệt cách chấm của ISIC (AUC, AP, Acc, Sens, Spec, Dice)
  inference.py              # suy luận test + kiểm tra định dạng file nộp
  profiling.py              # số tham số, GFLOPs, VRAM cho file log
  engine.py                 # train 1 fold, AMP, accumulation, early stopping, TTA
scripts/
  setup_data.py             # tải zip từ Google Drive / dùng thư mục có sẵn, giải nén, kiểm tra
  make_folds.py             # 5-fold phân tầng -> splits/folds_5.csv
  train.py                  # train mô hình ảnh qua các fold -> outputs/<run>/
  run_grid.py               # chạy lưới experiments/*.yaml, song song mỗi GPU 1 run, resume, giới hạn giờ
  train_tabular.py          # 9 mô hình ML trên metadata
  summarize.py              # bảng báo cáo: backbone x view, imbalance x view, ML, top run
  predict.py                # trung bình các fold + TTA -> submission.csv
  evaluate.py               # metric theo lớp + confusion matrix từ OOF
  compare_runs.py           # bảng xếp hạng mọi run -> outputs/leaderboard.csv
  run_baselines.sh          # chạy toàn bộ baseline
  run_ablation_views.sh     # 1 config x {clin, derm, clin+derm}
  run_imbalance.sh          # 1 baseline x mọi overlay trong configs/imbalance/
tests/test_smoke.py         # kiểm tra nhanh mọi baseline, transform, metadata, fold, metric, lưới
```

Mỗi run lưu trong `outputs/<run>/`:

| File | Nội dung |
|---|---|
| `train.log` | Tên model, kiến trúc, số tham số, GFLOPs, thiết bị, phân bố lớp, cấu hình loss/optimizer; mỗi epoch: lr, train loss/acc, val loss/acc, AUC, AP, Acc, Sens, Spec, Dice, thời gian, VRAM; bảng leaderboard theo lớp; thông tin file nộp |
| `fold*/history.csv` | Toàn bộ số liệu theo epoch (để vẽ learning curve) |
| `fold*/best.pt` | Checkpoint tốt nhất theo `train.monitor` (cRT: thêm `best_crt.pt`) |
| `oof.csv` | Xác suất softmax thô trên val |
| `metrics.json` | Metric tổng + theo fold + theo lớp |
| `submission.csv` | **File nộp test**: `lesion_id` + 11 lớp, đã hậu xử lý, đã kiểm tra định dạng |
| `test_probs.csv` | Xác suất softmax thô trên test (dùng để ensemble) |

## Cài đặt

```bash
# PyTorch bản CUDA (máy đang cài bản CPU -> GPU chưa được dùng). GTX 1650 dùng được cu126:
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # phải in True
```

## Sử dụng

```bash
python scripts/make_folds.py                         # chỉ cần chạy 1 lần (đã tạo sẵn)
python -m pytest tests -q                            # kiểm tra nhanh

# Mô hình ML trên metadata (vài chục giây trên CPU)
python scripts/train_tabular.py --model all

# Baseline ảnh (mặc định chỉ fold 0)
python scripts/train.py --config configs/baselines/cnn_efficientnet_b0.yaml
python scripts/train.py --config configs/baselines/vit_small.yaml --set model.views=[derm]

# Ablation nhánh ảnh: clin / derm / clin+derm
bash scripts/run_ablation_views.sh configs/baselines/cnn_efficientnet_b0.yaml

# Đã chọn được method -> chạy đủ 5 fold
python scripts/train.py --config <config> --set train.folds=[0,1,2,3,4]

python scripts/evaluate.py --run outputs/<run>      # bảng theo lớp + confusion matrix
python scripts/predict.py  --run outputs/<run>      # sinh lại file nộp (train.py đã tự làm)
python scripts/compare_runs.py                      # bảng xếp hạng giống leaderboard

# Chạy pipeline với vài batch để kiểm tra (không cần GPU)
python scripts/train.py --config configs/baselines/cnn_resnet50.yaml --set debug=true model.pretrained=false
```

Ghi đè config từ dòng lệnh bằng `--set`: `--set data.img_size=384 train.epochs=30 loss.name=focal`.

## Quy trình thực nghiệm

1. **Giai đoạn chọn method**: mọi run mặc định chỉ train **fold 0** (`train.folds: [0]`), so sánh trên cùng 1.048 lesion val.
2. **Ablation nhánh ảnh**: `model.views` = `[clin]` | `[derm]` | `[clin, derm]`. Tên run tự gắn view
   (ví dụ `vit_small_derm_20261008_...`).
3. **Chốt method**: chạy đủ 5 fold, báo cáo OOF.

`compare_runs.py` nhóm theo cột `val` (`fold0` hoặc `5-fold OOF`). Chỉ so sánh các run cùng nhóm.

> Lưu ý: fold 0 chỉ có 2 MAL_OTH, 9 BEN_OTH, 9 VASC, 10 DF, 10 INF. Đoán đúng hoặc sai 1 mẫu của các lớp này
> làm macro-F1 dao động đáng kể, nên chênh lệch nhỏ (< ~0.02) giữa hai run trên fold 0 chưa đủ kết luận.

## Xử lý mất cân bằng

Overlay trong `configs/imbalance/` được truyền **sau** config baseline, mỗi file chỉ thay đổi 1 yếu tố:

```bash
python scripts/train.py --config configs/baselines/<backbone>.yaml configs/imbalance/logit_adjusted.yaml
bash scripts/run_imbalance.sh configs/baselines/<backbone>.yaml      # chạy tất cả overlay
```

| Overlay | Tag | Kỹ thuật |
|---|---|---|
| `ce_plain` | ce_plain | Đối chứng: CE không bù |
| `ce_sqrt_inv` | ce_sqrtinv | CE trọng số √(1/n) — mặc định của baseline |
| `ce_inv` | ce_inv | CE trọng số 1/n |
| `cb_effective_num` | cb_b0999 | Class-Balanced loss, β=0.999 (Cui 2019) |
| `cb_focal` | cb_focal | Class-Balanced Focal, γ=2 |
| `logit_adjusted` | la_t1 | Logit-adjusted loss, τ=1 (Menon 2021) |
| `sampler_q05` | samp_q05 | Oversampling P(c) ∝ n_c^0.5 |
| `sampler_q1` | samp_q1 | Oversampling cân bằng hoàn toàn |
| `crt` | crt | Decoupling: train tự nhiên → đóng băng → train lại head với sampler cân bằng (Kang 2020) |

Tham số chỉnh qua `--set`, ví dụ `loss.la_tau=1.5`, `loss.cb_beta=0.9999`, `train.sampler_q=0.3`, `crt.epochs=5`.
Run có cRT lưu thêm `fold*/oof_stage1.csv` (kết quả trước cRT) để so sánh trong cùng một lần chạy, và
`predict.py` tự dùng `best_crt.pt`. Nếu vừa dùng sampler vừa reweight, log sẽ cảnh báo bù hai lần.

## Multimodal MoE (`model.name: moe`)

Mỗi lesion → token d = 256: ảnh clinical, ảnh dermoscopy (theo `model.views`), demographics + vị trí (20 chiều),
MONET (14 chiều). Metadata luôn được dùng; encoder ảnh (timm) dùng chung cho 2 ảnh, kèm embedding phân biệt nguồn.

| Hướng (`configs/moe/`) | Trộn token | Head |
|---|---|---|
| **A** | Gate theo nguồn: mỗi nguồn 1 expert MLP, gate theo từng lesion trộn đặc trưng | 1 |
| **B** | Transformer: attention giữa [CLS + token], FFN thay bằng sparse MoE (4 expert, top-2, load-balancing) | 1 |
| **C** | Như B | 3 head long-tail: logit-adjusted τ = 0 / 0.5 / 1, suy luận = trung bình xác suất |
| **D** | Như A | 3 head long-tail |

Chung: head phụ cho từng nguồn (`moe.aux_weight`), modality dropout (`moe.modality_dropout`), train end-to-end.
C/D tự có loss → chỉ kết hợp với overlay sampler. `oof.csv` của run MoE có thêm cột `gate_*` (A/D) hoặc `expert_*`
(B/C); `summarize.py` dùng chúng để in trọng số gate / expert theo lớp.

```bash
python scripts/train.py --config configs/baselines/vit_base.yaml configs/moe/B.yaml configs/imbalance/sampler_q05.yaml
python scripts/run_grid.py experiments/stage3_moe.yaml --backbones vit_base --archs A,B,C,D     --overlays ce_sqrt_inv,sampler_q05 --views clin,derm,clin+derm --dry-run
```

## Đánh giá — giống leaderboard ISIC MILK10k

Metric được tính **đúng như code chấm điểm chính thức** (`isic-challenge-scoring` 5.8.0). Đã kiểm chứng trên cùng
file OOF: sai khác so với code chính thức ≤ 1e-16.

- Mỗi lớp được chấm như bài toán nhị phân, dự đoán dương khi điểm **> 0.5**. Lesion có thể dương ở 0, 1 hoặc nhiều lớp.
- Leaderboard = **trung bình macro trên 11 lớp** của: AUC, Average Precision, Accuracy, Sensitivity, Specificity,
  **Dice Coefficient** (metric xếp hạng = macro F1). Chọn checkpoint theo Dice (`train.monitor: dice`).
- Quy ước ISIC: chỉ số có mẫu số bằng 0 được tính 1.0 (ví dụ một lớp vắng mặt và không bị dự đoán thì Dice = 1).
- Log có thêm `top1_acc` (accuracy đa lớp thông thường) và `balanced_acc` (metric tổng hợp của ISIC).

**Hậu xử lý file nộp** (`predict.postprocess`). Vì ngưỡng 0.5 áp lên từng lớp, nộp softmax thô thì các lesion có xác
suất cao nhất < 0.5 sẽ không có lớp dương nào. `top1` đưa lớp argmax lên 0.5+0.5p (> 0.5) và các lớp khác xuống 0.5p
(< 0.5), nên mỗi lesion có đúng 1 lớp dương. Trên baseline HGB (fold 0), Dice tăng từ 0.239 (softmax) lên **0.298**
(top1). Metric val luôn tính trên điểm **sau** hậu xử lý, nên khớp với điểm leaderboard.

**File nộp** `submission.csv`: cột `lesion_id, AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC`,
479 dòng, giá trị trong [0, 1]. Trang cuộc thi ghi cột đầu là `lesion`, nhưng code chấm điểm chỉ nhận `image` hoặc
`lesion_id`. Nếu hệ thống báo lỗi tên cột, đổi tên cột đầu thành `lesion`.

## Kết quả baseline (fold 0)

| Mô hình | Đầu vào | AUC | AP | Accuracy | Sensitivity | Specificity | **Dice** |
|---|---|---|---|---|---|---|---|
| Logistic Regression | metadata | 0.825 | 0.353 | 0.928 | 0.356 | 0.951 | **0.336** |
| SVM (RBF) | metadata | 0.825 | 0.343 | 0.938 | 0.315 | 0.952 | **0.324** |
| HistGradientBoosting | metadata | 0.832 | 0.323 | 0.934 | 0.296 | 0.952 | **0.298** |
| Extra Trees | metadata | 0.838 | 0.338 | 0.935 | 0.261 | 0.948 | **0.268** |
| Random Forest | metadata | 0.821 | 0.346 | 0.938 | 0.259 | 0.950 | **0.257** |
| KNN (k=25) | metadata | 0.735 | 0.302 | 0.932 | 0.230 | 0.944 | **0.232** |
| CNN / Transformer | ảnh | _chạy notebook 01_ | | | | | |

Mô hình ML dùng trọng số mẫu √balanced. LightGBM, XGBoost, CatBoost chạy trong notebook 03 (Kaggle cài sẵn).

Config trong `configs/baselines/` nhắm GPU 4GB (batch nhỏ + gradient checkpointing). Trên Kaggle, profile
`configs/env/kaggle_t4.yaml` nâng lên batch 16 × accum 2 và tắt checkpointing (ResNet-152 ở 320px dùng batch 8 × 4);
batch hiệu dụng luôn là 32.

## Thêm mô hình mới

1. Viết class trong `src/milk10k/models/`, có `forward(batch)`, `param_groups(lr, mult)`, `normalization()`;
   đặt `uses_metadata = True` nếu dùng `batch["meta"]`.
2. Đăng ký vào `MODELS` trong `src/milk10k/models/__init__.py`.
3. Tạo config với `model.name: <tên>`. Engine, đánh giá và dự đoán dùng lại được nguyên vẹn.
