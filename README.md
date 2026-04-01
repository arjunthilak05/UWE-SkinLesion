# Uncertainty-Weighted Ensemble for Dermoscopic Skin Lesion Classification

11-model heterogeneous ensemble with MC Dropout uncertainty weighting for 7-class dermoscopic skin lesion classification.

**85.6% balanced accuracy** on HAM10000 (5-fold CV, lesion-ID grouped).

## Key Results

| Method | Balanced Accuracy | AUC |
|--------|------------------|-----|
| Best Single (DINOv2 ViT-B) | 80.0 +/- 1.3% | 0.941 |
| Simple Average (11 models) | 81.9 +/- 1.1% | 0.966 |
| UWE (ours, no val tuning) | 82.4 +/- 1.0% | 0.969 |
| **Dirichlet Optimized** | **85.6 +/- 1.3%** | **0.973** |

## Models

| # | Model | Params | Role |
|---|-------|--------|------|
| 1 | DINOv2 ViT-B/14 | 86M | Best single model (34.9% weight) |
| 2 | DINOv2 ViT-S/14 | 22M | 2nd strongest (22.6% weight) |
| 3 | EfficientNetV2-S | 20.8M | Strong CNN |
| 4 | EfficientNet-B4 | 18.5M | Baseline CNN |
| 5 | Swin-T | 27.9M | Hierarchical ViT |
| 6 | DenseNet-201 | 19.1M | Dense CNN |
| 7 | ViT-B/16 | 86M | Supervised ViT |
| 8 | ConvNeXt-T | 28.2M | Modern CNN |
| 9 | ResNet-50 | 24.6M | Classic baseline |
| 10 | Global EB4 | 18.5M | Dual-pathway (full image) |
| 11 | Local R50 | 24.6M | Dual-pathway (lesion crop) |

## Setup

```bash
pip install -r requirements.txt
```

Requires: PyTorch >= 2.0, timm, segmentation-models-pytorch, albumentations, scikit-learn

## Training

```bash
# Train any model
python scripts/train_baselines.py --model_name "vit_base_patch14_dinov2.lvd142m" \
    --config configs/default.yaml --fold 0 --device cuda

# Train global pathway (EfficientNet-B4)
python scripts/train_global.py --config configs/default.yaml --fold 0

# Train local pathway (U-Net segmentation + ResNet-50 on crops)
python scripts/train_segmentation.py --config configs/default.yaml --fold 0
python scripts/generate_masks.py --config configs/default.yaml
python scripts/train_local.py --config configs/default.yaml --fold 0
```

## Ensemble

```bash
# UWE (MC Dropout uncertainty-weighted)
python scripts/ensemble/ensemble_uwe.py

# Dirichlet weight optimization
python scripts/ensemble/ensemble_optimize.py
```

## Data

- HAM10000 (ISIC 2018): 10,015 images
- ISIC 2019: 14,885 additional images (filtered to 7 matching classes)
- Total: 24,900 images, 7 classes
- 5-fold StratifiedGroupKFold by lesion_id

## Citation

If you use this code, please cite our paper (citation pending).
