<div align="center">
  <h1>🔬 UWE-SkinLesion</h1>
  <p><b>Uncertainty-Weighted Ensemble for Dermoscopic Skin Lesion Classification</b></p>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](#)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)](#)
  [![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](#)
  
  <p align="center">
    A robust, 11-model heterogeneous ensemble leveraging MC Dropout uncertainty weighting to achieve state-of-the-art results in 7-class dermoscopic skin lesion classification.
  </p>
</div>

<hr>

## ✨ Key Highlights
- **Stellar Performance**: Achieved **85.6% balanced accuracy** on HAM10000 (5-fold CV, lesion-ID grouped).
- **Heterogeneous Architecture**: Combines ViTs, modern CNNs, and dual-pathway local/global models.
- **Uncertainty Weighting**: Incorporates MC Dropout to weight predictions based on model confidence.

## 📊 Results Comparison

| Method | Balanced Accuracy | AUC |
| :--- | :---: | :---: |
| Best Single Model (DINOv2 ViT-B) | 80.0 ± 1.3% | 0.941 |
| Simple Average (11 models) | 81.9 ± 1.1% | 0.966 |
| UWE (ours, no val tuning) | 82.4 ± 1.0% | 0.969 |
| **Dirichlet Optimized UWE** | **85.6 ± 1.3%** | **0.973** |

## 🧠 Model Ensemble Architecture

| # | Architecture | Params | Role in Ensemble | Weight (Optimal) |
|---|---|---|---|---|
| 1 | **DINOv2 ViT-B/14** | 86M | Primary foundational model | 34.9% |
| 2 | **DINOv2 ViT-S/14** | 22M | Secondary foundational model | 22.6% |
| 3 | **EfficientNetV2-S** | 20.8M | Advanced sequential CNN | - |
| 4 | **EfficientNet-B4** | 18.5M | Standard baseline CNN | - |
| 5 | **Swin-T** | 27.9M | Hierarchical Vision Transformer | - |
| 6 | **DenseNet-201** | 19.1M | High-density feature CNN | - |
| 7 | **ViT-B/16** | 86M | Supervised Vision Transformer | - |
| 8 | **ConvNeXt-T** | 28.2M | Modern CNN matching ViT scalability | - |
| 9 | **ResNet-50** | 24.6M | Classic robust baseline | - |
| 10| **Global EB4** | 18.5M | Full image context pathway | - |
| 11| **Local R50** | 24.6M | Localized lesion crop pathway | - |

## 🚀 Setup & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/arjunthilak05/UWE-SkinLesion.git
cd UWE-SkinLesion
pip install -r requirements.txt
```

> **Requirements**: PyTorch >= 2.0, `timm`, `segmentation-models-pytorch`, `albumentations`, `scikit-learn`

## ⚙️ Usage

### 1. Training Individual Models

```bash
# Train any standard baseline
python scripts/train_baselines.py --model_name "vit_base_patch14_dinov2.lvd142m" \
    --config configs/default.yaml --fold 0 --device cuda

# Train the global context pathway (using EfficientNet-B4)
python scripts/train_global.py --config configs/default.yaml --fold 0
```

### 2. Dual Pathway Training (Local Cropping)
```bash
# Train the local pathway (U-Net segmentation + ResNet-50 on crops)
python scripts/train_segmentation.py --config configs/default.yaml --fold 0
python scripts/generate_masks.py --config configs/default.yaml
python scripts/train_local.py --config configs/default.yaml --fold 0
```

### 3. Running the Ensemble
```bash
# Evaluate using Uncertainty-Weighted Ensemble (MC Dropout)
python scripts/ensemble/ensemble_uwe.py

# Run optimal Dirichlet weight optimization
python scripts/ensemble/ensemble_optimize.py
```

## 📁 Dataset Details

The models were evaluated and trained using the ISIC archive datasets:
- **HAM10000 (ISIC 2018)**: 10,015 images
- **ISIC 2019**: 14,885 additional images (filtered carefully to the 7 directly matching diagnosis classes)
- **Cumulative Total**: 24,900 images spanning 7 specific classes.
- **Validation Scheme**: 5-fold `StratifiedGroupKFold` grouped uniquely by `lesion_id` to prevent data leakage.

## ✒️ Author
**Arjun Thilak** (@arjunthilak05)

## 📎 Citation

If you utilize this repository for your research, please cite our forthcoming paper:

```bibtex
@article{thilak_uwe_skinlesion,
  title={Uncertainty-Weighted Ensemble for Dermoscopic Skin Lesion Classification},
  author={Thilak, Arjun},
  year={2026},
  note={Citation details pending publication}
}
```
