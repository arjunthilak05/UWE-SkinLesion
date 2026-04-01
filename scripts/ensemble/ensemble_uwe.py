"""Uncertainty-Weighted Ensemble (UWE) with MC Dropout.

Novel contribution: Per-sample, per-model uncertainty weighting using
Monte Carlo Dropout. Models that are more certain about a specific sample
get higher weight for that sample. Combined with metadata stacking.

Key idea:
  1. For each model, enable dropout at test time
  2. Run T=20 forward passes per sample
  3. Compute mean prediction and prediction variance (epistemic uncertainty)
  4. Weight = 1 / (variance + epsilon) for each model per sample
  5. Normalize weights across models
  6. Feed uncertainty-weighted probabilities + metadata into stacking meta-learner
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path("C:/Users/Hp/Downloads/skin")))
sys.path.insert(0, str(Path("D:/skin_data/scripts")))

from add_metadata_features import encode_metadata
from src.dataset import HAM10000Dataset
from src.models.baseline import BaselineClassifier
from src.models.global_classifier import GlobalClassifier
from src.models.local_classifier import LocalClassifier
from src.models.temperature import TemperatureScaler
from src.transforms import get_val_transforms
from src.utils import get_device, load_config, seed_everything


def enable_mc_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model, images, T=20):
    """Run T stochastic forward passes and return mean probs + uncertainty.

    Args:
        model: Model with dropout layers
        images: Input tensor (B, C, H, W)
        T: Number of MC samples

    Returns:
        mean_probs: (B, num_classes) - mean prediction across T passes
        uncertainty: (B,) - prediction variance (epistemic uncertainty)
    """
    model.eval()
    enable_mc_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())

    # Stack: (T, B, C)
    stacked = torch.stack(all_probs)
    mean_probs = stacked.mean(dim=0)  # (B, C)

    # Epistemic uncertainty: variance of predicted class probabilities
    # Use predictive entropy or variance of max prob
    variance = stacked.var(dim=0).mean(dim=1)  # (B,) mean variance across classes

    model.eval()  # restore full eval mode
    return mean_probs, variance


def collect_mc_logits(model, loader, device, T=20):
    """Collect MC Dropout predictions from a DataLoader."""
    all_mean_probs = []
    all_uncertainties = []
    all_labels = []
    all_metadata = []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        mean_probs, uncertainty = mc_dropout_predict(model, images, T=T)
        all_mean_probs.append(mean_probs)
        all_uncertainties.append(uncertainty)
        all_labels.append(batch["label"])

        # Collect metadata
        m = batch.get("metadata", {})
        bs = images.size(0)
        for i in range(bs):
            meta_dict = {}
            for k, v in m.items():
                if isinstance(v, (list, tuple)):
                    meta_dict[k] = v[i]
                elif hasattr(v, '__getitem__') and hasattr(v, 'shape'):
                    meta_dict[k] = v[i]
                else:
                    meta_dict[k] = v
            all_metadata.append(meta_dict)

    return (torch.cat(all_mean_probs),
            torch.cat(all_uncertainties),
            torch.cat(all_labels),
            all_metadata)


def compute_metrics(labels, probs):
    preds = probs.argmax(1) if probs.ndim > 1 else probs
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro") if probs.ndim > 1 else 0.0
    except:
        auc = 0.0
    return {"bal_acc": bal_acc, "macro_f1": f1, "macro_auc": auc}


def main():
    seed_everything(42)
    device = get_device("cuda")
    cfg = load_config(Path("C:/Users/Hp/Downloads/skin/configs/default.yaml"))

    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    norm = aug_cfg.get("normalize", {})
    mean = tuple(norm.get("mean", [0.7635, 0.5461, 0.5705]))
    std = tuple(norm.get("std", [0.1404, 0.1521, 0.1697]))

    data_dir = Path(data_cfg["data_dir"])
    images_dir = data_dir / data_cfg["images_subdir"]
    crops_dir = data_dir / "cropped_lesions"
    val_csv = Path(data_cfg.get("splits_dir", "data/splits")) / "fold0_val.csv"

    T_MC = 20  # MC Dropout samples
    EPSILON = 1e-8

    MODELS = {
        "eb4_v2": {"ckpt": "D:/skin_data/checkpoints/global_classifier/best_v2.pt",
                    "type": "global", "input_size": 380, "images_dir": images_dir},
        "r50_v2": {"ckpt": "D:/skin_data/checkpoints/local_classifier/best_v2.pt",
                    "type": "local", "input_size": 224, "images_dir": crops_dir},
        "eb4_v3": {"ckpt": "D:/skin_data/checkpoints/global_classifier/best.pt",
                    "type": "global", "input_size": 380, "images_dir": images_dir},
        "r50_v3": {"ckpt": "D:/skin_data/checkpoints/local_classifier/best.pt",
                    "type": "local", "input_size": 224, "images_dir": crops_dir},
        "swin_t": {"ckpt": "D:/skin_data/checkpoints/baselines/swin_tiny_patch4_window7_224/best.pt",
                    "type": "baseline", "model_name": "swin_tiny_patch4_window7_224",
                    "input_size": 224, "images_dir": images_dir},
        "convnext_t": {"ckpt": "D:/skin_data/checkpoints/baselines/convnext_tiny/best.pt",
                       "type": "baseline", "model_name": "convnext_tiny",
                       "input_size": 224, "images_dir": images_dir},
        "densenet201": {"ckpt": "D:/skin_data/checkpoints/baselines/densenet201/best.pt",
                        "type": "baseline", "model_name": "densenet201",
                        "input_size": 224, "images_dir": images_dir},
        "vit_b16": {"ckpt": "D:/skin_data/checkpoints/baselines/vit_base_patch16_224/best.pt",
                    "type": "baseline", "model_name": "vit_base_patch16_224",
                    "input_size": 224, "images_dir": images_dir},
        "effv2_s": {"ckpt": "D:/skin_data/checkpoints/baselines/tf_efficientnetv2_s/best.pt",
                    "type": "baseline", "model_name": "tf_efficientnetv2_s",
                    "input_size": 384, "images_dir": images_dir},
    }

    # Check if DINOv2 checkpoint exists
    dinov2_ckpt = Path("D:/skin_data/checkpoints/baselines/vit_small_patch14_dinov2.lvd142m/best.pt")
    if dinov2_ckpt.exists():
        MODELS["dinov2_s"] = {
            "ckpt": str(dinov2_ckpt),
            "type": "baseline", "model_name": "vit_small_patch14_dinov2.lvd142m",
            "input_size": 518, "images_dir": images_dir,
        }

    bs, nw = 32, 4

    # Collect MC Dropout predictions from all models
    all_mean_probs = {}
    all_uncertainties = {}
    labels = None
    metadata_features = None
    first_global = True

    for name, info in MODELS.items():
        ckpt_path = Path(info["ckpt"])
        if not ckpt_path.exists():
            print(f"[SKIP] {name}")
            continue

        print(f"\n[{name}] MC Dropout inference (T={T_MC})...")
        tf = get_val_transforms(info["input_size"], mean, std)
        ds = HAM10000Dataset(
            csv_path=val_csv, images_dir=info["images_dir"],
            transform=tf, img_size=info["input_size"], mode="classification",
            filter_existing=True,
        )
        loader = DataLoader(ds, batch_size=bs, num_workers=nw, pin_memory=True)

        if info["type"] == "global":
            model = GlobalClassifier.from_checkpoint(ckpt_path, cfg["global_classifier"], device=device)
        elif info["type"] == "local":
            model = LocalClassifier.from_checkpoint(ckpt_path, cfg["local_classifier"], device=device)
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            model_cfg = ckpt.get("config", {})
            cur_bs = 16 if info["input_size"] > 400 else bs
            model = BaselineClassifier(
                model_name=info["model_name"], pretrained=False, num_classes=7,
                drop_rate=model_cfg.get("drop_rate", 0.3),
                hidden_dim=model_cfg.get("hidden_dim", 512),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            if info["input_size"] > 400:
                loader = DataLoader(ds, batch_size=16, num_workers=nw, pin_memory=True)

        mean_probs, uncertainty, lbls, meta_list = collect_mc_logits(model, loader, device, T=T_MC)

        all_mean_probs[name] = mean_probs.numpy()
        all_uncertainties[name] = uncertainty.numpy()

        if labels is None:
            labels = lbls.numpy()

        if first_global and info["type"] == "global":
            meta_feats = []
            for m in meta_list:
                meta_feats.append(encode_metadata(m.get("age"), m.get("sex"), m.get("localization")))
            metadata_features = np.stack(meta_feats)
            first_global = False

        # Report individual metrics
        m = compute_metrics(lbls.numpy(), mean_probs.numpy())
        mean_unc = uncertainty.mean().item()
        print(f"  {name}: bal_acc={m['bal_acc']:.4f}, mean_uncertainty={mean_unc:.6f}")

        del model
        torch.cuda.empty_cache()

    # Align sizes
    model_names = list(all_mean_probs.keys())
    min_n = min(all_mean_probs[n].shape[0] for n in model_names)
    for n in model_names:
        all_mean_probs[n] = all_mean_probs[n][:min_n]
        all_uncertainties[n] = all_uncertainties[n][:min_n]
    labels = labels[:min_n]
    if metadata_features is not None:
        metadata_features = metadata_features[:min_n]

    print(f"\n{'='*60}")
    print(f"UNCERTAINTY-WEIGHTED ENSEMBLE (UWE)")
    print(f"{len(model_names)} models, {min_n} samples, T={T_MC} MC passes")
    print(f"{'='*60}")

    # === Strategy 1: Simple average (baseline) ===
    prob_stack = np.stack([all_mean_probs[n] for n in model_names])
    avg_probs = prob_stack.mean(axis=0)
    m = compute_metrics(labels, avg_probs)
    print(f"\n--- Simple Average ---")
    print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

    # === Strategy 2: Uncertainty-Weighted Ensemble (UWE) ===
    # Weight = 1 / (uncertainty + epsilon), per sample per model
    print(f"\n--- Uncertainty-Weighted Ensemble (UWE) ---")
    n_models = len(model_names)
    n_samples = min_n
    n_classes = 7

    # Build per-sample weights: (n_models, n_samples)
    unc_matrix = np.stack([all_uncertainties[n] for n in model_names])  # (M, N)
    inv_unc = 1.0 / (unc_matrix + EPSILON)  # (M, N)
    weights = inv_unc / inv_unc.sum(axis=0, keepdims=True)  # normalize per sample

    # Weighted combination
    uwe_probs = np.zeros((n_samples, n_classes))
    for i, name in enumerate(model_names):
        uwe_probs += weights[i, :, np.newaxis] * all_mean_probs[name]

    m = compute_metrics(labels, uwe_probs)
    print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

    # Report per-model mean weight
    for i, name in enumerate(model_names):
        print(f"    {name:20s}: mean_weight={weights[i].mean():.4f}, mean_unc={unc_matrix[i].mean():.6f}")

    # === Strategy 3: UWE + Metadata Stacking ===
    print(f"\n--- UWE + Metadata Stacking ---")
    # Stack UWE probs (already uncertainty-weighted) + raw probs + uncertainty features + metadata
    X_uwe = uwe_probs  # (N, 7) - uncertainty-weighted ensemble probs
    X_raw = np.concatenate([all_mean_probs[n] for n in model_names], axis=1)  # (N, M*7)
    X_unc = np.stack([all_uncertainties[n] for n in model_names], axis=1)  # (N, M) uncertainty features

    if metadata_features is not None:
        scaler = StandardScaler()
        meta_scaled = scaler.fit_transform(metadata_features)
        X_full = np.concatenate([X_uwe, X_raw, X_unc, meta_scaled], axis=1)
        feature_desc = f"UWE({n_classes}) + raw({n_models*n_classes}) + unc({n_models}) + meta(17) = {X_full.shape[1]}"
    else:
        X_full = np.concatenate([X_uwe, X_raw, X_unc], axis=1)
        feature_desc = f"UWE({n_classes}) + raw({n_models*n_classes}) + unc({n_models}) = {X_full.shape[1]}"

    print(f"  Features: {feature_desc}")

    meta_lr = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
    meta_lr.fit(X_full, labels)
    stack_probs = meta_lr.predict_proba(X_full)
    m = compute_metrics(labels, stack_probs)
    print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

    # === Strategy 4: Previous best (raw probs + metadata stacking, no UWE) ===
    print(f"\n--- Previous Best (raw stacking + metadata, no UWE) ---")
    if metadata_features is not None:
        X_prev = np.concatenate([X_raw, meta_scaled], axis=1)
    else:
        X_prev = X_raw
    prev_lr = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
    prev_lr.fit(X_prev, labels)
    prev_probs = prev_lr.predict_proba(X_prev)
    m_prev = compute_metrics(labels, prev_probs)
    print(f"  bal_acc={m_prev['bal_acc']:.4f}, f1={m_prev['macro_f1']:.4f}, auc={m_prev['macro_auc']:.4f}")

    # === Per-class analysis for UWE ===
    print(f"\n--- Per-Class UWE Analysis ---")
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    uwe_preds = uwe_probs.argmax(1)
    for c_idx, c_name in enumerate(classes):
        mask = labels == c_idx
        if mask.sum() == 0:
            continue
        c_unc = np.stack([all_uncertainties[n][mask] for n in model_names])  # (M, n_c)
        c_acc = (uwe_preds[mask] == c_idx).mean()
        c_mean_unc = c_unc.mean()
        print(f"  {c_name:8s}: n={mask.sum():4d}, accuracy={c_acc:.3f}, mean_uncertainty={c_mean_unc:.6f}")

    # Save results
    results = {
        "n_models": len(model_names),
        "models": model_names,
        "T_mc": T_MC,
        "n_samples": int(min_n),
        "simple_average": float(balanced_accuracy_score(labels, avg_probs.argmax(1))),
        "uwe_only": float(balanced_accuracy_score(labels, uwe_probs.argmax(1))),
        "uwe_stacking_metadata": float(balanced_accuracy_score(labels, stack_probs.argmax(1))),
        "previous_best": float(balanced_accuracy_score(labels, prev_probs.argmax(1))),
        "per_model_mean_uncertainty": {n: float(all_uncertainties[n].mean()) for n in model_names},
    }
    with open("D:/skin_data/results/ensemble_uwe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  Simple average:              {results['simple_average']*100:.2f}%")
    print(f"  UWE only:                    {results['uwe_only']*100:.2f}%")
    print(f"  Previous (raw+meta stack):   {results['previous_best']*100:.2f}%")
    print(f"  UWE + meta stacking:         {results['uwe_stacking_metadata']*100:.2f}%")
    print(f"\n[DONE] Results saved to D:/skin_data/results/ensemble_uwe_results.json")


if __name__ == "__main__":
    main()
