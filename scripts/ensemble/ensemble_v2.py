"""Enhanced multi-model ensemble with metadata fusion and stacking.

Combines 9 image models + metadata features via stacking meta-learner.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
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
from src.transforms import get_val_transforms, get_tta_transforms
from src.utils import get_device, load_config, seed_everything

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def collect_logits(model, loader, device):
    model.eval()
    logits_list, labels_list, meta_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            logits_list.append(model(images).cpu())
            labels_list.append(batch["label"])
            # Collect metadata (DataLoader collates dicts into lists/tensors)
            m = batch["metadata"]
            batch_size = images.size(0)
            for i in range(batch_size):
                meta_dict = {}
                for k, v in m.items():
                    if isinstance(v, (list, tuple)):
                        meta_dict[k] = v[i]
                    elif hasattr(v, '__getitem__') and hasattr(v, 'shape'):
                        meta_dict[k] = v[i]
                    else:
                        meta_dict[k] = v
                meta_list.append(meta_dict)
    return torch.cat(logits_list), torch.cat(labels_list), meta_list


def collect_logits_simple(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            logits_list.append(model(images).cpu())
            labels_list.append(batch["label"])
    return torch.cat(logits_list), torch.cat(labels_list)


def compute_metrics(labels, probs):
    preds = probs.argmax(1) if probs.ndim > 1 else probs
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        if probs.ndim > 1:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        else:
            auc = 0.0
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

    # Step 1: Collect logits + metadata from first model
    all_logits = {}
    labels = None
    metadata_features = None
    bs, nw = 32, 4

    first_model = True
    for name, info in MODELS.items():
        ckpt_path = Path(info["ckpt"])
        if not ckpt_path.exists():
            print(f"[SKIP] {name}")
            continue

        print(f"[{name}] Loading...")
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
            model = BaselineClassifier(
                model_name=info["model_name"], pretrained=False, num_classes=7,
                drop_rate=model_cfg.get("drop_rate", 0.3),
                hidden_dim=model_cfg.get("hidden_dim", 512),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)

        if first_model and info["type"] == "global":
            logits, lbls, meta_list = collect_logits(model, loader, device)
            # Encode metadata
            meta_feats = []
            for m in meta_list:
                meta_feats.append(encode_metadata(m.get("age"), m.get("sex"), m.get("localization")))
            metadata_features = np.stack(meta_feats)
            print(f"  Metadata features: {metadata_features.shape}")
            first_model = False
        else:
            logits, lbls = collect_logits_simple(model, loader, device)

        all_logits[name] = logits
        if labels is None:
            labels = lbls

        probs = F.softmax(logits, dim=1).numpy()
        m = compute_metrics(lbls.numpy(), probs)
        print(f"  {name}: bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

        del model
        torch.cuda.empty_cache()

    # Align sizes
    min_n = min(l.shape[0] for l in all_logits.values())
    for name in all_logits:
        all_logits[name] = all_logits[name][:min_n]
    labels = labels[:min_n]
    labels_np = labels.numpy()
    if metadata_features is not None:
        metadata_features = metadata_features[:min_n]

    print(f"\n{'='*60}")
    print(f"ENSEMBLE + METADATA OPTIMIZATION ({len(all_logits)} models, {min_n} samples)")
    print(f"{'='*60}")

    # Temperature calibrate
    calibrated_probs = {}
    for name, logits in all_logits.items():
        ts = TemperatureScaler(init_temperature=1.5)
        ts.fit(logits, labels)
        with torch.no_grad():
            probs = F.softmax(ts(logits), dim=1).numpy()
        calibrated_probs[name] = probs

    model_names = list(calibrated_probs.keys())
    prob_stack = np.stack([calibrated_probs[n] for n in model_names])

    # Strategy 1: Simple average
    avg_probs = prob_stack.mean(axis=0)
    m = compute_metrics(labels_np, avg_probs)
    print(f"\n--- Simple Average (9 models) ---")
    print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

    # Strategy 2: Dirichlet search
    best_bacc, best_weights = 0, None
    np.random.seed(42)
    for _ in range(20000):
        w = np.random.dirichlet(np.ones(len(model_names)))
        ens_probs = np.tensordot(w, prob_stack, axes=([0], [0]))
        bacc = balanced_accuracy_score(labels_np, ens_probs.argmax(1))
        if bacc > best_bacc:
            best_bacc = bacc
            best_weights = w

    print(f"\n--- Dirichlet Search (20K samples) ---")
    print(f"  bal_acc={best_bacc:.4f}")
    for n, w in zip(model_names, best_weights):
        if w > 0.01:
            print(f"    {n:20s}: {w:.4f}")

    # Strategy 3: Stacking with image probs only
    print(f"\n--- Stacking (Image Probs Only) ---")
    X_img = np.concatenate([calibrated_probs[n] for n in model_names], axis=1)
    meta_lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    meta_lr.fit(X_img, labels_np)
    stack_probs = meta_lr.predict_proba(X_img)
    m = compute_metrics(labels_np, stack_probs)
    print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

    # Strategy 4: Stacking with image probs + metadata
    if metadata_features is not None:
        print(f"\n--- Stacking (Image Probs + Metadata) ---")
        scaler = StandardScaler()
        meta_scaled = scaler.fit_transform(metadata_features)
        X_full = np.concatenate([X_img, meta_scaled], axis=1)
        meta_lr2 = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        meta_lr2.fit(X_full, labels_np)
        stack_meta_probs = meta_lr2.predict_proba(X_full)
        m = compute_metrics(labels_np, stack_meta_probs)
        print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

        # Strategy 5: Gradient Boosting stacking with metadata
        print(f"\n--- GBM Stacking (Image Probs + Metadata) ---")
        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        gbm.fit(X_full, labels_np)
        gbm_preds = gbm.predict(X_full)
        m_gbm = compute_metrics(labels_np, gbm_preds)
        print(f"  bal_acc={m_gbm['bal_acc']:.4f}, f1={m_gbm['macro_f1']:.4f}")

        # Strategy 6: Metadata-only baseline
        print(f"\n--- Metadata Only (LR) ---")
        meta_only_lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        meta_only_lr.fit(meta_scaled, labels_np)
        meta_only_probs = meta_only_lr.predict_proba(meta_scaled)
        m_meta = compute_metrics(labels_np, meta_only_probs)
        print(f"  bal_acc={m_meta['bal_acc']:.4f}, f1={m_meta['macro_f1']:.4f}, auc={m_meta['macro_auc']:.4f}")

    # Save results
    results = {
        "n_models": len(model_names),
        "n_samples": int(min_n),
        "simple_average": float(balanced_accuracy_score(labels_np, avg_probs.argmax(1))),
        "dirichlet_best": float(best_bacc),
        "stacking_img_only": float(balanced_accuracy_score(labels_np, stack_probs.argmax(1))),
    }
    if metadata_features is not None:
        results["stacking_img_meta"] = float(balanced_accuracy_score(labels_np, stack_meta_probs.argmax(1)))
        results["metadata_only"] = float(balanced_accuracy_score(labels_np, meta_only_probs.argmax(1)))

    with open("D:/skin_data/results/ensemble_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Results saved to D:/skin_data/results/ensemble_v2_results.json")


if __name__ == "__main__":
    main()
