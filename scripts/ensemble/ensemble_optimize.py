"""Multi-model ensemble optimization with temperature scaling.

Collects logits from all trained models, calibrates each,
and finds optimal ensemble weights via multiple strategies.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path("C:/Users/Hp/Downloads/skin")))

from src.dataset import HAM10000Dataset
from src.models.baseline import BaselineClassifier
from src.models.global_classifier import GlobalClassifier
from src.models.local_classifier import LocalClassifier
from src.models.temperature import TemperatureScaler
from src.transforms import get_val_transforms
from src.utils import get_device, load_config, seed_everything

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def collect_logits(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            logits_list.append(model(images).cpu())
            labels_list.append(batch["label"])
    return torch.cat(logits_list), torch.cat(labels_list)


def compute_metrics(labels, probs):
    preds = probs.argmax(1)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
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

    # Define all models
    MODELS = {
        # v2 models (trained without oversampling, higher individual acc)
        "eb4_v2": {
            "ckpt": "D:/skin_data/checkpoints/global_classifier/best_v2.pt",
            "type": "global", "input_size": 380, "images_dir": images_dir,
        },
        "r50_v2": {
            "ckpt": "D:/skin_data/checkpoints/local_classifier/best_v2.pt",
            "type": "local", "input_size": 224, "images_dir": crops_dir,
        },
        # v3 models (trained with oversampling)
        "eb4_v3": {
            "ckpt": "D:/skin_data/checkpoints/global_classifier/best.pt",
            "type": "global", "input_size": 380, "images_dir": images_dir,
        },
        "r50_v3": {
            "ckpt": "D:/skin_data/checkpoints/local_classifier/best.pt",
            "type": "local", "input_size": 224, "images_dir": crops_dir,
        },
        # New baselines
        "swin_t": {
            "ckpt": "D:/skin_data/checkpoints/baselines/swin_tiny_patch4_window7_224/best.pt",
            "type": "baseline", "model_name": "swin_tiny_patch4_window7_224",
            "input_size": 224, "images_dir": images_dir,
        },
        "convnext_t": {
            "ckpt": "D:/skin_data/checkpoints/baselines/convnext_tiny/best.pt",
            "type": "baseline", "model_name": "convnext_tiny",
            "input_size": 224, "images_dir": images_dir,
        },
        "densenet201": {
            "ckpt": "D:/skin_data/checkpoints/baselines/densenet201/best.pt",
            "type": "baseline", "model_name": "densenet201",
            "input_size": 224, "images_dir": images_dir,
        },
        "vit_b16": {
            "ckpt": "D:/skin_data/checkpoints/baselines/vit_base_patch16_224/best.pt",
            "type": "baseline", "model_name": "vit_base_patch16_224",
            "input_size": 224, "images_dir": images_dir,
        },
        "effv2_s": {
            "ckpt": "D:/skin_data/checkpoints/baselines/tf_efficientnetv2_s/best.pt",
            "type": "baseline", "model_name": "tf_efficientnetv2_s",
            "input_size": 384, "images_dir": images_dir,
        },
    }

    # Collect logits from all models
    all_logits = {}
    labels = None
    bs = 32
    nw = 4

    for name, info in MODELS.items():
        ckpt_path = Path(info["ckpt"])
        if not ckpt_path.exists():
            print(f"[SKIP] {name} — checkpoint not found: {ckpt_path}")
            continue

        print(f"\n[INFO] Loading {name}...")
        input_size = info["input_size"]
        tf = get_val_transforms(input_size, mean, std)

        ds = HAM10000Dataset(
            csv_path=val_csv, images_dir=info["images_dir"],
            transform=tf, img_size=input_size, mode="classification",
            filter_existing=True,
        )
        loader = DataLoader(ds, batch_size=bs, num_workers=nw, pin_memory=True)

        # Load model
        if info["type"] == "global":
            model = GlobalClassifier.from_checkpoint(ckpt_path, cfg["global_classifier"], device=device)
        elif info["type"] == "local":
            model = LocalClassifier.from_checkpoint(ckpt_path, cfg["local_classifier"], device=device)
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            model_cfg = ckpt.get("config", {})
            model = BaselineClassifier(
                model_name=info["model_name"],
                pretrained=False,
                num_classes=7,
                drop_rate=model_cfg.get("drop_rate", 0.3),
                hidden_dim=model_cfg.get("hidden_dim", 512),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)

        logits, lbls = collect_logits(model, loader, device)
        all_logits[name] = logits
        if labels is None:
            labels = lbls
        print(f"  {name}: {logits.shape[0]} samples, logits shape {logits.shape}")

        # Individual accuracy
        probs = F.softmax(logits, dim=1).numpy()
        m = compute_metrics(lbls.numpy(), probs)
        print(f"  {name}: bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

        del model
        torch.cuda.empty_cache()

    # Align all logits to same size
    min_n = min(l.shape[0] for l in all_logits.values())
    for name in all_logits:
        all_logits[name] = all_logits[name][:min_n]
    labels = labels[:min_n]
    labels_np = labels.numpy()

    print(f"\n{'='*60}")
    print(f"ENSEMBLE OPTIMIZATION ({len(all_logits)} models, {min_n} samples)")
    print(f"{'='*60}")

    # Temperature calibrate each model
    print("\n--- Temperature Scaling ---")
    calibrated_probs = {}
    for name, logits in all_logits.items():
        ts = TemperatureScaler(init_temperature=1.5)
        result = ts.fit(logits, labels)
        with torch.no_grad():
            cal_logits = ts(logits)
            probs = F.softmax(cal_logits, dim=1).numpy()
        calibrated_probs[name] = probs
        print(f"  {name}: T={result['temperature']:.3f}, ECE {result['ece_before']:.4f} -> {result['ece_after']:.4f}")

    model_names = list(calibrated_probs.keys())
    prob_stack = np.stack([calibrated_probs[n] for n in model_names])  # (M, N, C)

    # Strategy 1: Simple average
    avg_probs = prob_stack.mean(axis=0)
    m = compute_metrics(labels_np, avg_probs)
    print(f"\n--- Simple Average ---")
    print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

    # Strategy 2: Dirichlet weight search
    print(f"\n--- Dirichlet Weight Search (10K samples) ---")
    n_models = len(model_names)
    best_bacc, best_weights = 0, None
    np.random.seed(42)
    for _ in range(10000):
        w = np.random.dirichlet(np.ones(n_models))
        ens_probs = np.tensordot(w, prob_stack, axes=([0], [0]))
        bacc = balanced_accuracy_score(labels_np, ens_probs.argmax(1))
        if bacc > best_bacc:
            best_bacc = bacc
            best_weights = w

    m = compute_metrics(labels_np, np.tensordot(best_weights, prob_stack, axes=([0], [0])))
    print(f"  Best weights:")
    for n, w in zip(model_names, best_weights):
        print(f"    {n:20s}: {w:.4f}")
    print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

    # Strategy 3: Stacking with LogisticRegression
    print(f"\n--- Stacking (Logistic Regression) ---")
    X = np.concatenate([calibrated_probs[n] for n in model_names], axis=1)  # (N, M*C)
    meta = LogisticRegression(C=1.0, multi_class="multinomial", max_iter=2000, solver="lbfgs")
    meta.fit(X, labels_np)
    stack_probs = meta.predict_proba(X)
    m = compute_metrics(labels_np, stack_probs)
    print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

    # Strategy 4: Top-K models only
    print(f"\n--- Top-5 Models Average ---")
    individual_accs = {}
    for name in model_names:
        bacc = balanced_accuracy_score(labels_np, calibrated_probs[name].argmax(1))
        individual_accs[name] = bacc
    top5 = sorted(individual_accs, key=individual_accs.get, reverse=True)[:5]
    top5_probs = np.stack([calibrated_probs[n] for n in top5]).mean(axis=0)
    m = compute_metrics(labels_np, top5_probs)
    print(f"  Models: {top5}")
    print(f"  bal_acc={m['bal_acc']:.4f}, f1={m['macro_f1']:.4f}, auc={m['macro_auc']:.4f}")

    # Save results
    results = {
        "models": model_names,
        "individual_bal_acc": {n: float(individual_accs[n]) for n in model_names},
        "simple_average": float(balanced_accuracy_score(labels_np, avg_probs.argmax(1))),
        "dirichlet_best": {
            "bal_acc": float(best_bacc),
            "weights": {n: float(w) for n, w in zip(model_names, best_weights)},
        },
        "stacking": float(balanced_accuracy_score(labels_np, stack_probs.argmax(1))),
    }
    with open("D:/skin_data/results/ensemble_optimization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Results saved to D:/skin_data/results/ensemble_optimization.json")


if __name__ == "__main__":
    main()
