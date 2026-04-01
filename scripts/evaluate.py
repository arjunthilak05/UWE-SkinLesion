"""Comprehensive evaluation of the dual-pathway system (Stage 6).

Runs ablation studies, per-class metrics, optional k-fold CV, and TTA.

Usage::

    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --config configs/default.yaml --folds 0 1 2 3 4
    python scripts/evaluate.py --config configs/default.yaml --tta
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HAM10000Dataset
from src.metrics import balanced_accuracy, macro_auc, macro_f1, per_class_auc
from src.models.gating import ConfidenceEnsemble
from src.models.global_classifier import GlobalClassifier
from src.models.local_classifier import LocalClassifier
from src.models.temperature import TemperatureScaler
from src.transforms import get_val_transforms, get_tta_transforms
from src.utils import get_device, load_config, seed_everything

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Evaluate dual-pathway system.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--folds", type=int, nargs="+", default=[0], help="Fold indices to evaluate.")
    p.add_argument("--smoke_test", type=int, default=0)
    p.add_argument("--tta", action="store_true", help="Enable 10x TTA.")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


# =========================================================================
# Logit collection
# =========================================================================


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and labels from a DataLoader."""
    model.eval()
    logits_list, labels_list = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        logits_list.append(model(images).cpu())
        labels_list.append(batch["label"])
    return torch.cat(logits_list), torch.cat(labels_list)


@torch.no_grad()
def collect_logits_tta(
    model: torch.nn.Module,
    csv_path: Path,
    images_dir: Path,
    img_size: int,
    mean: tuple,
    std: tuple,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = "classification",
    filter_existing: bool = False,
    desc: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect averaged logits over 10 TTA transforms."""
    model.eval()
    tta_transforms = get_tta_transforms(img_size, mean, std)
    all_logits = None
    labels = None

    for t_idx, tta_tf in enumerate(tta_transforms):
        ds = HAM10000Dataset(
            csv_path=csv_path, images_dir=images_dir,
            transform=tta_tf, img_size=img_size, mode=mode,
            filter_existing=filter_existing,
        )
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        logits_list, labels_list = [], []
        for batch in tqdm(loader, desc=f"{desc} TTA {t_idx+1}/{len(tta_transforms)}", leave=False):
            images = batch["image"].to(device)
            logits_list.append(model(images).cpu())
            labels_list.append(batch["label"])

        fold_logits = torch.cat(logits_list)
        if all_logits is None:
            all_logits = fold_logits
            labels = torch.cat(labels_list)
        else:
            all_logits = all_logits + fold_logits

    all_logits = all_logits / len(tta_transforms)
    return all_logits, labels


# =========================================================================
# Metric computation
# =========================================================================


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics."""
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_acc = float(np.mean(y_pred == y_true))

    prec, rec, f1_arr, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    specificity = []
    for c in range(NUM_CLASSES):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        specificity.append(float(tn / max(tn + fp, 1)))

    pc_auc = []
    for c in range(NUM_CLASSES):
        binary = (y_true == c).astype(int)
        if 0 < binary.sum() < len(binary):
            try:
                pc_auc.append(float(roc_auc_score(binary, y_probs[:, c])))
            except ValueError:
                pc_auc.append(0.0)
        else:
            pc_auc.append(0.0)
    valid = [a for a in pc_auc if a > 0]
    m_auc = float(np.mean(valid)) if valid else 0.0

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(prec[i]), "recall": float(rec[i]),
            "specificity": float(specificity[i]), "f1": float(f1_arr[i]),
            "auc": float(pc_auc[i]), "support": int(sup[i]),
        }

    return {
        "balanced_accuracy": bal_acc, "macro_f1": mf1, "macro_auc": m_auc,
        "weighted_accuracy": weighted_acc, "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


# =========================================================================
# Ablation
# =========================================================================


def run_ablation(
    p_global: np.ndarray,
    p_local: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict]:
    """Run all ablation configurations."""
    results = {}
    p_g_t = torch.from_numpy(p_global)
    p_l_t = torch.from_numpy(p_local)

    # 1. Global only
    results["global_only"] = compute_full_metrics(labels, p_global.argmax(1), p_global)

    # 2. Local only
    results["local_only"] = compute_full_metrics(labels, p_local.argmax(1), p_local)

    # 3. Naive average
    p_avg = 0.5 * p_global + 0.5 * p_local
    results["naive_average"] = compute_full_metrics(labels, p_avg.argmax(1), p_avg)

    # 4. Optimized fixed-weight
    best_w, best_ba = 0.5, 0.0
    for w in np.arange(0.0, 1.05, 0.05):
        p_mix = w * p_global + (1 - w) * p_local
        ba = float(balanced_accuracy_score(labels, p_mix.argmax(1)))
        if ba > best_ba:
            best_ba, best_w = ba, w
    p_fixed = best_w * p_global + (1 - best_w) * p_local
    results["fixed_weight"] = compute_full_metrics(labels, p_fixed.argmax(1), p_fixed)
    results["fixed_weight"]["best_w_global"] = float(best_w)

    # 5. Confidence ensemble
    best_tau, best_ba_conf = 2.0, 0.0
    for tau in np.arange(0.5, 5.25, 0.25):
        ens = ConfidenceEnsemble(tau=float(tau))
        out = ens(p_g_t, p_l_t)
        ba = float(balanced_accuracy_score(labels, out["p_final"].numpy().argmax(1)))
        if ba > best_ba_conf:
            best_ba_conf, best_tau = ba, float(tau)

    ens = ConfidenceEnsemble(tau=best_tau)
    out = ens(p_g_t, p_l_t)
    p_conf = out["p_final"].numpy()
    results["confidence_ensemble"] = compute_full_metrics(labels, p_conf.argmax(1), p_conf)
    results["confidence_ensemble"]["best_tau"] = best_tau
    results["confidence_ensemble"]["mean_w_global"] = float(out["w_global"].mean())

    return results


# =========================================================================
# Single fold evaluation
# =========================================================================


def evaluate_fold(
    fold: int, cfg: dict, device: torch.device, smoke_test: int = 0, use_tta: bool = False,
) -> dict[str, Any]:
    """Evaluate a single fold."""
    global_cfg = cfg["global_classifier"]
    local_cfg = cfg["local_classifier"]
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg.get("training", {})

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    global_ckpt = ckpt_dir / "global_classifier" / "best.pt"
    local_ckpt = ckpt_dir / "local_classifier" / "best.pt"
    cal_path = ckpt_dir / "calibration_state.pt"

    data_dir = Path(data_cfg["data_dir"])
    images_dir = data_dir / data_cfg["images_subdir"]
    crops_dir = data_dir / "cropped_lesions"
    val_csv = Path(data_cfg.get("splits_dir", "data/splits")) / f"fold{fold}_val.csv"

    for name, path in [("Global", global_ckpt), ("Local", local_ckpt), ("Val CSV", val_csv)]:
        if not path.exists():
            print(f"  [WARN] {name} not found: {path} — using random predictions")
            return _random_results(smoke_test)

    norm = aug_cfg.get("normalize", {})
    mean = tuple(norm.get("mean", [0.7635, 0.5461, 0.5705]))
    std = tuple(norm.get("std", [0.1404, 0.1521, 0.1697]))

    global_ds = HAM10000Dataset(
        csv_path=val_csv, images_dir=images_dir,
        transform=get_val_transforms(global_cfg.get("input_size", 380), mean, std),
        img_size=global_cfg.get("input_size", 380), mode="classification",
    )
    local_ds = HAM10000Dataset(
        csv_path=val_csv, images_dir=crops_dir,
        transform=get_val_transforms(local_cfg.get("input_size", 224), mean, std),
        img_size=local_cfg.get("input_size", 224), mode="classification",
        filter_existing=True,
    )

    if smoke_test > 0:
        n = min(smoke_test, len(global_ds), len(local_ds))
        global_ds = Subset(global_ds, list(range(n)))
        local_ds = Subset(local_ds, list(range(n)))

    bs = data_cfg.get("batch_size", 32)
    nw = data_cfg.get("num_workers", 4)
    g_loader = DataLoader(global_ds, batch_size=bs, num_workers=nw, pin_memory=True)
    l_loader = DataLoader(local_ds, batch_size=bs, num_workers=nw, pin_memory=True)

    global_model = GlobalClassifier.from_checkpoint(global_ckpt, global_cfg, device=device)
    local_model = LocalClassifier.from_checkpoint(local_ckpt, local_cfg, device=device)

    if use_tta:
        print(f"  [INFO] Using 10x TTA")
        logits_g, labels_g = collect_logits_tta(
            global_model, val_csv, images_dir,
            global_cfg.get("input_size", 380), mean, std, device,
            bs, nw, mode="classification", desc=f"Fold {fold} global",
        )
        logits_l, labels_l = collect_logits_tta(
            local_model, val_csv, crops_dir,
            local_cfg.get("input_size", 224), mean, std, device,
            bs, nw, mode="classification", filter_existing=True, desc=f"Fold {fold} local",
        )
    else:
        logits_g, labels_g = collect_logits(global_model, g_loader, device, f"Fold {fold} global")
        logits_l, labels_l = collect_logits(local_model, l_loader, device, f"Fold {fold} local")

    n = min(len(logits_g), len(logits_l))
    logits_g, logits_l = logits_g[:n], logits_l[:n]
    labels = labels_g[:n]

    if cal_path.exists():
        cal = torch.load(cal_path, map_location="cpu")
        temp_g, temp_l = TemperatureScaler(), TemperatureScaler()
        temp_g.load_state_dict(cal["temp_global_state"])
        temp_l.load_state_dict(cal["temp_local_state"])
    else:
        temp_g, temp_l = TemperatureScaler(1.0), TemperatureScaler(1.0)

    with torch.no_grad():
        p_global = F.softmax(temp_g(logits_g), dim=1).numpy()
        p_local = F.softmax(temp_l(logits_l), dim=1).numpy()

    return {"fold": fold, "n_samples": n, "ablation": run_ablation(p_global, p_local, labels.numpy())}


def _random_results(smoke_test: int) -> dict:
    n = smoke_test if smoke_test > 0 else 500
    labels = np.random.randint(0, 7, n)
    p = np.random.dirichlet(np.ones(7), n)
    m = compute_full_metrics(labels, p.argmax(1), p)
    return {"fold": -1, "n_samples": n,
            "ablation": {k: m for k in ["global_only", "local_only", "naive_average", "fixed_weight", "confidence_ensemble"]},
            "note": "random predictions"}


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("training", {}).get("seed", 42))
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")

    all_results = []
    for fold in args.folds:
        print(f"\n{'='*60}\nEvaluating Fold {fold}\n{'='*60}")
        all_results.append(evaluate_fold(fold, cfg, device, args.smoke_test, use_tta=args.tta))

    # --- Summary ---
    print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
    configs = ["global_only", "local_only", "naive_average", "fixed_weight", "confidence_ensemble"]
    header = f"{'Configuration':<25s} {'Bal Acc':>10s} {'Macro F1':>10s} {'Macro AUC':>10s}"
    print(header)
    print("-" * len(header))

    summary = {}
    for cn in configs:
        ba = [r["ablation"][cn]["balanced_accuracy"] for r in all_results]
        f1 = [r["ablation"][cn]["macro_f1"] for r in all_results]
        au = [r["ablation"][cn]["macro_auc"] for r in all_results]
        if len(all_results) > 1:
            print(f"{cn:<25s} {np.mean(ba):.4f}+/-{np.std(ba):.4f}  {np.mean(f1):.4f}+/-{np.std(f1):.4f}  {np.mean(au):.4f}+/-{np.std(au):.4f}")
        else:
            print(f"{cn:<25s} {ba[0]:>10.4f} {f1[0]:>10.4f} {au[0]:>10.4f}")
        summary[cn] = {"bal_acc_mean": float(np.mean(ba)), "bal_acc_std": float(np.std(ba)),
                        "f1_mean": float(np.mean(f1)), "f1_std": float(np.std(f1)),
                        "auc_mean": float(np.mean(au)), "auc_std": float(np.std(au))}

    results_dir = Path(cfg.get("training", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "evaluation_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"folds": all_results, "summary": summary}, fh, indent=2, default=str)
    print(f"\n[INFO] Results saved to {out_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
