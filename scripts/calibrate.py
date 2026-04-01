"""Calibrate both classifiers and grid-search the ensemble tau (Stage 5).

Usage::

    python scripts/calibrate.py --config configs/default.yaml
    python scripts/calibrate.py --config configs/default.yaml --smoke_test 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HAM10000Dataset
from src.metrics import balanced_accuracy, macro_auc, macro_f1
from src.models.gating import ConfidenceEnsemble
from src.models.global_classifier import GlobalClassifier
from src.models.local_classifier import LocalClassifier
from src.models.temperature import TemperatureScaler, expected_calibration_error
from src.transforms import get_val_transforms
from src.utils import get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Calibrate classifiers and search ensemble tau.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--global_ckpt", type=Path, default=None, help="Override global checkpoint path.")
    parser.add_argument("--local_ckpt", type=Path, default=None, help="Override local checkpoint path.")
    parser.add_argument("--smoke_test", type=int, default=0, help="Limit to N samples.")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Collecting",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference on a DataLoader and collect logits + labels.

    Args:
        model: Classifier in eval mode.
        loader: DataLoader.
        device: Compute device.
        desc: Progress bar description.

    Returns:
        Tuple of ``(all_logits (N, C), all_labels (N,))``.
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        labels = batch["label"]

        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def grid_search_tau(
    p_global: torch.Tensor,
    p_local: torch.Tensor,
    labels: torch.Tensor,
    tau_range: np.ndarray,
) -> dict[str, list]:
    """Grid search the ensemble tau on balanced accuracy.

    Args:
        p_global: Calibrated global probabilities ``(N, C)``.
        p_local: Calibrated local probabilities ``(N, C)``.
        labels: Ground-truth labels ``(N,)``.
        tau_range: Array of tau values to try.

    Returns:
        Dict with ``tau_values``, ``bal_acc``, ``macro_f1``, ``best_tau``,
        ``best_bal_acc``.
    """
    results: dict[str, list] = {
        "tau_values": [],
        "bal_acc": [],
        "macro_f1": [],
    }

    best_tau = 1.0
    best_bal_acc = 0.0

    for tau in tau_range:
        ens = ConfidenceEnsemble(tau=float(tau))
        out = ens(p_global, p_local)
        preds = out["p_final"].argmax(dim=1)

        ba = balanced_accuracy(labels, preds)
        mf1 = macro_f1(labels, preds)

        results["tau_values"].append(float(tau))
        results["bal_acc"].append(ba)
        results["macro_f1"].append(mf1)

        if ba > best_bal_acc:
            best_bal_acc = ba
            best_tau = float(tau)

    results["best_tau"] = best_tau
    results["best_bal_acc"] = best_bal_acc
    return results


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    global_cfg = cfg["global_classifier"]
    local_cfg = cfg["local_classifier"]
    ens_cfg = cfg.get("ensemble", {})
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg.get("training", {})

    seed_everything(train_cfg.get("seed", 42))
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")

    # --- Paths ---
    global_ckpt = args.global_ckpt or Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "global_classifier" / "best.pt"
    local_ckpt = args.local_ckpt or Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "local_classifier" / "best.pt"

    for name, ckpt_path in [("Global", global_ckpt), ("Local", local_ckpt)]:
        if not ckpt_path.exists():
            print(f"[ERROR] {name} checkpoint not found: {ckpt_path}")
            print(f"       Train the {name.lower()} classifier first.")
            sys.exit(1)

    data_dir = Path(data_cfg["data_dir"])
    images_dir = data_dir / data_cfg["images_subdir"]
    crops_dir = data_dir / "cropped_lesions"
    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    val_csv = splits_dir / f"fold{args.fold}_val.csv"

    if not val_csv.exists():
        print(f"[ERROR] Val CSV not found: {val_csv}")
        sys.exit(1)

    # --- Transforms ---
    norm = aug_cfg.get("normalize", {})
    mean = tuple(norm.get("mean", [0.7635, 0.5461, 0.5705]))
    std = tuple(norm.get("std", [0.1404, 0.1521, 0.1697]))

    global_tf = get_val_transforms(global_cfg.get("input_size", 380), mean, std)
    local_tf = get_val_transforms(local_cfg.get("input_size", 224), mean, std)

    # --- Datasets ---
    global_ds = HAM10000Dataset(
        csv_path=val_csv, images_dir=images_dir,
        transform=global_tf, img_size=global_cfg.get("input_size", 380),
        mode="classification",
    )
    local_ds = HAM10000Dataset(
        csv_path=val_csv, images_dir=crops_dir,
        transform=local_tf, img_size=local_cfg.get("input_size", 224),
        mode="classification", filter_existing=True,
    )

    if args.smoke_test > 0:
        n = min(args.smoke_test, len(global_ds), len(local_ds))
        global_ds = Subset(global_ds, list(range(n)))
        local_ds = Subset(local_ds, list(range(n)))
        print(f"[SMOKE TEST] {n} samples")

    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)

    global_loader = DataLoader(global_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    local_loader = DataLoader(local_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # --- Load models ---
    print(f"[INFO] Loading global model from {global_ckpt}")
    global_model = GlobalClassifier.from_checkpoint(global_ckpt, global_cfg, device=device)

    print(f"[INFO] Loading local model from {local_ckpt}")
    local_model = LocalClassifier.from_checkpoint(local_ckpt, local_cfg, device=device)

    # =====================================================================
    # Step 1: Collect logits
    # =====================================================================
    print("\n" + "=" * 60)
    print("Step 1: Collecting validation logits")
    print("=" * 60)

    logits_g, labels_g = collect_logits(global_model, global_loader, device, "Global logits")
    logits_l, labels_l = collect_logits(local_model, local_loader, device, "Local logits")

    print(f"  Global: {logits_g.shape[0]} samples, {logits_g.shape[1]} classes")
    print(f"  Local:  {logits_l.shape[0]} samples, {logits_l.shape[1]} classes")

    # Ensure same sample count (local may have fewer due to missing crops)
    n_common = min(len(logits_g), len(logits_l))
    logits_g = logits_g[:n_common]
    logits_l = logits_l[:n_common]
    labels = labels_g[:n_common]

    # =====================================================================
    # Step 2: Temperature scaling
    # =====================================================================
    print("\n" + "=" * 60)
    print("Step 2: Temperature scaling calibration")
    print("=" * 60)

    temp_global = TemperatureScaler(init_temperature=ens_cfg.get("temperature_tau", 1.5))
    result_g = temp_global.fit(
        logits_g, labels,
        lr=ens_cfg.get("calibration_lr", 0.01),
        max_iter=ens_cfg.get("calibration_max_iter", 200),
    )

    print(f"\n  Global classifier:")
    print(f"    Temperature:  {result_g['temperature']:.4f}")
    print(f"    ECE before:   {result_g['ece_before']:.4f}")
    print(f"    ECE after:    {result_g['ece_after']:.4f}")
    print(f"    NLL before:   {result_g['nll_before']:.4f}")
    print(f"    NLL after:    {result_g['nll_after']:.4f}")

    temp_local = TemperatureScaler(init_temperature=ens_cfg.get("temperature_tau", 1.5))
    result_l = temp_local.fit(
        logits_l, labels,
        lr=ens_cfg.get("calibration_lr", 0.01),
        max_iter=ens_cfg.get("calibration_max_iter", 200),
    )

    print(f"\n  Local classifier:")
    print(f"    Temperature:  {result_l['temperature']:.4f}")
    print(f"    ECE before:   {result_l['ece_before']:.4f}")
    print(f"    ECE after:    {result_l['ece_after']:.4f}")
    print(f"    NLL before:   {result_l['nll_before']:.4f}")
    print(f"    NLL after:    {result_l['nll_after']:.4f}")

    # =====================================================================
    # Step 3: Get calibrated probabilities
    # =====================================================================
    with torch.no_grad():
        p_global = F.softmax(temp_global(logits_g), dim=1)
        p_local = F.softmax(temp_local(logits_l), dim=1)

    # Individual model performance
    preds_g = p_global.argmax(dim=1)
    preds_l = p_local.argmax(dim=1)
    ba_g = balanced_accuracy(labels, preds_g)
    ba_l = balanced_accuracy(labels, preds_l)
    f1_g = macro_f1(labels, preds_g)
    f1_l = macro_f1(labels, preds_l)

    print(f"\n  Calibrated individual performance:")
    print(f"    Global: bal_acc={ba_g:.4f}, macro_f1={f1_g:.4f}")
    print(f"    Local:  bal_acc={ba_l:.4f}, macro_f1={f1_l:.4f}")

    # =====================================================================
    # Step 4: Grid search tau
    # =====================================================================
    print("\n" + "=" * 60)
    print("Step 3: Grid search ensemble tau")
    print("=" * 60)

    tau_range = np.arange(0.5, 5.25, 0.25)
    gs_results = grid_search_tau(p_global, p_local, labels, tau_range)

    print(f"\n  {'tau':>6s}  {'bal_acc':>10s}  {'macro_f1':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")
    for tau, ba, mf1 in zip(gs_results["tau_values"], gs_results["bal_acc"], gs_results["macro_f1"]):
        marker = " <-- best" if tau == gs_results["best_tau"] else ""
        print(f"  {tau:6.2f}  {ba:10.4f}  {mf1:10.4f}{marker}")

    print(f"\n  Best tau:        {gs_results['best_tau']:.2f}")
    print(f"  Best bal_acc:    {gs_results['best_bal_acc']:.4f}")

    # =====================================================================
    # Step 5: Save calibration parameters
    # =====================================================================
    print("\n" + "=" * 60)
    print("Step 4: Saving calibration parameters")
    print("=" * 60)

    cal_params = {
        "global_temperature": result_g["temperature"],
        "local_temperature": result_l["temperature"],
        "best_tau": gs_results["best_tau"],
        "best_ensemble_bal_acc": gs_results["best_bal_acc"],
        "global_ece_before": result_g["ece_before"],
        "global_ece_after": result_g["ece_after"],
        "local_ece_before": result_l["ece_before"],
        "local_ece_after": result_l["ece_after"],
        "global_bal_acc": ba_g,
        "local_bal_acc": ba_l,
        "global_macro_f1": f1_g,
        "local_macro_f1": f1_l,
        "tau_grid_search": {
            "tau_values": gs_results["tau_values"],
            "bal_acc": gs_results["bal_acc"],
            "macro_f1": gs_results["macro_f1"],
        },
    }

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    cal_path = ckpt_dir / "calibration_params.json"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with cal_path.open("w", encoding="utf-8") as fh:
        json.dump(cal_params, fh, indent=2)
    print(f"  Saved to: {cal_path}")

    # Save temperature scaler states
    torch.save({
        "temp_global_state": temp_global.state_dict(),
        "temp_local_state": temp_local.state_dict(),
        "best_tau": gs_results["best_tau"],
    }, ckpt_dir / "calibration_state.pt")
    print(f"  Saved to: {ckpt_dir / 'calibration_state.pt'}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"  Global ECE:  {result_g['ece_before']:.4f} → {result_g['ece_after']:.4f}  (T={result_g['temperature']:.4f})")
    print(f"  Local ECE:   {result_l['ece_before']:.4f} → {result_l['ece_after']:.4f}  (T={result_l['temperature']:.4f})")
    print(f"  Ensemble:    tau={gs_results['best_tau']:.2f}, bal_acc={gs_results['best_bal_acc']:.4f}")
    print(f"  Improvement: global={ba_g:.4f}, local={ba_l:.4f} → ensemble={gs_results['best_bal_acc']:.4f}")
    print("=" * 60)
    print("[DONE]")


if __name__ == "__main__":
    main()
