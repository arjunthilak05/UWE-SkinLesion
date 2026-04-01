"""Evaluate all baseline checkpoints and generate the SOTA comparison table.

Usage::

    python scripts/evaluate_baselines.py --config configs/default.yaml
    python scripts/evaluate_baselines.py --config configs/default.yaml --smoke_test 200
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HAM10000Dataset
from src.models.baseline import BaselineClassifier
from src.transforms import get_val_transforms
from src.utils import get_device, load_config, seed_everything

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7

BASELINES = [
    "resnet50",
    "vgg16_bn",
    "densenet121",
    "efficientnet_b4",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
]

DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "vgg16_bn": "VGG-16-BN",
    "densenet121": "DenseNet-121",
    "efficientnet_b4": "EfficientNet-B4",
    "vit_base_patch16_224": "ViT-B/16",
    "swin_tiny_patch4_window7_224": "Swin-T",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baselines and generate comparison table.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--smoke_test", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "",
) -> dict[str, float]:
    """Evaluate a single model on a DataLoader."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        all_logits.append(model(images).cpu())
        all_labels.append(batch["label"])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    bal_acc = float(balanced_accuracy_score(labels, preds))
    mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    # Macro AUC
    aucs = []
    for c in range(NUM_CLASSES):
        binary = (labels == c).astype(int)
        if 0 < binary.sum() < len(binary):
            try:
                aucs.append(float(roc_auc_score(binary, probs[:, c])))
            except ValueError:
                pass
    m_auc = float(np.mean(aucs)) if aucs else 0.0

    return {"balanced_accuracy": bal_acc, "macro_f1": mf1, "macro_auc": m_auc}


def _booktabs(rows: list[list[str]], header: list[str]) -> str:
    """Format as booktabs LaTeX table."""
    ncols = len(header)
    col_spec = "l" + "c" * (ncols - 1)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg.get("training", {})

    seed_everything(train_cfg.get("seed", 42))
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")

    data_dir = Path(data_cfg["data_dir"])
    images_dir = data_dir / data_cfg["images_subdir"]
    val_csv = Path(data_cfg.get("splits_dir", "data/splits")) / f"fold{args.fold}_val.csv"

    if not val_csv.exists():
        print(f"[ERROR] Val CSV not found: {val_csv}")
        sys.exit(1)

    norm = aug_cfg.get("normalize", {})
    mean = tuple(norm.get("mean", [0.7635, 0.5461, 0.5705]))
    std = tuple(norm.get("std", [0.1404, 0.1521, 0.1697]))

    ckpt_base = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "baselines"

    results_rows: list[dict[str, Any]] = []

    print(f"\n{'='*65}")
    print(f"{'Model':<30s} {'Bal Acc':>10s} {'Macro F1':>10s} {'Macro AUC':>10s} {'#Params':>10s}")
    print(f"{'='*65}")

    for model_name in BASELINES:
        ckpt_path = ckpt_base / model_name / "best.pt"

        if not ckpt_path.exists():
            print(f"{DISPLAY_NAMES[model_name]:<30s}  ** checkpoint not found: {ckpt_path} **")
            results_rows.append({
                "model": model_name, "display": DISPLAY_NAMES[model_name],
                "bal_acc": 0.0, "macro_f1": 0.0, "macro_auc": 0.0, "params": "—",
            })
            continue

        img_size = BaselineClassifier.get_input_size(model_name)
        val_tf = get_val_transforms(img_size, mean, std)

        val_ds = HAM10000Dataset(
            csv_path=val_csv, images_dir=images_dir,
            transform=val_tf, img_size=img_size, mode="classification",
        )
        if args.smoke_test > 0:
            val_ds = Subset(val_ds, list(range(min(args.smoke_test, len(val_ds)))))

        loader = DataLoader(
            val_ds, batch_size=32, shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
        )

        # Load model
        model = BaselineClassifier(
            model_name=model_name, pretrained=False, num_classes=NUM_CLASSES,
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        params_str = f"{total_params / 1e6:.1f}M"

        metrics = evaluate_model(model, loader, device, desc=DISPLAY_NAMES[model_name])

        print(
            f"{DISPLAY_NAMES[model_name]:<30s} "
            f"{metrics['balanced_accuracy']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['macro_auc']:>10.4f} "
            f"{params_str:>10s}"
        )

        results_rows.append({
            "model": model_name,
            "display": DISPLAY_NAMES[model_name],
            "bal_acc": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_auc": metrics["macro_auc"],
            "params": params_str,
        })

    # --- Add proposed method placeholder ---
    results_rows.append({
        "model": "proposed",
        "display": "\\textbf{Proposed (Ours)}",
        "bal_acc": 0.0,
        "macro_f1": 0.0,
        "macro_auc": 0.0,
        "params": "43.0M",
        "note": "Fill from evaluation_results.json",
    })

    # Try to load proposed results
    eval_results_path = Path("results/evaluation_results.json")
    if eval_results_path.exists():
        with eval_results_path.open() as f:
            eval_data = json.load(f)
        ens = eval_data.get("summary", {}).get("confidence_ensemble", {})
        if ens:
            results_rows[-1]["bal_acc"] = ens.get("bal_acc_mean", 0.0)
            results_rows[-1]["macro_f1"] = ens.get("f1_mean", 0.0)
            results_rows[-1]["macro_auc"] = ens.get("auc_mean", 0.0)

    print(f"{'='*65}")

    # --- Save tables ---
    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = table_dir / "table2_sota_comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "bal_acc", "macro_f1", "macro_auc", "params"])
        for r in results_rows:
            w.writerow([
                r["display"].replace("\\textbf{", "").replace("}", ""),
                f"{r['bal_acc']:.4f}" if r["bal_acc"] > 0 else "—",
                f"{r['macro_f1']:.4f}" if r["macro_f1"] > 0 else "—",
                f"{r['macro_auc']:.4f}" if r["macro_auc"] > 0 else "—",
                r["params"],
            ])
    print(f"\n[INFO] CSV saved to {csv_path}")

    # LaTeX
    tex_rows = []
    for r in results_rows:
        is_proposed = "proposed" in r["model"]
        fmt = lambda v: f"\\textbf{{{v:.4f}}}" if is_proposed and v > 0 else (f"{v:.4f}" if v > 0 else "---")
        name = r["display"] if is_proposed else r["display"]
        tex_rows.append([
            name, fmt(r["bal_acc"]), fmt(r["macro_f1"]), fmt(r["macro_auc"]), r["params"],
        ])

    tex = _booktabs(tex_rows, ["Method", "Bal.\\ Acc.", "Macro F1", "Macro AUC", "\\#Params"])
    tex_path = table_dir / "table2_sota_comparison.tex"
    tex_path.write_text(tex)
    print(f"[INFO] LaTeX saved to {tex_path}")

    # JSON
    json_path = Path("results") / "baseline_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results_rows, f, indent=2)
    print(f"[INFO] JSON saved to {json_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
