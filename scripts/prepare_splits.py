"""Create GroupKFold cross-validation splits for HAM10000.

Groups by ``lesion_id`` so that multiple images of the same lesion never
leak across folds.  Stratification is approximated by sorting groups by
their majority class before assignment.

Usage::

    python scripts/prepare_splits.py \
        --data_dir data/HAM10000 \
        --output_dir data/splits \
        --n_folds 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, seed_everything

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create GroupKFold splits for HAM10000."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/HAM10000"),
        help="Directory containing HAM10000_metadata.csv (and images/).",
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=None,
        help="Explicit path to metadata CSV (overrides data_dir lookup).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory to write fold CSVs.",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of folds."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def find_metadata_csv(data_dir: Path, explicit: Path | None) -> Path:
    """Locate the metadata CSV, searching several common locations.

    Args:
        data_dir: Base data directory.
        explicit: If provided, use this path directly.

    Returns:
        Resolved Path to the CSV.

    Raises:
        FileNotFoundError: If no CSV can be found.
    """
    if explicit and explicit.exists():
        return explicit

    candidates = [
        data_dir / "HAM10000_metadata.csv",
        data_dir / "metadata.csv",
        data_dir.parent / "raw" / "HAM10000_metadata.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find HAM10000_metadata.csv in any of: {candidates}"
    )


def compute_class_weights(labels: np.ndarray, num_classes: int) -> dict[str, float]:
    """Compute inverse-frequency class weights normalised to sum to ``num_classes``.

    Args:
        labels: Integer label array.
        num_classes: Total number of classes.

    Returns:
        Dict mapping class name to weight.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum() * num_classes
    return {CLASSES[i]: round(float(weights[i]), 6) for i in range(num_classes)}


def print_fold_distribution(
    df: pd.DataFrame, folds: np.ndarray, n_folds: int
) -> None:
    """Pretty-print per-fold class counts.

    Args:
        df: Full metadata DataFrame.
        folds: Array of fold indices (one per row).
        n_folds: Number of folds.
    """
    header = f"{'Class':<10}" + "".join(f"{'Fold ' + str(i):>10}" for i in range(n_folds)) + f"{'Total':>10}"
    print("\n" + "=" * (10 + 10 * (n_folds + 1)))
    print("GroupKFold Class Distribution")
    print("=" * (10 + 10 * (n_folds + 1)))
    print(header)
    print("-" * (10 + 10 * (n_folds + 1)))

    for cls in CLASSES:
        row = f"{cls:<10}"
        total = 0
        for fold in range(n_folds):
            count = int(((df["dx"] == cls) & (folds == fold)).sum())
            total += count
            row += f"{count:>10d}"
        row += f"{total:>10d}"
        print(row)

    print("-" * (10 + 10 * (n_folds + 1)))
    totals = f"{'TOTAL':<10}"
    grand = 0
    for fold in range(n_folds):
        c = int((folds == fold).sum())
        grand += c
        totals += f"{c:>10d}"
    totals += f"{grand:>10d}"
    print(totals)
    print("=" * (10 + 10 * (n_folds + 1)))


def main() -> None:
    """Entry point."""
    args = parse_args()
    seed_everything(args.seed)

    # Load config (best-effort, not strictly required)
    cfg: dict = {}
    if args.config.exists():
        cfg = load_config(args.config)

    # --- Locate and load metadata ---
    csv_path = find_metadata_csv(args.data_dir, args.metadata_csv)
    print(f"[INFO] Loading metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows, {df['lesion_id'].nunique()} unique lesions.")

    # --- Validate ---
    for col in ("image_id", "lesion_id", "dx"):
        if col not in df.columns:
            print(f"[ERROR] Missing column: {col}")
            sys.exit(1)
    unknown = set(df["dx"].unique()) - set(CLASSES)
    if unknown:
        print(f"[ERROR] Unknown dx values: {unknown}")
        sys.exit(1)

    # --- Integer labels and groups ---
    labels = df["dx"].map(LABEL_MAP).values
    groups = df["lesion_id"].values

    # --- StratifiedGroupKFold ---
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_indices = np.full(len(df), -1, dtype=int)

    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, labels, groups)):
        fold_indices[val_idx] = fold_idx

    df["fold"] = fold_indices

    # --- Print distribution ---
    print_fold_distribution(df, fold_indices, args.n_folds)

    # --- Save fold CSVs ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(args.n_folds):
        fold_val = df[df["fold"] == fold].copy()
        fold_train = df[df["fold"] != fold].copy()

        fold_val.to_csv(args.output_dir / f"fold{fold}_val.csv", index=False)
        fold_train.to_csv(args.output_dir / f"fold{fold}_train.csv", index=False)
        print(
            f"[INFO] Fold {fold}: train={len(fold_train)}, val={len(fold_val)}"
        )

    # Also save a master CSV with fold assignments
    df.to_csv(args.output_dir / "all_folds.csv", index=False)

    # --- Compute and save class weights ---
    weights = compute_class_weights(labels, num_classes=len(CLASSES))
    weights_path = Path("configs") / "class_weights.json"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump(weights, fh, indent=2)
    print(f"\n[INFO] Class weights saved to {weights_path}")
    for cls, w in weights.items():
        print(f"  {cls:<10} {w:.6f}")

    # --- Save split statistics ---
    stats = {
        "total_images": len(df),
        "total_lesions": int(df["lesion_id"].nunique()),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "class_counts": {cls: int((df["dx"] == cls).sum()) for cls in CLASSES},
        "class_weights": weights,
        "folds": {},
    }
    for fold in range(args.n_folds):
        fold_mask = fold_indices == fold
        stats["folds"][f"fold_{fold}"] = {
            "val_count": int(fold_mask.sum()),
            "train_count": int((~fold_mask).sum()),
            "val_class_counts": {
                cls: int(((df["dx"] == cls) & fold_mask).sum()) for cls in CLASSES
            },
        }

    stats_path = args.output_dir / "split_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[INFO] Split statistics saved to {stats_path}")
    print("\n[DONE] All folds generated successfully.")


if __name__ == "__main__":
    main()
