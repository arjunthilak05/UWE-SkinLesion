"""Add numeric metadata features to dataset samples for fusion models.

Encodes age, sex, localization into a fixed-size tensor that can be
concatenated with image features before the classification head.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("C:/Users/Hp/Downloads/skin")))

# 15 localization sites in HAM10000
LOCALIZATIONS = [
    "back", "lower extremity", "trunk", "upper extremity", "abdomen",
    "face", "chest", "foot", "unknown", "neck", "scalp", "hand",
    "ear", "genital", "acral",
]
LOC_TO_IDX = {loc: i for i, loc in enumerate(LOCALIZATIONS)}


def encode_metadata(age, sex, localization):
    """Encode a single sample's metadata into a numpy array of shape (17,).

    Features:
    - age: 1 float (normalized to [0,1], NaN -> 0.5)
    - sex: 1 float (male=0, female=1, unknown=0.5)
    - localization: 15 floats (one-hot)
    """
    # Age — handle tensors, numpy, scalars, NaN
    try:
        import torch
        if isinstance(age, torch.Tensor):
            age = age.item()
    except:
        pass
    try:
        age_val = float(age)
        if np.isnan(age_val):
            age_feat = 0.5
        else:
            age_feat = age_val / 100.0
    except (TypeError, ValueError):
        age_feat = 0.5

    # Sex
    sex_str = str(sex).lower().strip()
    if sex_str == "male":
        sex_feat = 0.0
    elif sex_str == "female":
        sex_feat = 1.0
    else:
        sex_feat = 0.5

    # Localization one-hot
    loc_feat = np.zeros(len(LOCALIZATIONS), dtype=np.float32)
    loc_str = str(localization).lower().strip()
    if loc_str in LOC_TO_IDX:
        loc_feat[LOC_TO_IDX[loc_str]] = 1.0

    return np.concatenate([[age_feat, sex_feat], loc_feat]).astype(np.float32)


def encode_metadata_batch(metadata_list):
    """Convert list of metadata dicts to (B, 17) numpy array."""
    features = []
    for m in metadata_list:
        features.append(encode_metadata(
            m.get("age"), m.get("sex"), m.get("localization")
        ))
    return np.stack(features)


if __name__ == "__main__":
    # Quick test
    test = encode_metadata(45.0, "male", "back")
    print(f"Shape: {test.shape}, Values: {test}")
    test2 = encode_metadata(None, "unknown", "missing")
    print(f"NaN test: {test2}")
