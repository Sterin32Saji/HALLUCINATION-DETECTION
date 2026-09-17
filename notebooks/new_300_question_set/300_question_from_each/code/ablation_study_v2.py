"""
Reproducible custom-detector ablation study.

This script mirrors the final custom-detector evaluation protocol:
train + validation records are used for fitting, and the held-out 45-record
test split is used only for evaluation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import (  # noqa: E402
    ABLATION_CSV_PATH,
    ABLATION_JSON_PATH,
    CUSTOM_SUMMARY_PATH,
    FEATURES_PATH,
    MERGED_RAW_PATH,
    custom_feature_groups,
    run_ablation_study,
)


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    raw = pd.read_csv(MERGED_RAW_PATH)
    summary = json.loads(CUSTOM_SUMMARY_PATH.read_text(encoding="utf-8"))

    labels = raw.set_index("id").loc[features["id"], "human"].astype(int).reset_index(drop=True)
    split_bundle = {
        "summary": summary,
        "train_idx": pd.Index(summary["train_indices"]).to_numpy(),
        "val_idx": pd.Index(summary["val_indices"]).to_numpy(),
        "test_idx": pd.Index(summary["test_indices"]).to_numpy(),
    }

    result = run_ablation_study(features, labels, split_bundle)
    print("Ablation study complete")
    print("Protocol:", result["protocol"])
    print("Feature groups:", {name: len(cols) for name, cols in custom_feature_groups().items()})
    print("Full-model F1:", round(result["full_model_metrics"]["f1"], 4))
    print("CSV:", ABLATION_CSV_PATH)
    print("JSON:", ABLATION_JSON_PATH)


if __name__ == "__main__":
    main()
