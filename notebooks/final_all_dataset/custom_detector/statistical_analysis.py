from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "artifacts" / "features.csv"
COMBINED_PATH = ROOT.parent / "150_filtered" / "combined_scores_final.csv"
CUSTOM_MODEL_PATH = ROOT / "artifacts" / "custom_hallucination_detector.pkl"
HYBRID_MODEL_PATH = ROOT.parent / "hybrid_detector_model.pkl"
OUTPUT_PATH = ROOT / "artifacts" / "statistical_analysis.json"


def _bootstrap_ci(values: np.ndarray, ci_level: float = 95.0) -> tuple[float, float]:
    lower, upper = np.percentile(values, [(100 - ci_level) / 2, (100 + ci_level) / 2])
    return float(lower), float(upper)


def run_analysis() -> dict:
    features = pd.read_csv(FEATURES_PATH)
    combined = pd.read_csv(COMBINED_PATH)
    merged = features.merge(
        combined[["id", "human", "baseline", "selfcheck", "ragas", "similarity"]],
        on="id",
        how="inner",
        suffixes=("_feat", "_hyb"),
    )
    y = merged["human_feat"].astype(int)

    custom_model = joblib.load(CUSTOM_MODEL_PATH)["pipeline"]
    hybrid_model = joblib.load(HYBRID_MODEL_PATH)
    custom_features = [col for col in features.columns if col not in {"id", "human"}]

    idx = np.arange(len(merged))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    test_df = merged.iloc[test_idx].copy()
    y_true = y.iloc[test_idx].to_numpy()

    custom_pred = custom_model.predict(test_df[custom_features])
    hybrid_pred = hybrid_model.predict(test_df[["baseline", "ragas", "selfcheck", "similarity"]])

    metrics = {
        "custom": {
            "accuracy": float(accuracy_score(y_true, custom_pred)),
            "precision": float(precision_score(y_true, custom_pred, zero_division=0)),
            "recall": float(recall_score(y_true, custom_pred, zero_division=0)),
            "f1": float(f1_score(y_true, custom_pred, zero_division=0)),
        },
        "hybrid": {
            "accuracy": float(accuracy_score(y_true, hybrid_pred)),
            "precision": float(precision_score(y_true, hybrid_pred, zero_division=0)),
            "recall": float(recall_score(y_true, hybrid_pred, zero_division=0)),
            "f1": float(f1_score(y_true, hybrid_pred, zero_division=0)),
        },
    }

    rng = np.random.default_rng(42)
    reps = 5000
    custom_f1_boot = []
    hybrid_f1_boot = []
    diff_f1_boot = []
    for _ in range(reps):
        sample = rng.choice(len(test_idx), size=len(test_idx), replace=True)
        yb = y_true[sample]
        custom_b = custom_pred[sample]
        hybrid_b = hybrid_pred[sample]
        custom_f1_boot.append(f1_score(yb, custom_b, zero_division=0))
        hybrid_f1_boot.append(f1_score(yb, hybrid_b, zero_division=0))
        diff_f1_boot.append(f1_score(yb, custom_b, zero_division=0) - f1_score(yb, hybrid_b, zero_division=0))

    # McNemar exact paired test
    correct_custom = custom_pred == y_true
    correct_hybrid = hybrid_pred == y_true
    b = int(np.sum((~correct_custom) & correct_hybrid))
    c = int(np.sum(correct_custom & (~correct_hybrid)))
    n = b + c
    if n > 0:
        p_value = float(binomtest(max(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    else:
        p_value = 1.0

    summary = {
        "n_test_samples": int(len(test_idx)),
        "custom": metrics["custom"],
        "hybrid": metrics["hybrid"],
        "bootstrap": {
            "iterations": reps,
            "custom_f1_95ci": list(_bootstrap_ci(np.asarray(custom_f1_boot))),
            "hybrid_f1_95ci": list(_bootstrap_ci(np.asarray(hybrid_f1_boot))),
            "difference_f1_95ci": list(_bootstrap_ci(np.asarray(diff_f1_boot))),
        },
        "paired_test": {
            "disagreements_model_a_wrong_model_b_right": b,
            "disagreements_model_a_right_model_b_wrong": c,
            "mcnemar_exact_p_value": p_value,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


if __name__ == "__main__":
    result = run_analysis()
    print(json.dumps({
        "custom_f1": result["custom"]["f1"],
        "hybrid_f1": result["hybrid"]["f1"],
        "custom_f1_95ci": result["bootstrap"]["custom_f1_95ci"],
        "hybrid_f1_95ci": result["bootstrap"]["hybrid_f1_95ci"],
        "difference_f1_95ci": result["bootstrap"]["difference_f1_95ci"],
        "mcnemar_p": result["paired_test"]["mcnemar_exact_p_value"],
    }, indent=2))
