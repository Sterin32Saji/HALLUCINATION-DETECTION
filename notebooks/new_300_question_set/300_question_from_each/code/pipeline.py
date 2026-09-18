from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
NOTEBOOKS_DIR = ROOT_DIR.parent
SOURCE_DIR = PROJECT_DIR
RESULTS_DIR = PROJECT_DIR / "analysis_results"
DATA_DIR = RESULTS_DIR / "data"
MODEL_DIR = RESULTS_DIR / "models"
PLOT_DIR = RESULTS_DIR / "plots"
REPORT_DIR = RESULTS_DIR / "reports"
ABLATION_DIR = RESULTS_DIR / "ablation_results"

for directory in (DATA_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR, ABLATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)

OLD_CUSTOM_DIR = NOTEBOOKS_DIR / "final_all_dataset" / "custom_detector"
if str(OLD_CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_CUSTOM_DIR))

from feature_extractor import CustomFeatureExtractor  # type: ignore


BASELINE_PATH = SOURCE_DIR / "baseline_score_filtered.jsonl"
RAGAS_PATH = SOURCE_DIR / "ragas_hallucination_results_filtered.jsonl"
SC_PATH = SOURCE_DIR / "selfcheck_dataset_filtered.jsonl"
SIM_PATH = SOURCE_DIR / "similarity_dataset_filtered.jsonl"
HUMAN_PATH = SOURCE_DIR / "human_label.jsonl"

MERGED_RAW_PATH = DATA_DIR / "merged_300_raw.csv"
COMBINED_PATH = DATA_DIR / "combined_scores_final.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
ALIGNMENT_PATH = REPORT_DIR / "alignment_report.json"
CUSTOM_SUMMARY_PATH = REPORT_DIR / "custom_training_summary.json"
HYBRID_SUMMARY_PATH = REPORT_DIR / "hybrid_training_summary.json"
METHOD_METRICS_PATH = REPORT_DIR / "method_metrics.csv"
METHOD_SUMMARY_PATH = REPORT_DIR / "method_metrics.json"
HELDOUT_PERFORMANCE_PATH = PLOT_DIR / "heldout_performance_comparison.png"
STATISTICS_PATH = REPORT_DIR / "statistical_analysis.json"
ABLATION_CSV_PATH = ABLATION_DIR / "ablation_study.csv"
ABLATION_JSON_PATH = ABLATION_DIR / "ablation_study.json"
TEST_PREDICTIONS_PATH = DATA_DIR / "test_predictions.csv"
MASTER_EVALUATION_CSV_PATH = DATA_DIR / "master_evaluation_dataset.csv"
MASTER_EVALUATION_JSONL_PATH = DATA_DIR / "master_evaluation_dataset.jsonl"
ERROR_ANALYSIS_CSV_PATH = DATA_DIR / "error_analysis.csv"
ERROR_ANALYSIS_JSONL_PATH = DATA_DIR / "error_analysis.jsonl"
FULL_EVALUATION_JSON_PATH = REPORT_DIR / "evaluation_results.json"
FULL_EVALUATION_CSV_PATH = REPORT_DIR / "evaluation_results.csv"

METHOD_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "ragas": "RAGAS",
    "selfcheck": "SelfCheck-style",
    "similarity": "Retrieval Similarity",
    "hybrid": "Hybrid Model",
    "custom": "Custom Model",
}

RISK_CATEGORIES = [
    ("Low Risk", 0, 30),
    ("Medium Risk", 31, 60),
    ("High Risk", 61, 80),
    ("Critical Risk", 81, 100),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, ensure_ascii=False, allow_nan=False)


def _canonical_id_set(rows: list[dict[str, Any]], key: str = "id") -> set[str]:
    return {str(row[key]) for row in rows if key in row and row[key] is not None}


def check_alignment() -> dict[str, Any]:
    baseline = read_jsonl(BASELINE_PATH)
    ragas = read_jsonl(RAGAS_PATH)
    selfcheck = read_jsonl(SC_PATH)
    similarity = read_jsonl(SIM_PATH)
    human = read_jsonl(HUMAN_PATH)

    ids = {
        "baseline": _canonical_id_set(baseline, "id"),
        "ragas": _canonical_id_set(ragas, "original_id"),
        "selfcheck": _canonical_id_set(selfcheck, "id"),
        "similarity": _canonical_id_set(similarity, "id"),
        "human": _canonical_id_set(human, "id"),
    }

    counts = {
        "baseline": len(baseline),
        "ragas": len(ragas),
        "selfcheck": len(selfcheck),
        "similarity": len(similarity),
        "human": len(human),
    }

    master = set.union(*ids.values()) if ids else set()
    alignment_ok = all(len(s) == 300 for s in ids.values()) and all(s == ids["human"] for s in ids.values())
    missing_by_file = {
        name: sorted(master - id_set)[:10]
        for name, id_set in ids.items()
        if master - id_set
    }

    report = {
        "alignment_ok": alignment_ok,
        "counts": counts,
        "unique_id_counts": {name: len(values) for name, values in ids.items()},
        "missing_by_file_sample": missing_by_file,
    }
    write_json(ALIGNMENT_PATH, report)
    return report


def _load_frame(path: Path, rename_map: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(path))
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def build_merged_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    human = _load_frame(HUMAN_PATH)
    baseline = _load_frame(BASELINE_PATH)[["id", "hallucination"]].rename(
        columns={"hallucination": "baseline_hallucination"}
    )
    ragas = _load_frame(RAGAS_PATH, {"original_id": "id", "faithfulness": "ragas_faithfulness"})[
        ["id", "ragas_faithfulness"]
    ]
    selfcheck = _load_frame(SC_PATH)[["id", "selfcheck_hallucination"]]
    similarity = _load_frame(SIM_PATH, {"retrail_similarity": "similarity_score"})[["id", "similarity_score"]]

    merged = (
        human.merge(baseline, on="id", how="inner")
        .merge(ragas, on="id", how="inner")
        .merge(selfcheck, on="id", how="inner")
        .merge(similarity, on="id", how="inner")
        .sort_values("id")
        .reset_index(drop=True)
    )

    merged["human"] = merged["human_label"].astype(int)
    merged["baseline"] = merged["baseline_hallucination"].astype(int)
    merged["selfcheck"] = merged["selfcheck_hallucination"].astype(int)
    merged["ragas"] = (merged["ragas_faithfulness"].astype(float) < 0.5).astype(int)
    merged["similarity"] = (merged["similarity_score"].astype(float) < 0.5).astype(int)

    raw_cols = [
        "id",
        "question",
        "answer",
        "evidence",
        "question_type",
        "difficulty",
        "chunk_id",
        "source_doc",
        "rag_answer",
        "baseline_hallucination",
        "ragas_faithfulness",
        "selfcheck_hallucination",
        "similarity_score",
        "human_label",
        "human",
        "baseline",
        "ragas",
        "selfcheck",
        "similarity",
    ]
    merged_raw = merged[raw_cols].copy()
    merged_raw.to_csv(MERGED_RAW_PATH, index=False)

    combined = merged[["id", "human", "baseline", "selfcheck", "ragas", "similarity"]].copy()
    combined.to_csv(COMBINED_PATH, index=False)

    return merged_raw, combined


def build_features(merged_raw: pd.DataFrame) -> pd.DataFrame:
    extractor = CustomFeatureExtractor()
    feature_input = merged_raw[
        ["id", "question", "question_type", "difficulty", "rag_answer", "evidence"]
    ].copy()
    if feature_input["rag_answer"].isna().any() or feature_input["rag_answer"].astype(str).str.strip().eq("").any():
        raise ValueError("Every record needs a generated RAG answer for custom feature extraction.")
    feature_input = feature_input.rename(columns={"rag_answer": "answer"})
    features = extractor.transform(feature_input)
    features = features.sort_values("id").reset_index(drop=True)
    features.to_csv(FEATURES_PATH, index=False)
    return features


def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        sparse_threshold=0.0,
        remainder="drop",
    )


def model_registry(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_split=3,
            class_weight="balanced",
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def custom_feature_groups() -> dict[str, list[str]]:
    return {
        "Semantic": [
            "semantic_question_evidence",
            "semantic_question_answer",
            "semantic_answer_evidence",
        ],
        "Lexical": [
            "answer_evidence_keyword_overlap",
            "answer_evidence_jaccard",
            "answer_evidence_token_overlap",
            "answer_evidence_coverage",
            "answer_unsupported_word_ratio",
            "answer_evidence_length_ratio",
            "answer_token_count",
            "evidence_token_count",
            "question_evidence_keyword_overlap",
            "question_evidence_jaccard",
            "question_evidence_token_overlap",
            "question_evidence_source_coverage",
            "question_evidence_target_coverage",
        ],
        "Technical-Consistency": [
            "entity_overlap",
            "entity_missing",
            "entity_extra",
            "number_match",
            "number_mismatch",
            "number_extra",
            "date_match",
            "date_mismatch",
            "date_extra",
            "technical_term_overlap",
            "technical_term_missing",
            "technical_term_extra",
            "technical_consistency_mismatch",
            "evidence_entity_count",
            "answer_entity_count",
            "evidence_number_count",
            "answer_number_count",
            "question_evidence_entity_overlap",
            "question_evidence_number_overlap",
            "question_evidence_term_overlap",
        ],
        "Metadata": [
            "question_type",
            "difficulty",
        ],
    }


def train_model_on_columns(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_cols: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    random_state: int = 42,
) -> dict[str, Any]:
    categorical_cols = [col for col in ["question_type", "difficulty"] if col in feature_cols]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    pipe = Pipeline([
        ("preprocess", build_preprocessor(categorical_cols, numeric_cols)),
        ("model", model_registry(random_state)[model_name]),
    ])
    train_val_idx = np.concatenate([train_idx, val_idx])
    pipe.fit(features.iloc[train_val_idx][feature_cols], labels.iloc[train_val_idx])
    y_test = labels.iloc[test_idx]
    pred = pipe.predict(features.iloc[test_idx][feature_cols])
    return {"pipeline": pipe, "metrics": evaluate_predictions(y_test, pred)}


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = int(tp + tn + fp + fn)
    assert total == len(y_true), "Confusion-matrix count does not match evaluation set size."
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    false_negative_rate = float(fn / (fn + tp)) if (fn + tp) else 0.0
    return {
        "n": total,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Hallucinated", "Hallucinated"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_roc_curve(y_true: pd.Series, y_score: np.ndarray, output_path: Path, title: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return float(auc)


def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return float(ap)


def plot_performance_bar(metrics_df: pd.DataFrame, output_path: Path) -> None:
    ordered = metrics_df.sort_values("f1", ascending=False)
    x = np.arange(len(ordered))
    width = 0.2
    plt.figure(figsize=(10, 5))
    for offset, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
        plt.bar(x + (offset - 1.5) * width, ordered[metric], width, label=metric.title())
    plt.xticks(x, ordered["display_name"], rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Detector Performance on 300 Human-Labelled Records")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_heldout_performance(metrics_df: pd.DataFrame, output_path: Path) -> None:
    ordered = metrics_df.sort_values("f1", ascending=False)
    x = np.arange(len(ordered))
    metrics = ("accuracy", "precision", "recall", "f1")
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for position, metric in enumerate(metrics):
        ax.bar(x + (position - 1.5) * width, ordered[metric], width, label=metric.title())
    ax.set_xticks(x, [METHOD_DISPLAY_NAMES[method] for method in ordered["method"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Detector Performance on {int(ordered['n'].iloc[0])}-Record Held-Out Test Split")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_score_distribution(master: pd.DataFrame, score_col: str, output_path: Path, title: str) -> None:
    non_hall = master.loc[master["human_label"] == 0, score_col].dropna().astype(float)
    hall = master.loc[master["human_label"] == 1, score_col].dropna().astype(float)
    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 16)
    plt.hist(non_hall, bins=bins, alpha=0.65, label="Actual Non-Hallucinated")
    plt.hist(hall, bins=bins, alpha=0.65, label="Actual Hallucinated")
    plt.xlabel("Hallucination risk score")
    plt.ylabel("Record count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_threshold_analysis(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4.5))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.plot(df["threshold"], df[metric], label=metric.title())
    plt.xlabel("Hallucination-risk threshold")
    plt.ylabel("Metric value")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_risk_distribution(master: pd.DataFrame, output_path: Path) -> None:
    values = master["final_risk_score"].dropna().astype(float)
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=np.arange(0, 105, 5), color="#356fb3", alpha=0.8)
    for boundary in [30, 60, 80]:
        plt.axvline(boundary, color="#555", linestyle="--", linewidth=1)
    plt.xlabel("Final risk score (0-100)")
    plt.ylabel("Record count")
    plt.title("Final Risk Score Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def risk_level(score: float) -> str:
    if score <= 30:
        return "Low Risk"
    if score <= 60:
        return "Medium Risk"
    if score <= 80:
        return "High Risk"
    return "Critical Risk"


def integrity_report(raw_merged: pd.DataFrame, master: pd.DataFrame) -> dict[str, Any]:
    source_rows = {
        "baseline": read_jsonl(BASELINE_PATH),
        "ragas": read_jsonl(RAGAS_PATH),
        "selfcheck": read_jsonl(SC_PATH),
        "similarity": read_jsonl(SIM_PATH),
        "human": read_jsonl(HUMAN_PATH),
    }
    source_keys = {
        "baseline": "id",
        "ragas": "original_id",
        "selfcheck": "id",
        "similarity": "id",
        "human": "id",
    }

    duplicate_ids: dict[str, list[str]] = {}
    missing_ids: dict[str, list[str]] = {}
    human_ids = _canonical_id_set(source_rows["human"], "id")
    for name, rows in source_rows.items():
        ids = [str(row.get(source_keys[name])) for row in rows if row.get(source_keys[name]) is not None]
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        duplicate_ids[name] = duplicates
        missing_ids[name] = sorted(human_ids - set(ids))

    missing_prediction_records = {}
    for method in METHOD_DISPLAY_NAMES:
        pred_col = f"{method}_prediction"
        if pred_col in master:
            missing_prediction_records[method] = master.loc[master[pred_col].isna(), "id"].astype(str).tolist()

    label_values = sorted(raw_merged["human_label"].dropna().astype(int).unique().tolist())
    class_counts = raw_merged["human_label"].astype(int).value_counts().to_dict()
    return {
        "dataset_path": str(HUMAN_PATH.resolve()),
        "human_annotation_count": int(len(source_rows["human"])),
        "merged_evaluation_count": int(len(raw_merged)),
        "evaluated_prediction_count": int(len(master)),
        "human_label_values": label_values,
        "positive_class": {"value": 1, "meaning": "Hallucinated"},
        "negative_class": {"value": 0, "meaning": "Non-Hallucinated"},
        "hallucinated_examples": int(class_counts.get(1, 0)),
        "non_hallucinated_examples": int(class_counts.get(0, 0)),
        "duplicate_ids": duplicate_ids,
        "missing_ids_against_human": missing_ids,
        "missing_prediction_ids": missing_prediction_records,
        "all_counts_are_300": all(len(rows) == 300 for rows in source_rows.values()) and len(master) == 300,
        "ids_match_human": all(not values for values in missing_ids.values()),
        "labels_are_binary": label_values == [0, 1],
    }


def threshold_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = np.linspace(0, 1, 101)
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        metrics = evaluate_predictions(pd.Series(y_true), pred)
        rows.append({"threshold": float(threshold), **metrics})

    best = max(rows, key=lambda item: (item["f1"], item["recall"], item["precision"]))
    return rows, dict(best)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def evaluate_full_dataset(
    raw_merged: pd.DataFrame,
    features: pd.DataFrame,
    custom_artifact: dict[str, Any],
    hybrid_model: LogisticRegression,
) -> dict[str, Any]:
    master = raw_merged.copy().reset_index(drop=True)
    y_true = master["human"].astype(int).to_numpy()

    custom_pipeline: Pipeline = custom_artifact["pipeline"]
    custom_cols = custom_artifact["categorical_cols"] + custom_artifact["numeric_cols"]
    custom_x = features[custom_cols]
    custom_pred = custom_pipeline.predict(custom_x).astype(int)
    custom_prob = custom_pipeline.predict_proba(custom_x)[:, 1] if hasattr(custom_pipeline, "predict_proba") else custom_pred

    hybrid_x = master[["baseline", "selfcheck", "ragas", "similarity"]].astype(int)
    hybrid_pred = hybrid_model.predict(hybrid_x).astype(int)
    hybrid_prob = hybrid_model.predict_proba(hybrid_x)[:, 1] if hasattr(hybrid_model, "predict_proba") else hybrid_pred

    score_definitions = {
        "baseline": {
            "score_col": None,
            "prediction_col": "baseline_prediction",
            "continuous": False,
            "threshold": None,
            "direction": "Binary detector output: 1 means hallucinated.",
        },
        "ragas": {
            "score_col": "ragas_risk_score",
            "prediction_col": "ragas_prediction",
            "continuous": True,
            "threshold": 0.5,
            "direction": "Risk score = 1 - faithfulness; higher means more hallucination risk.",
        },
        "selfcheck": {
            "score_col": None,
            "prediction_col": "selfcheck_prediction",
            "continuous": False,
            "threshold": None,
            "direction": "Binary detector output: 1 means hallucinated.",
        },
        "similarity": {
            "score_col": "similarity_risk_score",
            "prediction_col": "similarity_prediction",
            "continuous": True,
            "threshold": 0.5,
            "direction": "Risk score = 1 - retrieval similarity; higher means more hallucination risk.",
        },
        "hybrid": {
            "score_col": "hybrid_score",
            "prediction_col": "hybrid_prediction",
            "continuous": True,
            "threshold": 0.5,
            "direction": "Logistic-regression probability for the hallucinated class.",
        },
        "custom": {
            "score_col": "custom_score",
            "prediction_col": "custom_prediction",
            "continuous": True,
            "threshold": 0.5,
            "direction": "Classifier probability for the hallucinated class.",
        },
    }

    master["baseline_score"] = np.nan
    master["baseline_prediction"] = master["baseline"].astype(int)
    master["ragas_score"] = master["ragas_faithfulness"].astype(float)
    master["ragas_risk_score"] = 1.0 - master["ragas_score"]
    master["ragas_prediction"] = master["ragas"].astype(int)
    master["selfcheck_score"] = np.nan
    master["selfcheck_prediction"] = master["selfcheck"].astype(int)
    master["similarity_raw_score"] = master["similarity_score"].astype(float)
    master["similarity_risk_score"] = 1.0 - master["similarity_raw_score"]
    master["similarity_prediction"] = master["similarity"].astype(int)
    master["hybrid_score"] = hybrid_prob
    master["hybrid_prediction"] = hybrid_pred
    master["custom_score"] = custom_prob
    master["custom_prediction"] = custom_pred

    risk_signal_cols = ["ragas_risk_score", "similarity_risk_score", "hybrid_score", "custom_score"]
    master["final_risk_probability"] = master[risk_signal_cols].mean(axis=1)
    master["final_risk_score"] = (master["final_risk_probability"] * 100).clip(0, 100)
    master["final_risk_level"] = master["final_risk_score"].apply(lambda value: risk_level(float(value)))

    metrics_rows: list[dict[str, Any]] = []
    confusion_matrices: dict[str, Any] = {}
    roc_curves: dict[str, Any] = {}
    pr_curves: dict[str, Any] = {}
    threshold_analysis: dict[str, Any] = {}

    for method, definition in score_definitions.items():
        pred = master[definition["prediction_col"]].astype(int).to_numpy()
        metrics = evaluate_predictions(pd.Series(y_true), pred)
        metrics_rows.append(
            {
                "method": method,
                "display_name": METHOD_DISPLAY_NAMES[method],
                "n": int(len(y_true)),
                **metrics,
            }
        )
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        confusion_matrices[method] = {
            "labels": {
                "actual_0": "Actual Non-Hallucinated",
                "actual_1": "Actual Hallucinated",
                "predicted_0": "Predicted Non-Hallucinated",
                "predicted_1": "Predicted Hallucinated",
            },
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
        plot_confusion_matrix(pd.Series(y_true), pred, PLOT_DIR / f"{method}_full_confusion_matrix.png", f"{METHOD_DISPLAY_NAMES[method]} Confusion Matrix")

        score_col = definition["score_col"]
        if definition["continuous"] and score_col:
            scores = master[score_col].astype(float).to_numpy()
            fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
            precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
            roc_auc = float(roc_auc_score(y_true, scores))
            average_precision = float(average_precision_score(y_true, scores))
            plot_roc_curve(pd.Series(y_true), scores, PLOT_DIR / f"{method}_full_roc_curve.png", f"{METHOD_DISPLAY_NAMES[method]} ROC Curve")
            plot_precision_recall_curve(y_true, scores, PLOT_DIR / f"{method}_precision_recall_curve.png", f"{METHOD_DISPLAY_NAMES[method]} Precision-Recall Curve")
            plot_score_distribution(master, score_col, PLOT_DIR / f"{method}_score_distribution.png", f"{METHOD_DISPLAY_NAMES[method]} Score Distribution")
            threshold_rows, best_threshold = threshold_curve(y_true, scores)
            plot_threshold_analysis(threshold_rows, PLOT_DIR / f"{method}_threshold_analysis.png", f"{METHOD_DISPLAY_NAMES[method]} Threshold Analysis")
            roc_curves[method] = {
                "auc": roc_auc,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            }
            pr_curves[method] = {
                "average_precision": average_precision,
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            }
            threshold_analysis[method] = {
                "existing_threshold": definition["threshold"],
                "best_f1_threshold_on_full_dataset": best_threshold,
                "leakage_note": "Best threshold is descriptive only because it is selected on the same 300-record evaluation dataset.",
                "curve": threshold_rows,
            }
        else:
            roc_curves[method] = {"available": False, "reason": "No meaningful continuous score is stored for this detector."}
            pr_curves[method] = {"available": False, "reason": "No meaningful continuous score is stored for this detector."}
            threshold_analysis[method] = {"available": False, "reason": "Detector output is binary only."}

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(FULL_EVALUATION_CSV_PATH, index=False)
    plot_performance_bar(metrics_df, PLOT_DIR / "performance_comparison.png")
    plot_risk_distribution(master, PLOT_DIR / "risk_score_distribution.png")

    error_rows: list[dict[str, Any]] = []
    for _, row in master.iterrows():
        detector_errors = []
        detector_correct = []
        for method in METHOD_DISPLAY_NAMES:
            pred = int(row[f"{method}_prediction"])
            if pred == int(row["human"]):
                detector_correct.append(method)
            else:
                detector_errors.append(method)
        error_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "human_label": int(row["human"]),
                "human_label_text": "Hallucinated" if int(row["human"]) == 1 else "Non-Hallucinated",
                "hallucination_type": row.get("hallucination_type", ""),
                "error_category": row.get("hallucination_type", "") or "Requires manual annotation",
                "baseline_prediction": int(row["baseline_prediction"]),
                "ragas_prediction": int(row["ragas_prediction"]),
                "selfcheck_prediction": int(row["selfcheck_prediction"]),
                "similarity_prediction": int(row["similarity_prediction"]),
                "hybrid_prediction": int(row["hybrid_prediction"]),
                "custom_prediction": int(row["custom_prediction"]),
                "detectors_incorrect": ",".join(detector_errors),
                "detectors_correct": ",".join(detector_correct),
                "all_detectors_correct": len(detector_errors) == 0,
                "any_false_negative": int(row["human"]) == 1 and any(int(row[f"{method}_prediction"]) == 0 for method in METHOD_DISPLAY_NAMES),
                "any_false_positive": int(row["human"]) == 0 and any(int(row[f"{method}_prediction"]) == 1 for method in METHOD_DISPLAY_NAMES),
            }
        )

    pd.DataFrame(error_rows).to_csv(ERROR_ANALYSIS_CSV_PATH, index=False)
    write_jsonl(ERROR_ANALYSIS_JSONL_PATH, error_rows)

    export_cols = [
        "id",
        "question",
        "answer",
        "evidence",
        "human_label",
        "baseline_score",
        "baseline_prediction",
        "ragas_score",
        "ragas_risk_score",
        "ragas_prediction",
        "selfcheck_score",
        "selfcheck_prediction",
        "similarity_raw_score",
        "similarity_risk_score",
        "similarity_prediction",
        "hybrid_score",
        "hybrid_prediction",
        "custom_score",
        "custom_prediction",
        "final_risk_score",
        "final_risk_level",
    ]
    master[export_cols].to_csv(MASTER_EVALUATION_CSV_PATH, index=False)
    write_jsonl(MASTER_EVALUATION_JSONL_PATH, master[export_cols].to_dict(orient="records"))

    integrity = integrity_report(raw_merged, master)
    risk_stats = {
        "definition": "Final risk probability is the unweighted mean of the available continuous hallucination-risk signals: RAGAS risk (1-faithfulness), retrieval-similarity risk (1-similarity), Hybrid probability, and Custom probability. The percentage score is probability * 100.",
        "inputs": risk_signal_cols,
        "weights": {col: 0.25 for col in risk_signal_cols},
        "categories": [{"label": label, "min": low, "max": high} for label, low, high in RISK_CATEGORIES],
        "note": "Risk category boundaries are design choices for interpretation, not scientifically validated thresholds.",
        "mean": float(master["final_risk_score"].mean()),
        "median": float(master["final_risk_score"].median()),
        "min": float(master["final_risk_score"].min()),
        "max": float(master["final_risk_score"].max()),
        "by_level": master["final_risk_level"].value_counts().to_dict(),
    }

    summary = {
        "evaluation_scope": "full_300_apparent_evaluation",
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "reproducibility": {
            "random_seed": 42,
            "positive_class": "1 = Hallucinated",
            "negative_class": "0 = Non-Hallucinated",
            "metric_method": "sklearn binary classification metrics with zero_division=0; confusion matrix labels=[0,1].",
            "leakage_note": "Hybrid and Custom full-300 metrics include records used for model training; use saved held-out test summaries for unbiased supervised-model performance.",
        },
        "data_integrity": integrity,
        "score_definitions": score_definitions,
        "metrics": metrics_rows,
        "confusion_matrices": confusion_matrices,
        "roc": roc_curves,
        "precision_recall": pr_curves,
        "threshold_analysis": threshold_analysis,
        "risk": risk_stats,
        "outputs": {
            "master_evaluation_csv": str(MASTER_EVALUATION_CSV_PATH.resolve()),
            "master_evaluation_jsonl": str(MASTER_EVALUATION_JSONL_PATH.resolve()),
            "evaluation_results_json": str(FULL_EVALUATION_JSON_PATH.resolve()),
            "evaluation_results_csv": str(FULL_EVALUATION_CSV_PATH.resolve()),
            "error_analysis_csv": str(ERROR_ANALYSIS_CSV_PATH.resolve()),
            "error_analysis_jsonl": str(ERROR_ANALYSIS_JSONL_PATH.resolve()),
            "plots_dir": str(PLOT_DIR.resolve()),
        },
    }
    write_json(FULL_EVALUATION_JSON_PATH, summary)
    return summary


def split_indices(y: pd.Series, random_state: int = 42, test_size: float = 0.15, val_size: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    val_ratio = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y.iloc[train_val_idx],
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def train_custom_detector(features: pd.DataFrame, labels: pd.Series, random_state: int = 42) -> dict[str, Any]:
    categorical_cols = ["question_type", "difficulty"]
    ignore_cols = {"id", *categorical_cols}
    numeric_cols = [col for col in features.columns if col not in ignore_cols]

    x = features[categorical_cols + numeric_cols]
    y = labels.astype(int).reset_index(drop=True)
    train_idx, val_idx, test_idx = split_indices(y, random_state=random_state)

    models = model_registry(random_state)
    results: list[dict[str, Any]] = []
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_val_f1 = -1.0

    x_train, x_val, y_train, y_val = x.iloc[train_idx], x.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor(categorical_cols, numeric_cols)),
            ("model", estimator),
        ])
        pipe.fit(x_train, y_train)
        val_pred = pipe.predict(x_val)
        val_metrics = evaluate_predictions(y_val, val_pred)
        results.append({"model": name, **val_metrics})
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_name = name
            best_pipeline = pipe

    assert best_name is not None and best_pipeline is not None

    x_train_final = x.iloc[np.concatenate([train_idx, val_idx])]
    y_train_final = y.iloc[np.concatenate([train_idx, val_idx])]
    best_pipeline.fit(x_train_final, y_train_final)

    x_test = x.iloc[test_idx]
    y_test = y.iloc[test_idx]
    test_pred = best_pipeline.predict(x_test)
    test_metrics = evaluate_predictions(y_test, test_pred)
    class_report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)

    roc_auc = None
    if hasattr(best_pipeline, "predict_proba"):
        probs = best_pipeline.predict_proba(x_test)
        if probs.shape[1] == 2:
            roc_auc = plot_roc_curve(y_test, probs[:, 1], PLOT_DIR / "custom_roc_curve.png", "Custom Detector ROC Curve")

    plot_confusion_matrix(y_test, test_pred, PLOT_DIR / "custom_confusion_matrix.png", "Custom Detector Confusion Matrix")

    preprocessor: ColumnTransformer = best_pipeline.named_steps["preprocess"]
    estimator = best_pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(feature_names))

    feature_importance = pd.DataFrame({"feature": feature_names, "importance": values}).sort_values(
        "importance", ascending=False
    )
    feature_importance.to_csv(REPORT_DIR / "custom_feature_importance.csv", index=False)

    model_path = MODEL_DIR / "custom_hallucination_detector.pkl"
    artifact = {
        "pipeline": best_pipeline,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "best_model": best_name,
    }
    joblib.dump(artifact, model_path)

    predictions = pd.DataFrame(
        {
            "id": features.iloc[test_idx]["id"].astype(str).values,
            "human": y_test.values,
            "custom_prediction": test_pred,
        }
    )
    predictions.to_csv(DATA_DIR / "custom_test_predictions.csv", index=False)

    summary = {
        "best_model": best_name,
        "validation_results": results,
        "test_metrics": test_metrics,
        "roc_auc": roc_auc,
        "model_path": str(model_path.resolve()),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "classification_report": class_report,
        "feature_importance_top": feature_importance.head(15).to_dict(orient="records"),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
    write_json(CUSTOM_SUMMARY_PATH, summary)
    return {
        "artifact": artifact,
        "summary": summary,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "x": x,
        "y": y,
        "pipeline": best_pipeline,
    }


def train_hybrid_detector(combined: pd.DataFrame, split_bundle: dict[str, Any], random_state: int = 42) -> dict[str, Any]:
    x = combined[["baseline", "selfcheck", "ragas", "similarity"]].astype(int)
    y = combined["human"].astype(int)
    train_idx = split_bundle["train_idx"]
    val_idx = split_bundle["val_idx"]
    test_idx = split_bundle["test_idx"]

    x_train = x.iloc[train_idx]
    y_train = y.iloc[train_idx]
    x_val = x.iloc[val_idx]
    y_val = y.iloc[val_idx]
    x_train_final = x.iloc[np.concatenate([train_idx, val_idx])]
    y_train_final = y.iloc[np.concatenate([train_idx, val_idx])]
    x_test = x.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    val_metrics = evaluate_predictions(y_val, val_pred)

    model.fit(x_train_final, y_train_final)
    test_pred = model.predict(x_test)
    test_metrics = evaluate_predictions(y_test, test_pred)
    class_report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)

    plot_confusion_matrix(y_test, test_pred, PLOT_DIR / "hybrid_confusion_matrix.png", "Hybrid Detector Confusion Matrix")
    model_path = MODEL_DIR / "hybrid_detector_model.pkl"
    joblib.dump(model, model_path)

    predictions = pd.DataFrame(
        {
            "id": combined.iloc[test_idx]["id"].astype(str).values,
            "human": y_test.values,
            "hybrid_prediction": test_pred,
        }
    )
    predictions.to_csv(DATA_DIR / "hybrid_test_predictions.csv", index=False)

    summary = {
        "model": "logistic_regression",
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": str(model_path.resolve()),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "classification_report": class_report,
        "weights": pd.DataFrame({"feature": x.columns, "weight": model.coef_[0]}).sort_values(
            "weight", ascending=False
        ).to_dict(orient="records"),
    }
    write_json(HYBRID_SUMMARY_PATH, summary)
    return {"model": model, "summary": summary}


def run_ablation_study(features: pd.DataFrame, labels: pd.Series, split_bundle: dict[str, Any]) -> dict[str, Any]:
    train_idx = split_bundle["train_idx"]
    val_idx = split_bundle["val_idx"]
    test_idx = split_bundle["test_idx"]
    feature_cols = [col for col in features.columns if col != "id"]
    groups = custom_feature_groups()
    model_name = split_bundle["summary"]["best_model"]

    grouped = sorted({feature for cols in groups.values() for feature in cols})
    missing = sorted(set(grouped) - set(feature_cols))
    ungrouped = sorted(set(feature_cols) - set(grouped))
    if missing or ungrouped:
        raise ValueError(f"Feature-group coverage mismatch. Missing={missing}; ungrouped={ungrouped}")

    full = train_model_on_columns(features, labels, feature_cols, train_idx, val_idx, test_idx, model_name)
    full_metrics = full["metrics"]

    rows: list[dict[str, Any]] = []
    for group_name, group_features in groups.items():
        kept_features = [col for col in feature_cols if col not in group_features]
        ablated = train_model_on_columns(features, labels, kept_features, train_idx, val_idx, test_idx, model_name)
        metrics = ablated["metrics"]
        rows.append(
            {
                "feature_group_removed": group_name,
                "features_removed": len(group_features),
                "features_remaining": len(kept_features),
                "full_f1": full_metrics["f1"],
                "ablated_f1": metrics["f1"],
                "f1_change": metrics["f1"] - full_metrics["f1"],
                "f1_change_percentage_points": (metrics["f1"] - full_metrics["f1"]) * 100,
                "full_recall": full_metrics["recall"],
                "ablated_recall": metrics["recall"],
                "recall_change": metrics["recall"] - full_metrics["recall"],
                "full_accuracy": full_metrics["accuracy"],
                "ablated_accuracy": metrics["accuracy"],
                "ablated_precision": metrics["precision"],
                "ablated_specificity": metrics["specificity"],
                "ablated_false_positive_rate": metrics["false_positive_rate"],
                "ablated_false_negative_rate": metrics["false_negative_rate"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "metrics": metrics,
            }
        )

    csv_columns = [
        "feature_group_removed",
        "features_removed",
        "features_remaining",
        "full_f1",
        "ablated_f1",
        "f1_change",
        "f1_change_percentage_points",
        "full_recall",
        "ablated_recall",
        "recall_change",
        "full_accuracy",
        "ablated_accuracy",
        "ablated_precision",
        "ablated_specificity",
        "ablated_false_positive_rate",
        "ablated_false_negative_rate",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    ablation_df = pd.DataFrame(rows)[csv_columns]
    ablation_df.to_csv(ABLATION_CSV_PATH, index=False)
    summary = {
        "protocol": (
            "Logistic Regression is retrained with the same final split protocol as the custom detector: "
            "train and validation records are combined for fitting, and metrics are computed only on the "
            "held-out 45-record test set. No model is fitted on test labels."
        ),
        "n_train_plus_validation": int(len(train_idx) + len(val_idx)),
        "n_test": int(len(test_idx)),
        "full_model_metrics": full_metrics,
        "feature_groups": {name: cols for name, cols in groups.items()},
        "ablation_rows": rows,
        "outputs": {
            "csv": str(ABLATION_CSV_PATH.resolve()),
            "json": str(ABLATION_JSON_PATH.resolve()),
        },
    }
    write_json(ABLATION_JSON_PATH, summary)
    return summary


def bootstrap_ci(values: np.ndarray, ci_level: float = 95.0) -> list[float]:
    lower, upper = np.percentile(values, [(100 - ci_level) / 2, (100 + ci_level) / 2])
    return [float(lower), float(upper)]


def evaluate_all_methods(
    raw_merged: pd.DataFrame,
    features: pd.DataFrame,
    split_bundle: dict[str, Any],
    custom_artifact: dict[str, Any],
    hybrid_model: LogisticRegression,
) -> dict[str, Any]:
    test_idx = split_bundle["test_idx"]
    test_raw = raw_merged.iloc[test_idx].copy()
    test_features = features.iloc[test_idx].copy()

    y_true = test_raw["human"].astype(int).to_numpy()

    custom_pipeline: Pipeline = custom_artifact["pipeline"]
    custom_cols = custom_artifact["categorical_cols"] + custom_artifact["numeric_cols"]
    custom_pred = custom_pipeline.predict(test_features[custom_cols])
    custom_score = (
        custom_pipeline.predict_proba(test_features[custom_cols])[:, 1]
        if hasattr(custom_pipeline, "predict_proba")
        else custom_pred
    )

    hybrid_x = test_raw[["baseline", "selfcheck", "ragas", "similarity"]].astype(int)
    hybrid_pred = hybrid_model.predict(hybrid_x)
    hybrid_score = hybrid_model.predict_proba(hybrid_x)[:, 1] if hasattr(hybrid_model, "predict_proba") else hybrid_pred

    predictions = {
        "baseline": test_raw["baseline"].astype(int).to_numpy(),
        "ragas": test_raw["ragas"].astype(int).to_numpy(),
        "selfcheck": test_raw["selfcheck"].astype(int).to_numpy(),
        "similarity": test_raw["similarity"].astype(int).to_numpy(),
        "hybrid": hybrid_pred,
        "custom": custom_pred,
    }
    score_outputs = {
        "ragas": 1.0 - test_raw["ragas_faithfulness"].astype(float).to_numpy(),
        "similarity": 1.0 - test_raw["similarity_score"].astype(float).to_numpy(),
        "hybrid": hybrid_score,
        "custom": custom_score,
    }

    metrics_rows: list[dict[str, Any]] = []
    bootstrapped_f1: dict[str, list[float]] = {}
    score_metrics: dict[str, dict[str, float | str]] = {}
    rng = np.random.default_rng(42)
    n_boot = 5000
    bootstrap_samples = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))

    for name, pred in predictions.items():
        metrics = evaluate_predictions(pd.Series(y_true), pred)
        metrics_rows.append({"method": name, **metrics})

        boot = []
        for sample in bootstrap_samples:
            boot.append(f1_score(y_true[sample], pred[sample], zero_division=0))
        bootstrapped_f1[name] = bootstrap_ci(np.asarray(boot))

        if name in score_outputs:
            scores = np.asarray(score_outputs[name], dtype=float)
            score_metrics[name] = {
                "roc_auc": float(roc_auc_score(y_true, scores)),
                "pr_auc_average_precision": float(average_precision_score(y_true, scores)),
                "score_direction": "Higher score means greater hallucination risk.",
            }

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(METHOD_METRICS_PATH, index=False)
    plot_heldout_performance(metrics_df, HELDOUT_PERFORMANCE_PATH)

    summary = {
        "n_test_samples": int(len(test_idx)),
        "metrics": metrics_rows,
        "continuous_score_metrics": score_metrics,
        "f1_bootstrap_95ci": bootstrapped_f1,
    }
    write_json(METHOD_SUMMARY_PATH, summary)

    test_predictions = test_raw[["id", "human"]].copy()
    for name, pred in predictions.items():
        test_predictions[f"{name}_prediction"] = pred
    for name, score in score_outputs.items():
        test_predictions[f"{name}_score"] = score
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    correct_custom = custom_pred == y_true
    correct_hybrid = hybrid_pred == y_true
    b = int(np.sum((~correct_custom) & correct_hybrid))
    c = int(np.sum(correct_custom & (~correct_hybrid)))
    disagreements = b + c
    p_value = float(binomtest(max(b, c), n=disagreements, p=0.5, alternative="two-sided").pvalue) if disagreements else 1.0

    diff_boot = []
    for sample in bootstrap_samples:
        yb = y_true[sample]
        custom_b = custom_pred[sample]
        hybrid_b = hybrid_pred[sample]
        custom_f1 = f1_score(yb, custom_b, zero_division=0)
        hybrid_f1 = f1_score(yb, hybrid_b, zero_division=0)
        diff_boot.append(custom_f1 - hybrid_f1)

    statistics = {
        "bootstrap": {
            "iterations": n_boot,
            "resampling_unit": "held-out test record",
            "random_seed": 42,
            "method": "percentile bootstrap",
            "note": "Intervals are reported as lower and upper bounds. With only 45 held-out records, they should be interpreted cautiously.",
            "f1_ci": bootstrapped_f1,
            "custom_f1_95ci": bootstrapped_f1["custom"],
            "hybrid_f1_95ci": bootstrapped_f1["hybrid"],
            "difference_f1_95ci": bootstrap_ci(np.asarray(diff_boot)),
        },
        "paired_test": {
            "custom_wrong_hybrid_right": b,
            "custom_right_hybrid_wrong": c,
            "mcnemar_exact_p_value": p_value,
        },
    }
    write_json(STATISTICS_PATH, statistics)
    return {
        "metrics": metrics_rows,
        "summary": summary,
        "statistics": statistics,
    }


def run_pipeline() -> dict[str, Any]:
    alignment = check_alignment()
    merged_raw, combined = build_merged_datasets()
    features = build_features(merged_raw)
    labels = merged_raw.set_index("id").loc[features["id"], "human"].astype(int)

    split_bundle = train_custom_detector(features, labels=labels)
    hybrid_bundle = train_hybrid_detector(combined, split_bundle)
    ablation = run_ablation_study(features, labels, split_bundle)

    evaluation = evaluate_all_methods(
        raw_merged=merged_raw,
        features=features,
        split_bundle=split_bundle,
        custom_artifact=split_bundle["artifact"],
        hybrid_model=hybrid_bundle["model"],
    )
    full_evaluation = evaluate_full_dataset(
        raw_merged=merged_raw,
        features=features,
        custom_artifact=split_bundle["artifact"],
        hybrid_model=hybrid_bundle["model"],
    )

    return {
        "alignment": alignment,
        "custom_training_summary": split_bundle["summary"],
        "hybrid_training_summary": hybrid_bundle["summary"],
        "ablation": ablation,
        "evaluation": evaluation,
        "full_evaluation": full_evaluation,
    }


def main() -> None:
    result = run_pipeline()
    print(json.dumps(
        {
            "alignment_ok": result["alignment"]["alignment_ok"],
            "custom_test_f1": result["custom_training_summary"]["test_metrics"]["f1"],
            "hybrid_test_f1": result["hybrid_training_summary"]["test_metrics"]["f1"],
            "metrics_path": str(METHOD_METRICS_PATH),
            "full_evaluation_path": str(FULL_EVALUATION_JSON_PATH),
            "statistics_path": str(STATISTICS_PATH),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
