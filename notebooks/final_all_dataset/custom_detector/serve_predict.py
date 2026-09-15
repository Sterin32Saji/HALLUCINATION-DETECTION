from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from feature_extractor import CustomFeatureExtractor


REQUIRED_RAW_FIELDS = [
    "id",
    "question",
    "answer",
    "evidence",
    "rag_answer",
    "question_type",
    "difficulty",
    "source_doc",
]


def _ensure_fields(row: dict) -> dict:
    out = dict(row)
    for key in REQUIRED_RAW_FIELDS:
        out.setdefault(key, "")
    return out


def _load_payload_from_stdin() -> list[dict]:
    payload = sys.stdin.read().strip()
    if not payload:
        raise ValueError("Empty request payload")

    obj = json.loads(payload)
    if isinstance(obj, dict):
        return [_ensure_fields(obj)]
    if isinstance(obj, list):
        return [_ensure_fields(item) for item in obj]

    raise ValueError("Payload must be a JSON object or array")


def run_prediction(model_path: str | Path, rows: list[dict]) -> dict:
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    categorical_cols = artifact["categorical_cols"]
    numeric_cols = artifact["numeric_cols"]

    raw_df = pd.DataFrame(rows)
    extractor = CustomFeatureExtractor()
    feature_df = extractor.transform(raw_df)

    x = feature_df[categorical_cols + numeric_cols]
    y_pred = pipeline.predict(x)

    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(x)

    predictions = []
    for idx, row in raw_df.iterrows():
        result = {
            "id": str(row.get("id", "")),
            "question": str(row.get("question", "")),
            "custom_prediction": int(y_pred[idx]),
        }
        if probs is not None and probs.shape[1] == 2:
            result["custom_probability_not_hallucinated"] = float(probs[idx][0])
            result["custom_probability_hallucinated"] = float(probs[idx][1])
        predictions.append(result)

    return {
        "count": len(predictions),
        "hallucinated": int(sum(x["custom_prediction"] for x in predictions)),
        "not_hallucinated": int(sum(1 for x in predictions if x["custom_prediction"] == 0)),
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve custom detector predictions from stdin JSON")
    parser.add_argument("--model", required=True, help="Path to custom_hallucination_detector.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_payload_from_stdin()
    result = run_prediction(args.model, rows)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
