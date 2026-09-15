from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from feature_extractor import CustomFeatureExtractor
from utils import load_dataset, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run custom hallucination detector inference")
    parser.add_argument("--model", required=True, help="Path to trained model artifact (.pkl)")
    parser.add_argument("--input", required=True, help="Input .jsonl or .csv dataset")
    parser.add_argument("--output", required=True, help="Output path (.jsonl or .csv)")
    return parser.parse_args()


def run_inference(model_path: str | Path, input_path: str | Path, output_path: str | Path) -> Path:
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    categorical_cols = artifact["categorical_cols"]
    numeric_cols = artifact["numeric_cols"]

    raw_df = load_dataset(input_path)
    extractor = CustomFeatureExtractor()
    feature_df = extractor.transform(raw_df)

    x = feature_df[categorical_cols + numeric_cols]

    y_pred = pipeline.predict(x)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(x)
    else:
        probs = None

    output = raw_df.copy()
    output["custom_prediction"] = y_pred.astype(int)

    if probs is not None and probs.shape[1] == 2:
        output["custom_probability_not_hallucinated"] = probs[:, 0]
        output["custom_probability_hallucinated"] = probs[:, 1]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".jsonl":
        write_jsonl(output.to_dict(orient="records"), out)
    elif out.suffix.lower() == ".csv":
        output.to_csv(out, index=False)
    else:
        raise ValueError("Output format must be .jsonl or .csv")

    return out


def main() -> None:
    args = parse_args()
    output = run_inference(args.model, args.input, args.output)
    print(f"Saved predictions to: {output.resolve()}")


if __name__ == "__main__":
    main()
