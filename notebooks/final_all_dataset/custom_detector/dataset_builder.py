from __future__ import annotations

import argparse
from pathlib import Path

from feature_extractor import CustomFeatureExtractor
from utils import load_dataset


def build_feature_dataset(input_path: str | Path, output_path: str | Path) -> Path:
    df = load_dataset(input_path)
    extractor = CustomFeatureExtractor()
    feature_df = extractor.transform(df)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out, index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature dataset for custom hallucination detector")
    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset path (.jsonl or .csv) with columns including answer/evidence/rag_answer/human",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output feature CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_feature_dataset(args.input, args.output)
    print(f"Saved engineered feature dataset to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
