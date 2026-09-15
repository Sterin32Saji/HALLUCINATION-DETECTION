from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "id",
    "question",
    "answer",
    "evidence",
    "rag_answer",
    "question_type",
    "difficulty",
    "source_doc",
    "human",
]

LABEL_ALIASES = ["human", "human_label", "label", "target", "hallucination"]


def _normalize_evidence(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    if value is None:
        return ""
    return str(value)


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
    return pd.DataFrame(rows)


def load_dataset(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if data_path.suffix.lower() == ".jsonl":
        df = load_jsonl(data_path)
    elif data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Supported dataset formats are .jsonl and .csv")

    out = df.copy()

    # Normalize supervision column to a canonical `human` label.
    if "human" not in out.columns:
        alias = next((name for name in LABEL_ALIASES if name in out.columns), None)
        if alias is not None:
            out["human"] = out[alias]

    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    out["question"] = out["question"].map(_safe_str)
    out["answer"] = out["answer"].map(_safe_str)
    out["evidence"] = out["evidence"].map(_normalize_evidence)
    out["rag_answer"] = out["rag_answer"].map(_safe_str)
    out["question_type"] = out["question_type"].map(_safe_str)
    out["difficulty"] = out["difficulty"].map(_safe_str)
    out["source_doc"] = out["source_doc"].map(_safe_str)
    # Accept boolean or numeric labels.
    out["human"] = out["human"].astype(int)

    return out


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
