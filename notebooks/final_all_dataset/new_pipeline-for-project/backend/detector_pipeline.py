from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from config import RAGAS_THRESHOLD, SIMILARITY_THRESHOLD


def _to_int01(value: object, field_name: str) -> int:
    if value in (None, ""):
        raise ValueError(f"Missing value for {field_name}")

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        ivalue = int(value)
        if ivalue not in (0, 1):
            raise ValueError(f"Expected 0/1 for {field_name}, got {value}")
        return ivalue

    svalue = str(value).strip()
    if svalue in {"0", "1"}:
        return int(svalue)

    raise ValueError(f"Expected 0/1 for {field_name}, got {value}")


def _to_float(value: object, field_name: str) -> float:
    if value in (None, ""):
        raise ValueError(f"Missing value for {field_name}")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Invalid float for {field_name}: {value}") from exc


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_jsonl_stream(file_stream: IO[bytes]) -> list[dict]:
    rows = []
    for raw in file_stream:
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _merge_rows(
    baseline_rows: list[dict],
    ragas_rows: list[dict],
    selfcheck_rows: list[dict],
    similarity_rows: list[dict],
    human_rows: list[dict] | None = None,
) -> list[dict]:
    merged: dict[str, dict] = {}

    for row in baseline_rows:
        rid = row.get("id")
        if rid is None:
            continue
        merged[rid] = {
            "id": rid,
            "question": row.get("question"),
            "baseline": _to_int01(row.get("hallucination"), "baseline.hallucination"),
        }

    for row in ragas_rows:
        rid = row.get("original_id")
        if rid is None or rid not in merged:
            continue
        faithfulness = _to_float(row.get("faithfulness"), "ragas.faithfulness")
        merged[rid]["ragas_raw_faithfulness"] = faithfulness
        merged[rid]["ragas"] = 1 if faithfulness < RAGAS_THRESHOLD else 0

    for row in selfcheck_rows:
        rid = row.get("id")
        if rid is None or rid not in merged:
            continue
        merged[rid]["selfcheck"] = _to_int01(
            row.get("selfcheck_hallucination"), "selfcheck.selfcheck_hallucination"
        )

    for row in similarity_rows:
        rid = row.get("id")
        if rid is None or rid not in merged:
            continue
        similarity = _to_float(row.get("retrail_similarity"), "similarity.retrail_similarity")
        merged[rid]["similarity_raw_score"] = similarity
        merged[rid]["similarity"] = 1 if similarity < SIMILARITY_THRESHOLD else 0

    if human_rows:
        for row in human_rows:
            rid = row.get("id")
            if rid is None or rid not in merged:
                continue
            merged[rid]["human"] = _to_int01(row.get("human_label"), "human.human_label")

    required = {"baseline", "ragas", "selfcheck", "similarity"}
    output = []
    for rid, row in merged.items():
        if required.issubset(row.keys()):
            output.append(row)

    return output


def build_records_from_paths(
    baseline_path: Path,
    ragas_path: Path,
    selfcheck_path: Path,
    similarity_path: Path,
    human_path: Path | None = None,
) -> list[dict]:
    baseline_rows = _read_jsonl(baseline_path)
    ragas_rows = _read_jsonl(ragas_path)
    selfcheck_rows = _read_jsonl(selfcheck_path)
    similarity_rows = _read_jsonl(similarity_path)
    human_rows = _read_jsonl(human_path) if human_path else None

    return _merge_rows(
        baseline_rows=baseline_rows,
        ragas_rows=ragas_rows,
        selfcheck_rows=selfcheck_rows,
        similarity_rows=similarity_rows,
        human_rows=human_rows,
    )


def build_records_from_streams(
    baseline_file: IO[bytes],
    ragas_file: IO[bytes],
    selfcheck_file: IO[bytes],
    similarity_file: IO[bytes],
    human_file: IO[bytes] | None = None,
) -> list[dict]:
    baseline_rows = _read_jsonl_stream(baseline_file)
    ragas_rows = _read_jsonl_stream(ragas_file)
    selfcheck_rows = _read_jsonl_stream(selfcheck_file)
    similarity_rows = _read_jsonl_stream(similarity_file)
    human_rows = _read_jsonl_stream(human_file) if human_file else None

    return _merge_rows(
        baseline_rows=baseline_rows,
        ragas_rows=ragas_rows,
        selfcheck_rows=selfcheck_rows,
        similarity_rows=similarity_rows,
        human_rows=human_rows,
    )
