from __future__ import annotations

import json
from typing import IO

from live_detectors import compute_all_detectors


def parse_jsonl_stream(file_stream: IO[bytes]) -> list[dict]:
    rows = []
    for raw in file_stream:
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def compute_detectors(raw: dict) -> dict:
    return compute_all_detectors(raw)


def build_feature_records_from_raw_rows(raw_rows: list[dict]) -> list[dict]:
    return [compute_detectors(row) for row in raw_rows]
