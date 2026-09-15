from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import (
    DEFAULT_BASELINE_FILE,
    DEFAULT_HUMAN_FILE,
    MODEL_FEATURES,
    DEFAULT_RAGAS_FILE,
    DEFAULT_SELFCHECK_FILE,
    DEFAULT_SIMILARITY_FILE,
    MODEL_PATH,
    RAGAS_THRESHOLD,
    SIMILARITY_THRESHOLD,
)
from detector_pipeline import build_records_from_paths
from model_service import HybridModelService
from raw_detector_pipeline import (
    build_feature_records_from_raw_rows,
    compute_detectors,
    parse_jsonl_stream,
)

app = Flask(__name__)
CORS(app)

model_service = HybridModelService()


def _summarize(predictions: list[dict]) -> dict:
    total = len(predictions)
    hallucinated = sum(x["hybrid_prediction"] for x in predictions)
    not_hallucinated = total - hallucinated

    accuracy = None
    human_available = all("human" in x for x in predictions)
    if human_available and total > 0:
        correct = sum(1 for x in predictions if x["human"] == x["hybrid_prediction"])
        accuracy = correct / total

    return {
        "total": total,
        "hallucinated": hallucinated,
        "not_hallucinated": not_hallucinated,
        "accuracy_against_human": accuracy,
    }


def _parse_binary_feature(body: dict, name: str) -> int:
    value = body.get(name)
    if value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    raise ValueError(f"Invalid '{name}'. Expected 0 or 1.")


@app.get("/api/health")
def health() -> tuple:
    return jsonify(
        {
            "ok": True,
            "model_path": str(MODEL_PATH),
            "ragas_threshold": RAGAS_THRESHOLD,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        }
    )


@app.post("/api/predict")
def predict_single() -> tuple:
    body = request.get_json(silent=True) or {}

    try:
        feature_record = compute_detectors(body)
    except Exception as exc:
        return jsonify({"error": f"Failed to compute detectors: {exc}"}), 400

    predictions = model_service.predict_records([feature_record])
    out = predictions[0]
    out["answer"] = feature_record.get("answer")
    out["evidence"] = feature_record.get("evidence")
    out["rag_answer"] = feature_record.get("rag_answer")
    out["ragas_raw_faithfulness"] = feature_record["ragas_raw_faithfulness"]
    out["similarity_raw_score"] = feature_record["similarity_raw_score"]
    out["detector_debug"] = feature_record["detector_debug"]
    return jsonify(out)


@app.post("/api/predict-model-input")
def predict_model_input() -> tuple:
    body = request.get_json(silent=True) or {}

    try:
        feature_record = {
            "id": str(body.get("id") or "debug-sample"),
            "question": body.get("question"),
            "baseline": _parse_binary_feature(body, "baseline"),
            "ragas": _parse_binary_feature(body, "ragas"),
            "selfcheck": _parse_binary_feature(body, "selfcheck"),
            "similarity": _parse_binary_feature(body, "similarity"),
        }
    except ValueError as exc:
        return jsonify({"error": str(exc), "required_features": MODEL_FEATURES}), 400

    try:
        prediction = model_service.predict_records([feature_record])[0]
    except Exception as exc:
        return jsonify({"error": f"Failed to run hybrid model: {exc}"}), 400

    return jsonify({**prediction, "mode": "direct-model-input"})


@app.post("/api/predict-from-files")
def predict_from_files() -> tuple:
    input_file = request.files.get("input_file")
    if input_file is None:
        return jsonify({"error": "Missing file field: input_file"}), 400

    try:
        raw_rows = parse_jsonl_stream(input_file.stream)
    except Exception as exc:
        return jsonify({"error": f"Failed to parse JSONL file: {exc}"}), 400

    if not raw_rows:
        return jsonify({"error": "Uploaded JSONL contains no objects."}), 400

    try:
        records = build_feature_records_from_raw_rows(raw_rows)
    except Exception as exc:
        return jsonify({"error": f"Failed to compute detectors: {exc}"}), 400

    predictions = model_service.predict_records(records)
    pred_by_id = {row["id"]: row for row in predictions}

    merged_output = []
    for row in records:
        pred = pred_by_id[row["id"]]
        merged_output.append(
            {
                **pred,
                "answer": row.get("answer"),
                "evidence": row.get("evidence"),
                "rag_answer": row.get("rag_answer"),
                "ragas_raw_faithfulness": row["ragas_raw_faithfulness"],
                "similarity_raw_score": row["similarity_raw_score"],
                "detector_debug": row["detector_debug"],
            }
        )

    summary = _summarize(merged_output)
    return jsonify({"summary": summary, "predictions": merged_output})


@app.post("/api/run-default")
def run_default() -> tuple:
    missing_paths = [
        path
        for path in [
            DEFAULT_BASELINE_FILE,
            DEFAULT_RAGAS_FILE,
            DEFAULT_SELFCHECK_FILE,
            DEFAULT_SIMILARITY_FILE,
            DEFAULT_HUMAN_FILE,
        ]
        if not Path(path).exists()
    ]
    if missing_paths:
        return jsonify(
            {
                "error": "Default detector files not found.",
                "missing": [str(x) for x in missing_paths],
            }
        ), 400

    records = build_records_from_paths(
        baseline_path=DEFAULT_BASELINE_FILE,
        ragas_path=DEFAULT_RAGAS_FILE,
        selfcheck_path=DEFAULT_SELFCHECK_FILE,
        similarity_path=DEFAULT_SIMILARITY_FILE,
        human_path=DEFAULT_HUMAN_FILE,
    )

    predictions = model_service.predict_records(records)
    human_by_id = {row["id"]: row["human"] for row in records if "human" in row}
    for row in predictions:
        if row["id"] in human_by_id:
            row["human"] = human_by_id[row["id"]]

    summary = _summarize(predictions)
    return jsonify({"summary": summary, "predictions": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
