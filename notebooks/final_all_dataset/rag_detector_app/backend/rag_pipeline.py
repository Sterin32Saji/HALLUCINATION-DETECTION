from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Reuse the independent custom detector feature engineering/model contract.
CUSTOM_DETECTOR_ROOT = Path(__file__).resolve().parents[1] / ".." / "custom_detector"
sys.path.append(str(CUSTOM_DETECTOR_ROOT.resolve()))
from feature_extractor import CustomFeatureExtractor  # type: ignore  # noqa: E402


APP_ROOT = Path(__file__).resolve().parent
RAG_STORE_DIR = APP_ROOT / "rag_store"
UPLOAD_DIR = APP_ROOT / "uploads"
MODEL_PATH = (CUSTOM_DETECTOR_ROOT / "artifacts" / "custom_hallucination_detector.pkl").resolve()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

_embedding_model: SentenceTransformer | None = None
_feature_extractor: CustomFeatureExtractor | None = None
_model_artifact: dict[str, Any] | None = None


def read_request() -> dict[str, Any]:
    payload = sys.stdin.read().strip()
    if not payload:
        raise ValueError("Empty payload")
    body = json.loads(payload)
    if not isinstance(body, dict):
        raise ValueError("Payload must be a JSON object")
    return body


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_feature_extractor() -> CustomFeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = CustomFeatureExtractor()
    return _feature_extractor


def get_model_artifact() -> dict[str, Any]:
    global _model_artifact
    if _model_artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Custom detector model not found: {MODEL_PATH}")
        _model_artifact = joblib.load(MODEL_PATH)
    return _model_artifact


def chunk_text(text: str) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + CHUNK_SIZE, len(clean))
        piece = clean[start:end]
        if piece:
            chunks.append(piece)
        if end >= len(clean):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    vectors = get_embedding_model().encode(texts, normalize_embeddings=True)
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    vector = get_embedding_model().encode([text], normalize_embeddings=True)
    return np.array(vector, dtype=np.float32)


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def rag_dir(index_id: str) -> Path:
    return RAG_STORE_DIR / index_id


def save_index(index_id: str, index: faiss.Index, metadata: dict[str, Any]) -> None:
    out_dir = rag_dir(index_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def load_index(index_id: str) -> tuple[faiss.Index, dict[str, Any]]:
    base = rag_dir(index_id)
    idx_path = base / "index.faiss"
    meta_path = base / "metadata.json"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"index_id not found: {index_id}")

    index = faiss.read_index(str(idx_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata


def generate_rag_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"Context {i+1}:\n{c}" for i, c in enumerate(contexts))
    user_prompt = (
        "You are a technical assistant. Use only the provided context. "
        "If information is missing, explicitly say the context is insufficient.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Ground all claims in context."},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("message", {}).get("content", "")).strip()


def score_with_custom_detector(record: dict[str, Any]) -> dict[str, Any]:
    artifact = get_model_artifact()
    pipeline = artifact["pipeline"]
    categorical_cols = artifact["categorical_cols"]
    numeric_cols = artifact["numeric_cols"]

    features = get_feature_extractor().transform(pd.DataFrame([record]))
    x = features[categorical_cols + numeric_cols]

    pred = int(pipeline.predict(x)[0])
    hallucination_score = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(x)
        if probs.shape[1] == 2:
            hallucination_score = float(probs[0][1])

    return {
        "hallucinated": bool(pred == 1),
        "custom_prediction": pred,
        "hallucination_score": hallucination_score,
    }


def append_interaction(index_id: str, row: dict[str, Any]) -> None:
    path = rag_dir(index_id) / "interactions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _infer_question_type(question: str) -> str:
    """Heuristically classify question type from the text."""
    q = question.lower()
    if any(w in q for w in ("how many", "what is the number", "count", "how much", "percentage")):
        return "quantitative"
    if any(w in q for w in ("compare", "difference", "versus", "vs", "contrast")):
        return "comparative"
    if any(w in q for w in ("why", "explain", "reason", "cause", "justify")):
        return "explanatory"
    if any(w in q for w in ("list", "what are", "enumerate", "name all", "which ones")):
        return "enumerative"
    if any(w in q for w in ("define", "what does", "what is", "meaning of")):
        return "definitional"
    return "factual"


def _infer_difficulty(question: str) -> str:
    """Heuristically estimate difficulty from question length and complexity."""
    words = question.split()
    if len(words) <= 8:
        return "easy"
    if len(words) <= 18:
        return "medium"
    return "hard"


def _log(msg: str) -> None:
    print(f"[rag_pipeline] {msg}", file=sys.stderr, flush=True)


def handle_upload(body: dict[str, Any]) -> dict[str, Any]:
    pdf_path = Path(str(body.get("pdf_path") or "")).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    _log(f"extracting text from PDF: {pdf_path.name}")
    text = extract_pdf_text(pdf_path)
    if not text:
        raise ValueError("Could not extract text from the uploaded PDF")
    _log(f"extracted {len(text):,} characters")

    _log("chunking text...")
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No chunks produced from PDF")
    _log(f"produced {len(chunks)} chunks")

    _log("embedding chunks (this may take a moment)...")
    vectors = embed_texts(chunks)
    _log(f"embedding done: shape {vectors.shape}")

    _log("building FAISS index...")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_id = str(uuid.uuid4())
    source_doc = pdf_path.name
    metadata = {
        "index_id": index_id,
        "source_doc": source_doc,
        "chunk_count": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": chunks,
    }
    save_index(index_id, index, metadata)
    _log(f"index saved: index_id={index_id}")

    return {
        "index_id": index_id,
        "source_doc": source_doc,
        "chunk_count": len(chunks),
        "sample_chunks": [
            {"chunk_id": f"{source_doc}_chunk_{i}", "preview": c[:180]} for i, c in enumerate(chunks[:5])
        ],
    }


def handle_ask(body: dict[str, Any]) -> dict[str, Any]:
    index_id = str(body.get("index_id") or "").strip()
    question = str(body.get("question") or "").strip()
    if not index_id:
        raise ValueError("Missing index_id")
    if not question:
        raise ValueError("Missing question")
    _log(f"ask: index_id={index_id}  question='{question[:80]}'")

    # auto-select top_k based on question length: short questions get 3 chunks,
    # longer/complex questions get up to 5
    _top_k_hint = int(body.get("top_k") or 0)
    if _top_k_hint > 0:
        top_k = max(1, min(10, _top_k_hint))
    else:
        top_k = 5 if len(question.split()) > 12 else 3

    index, metadata = load_index(index_id)
    chunks: list[str] = list(metadata.get("chunks", []))
    source_doc = str(metadata.get("source_doc") or "uploaded.pdf")

    qv = embed_query(question)
    scores, ids = index.search(qv, top_k)

    retrieved = []
    contexts = []
    for rank, idx in enumerate(ids[0].tolist()):
        if idx < 0 or idx >= len(chunks):
            continue
        text = chunks[idx]
        contexts.append(text)
        retrieved.append(
            {
                "chunk_id": f"{source_doc}_chunk_{idx}",
                "source_doc": source_doc,
                "score": float(scores[0][rank]),
                "text": text,
            }
        )

    if not contexts:
        raise ValueError("No evidence chunks retrieved")
    _log(f"retrieved {len(contexts)} chunk(s)")

    _log("generating RAG answer via Ollama...")
    rag_answer = generate_rag_answer(question, contexts)
    _log(f"RAG answer generated ({len(rag_answer)} chars)")
    evidence = "\n\n".join(contexts)

    reference_answer = str(body.get("reference_answer") or "").strip()
    if not reference_answer:
        reference_answer = evidence

    record = {
        "id": str(body.get("id") or str(uuid.uuid4())),
        "question": question,
        "answer": reference_answer,
        "evidence": evidence,
        "rag_answer": rag_answer,
        "question_type": _infer_question_type(question),
        "difficulty": _infer_difficulty(question),
        "source_doc": str(body.get("source_doc") or source_doc),
        "human": 0,
    }

    _log("scoring with custom hallucination detector...")
    score = score_with_custom_detector(record)
    _log(f"score: hallucinated={score['hallucinated']}  prob={score['hallucination_score']}")

    interaction = {
        **record,
        "retrieved_chunks": retrieved,
        "model_output": score,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    append_interaction(index_id, interaction)

    return {
        "id": record["id"],
        "question": question,
        "answer": rag_answer,
        "rag_answer": rag_answer,
        "evidence": evidence,
        "source": source_doc,
        "retrieved_chunks": retrieved,
        "hallucination_result": score,
    }


def main() -> None:
    req = read_request()
    action = str(req.get("action") or "").strip()

    if action == "upload_pdf":
        result = handle_upload(req)
    elif action == "ask":
        result = handle_ask(req)
    else:
        raise ValueError("Unsupported action. Use upload_pdf or ask")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
