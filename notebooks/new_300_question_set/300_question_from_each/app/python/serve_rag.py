from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

# The local FAISS and embedding libraries can crash when their native thread pools compete.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import joblib
import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from feature_extractor import CustomFeatureExtractor


APP_ROOT = Path(__file__).resolve().parent.parent
RAG_STORE_DIR = APP_ROOT / "rag_store"
CUSTOM_MODEL_PATH = APP_ROOT.parent / "analysis_results" / "models" / "custom_hallucination_detector.pkl"
HYBRID_MODEL_PATH = APP_ROOT.parent / "analysis_results" / "models" / "hybrid_detector_model.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def _read_stdin_json() -> dict[str, Any]:
    import sys

    payload = sys.stdin.read().strip()
    if not payload:
        raise ValueError("Empty request payload")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Payload must be a JSON object")
    return data


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    n = len(clean)
    while start < n:
        end = min(start + chunk_size, n)
        piece = clean[start:end]
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def _embed_texts(texts: list[str]) -> np.ndarray:
    model = _embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    arr = np.array(vectors, dtype=np.float32)
    return arr


def _embed_query(text: str) -> np.ndarray:
    model = _embedding_model()
    vector = model.encode([text], normalize_embeddings=True)
    return np.array(vector, dtype=np.float32)


def _extract_pdf_text(pdf_path: Path) -> tuple[str, str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    full_text = "\n".join(pages).strip()
    return full_text, pdf_path.name


def _index_dir(index_id: str) -> Path:
    return RAG_STORE_DIR / index_id


def _save_index(index_id: str, index: faiss.Index, metadata: dict[str, Any]) -> None:
    out_dir = _index_dir(index_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def _load_index(index_id: str) -> tuple[faiss.Index, dict[str, Any]]:
    in_dir = _index_dir(index_id)
    index_path = in_dir / "index.faiss"
    meta_path = in_dir / "metadata.json"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"RAG index not found for index_id={index_id}")

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata


def _ollama_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    prompt = (
        "You are a technical RAG assistant. Use ONLY the provided context to answer. "
        "If context is insufficient, say that clearly and do not invent facts.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely and grounded in the context."
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Answer with factual grounding from context."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    message = data.get("message", {})
    return str(message.get("content", "")).strip()


@lru_cache(maxsize=1)
def _model_artifact() -> dict[str, Any]:
    if not CUSTOM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Custom detector model not found: {CUSTOM_MODEL_PATH}")
    return joblib.load(CUSTOM_MODEL_PATH)


@lru_cache(maxsize=1)
def _hybrid_model() -> Any:
    if not HYBRID_MODEL_PATH.exists():
        raise FileNotFoundError(f"Hybrid detector model not found: {HYBRID_MODEL_PATH}")
    return joblib.load(HYBRID_MODEL_PATH)


@lru_cache(maxsize=1)
def _feature_extractor() -> CustomFeatureExtractor:
    return CustomFeatureExtractor()


def _hallucination_score(record: dict[str, Any]) -> dict[str, Any]:
    artifact = _model_artifact()
    pipeline = artifact["pipeline"]
    categorical_cols = artifact["categorical_cols"]
    numeric_cols = artifact["numeric_cols"]

    raw_df = pd.DataFrame([record])
    feature_df = _feature_extractor().transform(raw_df)
    x = feature_df[categorical_cols + numeric_cols]

    pred = int(pipeline.predict(x)[0])
    score = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(x)
        if probs.shape[1] == 2:
            score = float(probs[0][1])

    return {
        "custom_prediction": pred,
        "hallucination_score": score,
    }


def _vector_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    va = _embed_query(a)[0]
    vb = _embed_query(b)[0]
    sim = float(np.dot(va, vb))
    return max(0.0, min(1.0, sim))


def _hybrid_signals(question: str, rag_answer: str, evidence: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    answer_lower = rag_answer.lower()
    has_uncertainty = any(
        phrase in answer_lower
        for phrase in [
            "i could not find",
            "insufficient",
            "not in the provided",
            "cannot determine",
            "no relevant",
        ]
    )

    faithfulness = _vector_similarity(rag_answer, evidence)
    qa_similarity = _vector_similarity(question, rag_answer)
    top_retrieval = max((float(c.get("score") or 0.0) for c in retrieved), default=0.0)

    baseline = int(has_uncertainty or faithfulness < 0.45)
    ragas = int(faithfulness < 0.5)
    selfcheck = int((faithfulness < 0.45 and qa_similarity < 0.35) or (has_uncertainty and top_retrieval < 0.25))
    similarity = int(faithfulness < 0.5)

    return {
        "baseline": baseline,
        "selfcheck": selfcheck,
        "ragas": ragas,
        "similarity": similarity,
        "faithfulness_proxy": faithfulness,
        "qa_similarity_proxy": qa_similarity,
        "top_retrieval_score": top_retrieval,
        "uncertainty_phrase_detected": has_uncertainty,
    }


def _hybrid_score(signals: dict[str, Any]) -> dict[str, Any]:
    model = _hybrid_model()
    x = pd.DataFrame(
        [
            {
                "baseline": int(signals["baseline"]),
                "selfcheck": int(signals["selfcheck"]),
                "ragas": int(signals["ragas"]),
                "similarity": int(signals["similarity"]),
            }
        ]
    )
    pred = int(model.predict(x)[0])
    score = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        if probs.shape[1] == 2:
            score = float(probs[0][1])

    return {
        "hybrid_prediction": pred,
        "hybrid_score": score,
        "signals": signals,
    }


def _ingest_pdf(pdf_path_str: str) -> dict[str, Any]:
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text, source_doc = _extract_pdf_text(pdf_path)
    if not text:
        raise ValueError("No extractable text found in PDF")

    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError("No chunks generated from PDF")

    vectors = _embed_texts(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_id = str(uuid.uuid4())
    metadata = {
        "index_id": index_id,
        "source_doc": source_doc,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": chunks,
    }
    _save_index(index_id, index, metadata)

    sample = [
        {"chunk_id": f"{source_doc}_chunk_{i}", "text_preview": c[:180]} for i, c in enumerate(chunks[:5])
    ]
    return {
        "index_id": index_id,
        "source_doc": source_doc,
        "chunk_count": len(chunks),
        "sample_chunks": sample,
    }


def _ask_question(payload: dict[str, Any]) -> dict[str, Any]:
    index_id = str(payload.get("index_id") or "").strip()
    question = str(payload.get("question") or "").strip()
    if not index_id:
        raise ValueError("Missing index_id")
    if not question:
        raise ValueError("Missing question")

    top_k = int(payload.get("top_k") or 3)
    top_k = max(1, min(top_k, 8))

    index, metadata = _load_index(index_id)
    chunks: list[str] = list(metadata.get("chunks", []))
    source_doc = str(metadata.get("source_doc") or "uploaded.pdf")

    qv = _embed_query(question)
    scores, indices = index.search(qv, top_k)

    retrieved = []
    contexts = []
    for rank, idx in enumerate(indices[0].tolist()):
        if idx < 0 or idx >= len(chunks):
            continue
        text = chunks[idx]
        sim = float(scores[0][rank])
        contexts.append(text)
        retrieved.append(
            {
                "chunk_id": f"{source_doc}_chunk_{idx}",
                "score": sim,
                "text": text,
            }
        )

    if not contexts:
        raise ValueError("No relevant chunks found for question")

    rag_answer = _ollama_answer(question, contexts)
    evidence = "\n\n".join(contexts)

    row = {
        "id": str(payload.get("id") or str(uuid.uuid4())),
        "question": question,
        "answer": rag_answer,
        "evidence": evidence,
        "question_type": str(payload.get("question_type") or "unknown"),
        "difficulty": str(payload.get("difficulty") or "unknown"),
    }

    model_mode = str(payload.get("model_mode") or "custom").strip().lower()
    if model_mode not in {"custom", "hybrid", "both"}:
        model_mode = "custom"

    custom_result = _hallucination_score(row)
    signals = _hybrid_signals(question, rag_answer, evidence, retrieved)
    hybrid_result = _hybrid_score(signals)

    selected = {}
    if model_mode in {"custom", "both"}:
        selected["custom"] = custom_result
    if model_mode in {"hybrid", "both"}:
        selected["hybrid"] = hybrid_result

    return {
        "index_id": index_id,
        "question": question,
        "source_doc": source_doc,
        "rag_answer": rag_answer,
        "retrieved_chunks": retrieved,
        "hallucination": selected,
        "model_mode": model_mode,
    }


def main() -> None:
    req = _read_stdin_json()
    action = str(req.get("action") or "").strip()

    if action == "upload_pdf":
        pdf_path = str(req.get("pdf_path") or "").strip()
        result = _ingest_pdf(pdf_path)
    elif action == "ask":
        result = _ask_question(req)
    else:
        raise ValueError("Unsupported action. Use 'upload_pdf' or 'ask'.")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
