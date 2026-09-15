# rag_detector_app

An end-to-end app that does exactly this flow:

1. User uploads a technical PDF.
2. Backend extracts text and chunks it.
3. Backend embeds chunks and creates a FAISS vector index.
4. User asks a question.
5. Backend retrieves top-k evidence chunks (base RAG retrieval).
6. Backend generates a grounded `rag_answer` using Ollama.
7. Backend builds the required record fields (`question`, `answer`, `evidence`, `rag_answer`, `source_doc`, etc.).
8. Backend runs the trained custom hallucination model.
9. Frontend gets answer, rag-answer, source, and hallucination/not-hallucinated result.

## Folder Layout

- `backend/server.js` - Node API service
- `backend/rag_pipeline.py` - Python RAG + scoring pipeline
- `backend/requirements.txt` - Python dependencies
- `frontend/index.html` - Vanilla UI
- `frontend/app.js` - Frontend logic
- `frontend/styles.css` - Frontend styles

## Prerequisites

- Python environment with dependencies:

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

- Node dependencies:

```bash
cd backend
npm install
```

- Ollama running with `qwen2.5:7b` model.

## Start App

From `rag_detector_app/backend`:

```bash
npm run dev
```

Open:

- `http://127.0.0.1:6060/`

## API Endpoints

- `GET /api/health`
- `POST /api/upload-pdf` (multipart field: `pdf_file`)
- `POST /api/ask` (JSON body with `index_id`, `question`, optional metadata)

## /api/ask Response Fields

- `answer` - final answer returned to user (base RAG output)
- `rag_answer` - same generated RAG answer value
- `source` - source document name
- `evidence` - concatenated retrieved contexts
- `retrieved_chunks` - list of chunk ids, scores, and chunk text
- `hallucination_result`:
  - `hallucinated` (boolean)
  - `custom_prediction` (0/1)
  - `hallucination_score` (probability for hallucination class, if available)

## Saved Runtime Artifacts

- Uploaded PDFs: `backend/uploads/`
- Per-document index + metadata + interaction logs: `backend/rag_store/<index_id>/`
  - `index.faiss`
  - `metadata.json`
  - `interactions.jsonl`
