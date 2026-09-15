# 300-Question Custom Detector App

This app reuses the existing custom-detector frontend/backend bridge and is wired to the latest model trained in:

- `../analysis_results/models/custom_hallucination_detector.pkl`

## Structure

- `backend/server.js` - Node API + static frontend server
- `frontend/` - Multi-chat browser UI with persistent chat history, PDF upload, and hallucination-aware answers
- `python/serve_predict.py` - Python prediction bridge (stdin JSON)
- `python/serve_rag.py` - PDF indexing + retrieval + answer generation + scoring
- `python/feature_extractor.py` - Feature engineering used by runtime scoring
- `uploads/` - Temporary uploaded PDFs
- `rag_store/` - FAISS indexes and metadata created per uploaded PDF
- `data/chats/` - Persisted chat sessions (survive page reload)

## Prerequisites

- Conda env with project deps (example: `rag_project`)
- Ollama running locally (default endpoint `http://127.0.0.1:11434/api/chat`)
- Node.js 18+

## 1) Install Python dependencies

From `app/`:

```bash
/Users/sterinsaji/miniconda3/envs/rag_project/bin/python -m pip install -r requirements.txt
/Users/sterinsaji/miniconda3/envs/rag_project/bin/python -m spacy download en_core_web_sm
```

## 2) Install Node dependencies

From `app/backend/`:

```bash
npm install
```

## 3) Start the app server

From `app/backend/`:

```bash
CUSTOM_DETECTOR_PYTHON=/Users/sterinsaji/miniconda3/envs/rag_project/bin/python \
CUSTOM_DETECTOR_MODEL=../../analysis_results/models/custom_hallucination_detector.pkl \
PORT=5055 \
npm run dev
```

Optional environment variables:

- `OLLAMA_URL` (default: `http://127.0.0.1:11434/api/chat`)
- `OLLAMA_MODEL` (default: `qwen2.5:7b`)

## 4) Open UI

- `http://127.0.0.1:5055`

## Chat UX Features

- Create a new chat from the sidebar
- Upload one or more PDFs per chat
- Switch active document index in a chat
- Ask questions and receive:
	- RAG answer
	- Hallucination class
	- Hallucination risk score
	- Retrieved evidence chunks
- Reload-safe chat history (messages stored server-side under `data/chats/`)

## API Endpoints

- `GET /api/health`
- `GET /api/chats`
- `POST /api/chats`
- `GET /api/chats/:chatId`
- `POST /api/chats/:chatId/select-document`
- `POST /api/chats/:chatId/upload-pdf`
- `POST /api/chats/:chatId/ask`
- `POST /api/predict`
- `POST /api/predict-batch`
- `POST /api/predict-jsonl`
- `POST /api/rag/upload-pdf`
- `POST /api/rag/ask`


