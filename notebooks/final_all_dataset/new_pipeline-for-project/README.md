# Hybrid Detector App (Flask + React)

This project runs all existing detector outputs through the trained model:

- Model: `notebooks/final_all_dataset/hybrid_detector_model.pkl`
- Detectors used:
  - baseline (`hallucination`)
  - ragas (`faithfulness` -> binary using `< 0.7`)
  - selfcheck (`selfcheck_hallucination`)
  - similarity (`retrail_similarity` -> binary using `< 0.75`)

The app also supports raw QA input objects and computes detector outputs live for each request using logic ported from the existing detector notebooks in this repository.

For raw-object and raw-JSONL requests, the backend runs baseline, selfcheck, ragas, and similarity detectors first, then applies the same RAGAS/similarity thresholds used in training, and finally runs the hybrid model.

## Project Structure

- `backend/` Flask API for merging detector files and running model inference
- `frontend/` React app for default-run, file-upload flow, and single-sample scoring

## Backend Setup

From `new_pipeline-for-project/backend`:

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python app.py`

Backend runs at `http://127.0.0.1:5001`.

## Frontend Setup

From `new_pipeline-for-project/frontend`:

1. `npm install`
2. `npm run dev`

Frontend runs at `http://127.0.0.1:5173`.

To target a different API URL, create `frontend/.env`:

`VITE_API_BASE_URL=http://127.0.0.1:5001`

## API Endpoints

- `GET /api/health`
- `POST /api/predict`
  - body is a raw QA object with fields such as:
    - `id`, `question`, `answer`, `evidence`, `rag_answer`
- `POST /api/predict-from-files`
  - multipart file field:
    - `input_file` (JSONL file containing one raw QA object per line)
- `POST /api/run-default`
  - uses existing files under `notebooks/final_all_dataset/150_filtered`

## Output

Each prediction includes:

- `baseline`, `ragas`, `selfcheck`, `similarity` (binary model inputs)
- `hybrid_prediction`
- probabilities when available (`predict_proba`)

The frontend can download all predictions as `hybrid_predictions.csv`.
