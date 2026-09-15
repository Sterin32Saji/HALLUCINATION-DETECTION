import React, { useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5001";

const SAMPLE_OBJECT = {
  id: "741920fe-c7d4-4e38-a239-207741b9e271",
  question: "Which document specifies the details of HTTP semantics?",
  answer: "RFC 9110 HTTP Semantics June 2022.",
  evidence: "RFC 9110 HTTP Semantics June 2022",
  question_type: "identifier",
  difficulty: "easy",
  chunk_id: "docs/rfc9110.pdf_chunk_502",
  source_doc: "docs/rfc9110.pdf",
  rag_answer:
    "The document that specifies the details of HTTP semantics is RFC 9110 HTTP Semantics June 2022, as authored by Fielding, et al. This can be inferred from references within the provided context where it is repeatedly cited as the source for information on HTTP semantics.",
};

const SAMPLE_MODEL_INPUT = {
  id: "debug-1",
  question: "Debug sample",
  baseline: 0,
  ragas: 1,
  selfcheck: 1,
  similarity: 0,
};

function toCsv(rows) {
  if (!rows || rows.length === 0) {
    return "";
  }

  const columns = Object.keys(rows[0]);
  const escapeCell = (value) => {
    const raw = value === undefined || value === null ? "" : String(value);
    if (raw.includes(",") || raw.includes("\n") || raw.includes('"')) {
      return `"${raw.replaceAll('"', '""')}"`;
    }
    return raw;
  };

  const header = columns.join(",");
  const body = rows
    .map((row) => columns.map((col) => escapeCell(row[col])).join(","))
    .join("\n");

  return `${header}\n${body}`;
}

function downloadCsv(rows) {
  const csv = toCsv(rows);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "hybrid_predictions.csv";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [activePage, setActivePage] = useState("pipeline");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [progressTitle, setProgressTitle] = useState("");
  const [progressDetail, setProgressDetail] = useState("");
  const [summary, setSummary] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [directResult, setDirectResult] = useState(null);
  const [inputFile, setInputFile] = useState(null);
  const [objectJson, setObjectJson] = useState(JSON.stringify(SAMPLE_OBJECT, null, 2));
  const [modelInputJson, setModelInputJson] = useState(JSON.stringify(SAMPLE_MODEL_INPUT, null, 2));

  const displayRows = useMemo(() => predictions.slice(0, 12), [predictions]);

  async function callApi(path, options = {}, progress = {}) {
    setLoading(true);
    setError("");
    setProgressTitle(progress.title || "Starting request...");
    setProgressDetail(progress.beforeFetch || "Preparing data for the backend pipeline.");

    try {
      setProgressDetail(progress.duringFetch || "Backend is computing detector signals and hybrid prediction.");
      const response = await fetch(`${API_BASE_URL}${path}`, options);
      const json = await response.json();
      if (!response.ok) {
        throw new Error(json.error || "Request failed");
      }
      setProgressDetail(progress.afterFetch || "Prediction completed successfully.");
      return json;
    } catch (err) {
      setError(err.message || String(err));
      return null;
    } finally {
      setLoading(false);
    }
  }

  async function runDefaultFlow() {
    const result = await callApi(
      "/api/run-default",
      { method: "POST" },
      {
        title: "Running default dataset",
        beforeFetch: "Validating prepared detector files and reading records.",
        duringFetch: "Merging detector outputs and running hybrid model predictions.",
        afterFetch: "Default pipeline completed.",
      }
    );
    if (!result) return;
    setDirectResult(null);
    setSummary(result.summary);
    setPredictions(result.predictions);
  }

  async function runObjectFlow(event) {
    event.preventDefault();

    let payload;
    try {
      payload = JSON.parse(objectJson);
    } catch {
      setError("Invalid JSON object. Please fix formatting and try again.");
      return;
    }

    const result = await callApi(
      "/api/predict",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
      {
        title: "Processing one object",
        beforeFetch: "Preparing JSON payload and validating input fields.",
        duringFetch:
          "Backend is running Baseline, SelfCheck, RAGAS, and Similarity detectors, then the hybrid model.",
        afterFetch: "Single-object prediction completed.",
      }
    );
    if (!result) return;

    setDirectResult(null);
    setSummary({
      total: 1,
      hallucinated: result.hybrid_prediction,
      not_hallucinated: 1 - result.hybrid_prediction,
      accuracy_against_human: null,
    });
    setPredictions([result]);
  }

  async function runJsonlFlow(event) {
    event.preventDefault();

    if (!inputFile) {
      setError("Please select a JSONL file first.");
      return;
    }

    const formData = new FormData();
    formData.append("input_file", inputFile);

    const result = await callApi(
      "/api/predict-from-files",
      {
        method: "POST",
        body: formData,
      },
      {
        title: "Processing JSONL dataset",
        beforeFetch: "Uploading JSONL and parsing objects.",
        duringFetch:
          "Backend is computing all four detectors row-by-row and then running hybrid predictions.",
        afterFetch: "JSONL batch prediction completed.",
      }
    );
    if (!result) return;

    setDirectResult(null);
    setSummary(result.summary);
    setPredictions(result.predictions);
  }

  async function runDirectModelFlow(event) {
    event.preventDefault();

    let payload;
    try {
      payload = JSON.parse(modelInputJson);
    } catch {
      setError("Invalid model-input JSON. Please fix formatting and try again.");
      return;
    }

    const result = await callApi(
      "/api/predict-model-input",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
      {
        title: "Running direct model debug",
        beforeFetch: "Sending precomputed model features directly to hybrid model.",
        duringFetch: "Backend is bypassing detectors and scoring only from supplied features.",
        afterFetch: "Direct model prediction completed.",
      }
    );
    if (!result) return;

    setSummary(null);
    setPredictions([]);
    setDirectResult(result);
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-glow" />
        <h1>Hybrid Hallucination Detector Pipeline</h1>
        <p>
          Input raw QA objects or JSONL and compute detector signals live, then pass
          those signals to the trained hybrid model.
        </p>
        <div className="api-box">
          API base URL: <span>{API_BASE_URL}</span>
        </div>
        <div className="page-link-row">
          <a className="page-link" href="/custom-detector/index.html" target="_blank" rel="noreferrer">
            Open Custom Detector (Vanilla Page)
          </a>
        </div>
        <div className="view-switch" role="tablist" aria-label="App sections">
          <button
            type="button"
            className={activePage === "pipeline" ? "switch-btn active" : "switch-btn"}
            onClick={() => setActivePage("pipeline")}
          >
            Pipeline Mode
          </button>
          <button
            type="button"
            className={activePage === "direct" ? "switch-btn active" : "switch-btn"}
            onClick={() => setActivePage("direct")}
          >
            Direct Model Debug
          </button>
        </div>
      </header>

      {activePage === "pipeline" ? (
      <main className="grid">
        <section className="card card-default">
          <h2>1) Run Existing Repo Detector Files</h2>
          <p>
            Uses the existing detector files from this repository and predicts with
            the hybrid model.
          </p>
          <button onClick={runDefaultFlow} disabled={loading}>
            {loading ? "Running..." : "Run Default Pipeline"}
          </button>
        </section>

        <section className="card">
          <h2>2) Single Raw Object (JSON)</h2>
          <form onSubmit={runObjectFlow} className="form-stack">
            <label>
              Input object
              <textarea
                rows={12}
                value={objectJson}
                onChange={(e) => setObjectJson(e.target.value)}
              />
            </label>
            <p>
              Send one QA-style object and backend will compute all detector scores
              before hybrid prediction.
            </p>
            <button type="submit" disabled={loading}>
              {loading ? "Scoring..." : "Run Detectors + Predict"}
            </button>
          </form>
        </section>

        <section className="card">
          <h2>3) Raw Dataset (JSONL)</h2>
          <form onSubmit={runJsonlFlow} className="form-stack">
            <label>
              JSONL file with one object per line
              <input
                type="file"
                accept=".jsonl"
                onChange={(e) => setInputFile(e.target.files?.[0] || null)}
              />
            </label>
            <button type="submit" disabled={loading}>
              {loading ? "Processing..." : "Upload JSONL + Predict"}
            </button>
          </form>
        </section>
      </main>
      ) : (
      <main className="direct-grid">
        <section className="card">
          <h2>Direct Hybrid Model Input</h2>
          <p>
            Provide the exact hybrid model features directly for testing and debugging.
          </p>
          <form onSubmit={runDirectModelFlow} className="form-stack">
            <label>
              Model input JSON
              <textarea
                rows={14}
                value={modelInputJson}
                onChange={(e) => setModelInputJson(e.target.value)}
              />
            </label>
            <p>Required binary fields: baseline, ragas, selfcheck, similarity.</p>
            <button type="submit" disabled={loading}>
              {loading ? "Predicting..." : "Run Hybrid Model Only"}
            </button>
          </form>
        </section>

        {directResult ? (
          <section className="card">
            <h2>Direct Prediction Output</h2>
            <pre className="json-output">{JSON.stringify(directResult, null, 2)}</pre>
          </section>
        ) : null}
      </main>
      )}

      {loading ? (
        <section className="progress card">
          <h2>Pipeline Progress</h2>
          <div className="progress-row">
            <span className="progress-dot" aria-hidden="true" />
            <div>
              <strong>{progressTitle || "Running"}</strong>
              <p>{progressDetail || "Working on your request..."}</p>
            </div>
          </div>
        </section>
      ) : null}

      {error ? <div className="error-box">{error}</div> : null}

      {activePage === "pipeline" && summary ? (
        <section className="summary card">
          <h2>Summary</h2>
          <div className="summary-grid">
            <div>
              <span>Total</span>
              <strong>{summary.total}</strong>
            </div>
            <div>
              <span>Hallucinated</span>
              <strong>{summary.hallucinated}</strong>
            </div>
            <div>
              <span>Not Hallucinated</span>
              <strong>{summary.not_hallucinated}</strong>
            </div>
            <div>
              <span>Accuracy vs Human</span>
              <strong>
                {summary.accuracy_against_human === null
                  ? "N/A"
                  : summary.accuracy_against_human.toFixed(3)}
              </strong>
            </div>
          </div>
        </section>
      ) : null}

      {activePage === "pipeline" && predictions.length > 0 ? (
        <section className="results card">
          <div className="results-head">
            <h2>Predictions ({predictions.length})</h2>
            <button onClick={() => downloadCsv(predictions)}>Download CSV</button>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Baseline</th>
                  <th>RAGAS(bin)</th>
                  <th>SelfCheck</th>
                  <th>Similarity(bin)</th>
                  <th>Hybrid</th>
                </tr>
              </thead>
              <tbody>
                {displayRows.map((row, idx) => (
                  <tr key={row.id || String(idx)}>
                    <td>{row.id || "-"}</td>
                    <td>{row.baseline}</td>
                    <td>{row.ragas}</td>
                    <td>{row.selfcheck}</td>
                    <td>{row.similarity}</td>
                    <td>{row.hybrid_prediction}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </div>
  );
}
