import cors from "cors";
import express from "express";
import multer from "multer";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const CUSTOM_DETECTOR_ROOT = path.resolve(__dirname, "..");

const PORT = Number(process.env.PORT || 5055);
const PYTHON_BIN = process.env.CUSTOM_DETECTOR_PYTHON || "/Users/sterinsaji/miniconda3/envs/rag_project/bin/python";
const MODEL_PATH = process.env.CUSTOM_DETECTOR_MODEL || path.resolve(CUSTOM_DETECTOR_ROOT, "artifacts", "custom_hallucination_detector.pkl");
const SCRIPT_PATH = path.resolve(CUSTOM_DETECTOR_ROOT, "serve_predict.py");
const RAG_SCRIPT_PATH = path.resolve(CUSTOM_DETECTOR_ROOT, "serve_rag.py");
const FRONTEND_DIR = path.resolve(CUSTOM_DETECTOR_ROOT, "frontend");
const UPLOAD_DIR = path.resolve(CUSTOM_DETECTOR_ROOT, "uploads");

const app = express();
app.use(cors());
app.use(express.json({ limit: "4mb" }));
app.use(express.static(FRONTEND_DIR));

const upload = multer({ storage: multer.memoryStorage() });

function runPythonRag(payload) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [RAG_SCRIPT_PATH], {
      cwd: CUSTOM_DETECTOR_ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });

    proc.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });

    proc.on("error", (err) => reject(err));

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `RAG python process failed with exit code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch (err) {
        reject(new Error(`Invalid JSON from RAG python bridge: ${err.message}`));
      }
    });

    proc.stdin.write(JSON.stringify(payload));
    proc.stdin.end();
  });
}

function runPythonPredict(rows) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [SCRIPT_PATH, "--model", MODEL_PATH], {
      cwd: CUSTOM_DETECTOR_ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });

    proc.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });

    proc.on("error", (err) => {
      reject(err);
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Python process failed with exit code ${code}`));
        return;
      }

      try {
        const parsed = JSON.parse(stdout.trim());
        resolve(parsed);
      } catch (err) {
        reject(new Error(`Invalid JSON from python bridge: ${err.message}`));
      }
    });

    proc.stdin.write(JSON.stringify(rows));
    proc.stdin.end();
  });
}

function parseJsonlBuffer(buffer) {
  const text = buffer.toString("utf-8");
  const lines = text.split(/\r?\n/);
  const rows = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    rows.push(JSON.parse(trimmed));
  }

  return rows;
}

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    backend: "node",
    model_path: MODEL_PATH,
    script_path: SCRIPT_PATH,
    rag_script_path: RAG_SCRIPT_PATH,
  });
});

app.get("/", (_req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, "index.html"));
});

app.post("/api/predict", async (req, res) => {
  try {
    const row = req.body || {};
    const result = await runPythonPredict(row);
    const first = Array.isArray(result.predictions) ? result.predictions[0] : null;
    res.json(first || result);
  } catch (err) {
    res.status(400).json({ error: `Prediction failed: ${err.message}` });
  }
});

app.post("/api/predict-batch", async (req, res) => {
  try {
    if (!Array.isArray(req.body)) {
      return res.status(400).json({ error: "Body must be a JSON array" });
    }
    const result = await runPythonPredict(req.body);
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: `Batch prediction failed: ${err.message}` });
  }
});

app.post("/api/predict-jsonl", upload.single("input_file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Missing file field: input_file" });
    }
    const rows = parseJsonlBuffer(req.file.buffer);
    const result = await runPythonPredict(rows);
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: `JSONL prediction failed: ${err.message}` });
  }
});

app.post("/api/rag/upload-pdf", upload.single("pdf_file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Missing file field: pdf_file" });
    }

    if (!req.file.originalname.toLowerCase().endsWith(".pdf")) {
      return res.status(400).json({ error: "Uploaded file must be a PDF" });
    }

    const safeName = `${Date.now()}_${req.file.originalname.replace(/[^a-zA-Z0-9_.-]/g, "_")}`;
    const fs = await import("node:fs/promises");
    await fs.mkdir(UPLOAD_DIR, { recursive: true });
    const fullPath = path.join(UPLOAD_DIR, safeName);
    await fs.writeFile(fullPath, req.file.buffer);

    const result = await runPythonRag({ action: "upload_pdf", pdf_path: fullPath });
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: `PDF upload and indexing failed: ${err.message}` });
  }
});

app.post("/api/rag/ask", async (req, res) => {
  try {
    const body = req.body || {};
    const payload = {
      action: "ask",
      index_id: body.index_id,
      question: body.question,
      top_k: body.top_k,
      reference_answer: body.reference_answer,
      question_type: body.question_type,
      difficulty: body.difficulty,
      source_doc: body.source_doc,
      id: body.id,
    };

    const result = await runPythonRag(payload);
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: `RAG answer generation failed: ${err.message}` });
  }
});

app.listen(PORT, () => {
  console.log(`Custom detector Node backend running at http://127.0.0.1:${PORT}`);
});
