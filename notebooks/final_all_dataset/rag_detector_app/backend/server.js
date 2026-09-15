import cors from "cors";
import express from "express";
import multer from "multer";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const APP_ROOT = __dirname;
const FRONTEND_DIR = path.resolve(APP_ROOT, "../frontend");
const UPLOAD_DIR = path.resolve(APP_ROOT, "uploads");

const PORT = Number(process.env.PORT || 6060);
const PYTHON_BIN = process.env.RAG_DETECTOR_PYTHON || "/Users/sterinsaji/miniconda3/envs/rag_project/bin/python";
const SCRIPT_PATH = path.resolve(APP_ROOT, "rag_pipeline.py");

const app = express();
app.use(cors());
app.use(express.json({ limit: "5mb" }));
app.use(express.static(FRONTEND_DIR));

const upload = multer({ storage: multer.memoryStorage() });

function runPython(payload) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [SCRIPT_PATH], {
      cwd: APP_ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    proc.stderr.on("data", (chunk) => {
      stderr += String(chunk);
      // stream python progress lines to the terminal in real time
      process.stderr.write(String(chunk));
    });

    proc.on("error", (err) => reject(err));

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Python process failed with code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch (err) {
        reject(new Error(`Invalid JSON from python service: ${err.message}`));
      }
    });

    proc.stdin.write(JSON.stringify(payload));
    proc.stdin.end();
  });
}

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, backend: "node", python_bin: PYTHON_BIN, script_path: SCRIPT_PATH });
});

app.get("/api/indexes", async (_req, res) => {
  try {
    const RAG_STORE_DIR = path.resolve(APP_ROOT, "rag_store");
    let entries = [];
    try {
      entries = await fs.readdir(RAG_STORE_DIR);
    } catch {
      return res.json({ indexes: [] });
    }
    const indexes = [];
    for (const entry of entries) {
      const metaPath = path.join(RAG_STORE_DIR, entry, "metadata.json");
      try {
        const raw = await fs.readFile(metaPath, "utf8");
        const meta = JSON.parse(raw);
        indexes.push({
          index_id: meta.index_id,
          source_doc: meta.source_doc,
          chunk_count: meta.chunk_count,
          created_at: meta.created_at,
        });
      } catch {
        // skip corrupt/incomplete entries
      }
    }
    // newest first
    indexes.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
    res.json({ indexes });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/indexes/:indexId/interactions", async (req, res) => {
  try {
    const indexId = String(req.params.indexId || "").trim();
    if (!indexId) {
      return res.status(400).json({ error: "Missing indexId" });
    }

    const interactionsPath = path.resolve(APP_ROOT, "rag_store", indexId, "interactions.jsonl");
    let raw = "";
    try {
      raw = await fs.readFile(interactionsPath, "utf8");
    } catch {
      return res.json({ interactions: [] });
    }

    const interactions = raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line))
      .map((row) => ({
        id: row.id,
        question: row.question,
        answer: row.rag_answer || row.answer,
        rag_answer: row.rag_answer || row.answer,
        evidence: row.evidence,
        retrieved_chunks: Array.isArray(row.retrieved_chunks) ? row.retrieved_chunks : [],
        hallucination_result: row.model_output || null,
        created_at: row.created_at,
      }));

    res.json({ interactions });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/upload-pdf", upload.single("pdf_file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Missing file field: pdf_file" });
    }
    if (!req.file.originalname.toLowerCase().endsWith(".pdf")) {
      return res.status(400).json({ error: "Only PDF files are supported" });
    }

    await fs.mkdir(UPLOAD_DIR, { recursive: true });
    const safeName = `${Date.now()}_${req.file.originalname.replace(/[^a-zA-Z0-9_.-]/g, "_")}`;
    const fullPath = path.join(UPLOAD_DIR, safeName);
    await fs.writeFile(fullPath, req.file.buffer);

    console.log(`[upload] received: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)} KB) -> ${safeName}`);
    const result = await runPython({ action: "upload_pdf", pdf_path: fullPath });
    console.log(`[upload] done: index_id=${result.index_id}  chunks=${result.chunk_count}`);
    res.json(result);
  } catch (err) {
    console.error(`[upload] error: ${err.message}`);
    res.status(400).json({ error: `Upload/index failed: ${err.message}` });
  }
});

app.post("/api/ask", async (req, res) => {
  try {
    const body = req.body || {};
    console.log(`[ask] index_id=${body.index_id}  question="${String(body.question || "").slice(0, 80)}"`);
    const result = await runPython({
      action: "ask",
      index_id: body.index_id,
      question: body.question,
      top_k: body.top_k,
      reference_answer: body.reference_answer,
      question_type: body.question_type,
      difficulty: body.difficulty,
      source_doc: body.source_doc,
      id: body.id,
    });
    console.log(`[ask] done: hallucinated=${result.hallucination_result?.hallucinated}  score=${result.hallucination_result?.hallucination_score}`);
    res.json(result);
  } catch (err) {
    console.error(`[ask] error: ${err.message}`);
    res.status(400).json({ error: `Question processing failed: ${err.message}` });
  }
});

app.get("/", (_req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, "index.html"));
});

app.listen(PORT, () => {
  console.log(`rag_detector_app running at http://127.0.0.1:${PORT}`);
});
