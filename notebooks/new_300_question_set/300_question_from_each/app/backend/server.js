import cors from "cors";
import express from "express";
import multer from "multer";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const APP_ROOT = path.resolve(__dirname, "..");

const PORT = Number(process.env.PORT || 5055);
const PYTHON_BIN = process.env.CUSTOM_DETECTOR_PYTHON || "/Users/sterinsaji/miniconda3/envs/rag_project/bin/python";
const MODEL_PATH = process.env.CUSTOM_DETECTOR_MODEL
  || path.resolve(APP_ROOT, "..", "analysis_results", "models", "custom_hallucination_detector.pkl");
const SCRIPT_PATH = path.resolve(APP_ROOT, "python", "serve_predict.py");
const RAG_SCRIPT_PATH = path.resolve(APP_ROOT, "python", "serve_rag.py");
const FRONTEND_DIR = path.resolve(APP_ROOT, "frontend");
const UPLOAD_DIR = path.resolve(APP_ROOT, "uploads");
const DATA_DIR = path.resolve(APP_ROOT, "data");
const CHAT_DIR = path.resolve(DATA_DIR, "chats");
const ANALYSIS_DIR = path.resolve(APP_ROOT, "..", "analysis_results");
const REPORT_DIR = path.resolve(ANALYSIS_DIR, "reports");
const ANALYSIS_DATA_DIR = path.resolve(ANALYSIS_DIR, "data");
const PLOT_DIR = path.resolve(ANALYSIS_DIR, "plots");
const EVALUATION_RESULTS_PATH = path.resolve(REPORT_DIR, "evaluation_results.json");
const MASTER_EVALUATION_PATH = path.resolve(ANALYSIS_DATA_DIR, "master_evaluation_dataset.jsonl");
const ERROR_ANALYSIS_PATH = path.resolve(ANALYSIS_DATA_DIR, "error_analysis.jsonl");

const app = express();
app.use(cors());
app.use(express.json({ limit: "4mb" }));
app.use(express.static(FRONTEND_DIR));
app.use("/analysis-results/plots", express.static(PLOT_DIR));

const upload = multer({ storage: multer.memoryStorage() });

async function ensureDataDirs() {
  await fs.mkdir(UPLOAD_DIR, { recursive: true });
  await fs.mkdir(CHAT_DIR, { recursive: true });
}

function nowIso() {
  return new Date().toISOString();
}

function chatPath(chatId) {
  return path.resolve(CHAT_DIR, `${chatId}.json`);
}

async function writeJsonAtomic(filePath, payload) {
  const tempPath = `${filePath}.tmp`;
  await fs.writeFile(tempPath, JSON.stringify(payload, null, 2), "utf-8");
  await fs.rename(tempPath, filePath);
}

function createChatTitle(seed = "") {
  const clean = String(seed).replace(/\s+/g, " ").trim();
  if (!clean) {
    return "New Chat";
  }
  return clean.slice(0, 60);
}

function isDefaultChatTitle(value = "") {
  return value === "New Chat";
}

function toChatSummary(chat) {
  const activeDoc = (chat.documents || []).find((doc) => doc.indexId === chat.activeIndexId) || null;
  return {
    id: chat.id,
    title: chat.title,
    createdAt: chat.createdAt,
    updatedAt: chat.updatedAt,
    messageCount: Array.isArray(chat.messages) ? chat.messages.length : 0,
    activeIndexId: chat.activeIndexId || null,
    activeSourceDoc: activeDoc ? activeDoc.sourceDoc : null,
    documentCount: Array.isArray(chat.documents) ? chat.documents.length : 0,
  };
}

async function readChat(chatId) {
  await ensureDataDirs();
  const filePath = chatPath(chatId);
  const raw = await fs.readFile(filePath, "utf-8");
  return JSON.parse(raw);
}

async function saveChat(chat) {
  await ensureDataDirs();
  chat.updatedAt = nowIso();
  await writeJsonAtomic(chatPath(chat.id), chat);
  return chat;
}

async function createChat(title = "") {
  const chat = {
    id: randomUUID(),
    title: createChatTitle(title),
    createdAt: nowIso(),
    updatedAt: nowIso(),
    activeIndexId: null,
    documents: [],
    messages: [],
  };
  await saveChat(chat);
  return chat;
}

async function listChats() {
  await ensureDataDirs();
  const files = await fs.readdir(CHAT_DIR);
  const chats = [];
  for (const fileName of files) {
    if (!fileName.endsWith(".json")) {
      continue;
    }
    const raw = await fs.readFile(path.resolve(CHAT_DIR, fileName), "utf-8");
    chats.push(JSON.parse(raw));
  }
  chats.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  return chats;
}

function assertChatExists(chat) {
  if (!chat || !chat.id) {
    throw new Error("Chat not found");
  }
}

function pushMessage(chat, message) {
  chat.messages.push({
    id: randomUUID(),
    createdAt: nowIso(),
    ...message,
  });
}

function runPythonRag(payload) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [RAG_SCRIPT_PATH], {
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

async function readJsonFile(filePath) {
  const raw = await fs.readFile(filePath, "utf-8");
  return JSON.parse(raw);
}

async function readJsonlFile(filePath) {
  const raw = await fs.readFile(filePath, "utf-8");
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function compactEvaluation(summary) {
  return {
    evaluation_scope: summary.evaluation_scope,
    created_at_utc: summary.created_at_utc,
    reproducibility: summary.reproducibility,
    data_integrity: summary.data_integrity,
    metrics: summary.metrics,
    risk: summary.risk,
    outputs: summary.outputs,
  };
}

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    backend: "node",
    model_path: MODEL_PATH,
    script_path: SCRIPT_PATH,
    rag_script_path: RAG_SCRIPT_PATH,
    chat_store: CHAT_DIR,
  });
});

app.get("/api/evaluation/summary", async (_req, res) => {
  try {
    const summary = await readJsonFile(EVALUATION_RESULTS_PATH);
    res.json(compactEvaluation(summary));
  } catch (err) {
    res.status(404).json({ error: `Evaluation summary not available. Run code/pipeline.py first: ${err.message}` });
  }
});

app.get("/api/evaluation/metrics", async (_req, res) => {
  try {
    const summary = await readJsonFile(EVALUATION_RESULTS_PATH);
    res.json(summary.metrics || []);
  } catch (err) {
    res.status(404).json({ error: `Evaluation metrics not available: ${err.message}` });
  }
});

app.get("/api/evaluation/confusion-matrices", async (_req, res) => {
  try {
    const summary = await readJsonFile(EVALUATION_RESULTS_PATH);
    res.json(summary.confusion_matrices || {});
  } catch (err) {
    res.status(404).json({ error: `Confusion matrices not available: ${err.message}` });
  }
});

app.get("/api/evaluation/roc", async (_req, res) => {
  try {
    const summary = await readJsonFile(EVALUATION_RESULTS_PATH);
    res.json(summary.roc || {});
  } catch (err) {
    res.status(404).json({ error: `ROC data not available: ${err.message}` });
  }
});

app.get("/api/evaluation/precision-recall", async (_req, res) => {
  try {
    const summary = await readJsonFile(EVALUATION_RESULTS_PATH);
    res.json(summary.precision_recall || {});
  } catch (err) {
    res.status(404).json({ error: `Precision-recall data not available: ${err.message}` });
  }
});

app.get("/api/evaluation/distributions", async (_req, res) => {
  try {
    const rows = await readJsonlFile(MASTER_EVALUATION_PATH);
    res.json(rows.map((row) => ({
      id: row.id,
      human_label: row.human_label,
      ragas_risk_score: row.ragas_risk_score,
      similarity_risk_score: row.similarity_risk_score,
      hybrid_score: row.hybrid_score,
      custom_score: row.custom_score,
      final_risk_score: row.final_risk_score,
      final_risk_level: row.final_risk_level,
    })));
  } catch (err) {
    res.status(404).json({ error: `Distribution data not available: ${err.message}` });
  }
});

app.get("/api/evaluation/thresholds", async (_req, res) => {
  try {
    const summary = await readJsonFile(EVALUATION_RESULTS_PATH);
    res.json(summary.threshold_analysis || {});
  } catch (err) {
    res.status(404).json({ error: `Threshold analysis not available: ${err.message}` });
  }
});

app.get("/api/evaluation/risk", async (_req, res) => {
  try {
    const summary = await readJsonFile(EVALUATION_RESULTS_PATH);
    res.json(summary.risk || {});
  } catch (err) {
    res.status(404).json({ error: `Risk analysis not available: ${err.message}` });
  }
});

app.get("/api/evaluation/errors", async (_req, res) => {
  try {
    const rows = await readJsonlFile(ERROR_ANALYSIS_PATH);
    res.json(rows);
  } catch (err) {
    res.status(404).json({ error: `Error analysis not available: ${err.message}` });
  }
});

app.get("/api/evaluation/records", async (req, res) => {
  try {
    const rows = await readJsonlFile(MASTER_EVALUATION_PATH);
    const limit = Math.max(1, Math.min(Number(req.query.limit || 50), 300));
    res.json(rows.slice(0, limit));
  } catch (err) {
    res.status(404).json({ error: `Evaluation records not available: ${err.message}` });
  }
});

app.get("/api/evaluation/records/:id", async (req, res) => {
  try {
    const rows = await readJsonlFile(MASTER_EVALUATION_PATH);
    const record = rows.find((row) => String(row.id) === String(req.params.id));
    if (!record) {
      return res.status(404).json({ error: "Evaluation record not found" });
    }
    res.json(record);
  } catch (err) {
    res.status(404).json({ error: `Evaluation record not available: ${err.message}` });
  }
});

app.get("/api/chats", async (_req, res) => {
  try {
    const chats = await listChats();
    res.json(chats.map(toChatSummary));
  } catch (err) {
    res.status(500).json({ error: `Failed to load chats: ${err.message}` });
  }
});

app.post("/api/chats", async (req, res) => {
  try {
    const title = String((req.body || {}).title || "");
    const chat = await createChat(title);
    res.status(201).json(chat);
  } catch (err) {
    res.status(500).json({ error: `Failed to create chat: ${err.message}` });
  }
});

app.get("/api/chats/:chatId", async (req, res) => {
  try {
    const chat = await readChat(req.params.chatId);
    assertChatExists(chat);
    res.json(chat);
  } catch (err) {
    res.status(404).json({ error: `Failed to load chat: ${err.message}` });
  }
});

app.post("/api/chats/:chatId/select-document", async (req, res) => {
  try {
    const chat = await readChat(req.params.chatId);
    assertChatExists(chat);

    const indexId = String((req.body || {}).index_id || "").trim();
    if (!indexId) {
      return res.status(400).json({ error: "Missing index_id" });
    }

    const doc = (chat.documents || []).find((d) => d.indexId === indexId);
    if (!doc) {
      return res.status(404).json({ error: "Document index not found in this chat" });
    }

    chat.activeIndexId = indexId;
    await saveChat(chat);
    res.json({ ok: true, activeIndexId: chat.activeIndexId, activeSourceDoc: doc.sourceDoc });
  } catch (err) {
    res.status(400).json({ error: `Failed to select document: ${err.message}` });
  }
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

app.post("/api/chats/:chatId/upload-pdf", upload.single("pdf_file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Missing file field: pdf_file" });
    }
    if (!req.file.originalname.toLowerCase().endsWith(".pdf")) {
      return res.status(400).json({ error: "Uploaded file must be a PDF" });
    }

    const chat = await readChat(req.params.chatId);
    assertChatExists(chat);

    const safeName = `${Date.now()}_${req.file.originalname.replace(/[^a-zA-Z0-9_.-]/g, "_")}`;
    const fullPath = path.join(UPLOAD_DIR, safeName);
    await fs.writeFile(fullPath, req.file.buffer);

    const result = await runPythonRag({ action: "upload_pdf", pdf_path: fullPath });
    const docEntry = {
      indexId: result.index_id,
      sourceDoc: result.source_doc,
      chunkCount: result.chunk_count,
      uploadedFile: safeName,
      createdAt: nowIso(),
    };

    chat.documents = Array.isArray(chat.documents) ? chat.documents : [];
    chat.documents.push(docEntry);
    chat.activeIndexId = result.index_id;
    pushMessage(chat, {
      role: "system",
      text: `Uploaded ${result.source_doc} and created index ${result.index_id}.`,
      meta: { type: "document_upload", ...docEntry },
    });
    await saveChat(chat);

    res.json({
      ...result,
      chat: toChatSummary(chat),
      documents: chat.documents,
      activeIndexId: chat.activeIndexId,
    });
  } catch (err) {
    res.status(400).json({ error: `PDF upload and indexing failed: ${err.message}` });
  }
});

app.post("/api/rag/ask", async (req, res) => {
  try {
    const body = req.body || {};
    const modelMode = String(body.model_mode || "custom").trim().toLowerCase();
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
      model_mode: modelMode,
    };
    const result = await runPythonRag(payload);
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: `RAG answer generation failed: ${err.message}` });
  }
});

app.post("/api/chats/:chatId/ask", async (req, res) => {
  try {
    const chat = await readChat(req.params.chatId);
    assertChatExists(chat);

    const body = req.body || {};
    const modelMode = String(body.model_mode || "custom").trim().toLowerCase();
    const question = String(body.question || "").trim();
    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    const indexId = String(body.index_id || chat.activeIndexId || "").trim();
    if (!indexId) {
      return res.status(400).json({ error: "Please upload a PDF first for this chat" });
    }

    const doc = (chat.documents || []).find((d) => d.indexId === indexId) || null;
    const payload = {
      action: "ask",
      index_id: indexId,
      question,
      top_k: body.top_k,
      reference_answer: body.reference_answer,
      question_type: body.question_type,
      difficulty: body.difficulty,
      source_doc: doc ? doc.sourceDoc : body.source_doc,
      id: body.id,
      model_mode: modelMode,
    };

    pushMessage(chat, {
      role: "user",
      text: question,
      meta: {
        indexId,
        modelMode,
        questionType: String(body.question_type || "unknown"),
        difficulty: String(body.difficulty || "unknown"),
        topK: Number(body.top_k || 3),
      },
    });

    const result = await runPythonRag(payload);

    pushMessage(chat, {
      role: "assistant",
      text: result.rag_answer || "",
      meta: {
        indexId,
        modelMode: result.model_mode || modelMode,
        sourceDoc: result.source_doc,
        hallucination: result.hallucination || null,
        retrievedChunks: result.retrieved_chunks || [],
      },
    });

    if (isDefaultChatTitle(chat.title)) {
      chat.title = createChatTitle(question);
    }
    chat.activeIndexId = indexId;
    await saveChat(chat);

    res.json({
      ...result,
      chat: toChatSummary(chat),
      messages: chat.messages,
      documents: chat.documents,
      activeIndexId: chat.activeIndexId,
    });
  } catch (err) {
    res.status(400).json({ error: `Chat ask failed: ${err.message}` });
  }
});

ensureDataDirs()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Custom detector Node backend running at http://127.0.0.1:${PORT}`);
    });
  })
  .catch((err) => {
    console.error(`Failed to initialize data directories: ${err.message}`);
    process.exit(1);
  });
