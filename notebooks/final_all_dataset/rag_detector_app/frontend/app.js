const API = window.localStorage.getItem("ragDetectorApiBase") || "http://127.0.0.1:6060";

const screenLoad = document.getElementById("screen-load");
const screenAsk = document.getElementById("screen-ask");
const existingSection = document.getElementById("existing-section");
const indexList = document.getElementById("index-list");
const uploadForm = document.getElementById("upload-form");
const pdfFileInput = document.getElementById("pdf-file");
const fileNameDisplay = document.getElementById("file-name-display");
const uploadBtn = document.getElementById("upload-btn");
const uploadProgress = document.getElementById("upload-progress");
const progressFill = document.getElementById("progress-fill");
const progressText = document.getElementById("progress-text");
const uploadError = document.getElementById("upload-error");
const activeDocLabel = document.getElementById("active-doc-label");
const switchPdfBtn = document.getElementById("switch-pdf-btn");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const chatThread = document.getElementById("chat-thread");

const ACTIVE_INDEX_KEY = "ragDetectorActiveIndexId";
const ACTIVE_DOC_KEY = "ragDetectorActiveSourceDoc";

let activeIndexId = "";
let activeSourceDoc = "";

async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || "Request failed");
  return body;
}

function showScreen(name) {
  screenLoad.hidden = name !== "load";
  screenAsk.hidden = name !== "ask";
}

function setProgress(pct, msg) {
  uploadProgress.hidden = false;
  progressFill.style.width = `${pct}%`;
  progressText.textContent = msg;
}

function showUploadError(msg) {
  uploadError.textContent = msg;
  uploadError.hidden = false;
}

function clearUploadError() {
  uploadError.hidden = true;
  uploadError.textContent = "";
}

function saveActiveIndex(indexId, sourceDoc) {
  window.localStorage.setItem(ACTIVE_INDEX_KEY, indexId);
  window.localStorage.setItem(ACTIVE_DOC_KEY, sourceDoc);
}

function clearActiveIndex() {
  window.localStorage.removeItem(ACTIVE_INDEX_KEY);
  window.localStorage.removeItem(ACTIVE_DOC_KEY);
}

function resetChatThread() {
  chatThread.innerHTML = '<p class="chat-hint">Ask any question about the document.</p>';
}

function getBadgeState(hallucinationResult) {
  const isHallucinated = Boolean(hallucinationResult?.hallucinated);
  const risk = Number(hallucinationResult?.hallucination_score ?? 0);

  if (isHallucinated || risk >= 0.5) {
    return {
      className: "halluc-red",
      text: `Hallucinated | High risk (${(risk * 100).toFixed(1)}%)`,
    };
  }

  if (risk >= 0.2) {
    return {
      className: "halluc-yellow",
      text: `Not hallucinated | Medium confidence (${(risk * 100).toFixed(1)}%)`,
    };
  }

  return {
    className: "halluc-green",
    text: `Not hallucinated | High confidence (${(risk * 100).toFixed(1)}%)`,
  };
}

async function loadInteractions(indexId) {
  const { interactions } = await api(`/api/indexes/${indexId}/interactions`);
  return Array.isArray(interactions) ? interactions : [];
}

async function openIndex(indexId, sourceDoc) {
  activeIndexId = indexId;
  activeSourceDoc = sourceDoc;
  saveActiveIndex(indexId, sourceDoc);
  resetChatThread();
  activeDocLabel.textContent = `Document: ${sourceDoc.replace(/^\d+_/, "")}`;
  showScreen("ask");

  try {
    const interactions = await loadInteractions(indexId);
    if (interactions.length === 0) {
      resetChatThread();
      return;
    }

    chatThread.innerHTML = "";
    interactions.forEach((item) => {
      addChatMessage("user", item.question || "");
      addChatMessage("assistant", item);
    });
  } catch {
    resetChatThread();
  }
}

async function loadExistingIndexes() {
  try {
    const { indexes } = await api("/api/indexes");
    if (!indexes || indexes.length === 0) {
      existingSection.hidden = true;
      return;
    }

    existingSection.hidden = false;
    indexList.innerHTML = "";

    indexes.forEach((idx) => {
      const card = document.createElement("div");
      card.className = "index-card";

      const name = document.createElement("div");
      name.className = "index-card-name";
      name.textContent = String(idx.source_doc || "document.pdf").replace(/^\d+_/, "");

      const meta = document.createElement("div");
      meta.className = "index-card-meta";
      const date = idx.created_at ? new Date(idx.created_at).toLocaleString() : "";
      meta.textContent = `${idx.chunk_count || 0} chunks ${date ? ` | ${date}` : ""}`;

      const btn = document.createElement("button");
      btn.className = "btn-resume";
      btn.textContent = "Open";
      btn.addEventListener("click", () => {
        void openIndex(idx.index_id, idx.source_doc);
      });

      card.append(name, meta, btn);
      indexList.append(card);
    });
  } catch {
    existingSection.hidden = true;
  }
}

pdfFileInput.addEventListener("change", () => {
  const file = pdfFileInput.files?.[0];
  if (file) {
    fileNameDisplay.textContent = file.name;
    uploadBtn.disabled = false;
  } else {
    fileNameDisplay.textContent = "Choose PDF file...";
    uploadBtn.disabled = true;
  }
  clearUploadError();
});

const UPLOAD_STEPS = [
  [10, "Uploading file to server..."],
  [25, "Extracting text from PDF..."],
  [45, "Chunking document..."],
  [65, "Embedding chunks..."],
  [85, "Building vector index..."],
  [95, "Saving index to disk..."],
];

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearUploadError();

  const file = pdfFileInput.files?.[0];
  if (!file) {
    showUploadError("Please choose a PDF file first.");
    return;
  }

  uploadBtn.disabled = true;
  let stepIdx = 0;
  const tick = () => {
    if (stepIdx < UPLOAD_STEPS.length) {
      const [pct, msg] = UPLOAD_STEPS[stepIdx++];
      setProgress(pct, msg);
    }
  };

  tick();
  const timer = setInterval(tick, 3500);

  const formData = new FormData();
  formData.append("pdf_file", file);

  try {
    const result = await api("/api/upload-pdf", { method: "POST", body: formData });
    clearInterval(timer);
    setProgress(100, "Index ready");
    setTimeout(() => {
      uploadProgress.hidden = true;
      progressFill.style.width = "0%";
      uploadBtn.disabled = false;
      pdfFileInput.value = "";
      fileNameDisplay.textContent = "Choose PDF file...";
      void loadExistingIndexes();
      void openIndex(result.index_id, result.source_doc);
    }, 500);
  } catch (err) {
    clearInterval(timer);
    uploadProgress.hidden = true;
    progressFill.style.width = "0%";
    uploadBtn.disabled = false;
    showUploadError(String(err.message || err));
  }
});

switchPdfBtn.addEventListener("click", () => {
  clearActiveIndex();
  showScreen("load");
  void loadExistingIndexes();
});

function addChatMessage(role, content) {
  const hint = chatThread.querySelector(".chat-hint");
  if (hint) hint.remove();

  const wrap = document.createElement("div");
  wrap.className = `chat-msg chat-${role}`;

  if (role === "user") {
    wrap.textContent = content;
  } else {
    const answerEl = document.createElement("p");
    answerEl.className = "chat-answer";
    answerEl.textContent = content.answer || "";
    wrap.appendChild(answerEl);

    if (content.hallucination_result) {
      const badge = document.createElement("span");
      const badgeState = getBadgeState(content.hallucination_result);
      badge.className = `halluc-badge ${badgeState.className}`;
      badge.textContent = badgeState.text;
      wrap.appendChild(badge);
    }

    if (content.retrieved_chunks && content.retrieved_chunks.length) {
      const details = document.createElement("details");
      details.className = "chunks-details";
      const summary = document.createElement("summary");
      summary.textContent = `${content.retrieved_chunks.length} source chunk(s)`;
      details.appendChild(summary);
      content.retrieved_chunks.forEach((c, i) => {
        const p = document.createElement("p");
        p.className = "chunk-preview";
        p.textContent = `[${i + 1}] ${(c.text || "").slice(0, 200)}...`;
        details.appendChild(p);
      });
      wrap.appendChild(details);
    }
  }

  chatThread.appendChild(wrap);
  chatThread.scrollTop = chatThread.scrollHeight;
}

function addThinkingIndicator() {
  const el = document.createElement("div");
  el.className = "chat-msg chat-assistant chat-thinking";
  el.id = "chat-thinking";
  el.textContent = "Thinking...";
  chatThread.appendChild(el);
  chatThread.scrollTop = chatThread.scrollHeight;
  return el;
}

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!activeIndexId) return;

  const q = questionInput.value.trim();
  if (!q) return;

  questionInput.value = "";
  questionInput.disabled = true;
  askBtn.disabled = true;

  addChatMessage("user", q);
  const thinking = addThinkingIndicator();

  try {
    const result = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        index_id: activeIndexId,
        question: q,
        source_doc: activeSourceDoc,
      }),
    });
    thinking.remove();
    addChatMessage("assistant", result);
  } catch (err) {
    thinking.remove();
    addChatMessage("assistant", {
      answer: `Error: ${String(err.message || err)}`,
      hallucination_result: null,
      retrieved_chunks: [],
    });
  } finally {
    questionInput.disabled = false;
    askBtn.disabled = false;
    questionInput.focus();
  }
});

async function boot() {
  showScreen("load");
  await loadExistingIndexes();

  const savedIndexId = window.localStorage.getItem(ACTIVE_INDEX_KEY);
  const savedSourceDoc = window.localStorage.getItem(ACTIVE_DOC_KEY);
  if (savedIndexId && savedSourceDoc) {
    await openIndex(savedIndexId, savedSourceDoc);
  }
}

void boot();
