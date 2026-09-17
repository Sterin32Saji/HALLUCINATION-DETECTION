const API_BASE_URL = window.localStorage.getItem("customDetectorApiBase") || "http://127.0.0.1:5055";
const ACTIVE_CHAT_KEY = "ragAppActiveChatId";

const ui = {
  apiUrlLabel: document.getElementById("api-url-label"),
  chatList: document.getElementById("chat-list"),
  newChatBtn: document.getElementById("new-chat-btn"),
  chatTitle: document.getElementById("chat-title"),
  chatSubtitle: document.getElementById("chat-subtitle"),
  uploadForm: document.getElementById("upload-form"),
  pdfFile: document.getElementById("pdf-file"),
  documentSelect: document.getElementById("document-select"),
  askForm: document.getElementById("ask-form"),
  question: document.getElementById("question"),
  questionType: document.getElementById("question-type"),
  difficulty: document.getElementById("difficulty"),
  topK: document.getElementById("top-k"),
  modelCustom: document.getElementById("model-custom"),
  modelHybrid: document.getElementById("model-hybrid"),
  sendBtn: document.getElementById("send-btn"),
  uploadBtn: document.getElementById("upload-btn"),
  askLoading: document.getElementById("ask-loading"),
  refreshEvaluationBtn: document.getElementById("refresh-evaluation-btn"),
  evaluationSubtitle: document.getElementById("evaluation-subtitle"),
  integrityStrip: document.getElementById("integrity-strip"),
  metricsTable: document.getElementById("metrics-table"),
  performanceChart: document.getElementById("performance-chart"),
  confusionMethod: document.getElementById("confusion-method"),
  confusionMatrix: document.getElementById("confusion-matrix"),
  distributionMethod: document.getElementById("distribution-method"),
  distributionChart: document.getElementById("distribution-chart"),
  rocMethod: document.getElementById("roc-method"),
  rocChart: document.getElementById("roc-chart"),
  prMethod: document.getElementById("pr-method"),
  prChart: document.getElementById("pr-chart"),
  thresholdMethod: document.getElementById("threshold-method"),
  thresholdChart: document.getElementById("threshold-chart"),
  riskSummary: document.getElementById("risk-summary"),
  riskChart: document.getElementById("risk-chart"),
  statusBox: document.getElementById("status-box"),
  statusText: document.getElementById("status-text"),
  errorBox: document.getElementById("error-box"),
  errorOutput: document.getElementById("error-output"),
  messages: document.getElementById("messages"),
  mainPanel: document.querySelector(".main-panel"),
};

let chats = [];
let activeChat = null;
let isAsking = false;
let isUploading = false;
let isSwitchingDocument = false;
let isCreatingChat = false;
let evaluationState = null;

ui.apiUrlLabel.textContent = API_BASE_URL;

function setStatus(text, show = true) {
  ui.statusText.textContent = text;
  ui.statusBox.hidden = !show;
}

function clearStatus() {
  ui.statusBox.hidden = true;
}

function setError(message) {
  ui.errorOutput.textContent = message;
  ui.errorBox.hidden = false;
}

function clearError() {
  ui.errorBox.hidden = true;
}

function hasBusyState() {
  return isAsking || isUploading || isSwitchingDocument || isCreatingChat;
}

function syncBusyUi() {
  const busy = hasBusyState();
  ui.mainPanel.classList.toggle("is-busy", busy);

  ui.sendBtn.disabled = busy || isAsking;
  ui.uploadBtn.disabled = busy || isUploading;
  ui.newChatBtn.disabled = busy || isCreatingChat;
  ui.documentSelect.disabled = busy || ui.documentSelect.options.length <= 1;
  ui.pdfFile.disabled = busy;
  ui.question.disabled = busy;
  ui.questionType.disabled = busy;
  ui.difficulty.disabled = busy;
  ui.topK.disabled = busy;
  ui.modelCustom.disabled = busy;
  ui.modelHybrid.disabled = busy;
  ui.refreshEvaluationBtn.disabled = busy;

  ui.sendBtn.textContent = isAsking ? "Sending..." : "Send";
  ui.uploadBtn.textContent = isUploading ? "Uploading..." : "Upload PDF";
  ui.newChatBtn.textContent = isCreatingChat ? "Creating..." : "+ New Chat";
  ui.askLoading.hidden = !isAsking;
}

function withBusyFlag(flagName, fn) {
  return async (...args) => {
    if (hasBusyState()) {
      return;
    }
    if (flagName === "ask") {
      isAsking = true;
    }
    if (flagName === "upload") {
      isUploading = true;
    }
    if (flagName === "switch") {
      isSwitchingDocument = true;
    }
    if (flagName === "chat") {
      isCreatingChat = true;
    }
    syncBusyUi();
    try {
      await fn(...args);
    } finally {
      if (flagName === "ask") {
        isAsking = false;
      }
      if (flagName === "upload") {
        isUploading = false;
      }
      if (flagName === "switch") {
        isSwitchingDocument = false;
      }
      if (flagName === "chat") {
        isCreatingChat = false;
      }
      syncBusyUi();
    }
  };
}

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, options);
  const body = await res.json();
  if (!res.ok) {
    throw new Error(body.error || "Request failed");
  }
  return body;
}

function fmtDate(iso) {
  if (!iso) {
    return "";
  }
  return new Date(iso).toLocaleString();
}

function scoreClass(score) {
  if (score == null) {
    return "chip neutral";
  }
  if (score >= 0.75) {
    return "chip high";
  }
  if (score >= 0.45) {
    return "chip medium";
  }
  return "chip low";
}

function riskText(score) {
  if (score == null) {
    return "No score";
  }
  return `${(score * 100).toFixed(1)}% risk`;
}

function fmtMetric(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(3);
}

function safePercent(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return `${Number(value).toFixed(1)}%`;
}

function methodOptions(methods) {
  return methods.map((method) => `<option value="${method.method}">${method.display_name || method.method}</option>`).join("");
}

function renderMetricsTable(metrics) {
  const rows = [...metrics].sort((a, b) => b.f1 - a.f1);
  ui.metricsTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Detector</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>TP</th>
          <th>TN</th>
          <th>FP</th>
          <th>FN</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${row.display_name || row.method}</td>
            <td>${fmtMetric(row.accuracy)}</td>
            <td>${fmtMetric(row.precision)}</td>
            <td>${fmtMetric(row.recall)}</td>
            <td>${fmtMetric(row.f1)}</td>
            <td>${row.tp}</td>
            <td>${row.tn}</td>
            <td>${row.fp}</td>
            <td>${row.fn}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderBarChart(container, rows, valueKey, labelKey = "display_name") {
  const width = 520;
  const height = 240;
  const pad = 34;
  const values = rows.map((row) => Number(row[valueKey] || 0));
  const maxValue = Math.max(1, ...values);
  const barGap = 10;
  const barWidth = (width - pad * 2 - barGap * (rows.length - 1)) / rows.length;
  const bars = rows.map((row, index) => {
    const value = Number(row[valueKey] || 0);
    const barHeight = ((height - pad * 2) * value) / maxValue;
    const x = pad + index * (barWidth + barGap);
    const y = height - pad - barHeight;
    const label = String(row[labelKey] || row.method || "").replace(" Model", "").replace("Retrieval ", "");
    return `
      <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="4"></rect>
      <text x="${x + barWidth / 2}" y="${height - 12}" text-anchor="middle">${label}</text>
      <text x="${x + barWidth / 2}" y="${Math.max(14, y - 5)}" text-anchor="middle">${fmtMetric(value)}</text>
    `;
  }).join("");
  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line>
      ${bars}
    </svg>
  `;
}

function renderGroupedHistogram(container, rows, scoreKey) {
  const bins = Array.from({ length: 10 }, (_, index) => ({ min: index / 10, max: (index + 1) / 10, hall: 0, nonHall: 0 }));
  for (const row of rows) {
    const value = Number(row[scoreKey]);
    if (!Number.isFinite(value)) {
      continue;
    }
    const idx = Math.max(0, Math.min(9, Math.floor(value * 10)));
    if (Number(row.human_label) === 1) {
      bins[idx].hall += 1;
    } else {
      bins[idx].nonHall += 1;
    }
  }
  const width = 520;
  const height = 240;
  const pad = 34;
  const maxCount = Math.max(1, ...bins.flatMap((bin) => [bin.hall, bin.nonHall]));
  const groupWidth = (width - pad * 2) / bins.length;
  const barWidth = groupWidth / 3;
  const shapes = bins.map((bin, index) => {
    const x = pad + index * groupWidth;
    const h0 = ((height - pad * 2) * bin.nonHall) / maxCount;
    const h1 = ((height - pad * 2) * bin.hall) / maxCount;
    return `
      <rect class="nonhall" x="${x + barWidth * 0.4}" y="${height - pad - h0}" width="${barWidth}" height="${h0}" rx="3"></rect>
      <rect class="hall" x="${x + barWidth * 1.5}" y="${height - pad - h1}" width="${barWidth}" height="${h1}" rx="3"></rect>
      <text x="${x + groupWidth / 2}" y="${height - 12}" text-anchor="middle">${bin.min.toFixed(1)}</text>
    `;
  }).join("");
  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line>
      ${shapes}
      <text x="${pad}" y="18">Non-Hall</text>
      <text class="hall-label" x="${pad + 90}" y="18">Hall</text>
    </svg>
  `;
}

function renderLineChart(container, series, options = {}) {
  const width = 520;
  const height = 240;
  const pad = 34;
  const lines = series.map((item, index) => {
    const points = item.points.map(([xVal, yVal]) => {
      const x = pad + Number(xVal) * (width - pad * 2);
      const y = height - pad - Number(yVal) * (height - pad * 2);
      return `${x},${y}`;
    }).join(" ");
    return `<polyline class="line-${index}" points="${points}" fill="none"></polyline>`;
  }).join("");
  const legend = series.map((item, index) => `<span class="legend-item line-${index}">${item.label}</span>`).join("");
  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>
      ${options.diagonal ? `<line class="diagonal" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${pad}"></line>` : ""}
      ${lines}
      <text x="${width / 2}" y="${height - 8}" text-anchor="middle">${options.xLabel || ""}</text>
      <text x="12" y="${height / 2}" text-anchor="middle" transform="rotate(-90 12 ${height / 2})">${options.yLabel || ""}</text>
    </svg>
    <div class="chart-legend">${legend}</div>
  `;
}

function renderConfusionMatrix(method) {
  const cm = evaluationState?.confusion?.[method];
  if (!cm) {
    ui.confusionMatrix.textContent = "No confusion matrix available.";
    return;
  }
  ui.confusionMatrix.innerHTML = `
    <div></div><div>Actual Non-Hall</div><div>Actual Hall</div>
    <div>Pred Non-Hall</div><strong>${cm.tn}</strong><strong>${cm.fn}</strong>
    <div>Pred Hall</div><strong>${cm.fp}</strong><strong>${cm.tp}</strong>
  `;
}

function renderSelectedDistribution() {
  const keyMap = {
    ragas: "ragas_risk_score",
    similarity: "similarity_risk_score",
    hybrid: "hybrid_score",
    custom: "custom_score",
  };
  renderGroupedHistogram(ui.distributionChart, evaluationState.distributions, keyMap[ui.distributionMethod.value]);
}

function renderSelectedRoc() {
  const roc = evaluationState?.roc?.[ui.rocMethod.value];
  if (!roc || roc.available === false) {
    ui.rocChart.textContent = roc?.reason || "No ROC curve available.";
    return;
  }
  const points = roc.fpr.map((x, index) => [x, roc.tpr[index]]);
  renderLineChart(ui.rocChart, [{ label: `AUC ${fmtMetric(roc.auc)}`, points }], { xLabel: "False Positive Rate", yLabel: "True Positive Rate", diagonal: true });
}

function renderSelectedPr() {
  const pr = evaluationState?.pr?.[ui.prMethod.value];
  if (!pr || pr.available === false) {
    ui.prChart.textContent = pr?.reason || "No precision-recall curve available.";
    return;
  }
  const points = pr.recall.map((x, index) => [x, pr.precision[index]]);
  renderLineChart(ui.prChart, [{ label: `AP ${fmtMetric(pr.average_precision)}`, points }], { xLabel: "Recall", yLabel: "Precision" });
}

function renderSelectedThreshold() {
  const threshold = evaluationState?.thresholds?.[ui.thresholdMethod.value];
  if (!threshold || threshold.available === false) {
    ui.thresholdChart.textContent = threshold?.reason || "No threshold curve available.";
    return;
  }
  const curve = threshold.curve || [];
  renderLineChart(ui.thresholdChart, [
    { label: "Accuracy", points: curve.map((row) => [row.threshold, row.accuracy]) },
    { label: "Precision", points: curve.map((row) => [row.threshold, row.precision]) },
    { label: "Recall", points: curve.map((row) => [row.threshold, row.recall]) },
    { label: "F1", points: curve.map((row) => [row.threshold, row.f1]) },
  ], { xLabel: "Threshold", yLabel: "Metric" });
}

function renderRisk(summary, rows) {
  const byLevel = summary.by_level || {};
  ui.riskSummary.innerHTML = `
    <span>Mean ${safePercent(summary.mean)}</span>
    <span>Median ${safePercent(summary.median)}</span>
    <span>Low ${byLevel["Low Risk"] || 0}</span>
    <span>Medium ${byLevel["Medium Risk"] || 0}</span>
    <span>High ${byLevel["High Risk"] || 0}</span>
    <span>Critical ${byLevel["Critical Risk"] || 0}</span>
  `;
  renderBarChart(
    ui.riskChart,
    ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"].map((label) => ({ display_name: label.replace(" Risk", ""), count: byLevel[label] || 0 })),
    "count",
  );
}

function renderIntegrity(summary) {
  const data = summary.data_integrity || {};
  ui.evaluationSubtitle.textContent = `${data.evaluated_prediction_count || 0} records • ${data.hallucinated_examples || 0} hallucinated • ${data.non_hallucinated_examples || 0} non-hallucinated`;
  const checks = [
    ["300 annotations", data.human_annotation_count === 300],
    ["IDs aligned", data.ids_match_human],
    ["Binary labels", data.labels_are_binary],
    ["No missing predictions", Object.values(data.missing_prediction_ids || {}).every((items) => Array.isArray(items) && items.length === 0)],
  ];
  ui.integrityStrip.innerHTML = checks.map(([label, ok]) => `<span class="${ok ? "ok" : "bad"}">${label}: ${ok ? "OK" : "Check"}</span>`).join("");
}

async function loadEvaluation() {
  try {
    const [summary, confusion, distributions, roc, pr, thresholds] = await Promise.all([
      api("/api/evaluation/summary"),
      api("/api/evaluation/confusion-matrices"),
      api("/api/evaluation/distributions"),
      api("/api/evaluation/roc"),
      api("/api/evaluation/precision-recall"),
      api("/api/evaluation/thresholds"),
    ]);
    const continuousMethods = (summary.metrics || []).filter((row) => ["ragas", "similarity", "hybrid", "custom"].includes(row.method));
    evaluationState = { summary, confusion, distributions, roc, pr, thresholds };
    renderIntegrity(summary);
    renderMetricsTable(summary.metrics || []);
    renderBarChart(ui.performanceChart, [...(summary.metrics || [])].sort((a, b) => b.f1 - a.f1), "f1");
    ui.confusionMethod.innerHTML = methodOptions(summary.metrics || []);
    ui.distributionMethod.innerHTML = methodOptions(continuousMethods);
    ui.rocMethod.innerHTML = methodOptions(continuousMethods);
    ui.prMethod.innerHTML = methodOptions(continuousMethods);
    ui.thresholdMethod.innerHTML = methodOptions(continuousMethods);
    renderConfusionMatrix(ui.confusionMethod.value || "hybrid");
    renderSelectedDistribution();
    renderSelectedRoc();
    renderSelectedPr();
    renderSelectedThreshold();
    renderRisk(summary.risk || {}, distributions || []);
  } catch (err) {
    ui.evaluationSubtitle.textContent = "Evaluation artifacts are not available yet. Run the pipeline to generate them.";
    ui.integrityStrip.innerHTML = `<span class="bad">${String(err.message || err)}</span>`;
  }
}

function buildHallucinationReasons(hallucination, chunks) {
  const reasons = [];
  const score = Number(hallucination.hallucination_score);
  const predictedHallucinated = Number(hallucination.custom_prediction) === 1;

  if (predictedHallucinated) {
    reasons.push(`Model classified this answer as hallucinated with score ${riskText(score)}.`);
  } else {
    reasons.push(`Model classified this answer as grounded with score ${riskText(score)}.`);
  }

  if (Number.isFinite(score)) {
    if (score >= 0.75) {
      reasons.push("Risk score is high and above the strict alert threshold (>= 75%).");
    } else if (score >= 0.45) {
      reasons.push("Risk score is moderate and falls in the warning range (45%-75%).");
    } else {
      reasons.push("Risk score is low, indicating better alignment with retrieved evidence.");
    }
  }

  const safeChunks = Array.isArray(chunks) ? chunks : [];
  if (!safeChunks.length) {
    reasons.push("No retrieved evidence chunks were returned, which increases uncertainty.");
    return reasons;
  }

  const scores = safeChunks.map((c) => Number(c.score || 0)).filter((v) => Number.isFinite(v));
  const best = scores.length ? Math.max(...scores) : 0;
  const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

  if (best < 0.2) {
    reasons.push("Top evidence relevance score is weak (< 0.20), so grounding evidence is limited.");
  } else if (best > 0.55) {
    reasons.push("Top evidence relevance score is strong (> 0.55), suggesting substantial grounding context.");
  }

  if (avg < 0.15) {
    reasons.push("Average evidence relevance is low across retrieved chunks.");
  }

  return reasons;
}

function modelModeFromToggles() {
  const customChecked = Boolean(ui.modelCustom.checked);
  const hybridChecked = Boolean(ui.modelHybrid.checked);
  if (customChecked && hybridChecked) {
    return "both";
  }
  if (hybridChecked) {
    return "hybrid";
  }
  return "custom";
}

function ensureAtLeastOneModelSelected(changedKey) {
  if (!ui.modelCustom.checked && !ui.modelHybrid.checked) {
    if (changedKey === "custom") {
      ui.modelHybrid.checked = true;
    } else {
      ui.modelCustom.checked = true;
    }
  }
}

function normalizeModelResults(hallucination) {
  const out = {};
  if (!hallucination || typeof hallucination !== "object") {
    return out;
  }

  // Backward compatibility with old response shape.
  if (Object.prototype.hasOwnProperty.call(hallucination, "custom_prediction")) {
    out.custom = hallucination;
  }

  if (hallucination.custom && typeof hallucination.custom === "object") {
    out.custom = hallucination.custom;
  }

  if (hallucination.hybrid && typeof hallucination.hybrid === "object") {
    out.hybrid = hallucination.hybrid;
  }

  return out;
}

function appendSingleModelResult(container, modelName, result, chunks) {
  if (!result || typeof result !== "object") {
    return;
  }

  let prediction = null;
  let score = null;
  if (modelName === "custom") {
    prediction = Number(result.custom_prediction);
    score = result.hallucination_score;
  }
  if (modelName === "hybrid") {
    prediction = Number(result.hybrid_prediction);
    score = result.hybrid_score;
  }

  const chip = document.createElement("div");
  chip.className = scoreClass(score);
  const modelLabel = modelName === "hybrid" ? "Hybrid" : "Custom";
  const predText = prediction === 1 ? "Hallucinated" : "Not Hallucinated";
  chip.textContent = `${modelLabel}: ${predText} • ${riskText(score)}`;
  container.appendChild(chip);

  const reasonCard = document.createElement("section");
  reasonCard.className = `reason-card ${prediction === 1 ? "high" : ""}`;
  const reasonTitle = document.createElement("div");
  reasonTitle.className = "reason-title";
  reasonTitle.textContent = prediction === 1 ? `${modelLabel} reason this was flagged` : `${modelLabel} reason this seems grounded`;
  reasonCard.appendChild(reasonTitle);

  const reasonList = document.createElement("ul");
  reasonList.className = "reason-list";

  if (modelName === "custom") {
    for (const reason of buildHallucinationReasons(result, chunks)) {
      const li = document.createElement("li");
      li.textContent = reason;
      reasonList.appendChild(li);
    }
  } else {
    const signals = result.signals || {};
    const hybridReasons = [
      `Hybrid voting signals: baseline=${signals.baseline ?? "n/a"}, selfcheck=${signals.selfcheck ?? "n/a"}, ragas=${signals.ragas ?? "n/a"}, similarity=${signals.similarity ?? "n/a"}.`,
      `Faithfulness proxy=${Number(signals.faithfulness_proxy || 0).toFixed(3)}, top retrieval score=${Number(signals.top_retrieval_score || 0).toFixed(3)}.`,
    ];
    if (signals.uncertainty_phrase_detected) {
      hybridReasons.push("Answer includes uncertainty phrasing, which increased hybrid hallucination risk.");
    }
    for (const reason of hybridReasons) {
      const li = document.createElement("li");
      li.textContent = reason;
      reasonList.appendChild(li);
    }
  }

  reasonCard.appendChild(reasonList);
  container.appendChild(reasonCard);
}

function documentNameForActive(chat) {
  if (!chat || !chat.activeIndexId || !Array.isArray(chat.documents)) {
    return "No active document";
  }
  const doc = chat.documents.find((d) => d.indexId === chat.activeIndexId);
  return doc ? `${doc.sourceDoc} (${doc.chunkCount} chunks)` : "No active document";
}

function renderChatList() {
  ui.chatList.innerHTML = "";
  for (const chat of chats) {
    const button = document.createElement("button");
    button.className = `chat-item ${activeChat && activeChat.id === chat.id ? "active" : ""}`;
    button.type = "button";
    const title = chat.title || "Untitled chat";
    button.innerHTML = `
      <div class="chat-item-title">${title}</div>
      <div class="chat-item-meta">${fmtDate(chat.updatedAt)} • ${chat.messageCount || 0} msgs</div>
    `;
    button.addEventListener("click", () => loadChat(chat.id));
    ui.chatList.appendChild(button);
  }
}

function renderDocumentSelect(chat) {
  ui.documentSelect.innerHTML = "";
  const docs = Array.isArray(chat.documents) ? chat.documents : [];
  if (!docs.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No document uploaded yet";
    ui.documentSelect.appendChild(opt);
    ui.documentSelect.disabled = true;
    return;
  }

  for (const doc of docs) {
    const opt = document.createElement("option");
    opt.value = doc.indexId;
    opt.textContent = `${doc.sourceDoc} (${doc.chunkCount})`;
    if (doc.indexId === chat.activeIndexId) {
      opt.selected = true;
    }
    ui.documentSelect.appendChild(opt);
  }
  if (!hasBusyState()) {
    ui.documentSelect.disabled = false;
  }
}

function renderMessages(chat) {
  ui.messages.innerHTML = "";
  const messages = Array.isArray(chat.messages) ? chat.messages : [];

  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Start by uploading a PDF, then ask a question.";
    ui.messages.appendChild(empty);
    return;
  }

  for (const msg of messages) {
    const bubble = document.createElement("article");
    bubble.className = `msg ${msg.role}`;

    const head = document.createElement("header");
    head.className = "msg-head";
    const role = msg.role === "assistant" ? "Assistant" : msg.role === "user" ? "You" : "System";
    head.innerHTML = `<span>${role}</span><span>${fmtDate(msg.createdAt)}</span>`;
    bubble.appendChild(head);

    const body = document.createElement("div");
    body.className = "msg-body";
    body.textContent = msg.text || "";
    bubble.appendChild(body);

    if (msg.role === "assistant" && msg.meta && msg.meta.hallucination) {
      const modelResults = normalizeModelResults(msg.meta.hallucination);
      const chunks = Array.isArray(msg.meta.retrievedChunks) ? msg.meta.retrievedChunks : [];
      const customPred = modelResults.custom ? Number(modelResults.custom.custom_prediction) : null;
      const hybridPred = modelResults.hybrid ? Number(modelResults.hybrid.hybrid_prediction) : null;
      const anyFlagged = customPred === 1 || hybridPred === 1;
      if (anyFlagged) {
        bubble.classList.add("high-risk");
      }

      if (modelResults.custom) {
        appendSingleModelResult(bubble, "custom", modelResults.custom, chunks);
      }
      if (modelResults.hybrid) {
        appendSingleModelResult(bubble, "hybrid", modelResults.hybrid, chunks);
      }

      const details = document.createElement("details");
      details.className = "evidence-details";
      const summary = document.createElement("summary");
      summary.textContent = "View retrieved evidence";
      details.appendChild(summary);

      if (!chunks.length) {
        const noChunk = document.createElement("p");
        noChunk.textContent = "No chunk details available.";
        details.appendChild(noChunk);
      } else {
        for (const chunk of chunks) {
          const block = document.createElement("div");
          block.className = "chunk";
          block.innerHTML = `<div class="chunk-head">${chunk.chunk_id} • score ${(chunk.score || 0).toFixed(3)}</div><div class="chunk-text">${chunk.text || ""}</div>`;
          details.appendChild(block);
        }
      }
      bubble.appendChild(details);
    }

    ui.messages.appendChild(bubble);
  }

  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function persistActiveChatId(chatId) {
  window.localStorage.setItem(ACTIVE_CHAT_KEY, chatId);
}

function readActiveChatId() {
  return window.localStorage.getItem(ACTIVE_CHAT_KEY);
}

async function refreshChats() {
  chats = await api("/api/chats");
  renderChatList();
}

async function createNewChat() {
  const chat = await api("/api/chats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  await refreshChats();
  await loadChat(chat.id);
}

async function loadChat(chatId) {
  clearError();
  const chat = await api(`/api/chats/${chatId}`);
  activeChat = chat;
  persistActiveChatId(chat.id);

  ui.chatTitle.textContent = chat.title || "Chat";
  ui.chatSubtitle.textContent = documentNameForActive(chat);
  renderDocumentSelect(chat);
  renderMessages(chat);
  renderChatList();
  syncBusyUi();
}

async function bootstrap() {
  setStatus("Loading chats...");
  try {
    await loadEvaluation();
    await refreshChats();
    if (!chats.length) {
      await createNewChat();
    } else {
      const stored = readActiveChatId();
      const exists = stored && chats.find((c) => c.id === stored);
      await loadChat(exists ? stored : chats[0].id);
    }
    clearStatus();
  } catch (err) {
    setError(String(err.message || err));
    clearStatus();
  }
}

ui.newChatBtn.addEventListener("click", async () => {
  await withBusyFlag("chat", async () => {
  clearError();
  setStatus("Creating new chat...");
  try {
    await createNewChat();
    clearStatus();
  } catch (err) {
    setError(String(err.message || err));
    clearStatus();
  }
  })();
});

ui.documentSelect.addEventListener("change", async () => {
  await withBusyFlag("switch", async () => {
  if (!activeChat || !ui.documentSelect.value) {
    return;
  }
  clearError();
  setStatus("Switching active document...");
  try {
    await api(`/api/chats/${activeChat.id}/select-document`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index_id: ui.documentSelect.value }),
    });
    await loadChat(activeChat.id);
    clearStatus();
  } catch (err) {
    setError(String(err.message || err));
    clearStatus();
  }
  })();
});

ui.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await withBusyFlag("upload", async () => {
  clearError();

  if (!activeChat) {
    setError("No active chat. Create a new chat first.");
    return;
  }

  if (!ui.pdfFile.files || !ui.pdfFile.files[0]) {
    setError("Choose a PDF to upload.");
    return;
  }

  setStatus("Uploading PDF and building index...");
  try {
    const form = new FormData();
    form.append("pdf_file", ui.pdfFile.files[0]);
    await api(`/api/chats/${activeChat.id}/upload-pdf`, {
      method: "POST",
      body: form,
    });
    await refreshChats();
    await loadChat(activeChat.id);
    ui.pdfFile.value = "";
    setStatus("PDF indexed and attached to this chat.", true);
    setTimeout(() => clearStatus(), 1200);
  } catch (err) {
    setError(String(err.message || err));
    clearStatus();
  }
  })();
});

ui.askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await withBusyFlag("ask", async () => {
  clearError();

  if (!activeChat) {
    setError("No active chat selected.");
    return;
  }

  const question = ui.question.value.trim();
  if (!question) {
    setError("Please enter a question.");
    return;
  }

  const indexId = ui.documentSelect.value;
  if (!indexId) {
    setError("Please upload and select a PDF document first.");
    return;
  }

  setStatus("Generating answer and hallucination analysis...");
  try {
    await api(`/api/chats/${activeChat.id}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        index_id: indexId,
        question,
        question_type: ui.questionType.value.trim() || "unknown",
        difficulty: ui.difficulty.value.trim() || "unknown",
        top_k: Number(ui.topK.value || 3),
        model_mode: modelModeFromToggles(),
      }),
    });

    ui.question.value = "";

    await refreshChats();
    await loadChat(activeChat.id);
    clearStatus();
  } catch (err) {
    setError(String(err.message || err));
    clearStatus();
  }
  })();
});

ui.modelCustom.addEventListener("change", () => {
  ensureAtLeastOneModelSelected("custom");
});

ui.modelHybrid.addEventListener("change", () => {
  ensureAtLeastOneModelSelected("hybrid");
});

ui.refreshEvaluationBtn.addEventListener("click", loadEvaluation);

ui.confusionMethod.addEventListener("change", () => {
  renderConfusionMatrix(ui.confusionMethod.value);
});

ui.distributionMethod.addEventListener("change", renderSelectedDistribution);
ui.rocMethod.addEventListener("change", renderSelectedRoc);
ui.prMethod.addEventListener("change", renderSelectedPr);
ui.thresholdMethod.addEventListener("change", renderSelectedThreshold);

bootstrap();
