const API_BASE_URL = window.localStorage.getItem("customDetectorApiBase") || "http://127.0.0.1:5055";

const uploadForm = document.getElementById("upload-form");
const askForm = document.getElementById("ask-form");
const pdfFile = document.getElementById("pdf-file");
const indexInfo = document.getElementById("index-info");
const question = document.getElementById("question");
const referenceAnswer = document.getElementById("reference-answer");
const questionType = document.getElementById("question-type");
const difficulty = document.getElementById("difficulty");
const topK = document.getElementById("top-k");
const statusBox = document.getElementById("status-box");
const statusText = document.getElementById("status-text");
const resultBox = document.getElementById("result-box");
const resultOutput = document.getElementById("result-output");
const errorBox = document.getElementById("error-box");
const errorOutput = document.getElementById("error-output");
const apiUrlLabel = document.getElementById("api-url-label");

apiUrlLabel.textContent = API_BASE_URL;
question.value = "What is the main security guidance discussed in this document?";

let currentIndexId = "";
let currentSourceDoc = "";

function showStatus(text) {
  statusText.textContent = text;
  statusBox.hidden = false;
}

function clearMessages() {
  errorBox.hidden = true;
  resultBox.hidden = true;
}

function showError(message) {
  errorOutput.textContent = message;
  errorBox.hidden = false;
}

function showResult(obj) {
  resultOutput.textContent = JSON.stringify(obj, null, 2);
  resultBox.hidden = false;
}

async function callApi(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error || "Request failed");
  }
  return body;
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessages();
  indexInfo.hidden = true;

  if (!pdfFile.files || !pdfFile.files[0]) {
    showError("Please choose a PDF file first.");
    return;
  }

  showStatus("Uploading PDF and building chunk/FAISS index...");

  const form = new FormData();
  form.append("pdf_file", pdfFile.files[0]);

  try {
    const result = await callApi("/api/rag/upload-pdf", {
      method: "POST",
      body: form
    });
    currentIndexId = result.index_id;
    currentSourceDoc = result.source_doc;
    indexInfo.hidden = false;
    indexInfo.textContent = `Index ready: ${currentIndexId} | source: ${currentSourceDoc} | chunks: ${result.chunk_count}`;
    showStatus("Indexing completed. You can ask questions now.");
    showResult(result);
  } catch (err) {
    showStatus("Indexing failed.");
    showError(String(err.message || err));
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessages();

  if (!currentIndexId) {
    showError("Please upload a PDF first to create an index.");
    return;
  }

  const q = question.value.trim();
  if (!q) {
    showError("Please enter a question.");
    return;
  }

  showStatus("Retrieving chunks, generating RAG answer, and scoring hallucination...");

  try {
    const result = await callApi("/api/rag/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        index_id: currentIndexId,
        question: q,
        reference_answer: referenceAnswer.value.trim(),
        question_type: questionType.value.trim() || "unknown",
        difficulty: difficulty.value.trim() || "unknown",
        top_k: Number(topK.value || 3),
        source_doc: currentSourceDoc || undefined,
      })
    });
    showStatus("Answer and score generated successfully.");
    showResult(result);
  } catch (err) {
    showStatus("Generation failed.");
    showError(String(err.message || err));
  }
});
