const API_BASE_URL = window.localStorage.getItem("customDetectorApiBase") || "http://127.0.0.1:5055";

const SAMPLE = {
  id: "debug-custom-1",
  question: "Which document specifies HTTP semantics?",
  answer: "RFC 9110 HTTP Semantics June 2022.",
  evidence: "RFC 9110 HTTP Semantics June 2022",
  rag_answer: "HTTP semantics are specified by RFC 9110.",
  question_type: "identifier",
  difficulty: "easy",
  source_doc: "docs/rfc9110.pdf"
};

const singleForm = document.getElementById("single-form");
const jsonlForm = document.getElementById("jsonl-form");
const singleJson = document.getElementById("single-json");
const jsonlFile = document.getElementById("jsonl-file");
const statusBox = document.getElementById("status-box");
const statusText = document.getElementById("status-text");
const resultBox = document.getElementById("result-box");
const resultOutput = document.getElementById("result-output");
const errorBox = document.getElementById("error-box");
const errorOutput = document.getElementById("error-output");
const apiUrlLabel = document.getElementById("api-url-label");

singleJson.value = JSON.stringify(SAMPLE, null, 2);
apiUrlLabel.textContent = API_BASE_URL;

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

singleForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessages();

  let payload;
  try {
    payload = JSON.parse(singleJson.value);
  } catch {
    showError("Invalid JSON in Single Sample input.");
    return;
  }

  showStatus("Sending sample and running custom model prediction...");

  try {
    const result = await callApi("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    showStatus("Prediction completed.");
    showResult(result);
  } catch (err) {
    showStatus("Prediction failed.");
    showError(String(err.message || err));
  }
});

jsonlForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessages();

  if (!jsonlFile.files || !jsonlFile.files[0]) {
    showError("Please choose a JSONL file.");
    return;
  }

  showStatus("Uploading JSONL and running batch predictions...");

  const form = new FormData();
  form.append("input_file", jsonlFile.files[0]);

  try {
    const result = await callApi("/api/predict-jsonl", {
      method: "POST",
      body: form
    });
    showStatus("Batch prediction completed.");
    showResult(result);
  } catch (err) {
    showStatus("Batch prediction failed.");
    showError(String(err.message || err));
  }
});
