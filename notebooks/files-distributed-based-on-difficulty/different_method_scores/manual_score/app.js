(function () {
    "use strict";

    const TYPES = [
        "unsupported_expansion",
        "wrong_entity",
        "wrong_number",
        "contradiction",
        "fabricated_information",
        "incorrect_procedure",
        "missing_context"
    ];

    const API_BASE = "http://localhost:3000";

    const state = {
        records: [],
        filtered: [],
        current: 0,
        filename: "",
        dirty: false,
        filterMode: "all",
        searchTerm: "",
        saveTimer: null,
        saveInFlight: false,
        lastSaveError: "",
        toastTimer: null,
        serverConnected: false
    };

    const $ = function (id) {
        return document.getElementById(id);
    };

    const fileSelector = $("fileSelector");
    const refreshBtn = $("refreshBtn");
    const saveBtn = $("saveBtn");
    const downloadBtn = $("downloadBtn");
    const serverStatus = $("serverStatus");
    const status = $("status");
    const progress = $("progress-bar");
    const recordsEl = $("records");
    const search = $("search");
    const details = $("details");
    const badge = $("badge");
    const typeSelect = $("typeSelect");
    const reason = $("reason");
    const position = $("position");
    const toastEl = $("toast");
    const hallBtn = $("hallBtn");
    const okBtn = $("okBtn");
    const removeBtn = $("removeBtn");
    const copyBtn = $("copyBtn");
    const firstBtn = $("firstBtn");
    const prevBtn = $("prevBtn");
    const nextBtn = $("nextBtn");
    const lastBtn = $("lastBtn");

    function escapeHTML(value) {
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function toast(message) {
        toastEl.textContent = message;
        toastEl.classList.add("show");

        if (state.toastTimer) {
            clearTimeout(state.toastTimer);
        }

        state.toastTimer = window.setTimeout(function () {
            toastEl.classList.remove("show");
        }, 1800);
    }

    function currentRecord() {
        if (!state.filtered.length) {
            return null;
        }

        return state.records[state.filtered[state.current]] || null;
    }

    function serialize() {
        return state.records.map(function (record) {
            return JSON.stringify(record);
        }).join("\n") + (state.records.length ? "\n" : "");
    }

    function updateStatus(extraMessage) {
        const labeled = state.records.filter(function (record) {
            return record.human_label === 0 || record.human_label === 1;
        }).length;

        const parts = [state.filename || "No file loaded"];

        if (state.records.length) {
            parts.push(state.records.length + " records");
            parts.push(labeled + " labeled");
        }

        if (state.serverConnected && state.filename) {
            parts.push("autosave enabled");
        }

        if (state.saveInFlight) {
            parts.push("saving...");
        } else if (state.dirty) {
            parts.push("unsaved changes");
        } else if (state.records.length) {
            parts.push("saved");
        }

        if (state.lastSaveError) {
            parts.push(state.lastSaveError);
        }

        if (extraMessage) {
            parts.push(extraMessage);
        }

        status.textContent = parts.join(" | ");
        progress.style.width = state.records.length ? String((labeled / state.records.length) * 100) + "%" : "0%";
        updateServerStatus();
    }

    function updateServerStatus() {
        if (state.serverConnected) {
            serverStatus.textContent = "● Online";
            serverStatus.style.color = "#157347";
            serverStatus.style.background = "#dff6e8";
        } else {
            serverStatus.textContent = "● Offline";
            serverStatus.style.color = "#b42318";
            serverStatus.style.background = "#fde7e5";
        }
    }

    function updateBadge(record) {
        badge.className = "badge";

        if (!record) {
            badge.classList.add("none");
            badge.textContent = "No record";
            return;
        }

        if (record.human_label === 1) {
            badge.classList.add("hall");
            badge.textContent = "Hallucinated";
            return;
        }

        if (record.human_label === 0) {
            badge.classList.add("ok");
            badge.textContent = "Correct";
            return;
        }

        badge.classList.add("none");
        badge.textContent = "Unlabeled";
    }

    function buildWelcomeState(message) {
        details.innerHTML = [
            '<section class="hero-card">',
            '<h2 class="hero-title">',
            escapeHTML(message),
            '</h2>',
            '<p class="hero-copy">Select a JSONL file from the dropdown to start labeling. Each edit will autosave to the server after a short delay. You can also download a local copy at any time.</p>',
            '<div class="shortcut-grid">',
            '<div class="shortcut"><strong>H</strong><span>Mark hallucinated</span></div>',
            '<div class="shortcut"><strong>N</strong><span>Mark correct</span></div>',
            '<div class="shortcut"><strong>U</strong><span>Remove label</span></div>',
            '<div class="shortcut"><strong>&larr; / &rarr;</strong><span>Move records</span></div>',
            '</div>',
            '</section>'
        ].join("");
    }

    function renderList() {
        recordsEl.innerHTML = "";

        state.filtered.forEach(function (recordIndex, positionIndex) {
            const record = state.records[recordIndex];
            const card = document.createElement("div");
            let title = record.question || record.id || "Untitled";
            let statusText = "Unlabeled";
            let statusClass = "status-none";

            card.className = "record";
            if (positionIndex === state.current) {
                card.classList.add("active");
            }

            if (String(title).length > 90) {
                title = String(title).slice(0, 90) + "...";
            }

            if (record.human_label === 1) {
                statusText = "Hallucinated";
                statusClass = "status-hall";
            } else if (record.human_label === 0) {
                statusText = "Correct";
                statusClass = "status-ok";
            }

            card.innerHTML = [
                '<div class="record-title">',
                escapeHTML(title),
                '</div>',
                '<div class="record-status ',
                statusClass,
                '">',
                escapeHTML(statusText),
                '</div>'
            ].join("");

            card.onclick = function () {
                state.current = positionIndex;
                renderList();
                renderCurrent();
            };

            recordsEl.appendChild(card);
        });

        position.textContent = state.filtered.length ? String(state.current + 1) + " / " + String(state.filtered.length) : "0 / 0";
    }

    function renderCurrent() {
        const record = currentRecord();

        updateBadge(record);

        if (!record) {
            buildWelcomeState(state.records.length ? "No records match the current filter." : "Review one record at a time.");
            typeSelect.value = "";
            reason.value = "";
            return;
        }

        typeSelect.innerHTML = '<option value="">No type</option>' + TYPES.map(function (type) {
            return '<option value="' + escapeHTML(type) + '">' + escapeHTML(type) + '</option>';
        }).join("");

        typeSelect.value = record.hallucination_type || "";
        reason.value = record.reason || "";

        const html = Object.keys(record).filter(function (key) {
            return key !== "human_label" && key !== "hallucination_type" && key !== "reason";
        }).map(function (key) {
            const value = typeof record[key] === "object" ? JSON.stringify(record[key], null, 2) : record[key];
            return [
                '<section class="card">',
                '<div class="card-title">',
                escapeHTML(key),
                '</div>',
                '<div class="card-body">',
                escapeHTML(value),
                '</div>',
                '</section>'
            ].join("");
        }).join("");

        details.innerHTML = html || '<section class="card"><div class="card-title">Record</div><div class="card-body">No additional fields to display.</div></section>';
    }

    function applyFilter() {
        state.filtered = [];

        state.records.forEach(function (record, index) {
            let visible = true;

            if (state.filterMode === "hall") {
                visible = record.human_label === 1;
            } else if (state.filterMode === "ok") {
                visible = record.human_label === 0;
            } else if (state.filterMode === "unlabeled") {
                visible = record.human_label !== 0 && record.human_label !== 1;
            }

            if (visible && state.searchTerm) {
                visible = JSON.stringify(record).toLowerCase().includes(state.searchTerm.toLowerCase());
            }

            if (visible) {
                state.filtered.push(index);
            }
        });

        if (state.current >= state.filtered.length) {
            state.current = Math.max(0, state.filtered.length - 1);
        }

        renderList();
        renderCurrent();
    }

    async function saveToServer() {
        if (!state.filename || !state.serverConnected) {
            return false;
        }

        state.saveInFlight = true;
        updateStatus();

        try {
            const response = await fetch(API_BASE + "/api/save/" + encodeURIComponent(state.filename), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ records: state.records })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || "Save failed");
            }

            state.dirty = false;
            state.lastSaveError = "";
            return true;
        } catch (error) {
            console.warn("Save failed", error);
            state.lastSaveError = "save failed: " + error.message;
            return false;
        } finally {
            state.saveInFlight = false;
            updateStatus();
        }
    }

    function scheduleAutoSave() {
        if (!state.filename || !state.serverConnected) {
            return;
        }

        if (state.saveTimer) {
            clearTimeout(state.saveTimer);
        }

        state.saveTimer = window.setTimeout(function () {
            saveToServer().then(function (saved) {
                if (saved) {
                    toast("Autosaved");
                }
            });
        }, 450);
    }

    function markDirty() {
        state.dirty = true;
        state.lastSaveError = "";
        updateStatus();
        scheduleAutoSave();
    }

    function setLabel(value) {
        const record = currentRecord();
        if (!record) {
            return;
        }

        record.human_label = value;
        markDirty();
        renderList();
        renderCurrent();
    }

    function removeLabel() {
        const record = currentRecord();
        if (!record) {
            return;
        }

        delete record.human_label;
        delete record.hallucination_type;
        delete record.reason;
        markDirty();
        renderList();
        renderCurrent();
    }

    function move(step) {
        if (!state.filtered.length) {
            return;
        }

        state.current = Math.min(Math.max(0, state.current + step), state.filtered.length - 1);
        renderList();
        renderCurrent();
    }

    function downloadJSONL() {
        if (!state.records.length) {
            toast("Nothing to download");
            return;
        }

        const blob = new Blob([serialize()], { type: "application/x-ndjson" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");

        anchor.href = url;
        anchor.download = state.filename || "labeled.jsonl";

        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);

        window.setTimeout(function () {
            URL.revokeObjectURL(url);
        }, 1000);

        state.dirty = false;
        state.lastSaveError = "";
        updateStatus("downloaded copy");
        toast("Downloaded JSONL");
    }

    async function saveJSONL() {
        if (!state.records.length) {
            toast("Nothing to save");
            return;
        }

        if (state.saveTimer) {
            clearTimeout(state.saveTimer);
            state.saveTimer = null;
        }

        const saved = await saveToServer();
        if (saved) {
            toast("Saved successfully");
        } else {
            toast("Save failed - check console");
        }
    }

    function loadRecords(records, name) {
        state.records = records;
        state.filtered = [];
        state.current = 0;
        state.filename = name || "";
        state.dirty = false;
        state.lastSaveError = "";

        applyFilter();
        updateStatus();
    }

    async function checkServerHealth() {
        try {
            const response = await fetch(API_BASE + "/api/health", {
                method: "GET",
                cache: "no-cache"
            });
            state.serverConnected = response.ok;
        } catch (error) {
            state.serverConnected = false;
        }
        updateStatus();
        return state.serverConnected;
    }

    async function loadFileList() {
        try {
            const response = await fetch(API_BASE + "/api/files");
            if (!response.ok) {
                throw new Error("Failed to fetch file list");
            }

            const data = await response.json();
            fileSelector.innerHTML = '<option value="">Select a file...</option>' +
                data.files.map(function (file) {
                    return '<option value="' + escapeHTML(file) + '">' + escapeHTML(file) + '</option>';
                }).join("");

            if (state.filename && data.files.includes(state.filename)) {
                fileSelector.value = state.filename;
            }

            toast("Found " + data.files.length + " files");
        } catch (error) {
            console.warn("Failed to load file list", error);
            toast("Failed to load file list");
        }
    }

    async function loadFileFromServer(filename) {
        if (!filename) {
            return;
        }

        try {
            const response = await fetch(API_BASE + "/api/load/" + encodeURIComponent(filename));
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || "Failed to load file");
            }

            const data = await response.json();
            loadRecords(data.records, data.filename);
            toast("Loaded " + data.count + " records from " + filename);
        } catch (error) {
            console.warn("Load failed", error);
            toast("Failed to load file: " + error.message);
        }
    }

    hallBtn.onclick = function () {
        setLabel(1);
    };

    okBtn.onclick = function () {
        setLabel(0);
    };

    removeBtn.onclick = removeLabel;

    copyBtn.onclick = function () {
        const record = currentRecord();
        if (!record) {
            return;
        }

        const text = JSON.stringify(record, null, 2);
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(function () {
                toast("Copied");
            }).catch(function () {
                toast("Clipboard blocked");
            });
            return;
        }

        toast("Clipboard unavailable");
    };

    firstBtn.onclick = function () {
        if (!state.filtered.length) {
            return;
        }

        state.current = 0;
        renderList();
        renderCurrent();
    };

    lastBtn.onclick = function () {
        if (!state.filtered.length) {
            return;
        }

        state.current = state.filtered.length - 1;
        renderList();
        renderCurrent();
    };

    prevBtn.onclick = function () {
        move(-1);
    };

    nextBtn.onclick = function () {
        move(1);
    };

    search.oninput = function () {
        state.searchTerm = search.value;
        applyFilter();
    };

    document.querySelectorAll('input[name="filter"]').forEach(function (input) {
        input.onchange = function () {
            state.filterMode = input.value;
            applyFilter();
        };
    });

    typeSelect.onchange = function () {
        const record = currentRecord();
        if (!record) {
            return;
        }

        if (typeSelect.value) {
            record.hallucination_type = typeSelect.value;
        } else {
            delete record.hallucination_type;
        }

        markDirty();
    };

    reason.oninput = function () {
        const record = currentRecord();
        if (!record) {
            return;
        }

        if (reason.value.trim()) {
            record.reason = reason.value;
        } else {
            delete record.reason;
        }

        markDirty();
    };

    fileSelector.onchange = function () {
        const filename = fileSelector.value;
        if (filename) {
            loadFileFromServer(filename);
        }
    };

    refreshBtn.onclick = function () {
        checkServerHealth().then(function (connected) {
            if (connected) {
                loadFileList();
            } else {
                toast("Server offline - start backend server");
            }
        });
    };

    saveBtn.onclick = function () {
        saveJSONL();
    };

    downloadBtn.onclick = downloadJSONL;

    document.addEventListener("keydown", function (event) {
        const activeTag = document.activeElement && document.activeElement.tagName;
        if (["INPUT", "TEXTAREA", "SELECT"].includes(activeTag)) {
            return;
        }

        if (event.key === "h" || event.key === "H") {
            setLabel(1);
        } else if (event.key === "n" || event.key === "N") {
            setLabel(0);
        } else if (event.key === "u" || event.key === "U") {
            removeLabel();
        } else if (event.key === "ArrowRight") {
            move(1);
        } else if (event.key === "ArrowLeft") {
            move(-1);
        }
    });

    window.addEventListener("beforeunload", function (event) {
        if (state.dirty) {
            event.preventDefault();
            event.returnValue = "";
        }
    });

    buildWelcomeState("Review one record at a time.");
    updateStatus();

    // Initialize: Check server and load file list
    checkServerHealth().then(function (connected) {
        if (connected) {
            loadFileList();
        } else {
            toast("Server offline - run: cd frontend-backend && npm install && npm start");
        }
    });
})();
