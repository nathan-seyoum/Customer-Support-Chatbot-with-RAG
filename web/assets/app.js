// ---------------------------------------------------------------------------
// Frontend controller.
// Three states are rendered into #result by cloning the <template>s in
// index.html: loading, failure, success. Keeping markup in templates and
// behavior in JS makes each state self-contained and easy to reason about.
// ---------------------------------------------------------------------------

const form         = document.getElementById("ask-form");
const input        = document.getElementById("q");
const askBtn       = document.getElementById("ask-btn");
const resultEl     = document.getElementById("result");
const suggestionEls = document.querySelectorAll("#suggestions button");

const tpl = {
  loading: document.getElementById("tpl-loading"),
  failure: document.getElementById("tpl-failure"),
  success: document.getElementById("tpl-success"),
};

// ────────────────────────── State renderers ──────────────────────────
function clearResult() {
  while (resultEl.firstChild) resultEl.removeChild(resultEl.firstChild);
}

// Map server-emitted stage names to user-facing labels. The pipeline
// (app/rag/pipeline.py) fires a stage event right *before* each step starts,
// so whatever shows here is what the server is actually doing right now.
const STAGE_LABELS = {
  retrieve: "Retrieving relevant passages",
  generate: "Generating a grounded answer",
  detect:   "Checking the answer for hallucinations",
};

function renderLoading() {
  clearResult();
  const node = tpl.loading.content.cloneNode(true);
  resultEl.appendChild(node);
  const sub = resultEl.querySelector("#loading-stage");
  return {
    setStage(stage, data) {
      if (!sub) return;
      let label = STAGE_LABELS[stage] || stage;
      if (stage === "generate" && data && typeof data.n_chunks === "number") {
        label = `Retrieved ${data.n_chunks} passage${data.n_chunks === 1 ? "" : "s"} — generating a grounded answer`;
      }
      sub.textContent = label;
    },
  };
}

function renderFailure(message, onRetry) {
  clearResult();
  const node = tpl.failure.content.cloneNode(true);
  node.querySelector(".failure-message").textContent =
    message || "We couldn't reach the support service. Please try again.";
  node.querySelector(".retry-btn").addEventListener("click", onRetry);
  resultEl.appendChild(node);
}

function renderSuccess(payload) {
  clearResult();
  const node = tpl.success.content.cloneNode(true);

  // --- Answer card
  const answerText = node.querySelector(".answer-text");
  answerText.innerHTML = formatAnswer(payload.answer);

  // --- Groundedness badge
  const badge = node.querySelector(".grounded-badge");
  const h = payload.hallucination || {};
  const score = typeof h.score === "number" ? h.score : 0;
  const flagged = !!h.flagged;
  if (flagged) {
    badge.classList.add("is-flagged");
    badge.textContent = `Hallucination flagged · ${score.toFixed(2)}`;
  } else if (score >= 0.7) {
    badge.classList.add("is-grounded");
    badge.textContent = `Grounded · ${score.toFixed(2)}`;
  } else {
    badge.classList.add("is-partial");
    badge.textContent = `Partially grounded · ${score.toFixed(2)}`;
  }

  // --- Latency footer
  const latency = node.querySelector(".latency");
  const lat = payload.latency_ms || {};
  latency.innerHTML = `
    <span>retrieve <strong>${fmtMs(lat.retrieve_ms)}</strong></span>
    <span>generate <strong>${fmtMs(lat.generate_ms)}</strong></span>
    <span>detect <strong>${fmtMs(lat.detect_ms)}</strong></span>
    <span>total <strong>${fmtMs(lat.total_ms)}</strong></span>
  `;

  // --- Citations
  const citeList = node.querySelector(".citation-list");
  (payload.citations || []).forEach(c => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="cite-body">
        <span class="cite-source">${escapeHtml(c.source)}</span>
        ${escapeHtml(truncate(c.text, 220))}
      </div>
      <span class="cite-score">${(c.score ?? 0).toFixed(2)}</span>
    `;
    citeList.appendChild(li);
  });

  // --- Per-sentence groundedness breakdown
  const halBody = node.querySelector(".hallucination-body");
  const sentences = h.per_sentence || [];
  if (sentences.length === 0) {
    halBody.innerHTML = `<p style="color: var(--ink-soft); margin: 0; font-size: 14px;">
      No factual sentences to evaluate.
    </p>`;
  } else {
    sentences.forEach(s => {
      const row = document.createElement("div");
      row.className = "sentence-row";
      const cls = s.entailment_score >= 0.7 ? "score-good"
                : s.entailment_score >= 0.4 ? "score-mid"
                : "score-bad";
      row.innerHTML = `
        <div class="sentence-text">${escapeHtml(s.sentence)}</div>
        <div class="sentence-score ${cls}">${(s.entailment_score ?? 0).toFixed(2)}</div>
      `;
      halBody.appendChild(row);
    });
  }

  resultEl.appendChild(node);
}

// ────────────────────────── Helpers ──────────────────────────
function fmtMs(v) {
  if (typeof v !== "number") return "—";
  return v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${Math.round(v)}ms`;
}

function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n).trimEnd() + "…" : s;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** Render `[#1]` style citations as superscripts. */
function formatAnswer(answer) {
  const escaped = escapeHtml(answer);
  return escaped.replace(/\[#(\d+)\]/g, (_, n) => `<sup>[${n}]</sup>`);
}

// ────────────────────────── Submit handler ──────────────────────────
async function ask(question) {
  if (!question || !question.trim()) return;
  askBtn.disabled = true;
  input.disabled = true;
  const loading = renderLoading();

  let aborted = false;
  const controller = new AbortController();
  // Generous timeout — local LLMs can be slow on first warm-up. The server
  // pre-warms the NLI model at startup, but if that was skipped we may still
  // hit a long first request, so we allow plenty of room.
  const timeoutId = setTimeout(() => { aborted = true; controller.abort(); }, 300_000);

  try {
    const resp = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });

    if (!resp.ok) {
      clearTimeout(timeoutId);
      const detail = await safeReadDetail(resp);
      renderFailure(`Service returned ${resp.status}: ${detail}`, () => ask(question));
      return;
    }

    let finalPayload = null;
    let streamError = null;
    await readSSE(resp.body, (event, data) => {
      if (event === "stage") {
        loading.setStage(data.stage, data);
      } else if (event === "result") {
        finalPayload = data;
      } else if (event === "error") {
        streamError = data && data.detail ? data.detail : "Server error during streaming.";
      }
    });
    clearTimeout(timeoutId);

    if (streamError) {
      renderFailure(streamError, () => ask(question));
    } else if (finalPayload) {
      renderSuccess(finalPayload);
    } else {
      renderFailure("Stream ended without a result.", () => ask(question));
    }
  } catch (err) {
    clearTimeout(timeoutId);
    const msg = aborted
      ? "The request timed out. The model may still be warming up — try again."
      : err && err.message
        ? err.message
        : "Network error.";
    renderFailure(msg, () => ask(question));
  } finally {
    askBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

// Minimal SSE parser. Reads a fetch response body and invokes `onEvent` for
// each complete `event:`/`data:` pair. Sticking with fetch (rather than
// EventSource) because the server expects POST + JSON body.
async function readSSE(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message";
      const dataLines = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length === 0) continue;
      let data;
      try { data = JSON.parse(dataLines.join("\n")); }
      catch { data = dataLines.join("\n"); }
      onEvent(event, data);
    }
  }
}

async function safeReadDetail(resp) {
  try {
    const body = await resp.json();
    return body.detail || resp.statusText;
  } catch {
    return resp.statusText || "unknown error";
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  ask(input.value);
});

suggestionEls.forEach(btn => {
  btn.addEventListener("click", () => {
    input.value = btn.dataset.q;
    ask(input.value);
  });
});
