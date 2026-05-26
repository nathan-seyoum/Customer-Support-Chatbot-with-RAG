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

function renderLoading() {
  clearResult();
  const node = tpl.loading.content.cloneNode(true);
  resultEl.appendChild(node);
  // Rotate the sub-stage so the user sees the system is doing work.
  const stages = [
    "Retrieving relevant passages",
    "Generating a grounded answer",
    "Checking the answer for hallucinations",
  ];
  let i = 0;
  const sub = resultEl.querySelector("#loading-stage");
  const interval = setInterval(() => {
    i = (i + 1) % stages.length;
    if (sub) sub.textContent = stages[i];
  }, 1400);
  return () => clearInterval(interval);
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
  const stopLoading = renderLoading();

  let aborted = false;
  const controller = new AbortController();
  // Generous timeout — local LLMs can be slow on first warm-up.
  const timeoutId = setTimeout(() => { aborted = true; controller.abort(); }, 120_000);

  try {
    const resp = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    stopLoading();

    if (!resp.ok) {
      const detail = await safeReadDetail(resp);
      renderFailure(`Service returned ${resp.status}: ${detail}`, () => ask(question));
      return;
    }

    const payload = await resp.json();
    renderSuccess(payload);
  } catch (err) {
    clearTimeout(timeoutId);
    stopLoading();
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
