# Customer-Support Chatbot with RAG

A retrieval-augmented question-answering service for customer-support knowledge bases. Documents are chunked and indexed with embeddings; user questions retrieve relevant chunks, an LLM generates a cited answer from those chunks, and an NLI model checks whether the answer is actually grounded in what was retrieved. Everything is served behind a FastAPI HTTP API.

The whole stack runs **locally by default** (sentence-transformers + Ollama + ChromaDB + a local NLI model) and every component sits behind a small protocol so you can swap in OpenAI, Anthropic, or a different vector store by changing a single environment variable. A clean, Apple-inspired web UI is served from the same FastAPI process, and the whole thing comes up with `docker compose up`.

> See [DESIGN.md](./DESIGN.md) for the why behind each component, pro/con comparisons, and a curated list of documentation to read alongside this project.

---

## Architecture at a glance

```
                  ┌──────────────────────────────────────────────┐
                  │                   FastAPI                    │
                  │  POST /query   POST /ingest   GET /health    │
                  └────────────────────┬─────────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │   RAGPipeline   │
                              └────────┬────────┘
                ┌─────────────┬────────┼────────┬──────────────┐
                ▼             ▼        ▼        ▼              ▼
         ┌───────────┐ ┌────────────┐ ┌─────────────┐ ┌─────────────────┐
         │ Retriever │ │ Generator  │ │  Detector   │ │  Ingestion      │
         │ (MMR k=5) │ │ (grounded  │ │  (NLI       │ │  pipeline       │
         │           │ │  prompt +  │ │   entail-   │ │  (chunk+embed+  │
         │           │ │  cites)    │ │   ment)     │ │   upsert)       │
         └─────┬─────┘ └─────┬──────┘ └──────┬──────┘ └────────┬────────┘
               │             │               │                 │
               ▼             ▼               ▼                 ▼
         ┌───────────┐ ┌────────────┐ ┌─────────────┐ ┌─────────────────┐
         │ Embedder  │ │  LLM       │ │ NLI cross-  │ │ VectorStore     │
         │ (BGE)     │ │  (Ollama)  │ │ encoder     │ │ (Chroma)        │
         └───────────┘ └────────────┘ └─────────────┘ └─────────────────┘
```

Each box in the bottom row is selected by configuration. Replacing one is a 1-line env-var change — no code touches the rest of the system.

---

## Quick start (Docker — recommended)

If you just want to see it running:

```bash
docker compose up --build
```

This brings up three containers:

| Container               | Purpose                                                 |
|-------------------------|---------------------------------------------------------|
| `support-rag-ollama`    | Local LLM server on `11434`                             |
| `support-rag-ollama-pull` | One-shot: pulls `gemma3:1b` into the Ollama volume    |
| `support-rag-app`       | FastAPI + Apple-themed web UI on `8000`                 |

On first boot the app container auto-ingests `data/docs/` so the demo is immediately usable. Open **<http://localhost:8000>** and ask a question.

State is persisted across restarts in three named volumes (`ollama-data`, `chroma-data`, `hf-cache`).

To bake the embedder + NLI weights into the image (instant first request, but a ~1.5 GB image):

```bash
PREWARM=1 docker compose up --build
```

---

## Quick start (local dev)

### 1. Install

```bash
git clone <this repo>
cd Customer-Support-Chatbot-with-RAG
python -m venv .venv
.venv\Scripts\activate          # Windows
# or:  source .venv/bin/activate # macOS/Linux
pip install -e ".[dev]"
```

### 2. Start a local LLM (Ollama)

Install Ollama from <https://ollama.com>, then pull a small model:

```bash
ollama pull gemma3:1b      # small + fast default (~800 MB)
# or, for higher quality:
# ollama pull llama3.2:3b
```

Ollama runs as a background service on `http://localhost:11434` — the app talks to it via its HTTP API.

### 3. Configure

```bash
copy .env.example .env          # Windows
# or:  cp .env.example .env     # macOS/Linux
```

The defaults already point at a fully local stack.

### 4. Ingest the sample knowledge base

```bash
python -m app.ingestion.cli data/docs
```

This walks `data/docs/`, chunks each markdown file, embeds the chunks with BGE, and upserts them into ChromaDB at `./data/chroma/`.

First run downloads ~120 MB of model weights (BGE embedder) — subsequent runs are instant.

### 5. Run the API + web UI

```bash
uvicorn app.main:app --reload --port 8000
```

| URL                                          | What's there                       |
|----------------------------------------------|------------------------------------|
| <http://localhost:8000>                      | Apple-themed web UI                |
| <http://localhost:8000/docs>                 | Auto-generated Swagger UI          |
| <http://localhost:8000/health>               | Liveness + chunk count             |

### 6. Ask a question

Use the web UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"How long do refunds take?\"}"
```

Sample response:

```json
{
  "answer": "Refunds are issued to the original payment method within 5 to 7 business days after we receive and inspect the returned item [#1]. Credit-card refunds may take an additional 1–2 billing cycles to appear on your statement [#1].",
  "citations": [
    {"chunk_id": "returns_policy-0002-...", "source": "returns_policy.md", "score": 0.71, "text": "..."},
    {"chunk_id": "returns_policy-0001-...", "source": "returns_policy.md", "score": 0.68, "text": "..."}
  ],
  "hallucination": {
    "flagged": false,
    "score": 0.87,
    "threshold": 0.5,
    "per_sentence": [
      {"sentence": "Refunds are issued ... 5 to 7 business days.", "entailment_score": 0.94, "best_chunk_id": "returns_policy-0002-..."},
      {"sentence": "Credit-card refunds may take an additional 1–2 billing cycles ...", "entailment_score": 0.81, "best_chunk_id": "returns_policy-0002-..."}
    ]
  },
  "latency_ms": {"retrieve_ms": 28.4, "generate_ms": 612.7, "detect_ms": 184.2, "total_ms": 825.3}
}
```

---

## API

| Method | Path             | Body / Params                                                 | Purpose                                |
|--------|------------------|---------------------------------------------------------------|----------------------------------------|
| GET    | `/health`        | —                                                             | Liveness + store size                  |
| POST   | `/query`         | `{ "question": "...", "top_k": 5 }`                           | Run the full RAG pipeline              |
| POST   | `/ingest`        | `{ "documents": [{"id","source","text","metadata"}] }`        | Ingest documents from the request body |
| POST   | `/ingest/path?path=data/docs` | path query param                                 | Ingest from a server-side path (dev)   |

---

## Configuration

All knobs live in `.env` / environment variables. See `.env.example` for the full list. The most useful ones:

| Variable                  | Default                              | What it controls                          |
|---------------------------|--------------------------------------|-------------------------------------------|
| `EMBEDDER_PROVIDER`       | `local`                              | `local`, `openai`, `voyage`               |
| `EMBEDDER_MODEL`          | `BAAI/bge-small-en-v1.5`             | HF model id (local) or provider model id  |
| `LLM_PROVIDER`            | `ollama`                             | `ollama`, `openai`, `anthropic`           |
| `LLM_MODEL`               | `gemma3:1b`                          | Ollama model tag / API model id           |
| `NLI_MODEL`               | `cross-encoder/nli-deberta-v3-base`  | Any HF cross-encoder NLI model            |
| `VECTOR_STORE`            | `chroma`                             | (more backends pluggable)                 |
| `CHUNK_SIZE`              | `600`                                | Characters per chunk                      |
| `CHUNK_OVERLAP`           | `100`                                | Characters of overlap between chunks      |
| `TOP_K`                   | `5`                                  | Chunks returned to the LLM                |
| `MMR_LAMBDA`              | `0.5`                                | 1.0 = pure relevance, 0.0 = pure diversity|
| `HALLUCINATION_THRESHOLD` | `0.5`                                | Below this, answer is flagged             |

### Swap to OpenAI in one block

```env
EMBEDDER_PROVIDER=openai
EMBEDDER_MODEL=text-embedding-3-small
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

`pip install -e ".[openai]"` and you're done. No code changes.

---

## Tests

```bash
pytest -q
```

Tests use in-memory fakes for the embedder, LLM, and NLI model so the whole suite runs in under a second without any model downloads.

---

## Web UI

A single-page, vanilla-JS frontend is served at `/` directly from the FastAPI process. The aesthetic is deliberately Apple-inspired: SF Pro system font stack, near-white surfaces (#fbfbfd, #f5f5f7), one restrained accent (#0066cc), generous whitespace, and large rounded radii.

Three explicit UI states are implemented as `<template>` elements in [`web/index.html`](web/index.html) and cloned by [`web/assets/app.js`](web/assets/app.js):

| State    | What the user sees                                                                                  |
|----------|-----------------------------------------------------------------------------------------------------|
| Loading  | A subtle three-dot spinner + rotating sub-stage text (*Retrieving → Generating → Detecting*).       |
| Failure  | A friendly error card with the underlying message and a **Try again** button bound to a retry.      |
| Success  | The answer (with `[#1]` superscript citations), a colored groundedness badge, per-source chunks, per-stage latency, and a collapsible per-sentence groundedness breakdown. |

The groundedness badge color tracks the hallucination detector:

- **Grounded** (green) — score ≥ 0.7
- **Partially grounded** (amber) — 0 ≤ score < 0.7
- **Hallucination flagged** (red) — `flagged: true`

**Why no Streamlit?** Streamlit's component model fights the Apple aesthetic — you spend more time overriding its CSS than designing. A 250-line static page is simpler to ship (one process, one image, one port), gives full design control, and reuses the same API the CLI and Swagger UI hit. The whole UI is three files in `web/`.

---

## Docker

A multi-stage `Dockerfile` builds a small (~400 MB without pre-warmed models, ~1.5 GB with) runtime image:

- **Stage 1 (`builder`)** — installs the project into a venv with build tooling.
- **Stage 2 (`runtime`)** — `python:3.11-slim` + the venv. No compilers. Non-root `app` user. A Docker `HEALTHCHECK` hits `/health`.

`docker-compose.yml` wires three services:

- `ollama` — runs the local LLM server with a persistent volume for downloaded models.
- `ollama-pull` — one-shot sidecar that pulls `gemma3:1b` (configurable via `LLM_MODEL`) and exits. Depends on `ollama` being healthy.
- `app` — the FastAPI + web-UI container. Depends on `ollama` healthy *and* `ollama-pull` having completed successfully. On first boot it auto-ingests `data/docs/` if the Chroma volume is empty.

Override the model with an env var:

```bash
LLM_MODEL=qwen2.5:3b docker compose up --build
```

---

## CI/CD

GitHub Actions workflow at [.github/workflows/ci.yml](.github/workflows/ci.yml):

| Job      | When                       | What it does                                                |
|----------|----------------------------|-------------------------------------------------------------|
| `test`   | every push & PR            | `ruff check` + `pytest -q` on Python 3.11 *and* 3.12        |
| `docker` | push to `main`             | Builds the image and pushes to GHCR with a `sha-` + `latest` tag using `GITHUB_TOKEN` (no extra secrets to configure) |

After the first `main` push the image is available at:

```
ghcr.io/<your-github-username-or-org>/<repo-name>:latest
```

To pull and run on any machine with Docker + Ollama:

```bash
docker pull ghcr.io/<owner>/<repo>:latest
docker run -p 8000:8000 \
  -e LLM_BASE_URL=http://host.docker.internal:11434 \
  ghcr.io/<owner>/<repo>:latest
```

The workflow uses **GitHub Actions cache** for both `pip` (via `actions/setup-python`) and the Docker layer cache (`type=gha`) so subsequent CI runs are fast.

---

## Project layout

```
app/
  main.py               # FastAPI app + lifespan
  config.py             # Pydantic settings, all env-driven
  schemas.py            # API request/response models
  api/routes.py         # /query, /ingest, /health
  providers/            # Embedder + LLM protocols and impls
    base.py             #   protocols
    local.py            #   sentence-transformers + Ollama
    cloud.py            #   OpenAI + Anthropic + Voyage (optional)
    factory.py          #   build_embedder / build_llm
  stores/               # Vector-store protocol + impls
    base.py
    chroma_store.py
    factory.py
  rag/                  # Core RAG components
    chunker.py          #   recursive character splitter
    retriever.py        #   embedding lookup + MMR re-ranking
    generator.py        #   grounded prompt + inline citations
    hallucination.py    #   NLI-based groundedness checker
    pipeline.py         #   orchestrator
  ingestion/
    loaders.py          # md / txt / pdf loaders
    pipeline.py         # load → chunk → embed → upsert
    cli.py              # `python -m app.ingestion.cli <path>`
data/
  docs/                 # sample customer-support knowledge base
  chroma/               # ChromaDB persistence (gitignored)
tests/                  # fast tests with fakes
```


