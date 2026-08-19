# RAGawaz

RAGawaz is a multilingual voice retrieval-augmented generation application for Indic and English questions. It combines browser voice capture, multilingual speech-to-text, BAAI/bge-m3 embeddings, in-memory FAISS-HNSW and BM25 retrieval, defensive guardrails, streaming LLM answers, groundedness verification, and optional Indic text-to-speech.

The frontend is a Vite React application. The backend is a FastAPI service that loads the configured index once at startup and exposes text, streaming, voice, health, and benchmark endpoints.

## Key features

- Voice input with browser interim transcription and backend multilingual STT.
- Support for English, Hindi, Hinglish, Marathi, Tamil, Bengali, and additional Indic languages in the indexed corpus.
- Production embedding profile: **BAAI/bge-m3**, 1024 dimensions.
- Hybrid retrieval using FAISS-HNSW dense search plus BM25 sparse search and reciprocal-rank fusion.
- 301,108-vector multilingual corpus bundle when the production index is available.
- Safety, relevance, evidence-sufficiency, and groundedness guardrails.
- Server-sent event (SSE) streaming for low time-to-first-token responses.
- Groq, Cerebras, Sarvam, and mock LLM provider support.
- Sarvam Indic STT/TTS with Groq Whisper fallback.
- Benchmark dashboard backed by saved reports under `benchmarks/`.
- Responsive mobile conversation UI and desktop telemetry dashboard.

## Tech stack

| Area | Technology |
| --- | --- |
| Frontend | React 18, Vite, plain CSS, lucide-react |
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Embeddings | `BAAI/bge-m3` via sentence-transformers |
| Dense retrieval | FAISS `IndexHNSWFlat`, inner-product/cosine search |
| Sparse retrieval | Custom BM25-WAND retriever |
| LLM providers | Groq, Cerebras, Sarvam, mock fallback |
| Voice | Browser SpeechRecognition, MediaRecorder, Sarvam STT/TTS, Groq Whisper |
| Data/index storage | Local FAISS and pickle bundles |
| Testing | Backend tests and benchmark-specific Python runners |

## Prerequisites

- Python 3.11 or newer.
- Node.js 18 or newer and npm.
- Git.
- A microphone and a browser supporting `getUserMedia`; Chrome/Edge generally provide the best SpeechRecognition support.
- Optional API keys:
  - Groq for low-latency LLM and Whisper.
  - Sarvam for Indic STT/TTS and optional LLM.
  - Cerebras for the Cerebras provider.
- Enough RAM for the selected local index. The full BGE-M3/BM25 bundle is substantially larger than the lightweight starter corpus.

## Project structure

```text
.
├── app.py                         # Index preparation and Uvicorn startup
├── Dockerfile                     # Container definition
├── railway.json                   # Service configuration
├── requirements.txt               # Backend Python dependencies
├── .env.example                   # Backend environment template
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, startup, routes, static serving
│   │   ├── config.py              # Embedding, guardrail, and latency configuration
│   │   ├── rag/                   # Embedding, FAISS, BM25, hybrid retrieval, ingestion
│   │   ├── guardrails/            # Safety, relevance, evidence, groundedness checks
│   │   ├── harness/               # RAG orchestration and LLM provider adapters
│   │   └── voice/                 # Language detection and STT/TTS pipeline
│   ├── data/                      # Local index/corpus artifacts when present
│   ├── requirements.txt           # Minimal benchmark requirements
│   └── tests/                     # Backend tests
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Main voice/benchmark views
│   │   ├── hooks/useVoiceRAG.js   # Recording, STT, SSE, and state machine
│   │   ├── components/            # Conversation, voice, telemetry, benchmark UI
│   │   └── index.css              # Responsive design system
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example
├── benchmarks/
│   ├── benchmark_results.json     # Final stratified summary
│   ├── final_benchmark_report.json
│   ├── experiments/               # Retrieval, latency, multilingual reports
│   ├── phase_a/                   # Deterministic pre-LLM benchmark
│   ├── voice/                     # Voice/STT/retrieval validation
│   └── datasets/                  # Dataset generation utilities
├── resources/                     # Task/reference resources
└── .github/workflows/             # Repository automation
```

## Getting started locally

### 1. Clone the repository

```bash
git clone https://github.com/MdTowfikomer/RAGawaz.git
cd RAGawaz
```

If the repository is checked out under a different owner or URL, use that remote instead.

### 2. Create a Python environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure the backend

```powershell
Copy-Item .env.example .env
```

Set at least one LLM key for real answers:

```dotenv
GROQ_API_KEY=gsk_...
SARVAM_API_KEY=...
LLM_PROVIDER=groq
EMBEDDING_MODEL=bge_m3
```

The backend falls back to a mock provider if no LLM key is available. Never commit `.env` or API keys.

### 4. Install frontend dependencies

```powershell
Set-Location frontend
npm install
```

For a Vite development server pointing at a separately running backend, create `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 5. Start the backend

From the repository root:

```powershell
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

On startup the service loads a local FAISS/BM25 index when available. Otherwise it builds a small starter index from available local passages and warms retrieval across several languages.

### 6. Start the frontend

In a second terminal:

```powershell
Set-Location frontend
npm run dev
```

Open the Vite URL printed in the terminal, normally `http://localhost:5173`.

### 7. Production frontend build

```powershell
Set-Location frontend
npm run build
npm run preview
```

The backend can also serve `frontend/dist` when the compiled directory is available in the backend root.

## Environment variables

### Backend

| Variable | Required | Description | Default |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Recommended | Groq LLM and Whisper credentials | Empty |
| `SARVAM_API_KEY` | Optional | Sarvam Indic STT/TTS credentials | Empty |
| `CEREBRAS_API_KEY` | Optional | Cerebras provider credentials | Empty |
| `LLM_PROVIDER` | Optional | `groq`, `cerebras`, `sarvam`, or `mock` | Auto-select Groq, otherwise mock |
| `EMBEDDING_MODEL` | Optional | Active embedding profile; production is `bge_m3` | `bge_m3` |
| `EMBEDDING_DIM` | Optional | Embedding dimension override | `1024` |
| `MAX_PASSAGES` | Optional | Maximum passages when building a fallback index | `50000` |
| `SKIP_PARENT_TEXTS` | Optional | Avoid loading parent text pickle to reduce RAM | `false` |
| `INDEX_DATASET_REPO` | Optional | Dataset containing a prebuilt index bundle | `towfikomer/voice-rag-index` |
| `PORT` | Optional | HTTP port used by the startup script | `8080` |
| `HOST` | Optional | Host value for local setups | `127.0.0.1` |

`bge_m3` resolves to `BAAI/bge-m3` with 1024 dimensions. The `minilm` profile exists as a rollback/benchmark profile, but it is not the production embedding configuration.

### Frontend

| Variable | Description | Example |
| --- | --- | --- |
| `VITE_API_BASE_URL` | URL of the FastAPI backend | `http://127.0.0.1:8000` |

When omitted, the frontend uses same-origin API calls, which is useful when FastAPI serves the built frontend.

## API reference

Base URL: `http://localhost:8000`

### Health

```http
GET /api/health
```

Returns service status, loaded corpus statistics, active embedder, retriever, and index status.

### Text query

```http
POST /api/query
Content-Type: application/json

{
  "query": "What is the capital of India?",
  "provider": "groq",
  "top_k": 5
}
```

Returns the answer, status, refusal reason, groundedness score, retrieved chunks, and telemetry.

### Streaming text query

```http
POST /api/query/stream
Content-Type: application/json

{
  "query": "भारत की राजधानी क्या है?",
  "top_k": 5
}
```

The response is an SSE stream containing status, token, refusal, complete, and error events. The frontend uses this route first and falls back to `/api/query` if streaming fails.

### Speech-to-text

```http
POST /api/voice/stt
Content-Type: multipart/form-data

file=<audio file>
language_code=auto
```

Returns the transcript, detected language metadata, confidence, and STT latency.

### Full voice processing

```http
POST /api/voice/process
Content-Type: multipart/form-data

file=<audio file>
language_code=auto
```

Runs STT, RAG, and TTS and returns the answer, audio base64 payload, language metadata, and telemetry.

### Benchmark results

```http
GET /api/benchmark/results
```

Aggregates the saved final, retrieval, multilingual, and voice validation reports from `benchmarks/`.

## Architecture and request flow

```text
Browser microphone
    │
    ├── Browser SpeechRecognition (interim UI transcript)
    └── MediaRecorder audio
            │
            ▼
      POST /api/voice/stt
            │
            ▼
   Sarvam STT or Groq Whisper
            │
            ▼
       Query normalization
            │
            ▼
      Safety guardrail
            │
            ▼
   BGE-M3 query embedding
            │
            ├── FAISS-HNSW dense retrieval
            └── BM25 sparse retrieval
                    │
                    ▼
             RRF hybrid fusion
                    │
                    ▼
       Relevance/evidence gates
                    │
                    ▼
       Groq/Cerebras/Sarvam LLM
                    │
                    ▼
       Groundedness verification
                    │
                    ├── SSE answer stream
                    └── Optional Sarvam TTS
```

### Guardrail behavior

The orchestrator can short-circuit before an LLM call:

1. Safety guardrail detects unsafe or prompt-injection patterns.
2. Relevance gate rejects clearly out-of-domain queries.
3. Evidence sufficiency checks whether retrieval supports answering.
4. The LLM receives strict grounding instructions.
5. Groundedness verification checks answer support, numeric consistency, and degeneration.

Refusals are explicit statuses rather than silent empty responses.

## Benchmarks

Benchmark scripts and reports are kept in `benchmarks/`. The UI reads the saved reports through `/api/benchmark/results`.

### Final stratified benchmark

The saved `benchmarks/benchmark_results.json` reports:

| Metric | Result |
| --- | ---: |
| Total queries | 135 |
| Embed + retrieval P70 | 17.25 ms |
| Harness P50 / P70 / P95 | 31.44 / 46.43 / 47.28 ms |
| Voice pipeline P50 / P70 | 78.45 / 92.86 ms |
| Refusal accuracy | 100% |
| Groundedness rate | 99% |

### Multilingual retrieval matrix

`benchmarks/experiments/phase6d_matrix_results.json` evaluates 60 queries across Hindi, English, Hinglish, Marathi, Tamil, and Bengali:

- Global Recall@5: 90%.
- Global MRR: 0.8667.
- Adversarial accuracy: 6/8 (75%).
- Retrieval P50: 19.47 ms.
- Retrieval P95: 100.84 ms.

### Run benchmark commands

From the repository root with `.venv` activated:

```powershell
# Deterministic Phase A pre-LLM baseline
python benchmarks/phase_a/run_phase_a.py

# Final stratified evaluation
python benchmarks/final_stratified_benchmark.py

# Multilingual matrix
python benchmarks/experiments/phase6d_multilingual_matrix.py

# Voice/STT benchmark examples
python benchmarks/voice/benchmark_stt_realtime.py
python benchmarks/voice/production_smoke_bge_m3.py
```

The backend contains test modules under `backend/tests/`; run them with your configured Python test runner when working on backend changes. The frontend currently exposes only `dev`, `build`, and `preview` npm scripts.

Phase A is intentionally a pre-LLM benchmark. It asserts that no LLM or TTS call is made and records embedding, vector search, guardrail, and total pre-LLM latency.

## Troubleshooting

### The backend starts with a starter corpus

Check that the expected index files exist under:

```text
backend/data/multilingual_index_bundle/
```

Verify the configured index bundle contains:

```text
faiss.index
metadata_light.pkl
bm25_wand.pkl
manifest.json
```

### The frontend cannot reach the API

Set `frontend/.env.local` and restart Vite after changing environment variables.

### Voice input does not transcribe

1. Use localhost; browsers block microphone access on insecure origins.
2. Grant microphone permission.
3. Use Chrome or Edge for browser SpeechRecognition support.
4. Confirm `GROQ_API_KEY` or `SARVAM_API_KEY` is configured for backend STT.
5. Check `/api/voice/stt` and browser developer-console errors.

### The app returns mock answers

No supported LLM key is configured. Set `GROQ_API_KEY` and `LLM_PROVIDER=groq`, or explicitly use another configured provider.

### Index startup uses too much memory

Set:

```dotenv
SKIP_PARENT_TEXTS=true
```

Use a prebuilt index bundle instead of rebuilding from the raw corpus at every startup.

### Benchmark values are missing

Restart the backend after changing benchmark files and verify:

```http
GET /api/benchmark/results
```
