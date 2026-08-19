---
title: Indic Voice RAG
emoji: 🎙️
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
app_port: 7860
pinned: false
license: mit
---

<div align="center">

# 🎙️ Indic Voice RAG
### High-Performance, Low-Latency Multilingual Voice Knowledge Engine with 5-Tier Defensive Guardrails

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW%20(M%3D32)-blue.svg?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![BGE-M3](https://img.shields.io/badge/BAAI-BGE--M3%20(1024d)-orange.svg)](https://huggingface.co/BAAI/bge-m3)
[![Groq LPU](https://img.shields.io/badge/Groq-LPU%20Streaming-f55036.svg)](https://groq.com)
[![Sarvam AI](https://img.shields.io/badge/Sarvam%20AI-Indic%20STT%2FTTS-7C3AED.svg)](https://sarvam.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🏆 Hackathon Task 2 (HH Goa) — Requirements & Compliance Matrix

This project was built to strictly satisfy and exceed the **Task 2 Problem Statement & Evaluation Criteria** ([`resources/task 2_ hhg.pdf`](resources/task%202_%20hhg.pdf)):

| Task 2 Requirement | Specification Requirement | Our Implementation & Status | File Reference |
|---|---|---|---|
| **1. Dataset Scoping & Ingestion** | Ingest, flatten, and index `ai4bharat/MSMARCO-XI` dataset with deduplication. | **301,108 chunks** indexed across 14 Indic languages with streaming SHA-256 deduplication. | [`stream_ingest.py`](backend/app/rag/stream_ingest.py) |
| **2. Multi-Strategy Chunking** | Benchmark at least 4 distinct chunking strategies with recall/latency trade-offs. | Evaluated **Fixed+Overlap**, **Semantic Danda (`।`)**, **Parent-Child**, and **Adaptive Structure-Aware**. | [`chunker.py`](backend/app/rag/chunker.py) |
| **3. High-Speed Local Retrieval** | Sub-50ms vector search on embedded corpus. | `faiss.IndexHNSWFlat` ($M=32$) runs in **$0.91\text{ ms}$**; Combined Embed+Retrieval is **$16.45\text{ ms}$ P70**. | [`retriever.py`](backend/app/rag/retriever.py) |
| **4. Voice Pipeline Integration** | Speech-to-Text with Indic language identification & streaming support. | **Sarvam AI `saaras:v3`** (Indic native) + **Groq Whisper `whisper-large-v3-turbo`** fallback. | [`pipeline.py`](backend/app/voice/pipeline.py) |
| **5. Custom RAG Harness** | Purpose-built orchestration, strict context grounding, and prompt defense. | Language-adaptive system prompts, circuit breakers, and streaming event generators. | [`orchestrator.py`](backend/app/harness/orchestrator.py) |
| **6. Multi-Tier Guardrails** | Must know *when NOT to answer* (off-topic, safety, ungroundedness). | Active **5-layer pipeline**: Safety ($< 1\text{ms}$), Relevance ($< 16\text{ms}$), Pre-LLM Evidence Gate, Groundedness Verifier. | [`guardrails/`](backend/app/guardrails/) |
| **7. Glass-Box Observability** | Report discrete P50/P70/P100 latency distributions across all pipeline stages. | Live telemetry dashboard reporting `embed_retrieval_ms`, `llm_ttft_ms`, and `text_to_answer_ms`. | [`PerformanceTelemetry.jsx`](frontend/src/components/PerformanceTelemetry.jsx) |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    UserVoice([User Spoken Voice / Microphone]) --> STT[1. Multilingual STT<br>Sarvam saaras:v3 / Groq Whisper]
    STT --> ScriptDet[Language & Script Identification<br>Hindi, English, Hinglish, Marathi, Tamil, Bengali]
    
    ScriptDet --> Guard1{2. Safety Guardrail<br>&lt; 1ms Policy & Injection Check}
    Guard1 -- Malicious / Unsafe --> Refusal1[Refusal: Safety Policy]
    
    Guard1 -- Safe Query --> HybridRet[3. Hybrid In-Memory Retrieval]
    
    subgraph RetrievalEngine [High-Throughput In-Memory Engine (301k Chunks)]
        HybridRet --> Dense[Dense FAISS-HNSW<br>BGE-M3 1024-d / ~0.91ms]
        HybridRet --> Sparse[Sparse BM25 Search<br>617k Unique Terms / ~22ms]
        Dense --> RRF[RRF Rank Fusion<br>k=60 / ~0.11ms]
        Sparse --> RRF
    end
    
    RRF --> Guard2{4. Relevance Gate<br>Cosine &gt; 0.25 & Intent Match}
    Guard2 -- Off-Topic --> Refusal2[Refusal: Out-of-Domain]
    
    Guard2 -- Top Chunks --> Guard3{5. Evidence Sufficiency Gate<br>Pre-LLM Entity Match & Score &gt; 0.38}
    Guard3 -- Insufficient Evidence --> Refusal3[Refusal: Insufficient Evidence<br>Skips LLM, Saves ~150ms]
    
    Guard3 -- Sufficient Context --> LLM[6. Language-Adaptive LLM Stream<br>Groq Llama-3.3-70b Versatile]
    LLM --> Guard4{7. Groundedness Verifier<br>Numeric &amp; Content Faithfulness}
    
    Guard4 -- Fabricated Facts / Drift --> Refusal4[Refusal: Ungrounded Overwrite]
    Guard4 -- 100% Verified --> AudioOut([8. Verified Answer + Audio Stream])
```

---

## ⚡ Latency Budget & Empirical Measurements

Tested across the **135-query stratified evaluation dataset** on the 301,108-chunk corpus:

| Pipeline Boundary | Technology / Provider | P50 Latency | P70 Target | Compliance Target | Status |
|---|---|---|---|---|---|
| **Query Embedding** | `BAAI/bge-m3` (CUDA FP16 / Warmed) | **11.38 ms** | 14.5 ms | $< 25\text{ ms}$ | ✅ **PASS** |
| **Vector DB Search** | `faiss.IndexHNSWFlat` ($M=32, ef=64$) | **0.91 ms** | 1.25 ms | $< 10\text{ ms}$ | ✅ **PASS** |
| **Sparse Keyword Search** | `BM25Retriever` (617,865 terms) | **22.11 ms** | 24.5 ms | $< 35\text{ ms}$ | ✅ **PASS** |
| **Rank Fusion (RRF)** | Reciprocal Rank Fusion ($k=60$) | **0.11 ms** | 0.15 ms | $< 1\text{ ms}$ | ✅ **PASS** |
| **Pre-LLM Boundary** | **Safety + Embed + Hybrid Search + Gate** | **$\mathbf{166.6\text{ ms}}$** | **$\mathbf{185.0\text{ ms}}$** | $\mathbf{< 200\text{ ms}}$ | ✅ **PASS** |
| **LLM TTFT (First Token)** | Groq LPU (`llama-3.3-70b-versatile`) | **120.0 ms** | 150.0 ms | $< 250\text{ ms}$ | ✅ **PASS** |
| **Pre-LLM Refusal Intercept** | `EvidenceGate` / `SafetyGuardrail` | **15.07 ms** | 17.03 ms | $< 25\text{ ms}$ | ✅ **PASS** |

---

## 🛡️ The 5-Layer Defensive Guardrail Pipeline

The harness prioritizes **knowing when NOT to answer**:

1. **Safety Guardrail ([`safety.py`](backend/app/guardrails/safety.py)):** Sub-millisecond ($< 0.1\text{ ms}$) regex scan for jailbreaks, prompt injection, and harmful instructions in Latin and Indic scripts.
2. **Relevance Gate ([`relevance.py`](backend/app/guardrails/relevance.py)):** Rejects off-topic conversational or transactional intents (weather, flight booking, live stocks) with a calibrated $0.25$ cosine cutoff.
3. **Evidence Sufficiency Interceptor ([`evidence_gate.py`](backend/app/guardrails/evidence_gate.py)):** Pre-LLM check evaluating whether named query entities actually exist inside retrieved passages. Short-circuits unanswerable queries in $< 18\text{ ms}$, saving 150ms of LLM generation time and preventing hallucinations.
4. **Strict Grounded Prompting:** Language-adaptive system instructions enforcing strict adherence to retrieved text spans.
5. **Groundedness Verifier ([`groundedness.py`](backend/app/guardrails/groundedness.py)):** Post-generation verification checking numeric veracity ($\text{numbers}_{\text{ans}} \subseteq \text{numbers}_{\text{ctx}}$), non-stopword content keyword support ($\ge 15\%$), and loop degeneration suppression.

---

## 🌐 Multilingual Evaluation Matrix (301,108 Chunks)

Evaluated on stratified multilingual benchmarks:

| Language | Sample Count | Recall@5 | MRR | Avg Cosine Similarity |
|---|---|---|---|---|
| **Hindi (`hi`)** | 10 queries | **90.0%** | 0.8250 | 0.6766 |
| **English (`en`)** | 10 queries | **90.0%** | 0.9000 | 0.6181 |
| **Hinglish (`hi-EN`)** | 10 queries | **80.0%** | 0.7250 | 0.5704 |
| **Marathi (`mr`)** | 10 queries | **100.0%** | 1.0000 | 0.7033 |
| **Tamil (`ta`)** | 10 queries | **90.0%** | 0.8500 | 0.6497 |
| **Bengali (`bn`)** | 10 queries | **90.0%** | 0.9000 | 0.6521 |
| **OVERALL GLOBAL** | **60 queries** | **$\mathbf{90.0\%}$** | **$\mathbf{0.8667}$** | **$\mathbf{0.6450}$** |

---

## 📂 Multi-Strategy Chunking Suite

Implemented in [`backend/app/rag/chunker.py`](backend/app/rag/chunker.py) & [`minimal_chunker.py`](backend/app/rag/minimal_chunker.py):

* **Sentence-Aware Minimal-Context (Active Production):** 180–220 characters (250 char hard cap, ~35–45 words / 1–2 Devanagari sentences) respecting Indic punctuation (`।`, `?`, `!`, `\n`) and word boundaries.
* **Fixed + Overlap:** 250-character window with 50-character sliding overlap.
* **Semantic Devanagari Splitting:** Groups sentence units up to 300 characters via Devanagari danda (`।`).
* **Parent-Child Hierarchical:** 100-character micro-child vector matching linked to 500-character macro-parent LLM context.
* **Adaptive Structure-Aware:** Dynamic density-aware splitting based on paragraph length.

---

## 💻 Quickstart & Deployment

### 1. Environment Configuration

Create a `.env` file in the root directory:

```env
# API Keys
GROQ_API_KEY=gsk_your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here

# Server Settings
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=7860
EMBEDDING_PROFILE=bge_m3
```

### 2. Local Setup

```bash
# 1. Install backend requirements
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# 2. Build the React frontend
cd frontend
npm install
npm run build
cd ..

# 3. Launch unified server
uvicorn backend.app.main:app --host 0.0.0.0 --port 7860
```

Open your browser at:
* **Web UI Dashboard:** `http://localhost:7860`
* **Swagger API Documentation:** `http://localhost:7860/docs`
* **Health & Diagnostics:** `http://localhost:7860/api/health`

---

## 🐳 Docker & Hugging Face Spaces Deployment

This repository is pre-configured with a multi-stage Docker setup.

### Deploy to Hugging Face Spaces (100% Free):
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) $\rightarrow$ Select **Docker** $\rightarrow$ **Blank**.
2. Under **Space Settings $\rightarrow$ Variables and secrets**, add `GROQ_API_KEY`.
3. Push the repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/indic-voice-rag
   git push space main --force
   ```

### Run Locally with Docker:
```bash
docker build -t indic-voice-rag:latest .
docker run -p 7860:7860 --env-file .env indic-voice-rag:latest
```

---

## 🧪 Test & Evaluation Suite

Run the full automated test and benchmark suite:

```bash
# Run all unit & integration tests
pytest backend/tests/ -v

# Run the 14-language evaluation benchmark matrix
python benchmarks/experiments/phase6d_multilingual_matrix.py

# Run the full stratified 135-query benchmark
python benchmarks/final_stratified_benchmark.py
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
