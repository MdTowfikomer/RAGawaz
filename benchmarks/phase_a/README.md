# Phase A — Pre-LLM Benchmark Harness

## Purpose

Measures the **deterministic retrieval and guardrail pipeline** performance BEFORE any LLM call.

This establishes a reproducible baseline for tuning the pre-generation layers independently of LLM latency and cost.

## Pipeline Under Test

```
query
  ↓ normalization
  ↓ safety guardrail (regex/keyword blocklist)
  ↓ query embedding (MiniLM-384d)
  ↓ FAISS-HNSW vector search (top-10)
  ↓ relevance gate (cosine threshold)
  ↓ insufficient-evidence check (confidence threshold)
  ↓ STOP — no LLM generation
```

## Query Dataset

**File:** `phase_a_queries.jsonl` (50 queries, deterministic, seed=42)

| Category | Count | Source |
|----------|-------|--------|
| Answerable | 20 | Derived from actual corpus (is_selected=1) |
| Insufficient Evidence | 10 | In-domain queries absent from corpus |
| Off-Topic | 10 | Clearly outside knowledge scope |
| Unsafe | 10 | Safety blocklist trigger patterns |

### Regenerating the dataset

```bash
.venv\Scripts\python benchmarks/phase_a/generate_queries.py
```

## Running the Benchmark

```bash
.venv\Scripts\python benchmarks/phase_a/run_phase_a.py
```

### Output

- `results/baseline_results.json` — Full metrics + per-query telemetry
- `results/failures.jsonl` — Diagnostic records for failed retrievals

## Metrics Collected

### Per-Query
- `query_embedding_ms`, `vector_search_ms`, `reranking_ms`, `embed_retrieval_ms`
- `guardrail_ms`, `total_pre_llm_ms`
- `llm_called` (must always be `false`)
- `top1_score`, `top3_score`, `top5_score`, `top10_score`
- `retrieved_passage_ids`, `retrieved_is_selected`

### Retrieval Quality (answerable queries)
- Recall@1, Recall@5, Recall@10
- Hit Rate@1, @5, @10
- MRR

### Guardrail Accuracy
- Unsafe rejection recall
- Off-topic rejection recall
- Insufficient-evidence rejection recall
- False refusal rate on answerable queries
- Confusion matrix by category

### Latency (P50, P70, P95, MAX)
- Query embedding
- Vector search
- Embed + retrieval
- Guardrails
- Total pre-LLM pipeline

### Warm/Cold Diagnostic
- Cold latency (first run)
- Warm P50, P70, MAX (10 repeated runs)

## Configuration (frozen for baseline)

| Parameter | Value |
|-----------|-------|
| Embedder | paraphrase-multilingual-MiniLM-L12-v2 (384d) |
| Retriever | FAISS-HNSW (M=32, efSearch=64) |
| Chunk Strategy | fixed (250 chars, 50 overlap) |
| Relevance Threshold | 0.20 |
| Insufficient Evidence Threshold | 0.35 |
| Top-K | 10 |
| Corpus Size | 50,000 passages |

## Tests

```bash
.venv\Scripts\python -m pytest backend/tests/test_phase_a.py -v
```

## Important

- **No LLM calls** — asserted for all 50 queries
- **No TTS calls** — voice pipeline not invoked
- **Baseline only** — do not change any parameters until baseline is saved
- **Deterministic** — fixed seed, reproducible results
