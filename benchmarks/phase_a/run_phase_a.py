"""
Phase A — Pre-LLM Benchmark Runner.

Runs the deterministic retrieval/guardrail pipeline BEFORE any LLM call:
  query -> normalization -> safety guardrail -> embedding -> FAISS-HNSW retrieval
       -> relevance gate -> insufficient-evidence check -> STOP

No LLM is called. No TTS. No generation.

Outputs:
  - benchmarks/phase_a/results/baseline_results.json
  - benchmarks/phase_a/results/failures.jsonl
  - Terminal report with guardrail, retrieval, and latency metrics
  - Warm/cold diagnostic for representative queries
"""

import json
import os
import sys
import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.rag.ingest import load_passages_from_jsonl
from backend.app.rag.chunker import chunk_corpus
from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker


# ─── Paths ────────────────────────────────────────────────────────────────────
QUERIES_PATH = os.path.join(ROOT_DIR, "benchmarks", "phase_a", "phase_a_queries.jsonl")
RESULTS_DIR = os.path.join(ROOT_DIR, "benchmarks", "phase_a", "results")
BASELINE_PATH = os.path.join(RESULTS_DIR, "baseline_results.json")
FAILURES_PATH = os.path.join(RESULTS_DIR, "failures.jsonl")
CORPUS_PATH = os.path.join(ROOT_DIR, "backend", "data", "passages.jsonl")

# ─── Production Config (frozen for baseline) ─────────────────────────────────
MAX_PASSAGES = int(os.getenv("PHASE_A_MAX_PASSAGES", "50000"))
EMBEDDER_NAME = "minilm"
RETRIEVER_DIM = 384
HNSW_M = 32
HNSW_EF_SEARCH = 64
CHUNK_STRATEGY = "fixed"
RELEVANCE_THRESHOLD = 0.20
INSUFFICIENT_THRESHOLD = 0.35
TOP_K = 10


def log(msg: str):
    print(msg, flush=True)


def load_queries() -> List[Dict[str, Any]]:
    """Load phase_a_queries.jsonl."""
    queries = []
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            ls = line.strip()
            if ls:
                queries.append(json.loads(ls))
    return queries


def build_pipeline():
    """Build the pre-LLM pipeline components: embedder, retriever, guardrails."""
    log("Loading corpus...")
    passages = load_passages_from_jsonl(CORPUS_PATH, max_count=MAX_PASSAGES)
    log(f"  Loaded {len(passages)} passages")

    log("Chunking corpus (strategy={})...".format(CHUNK_STRATEGY))
    chunks = chunk_corpus(passages, strategy=CHUNK_STRATEGY)
    log(f"  Generated {len(chunks)} chunks")

    log("Loading embedder ({})...".format(EMBEDDER_NAME))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = get_embedding_provider(EMBEDDER_NAME, device=device)

    log("Embedding corpus chunks...")
    t0 = time.perf_counter()
    embeddings = embedder.embed([c["text"] for c in chunks], batch_size=128)
    embed_time = time.perf_counter() - t0
    log(f"  Embedded {len(chunks)} chunks in {embed_time:.1f}s")

    log("Building FAISS-HNSW index...")
    retriever = FAISSHNSWRetriever(dimension=RETRIEVER_DIM, m=HNSW_M, ef_search=HNSW_EF_SEARCH)
    retriever.index(chunks, embeddings)
    log("  Index built.")

    safety = SafetyGuardrail()
    relevance_gate = RelevanceGate(threshold=RELEVANCE_THRESHOLD)
    insufficient_checker = InsufficientEvidenceChecker(confidence_threshold=INSUFFICIENT_THRESHOLD)

    return embedder, retriever, safety, relevance_gate, insufficient_checker


def run_single_query(
    query_rec: Dict[str, Any],
    embedder,
    retriever,
    safety: SafetyGuardrail,
    relevance_gate: RelevanceGate,
    insufficient_checker: InsufficientEvidenceChecker,
) -> Dict[str, Any]:
    """Execute pre-LLM pipeline for a single query. Returns full result record."""
    query_text = query_rec["query"]
    t_total_start = time.perf_counter()

    # Step 1: Normalization
    normalized = " ".join(query_text.strip().split())

    # Step 2: Safety Guardrail
    t_guard_start = time.perf_counter()
    is_safe, safety_msg = safety.evaluate(normalized)
    guardrail_ms = (time.perf_counter() - t_guard_start) * 1000.0

    if not is_safe:
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return {
            "id": query_rec["id"],
            "category": query_rec["category"],
            "query": query_text,
            "expected_decision": query_rec["expected_decision"],
            "actual_decision": "refusal_safety",
            "query_embedding_ms": 0.0,
            "vector_search_ms": 0.0,
            "reranking_ms": 0.0,
            "embed_retrieval_ms": 0.0,
            "guardrail_ms": guardrail_ms,
            "total_pre_llm_ms": total_ms,
            "llm_called": False,
            "top1_score": 0.0,
            "top3_score": 0.0,
            "top5_score": 0.0,
            "top10_score": 0.0,
            "retrieved_passage_ids": [],
            "retrieved_is_selected": [],
            "relevant_passage_ids": query_rec.get("relevant_passage_ids", []),
        }

    # Step 3: Query Embedding
    t_embed_start = time.perf_counter()
    q_emb = embedder.embed_query(normalized)
    query_embedding_ms = (time.perf_counter() - t_embed_start) * 1000.0

    # Step 4: FAISS-HNSW Vector Search
    t_search_start = time.perf_counter()
    retrieved_chunks = retriever.search(q_emb, top_k=TOP_K)
    vector_search_ms = (time.perf_counter() - t_search_start) * 1000.0

    embed_retrieval_ms = query_embedding_ms + vector_search_ms

    scores = [c.score for c in retrieved_chunks]
    top1_score = scores[0] if len(scores) >= 1 else 0.0
    top3_score = max(scores[:3]) if len(scores) >= 3 else (max(scores) if scores else 0.0)
    top5_score = max(scores[:5]) if len(scores) >= 5 else (max(scores) if scores else 0.0)
    top10_score = max(scores[:10]) if len(scores) >= 10 else (max(scores) if scores else 0.0)

    retrieved_passage_ids = [c.passage_id for c in retrieved_chunks]
    retrieved_is_selected = [c.is_selected for c in retrieved_chunks]

    # Step 5: Relevance Gate
    t_guard2_start = time.perf_counter()
    is_relevant, rel_msg = relevance_gate.evaluate(scores[:5])
    guard2_ms = (time.perf_counter() - t_guard2_start) * 1000.0
    guardrail_ms += guard2_ms

    if not is_relevant:
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return {
            "id": query_rec["id"],
            "category": query_rec["category"],
            "query": query_text,
            "expected_decision": query_rec["expected_decision"],
            "actual_decision": "refusal_offtopic",
            "query_embedding_ms": query_embedding_ms,
            "vector_search_ms": vector_search_ms,
            "reranking_ms": 0.0,
            "embed_retrieval_ms": embed_retrieval_ms,
            "guardrail_ms": guardrail_ms,
            "total_pre_llm_ms": total_ms,
            "llm_called": False,
            "top1_score": top1_score,
            "top3_score": top3_score,
            "top5_score": top5_score,
            "top10_score": top10_score,
            "retrieved_passage_ids": retrieved_passage_ids,
            "retrieved_is_selected": retrieved_is_selected,
            "relevant_passage_ids": query_rec.get("relevant_passage_ids", []),
        }

    # Step 6: Insufficient Evidence Check
    t_guard3_start = time.perf_counter()
    has_evidence, insuff_msg = insufficient_checker.evaluate(scores[:5])
    guard3_ms = (time.perf_counter() - t_guard3_start) * 1000.0
    guardrail_ms += guard3_ms

    if not has_evidence:
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return {
            "id": query_rec["id"],
            "category": query_rec["category"],
            "query": query_text,
            "expected_decision": query_rec["expected_decision"],
            "actual_decision": "refusal_insufficient_evidence",
            "query_embedding_ms": query_embedding_ms,
            "vector_search_ms": vector_search_ms,
            "reranking_ms": 0.0,
            "embed_retrieval_ms": embed_retrieval_ms,
            "guardrail_ms": guardrail_ms,
            "total_pre_llm_ms": total_ms,
            "llm_called": False,
            "top1_score": top1_score,
            "top3_score": top3_score,
            "top5_score": top5_score,
            "top10_score": top10_score,
            "retrieved_passage_ids": retrieved_passage_ids,
            "retrieved_is_selected": retrieved_is_selected,
            "relevant_passage_ids": query_rec.get("relevant_passage_ids", []),
        }

    # Pipeline reached the end: would proceed to LLM, but we STOP here.
    total_ms = (time.perf_counter() - t_total_start) * 1000.0
    return {
        "id": query_rec["id"],
        "category": query_rec["category"],
        "query": query_text,
        "expected_decision": query_rec["expected_decision"],
        "actual_decision": "proceed_to_llm",
        "query_embedding_ms": query_embedding_ms,
        "vector_search_ms": vector_search_ms,
        "reranking_ms": 0.0,
        "embed_retrieval_ms": embed_retrieval_ms,
        "guardrail_ms": guardrail_ms,
        "total_pre_llm_ms": total_ms,
        "llm_called": False,
        "top1_score": top1_score,
        "top3_score": top3_score,
        "top5_score": top5_score,
        "top10_score": top10_score,
        "retrieved_passage_ids": retrieved_passage_ids,
        "retrieved_is_selected": retrieved_is_selected,
        "relevant_passage_ids": query_rec.get("relevant_passage_ids", []),
    }


def compute_retrieval_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Recall@K, MRR for answerable queries."""
    answerable = [r for r in results if r["category"] == "answerable"]
    if not answerable:
        return {}

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    mrr_sum = 0.0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0

    for r in answerable:
        relevant = set(r.get("relevant_passage_ids", []))
        retrieved = r.get("retrieved_passage_ids", [])

        # Recall@K: fraction of relevant passages found in top-K retrieved
        found_at_1 = len(relevant.intersection(retrieved[:1]))
        found_at_5 = len(relevant.intersection(retrieved[:5]))
        found_at_10 = len(relevant.intersection(retrieved[:10]))

        recall_at_1 += found_at_1 / len(relevant) if relevant else 0
        recall_at_5 += found_at_5 / len(relevant) if relevant else 0
        recall_at_10 += found_at_10 / len(relevant) if relevant else 0

        # Hit rate: did we find ANY relevant passage?
        hit_at_1 += 1 if found_at_1 > 0 else 0
        hit_at_5 += 1 if found_at_5 > 0 else 0
        hit_at_10 += 1 if found_at_10 > 0 else 0

        # MRR: reciprocal rank of first relevant passage
        rr = 0.0
        for rank, pid in enumerate(retrieved, start=1):
            if pid in relevant:
                rr = 1.0 / rank
                break
        mrr_sum += rr

    n = len(answerable)
    return {
        "recall_at_1": recall_at_1 / n,
        "recall_at_5": recall_at_5 / n,
        "recall_at_10": recall_at_10 / n,
        "hit_rate_at_1": hit_at_1 / n,
        "hit_rate_at_5": hit_at_5 / n,
        "hit_rate_at_10": hit_at_10 / n,
        "mrr": mrr_sum / n,
        "total_answerable": n,
    }


def compute_guardrail_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute guardrail accuracy by category."""
    categories = {"unsafe": [], "off_topic": [], "insufficient_evidence": [], "answerable": []}
    for r in results:
        cat = r["category"]
        if cat in categories:
            categories[cat].append(r)

    # Unsafe rejection recall
    unsafe = categories["unsafe"]
    unsafe_correct = sum(1 for r in unsafe if r["actual_decision"] == "refusal_safety")
    unsafe_recall = unsafe_correct / len(unsafe) if unsafe else 0.0

    # Off-topic rejection recall
    offtopic = categories["off_topic"]
    offtopic_correct = sum(1 for r in offtopic if r["actual_decision"] == "refusal_offtopic")
    offtopic_recall = offtopic_correct / len(offtopic) if offtopic else 0.0

    # Insufficient-evidence rejection recall
    insufficient = categories["insufficient_evidence"]
    insufficient_correct = sum(1 for r in insufficient if r["actual_decision"] == "refusal_insufficient_evidence")
    insufficient_recall = insufficient_correct / len(insufficient) if insufficient else 0.0

    # False refusal rate on answerable queries
    answerable = categories["answerable"]
    false_refusals = sum(1 for r in answerable if r["actual_decision"] != "proceed_to_llm")
    false_refusal_rate = false_refusals / len(answerable) if answerable else 0.0

    # Confusion matrix
    confusion = {}
    for r in results:
        key = (r["category"], r["actual_decision"])
        confusion[f"{key[0]}->{key[1]}"] = confusion.get(f"{key[0]}->{key[1]}", 0) + 1

    return {
        "unsafe_rejection_recall": unsafe_recall,
        "unsafe_correct": unsafe_correct,
        "unsafe_total": len(unsafe),
        "offtopic_rejection_recall": offtopic_recall,
        "offtopic_correct": offtopic_correct,
        "offtopic_total": len(offtopic),
        "insufficient_rejection_recall": insufficient_recall,
        "insufficient_correct": insufficient_correct,
        "insufficient_total": len(insufficient),
        "false_refusal_rate": false_refusal_rate,
        "false_refusals": false_refusals,
        "answerable_total": len(answerable),
        "confusion_matrix": confusion,
    }


def compute_latency_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute P50, P70, P95, MAX for all latency dimensions."""
    def percentiles(values):
        if not values:
            return {"p50": 0, "p70": 0, "p95": 0, "max": 0}
        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p70": float(np.percentile(arr, 70)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    # Only compute over queries that reached embedding (skip safety-blocked)
    embed_results = [r for r in results if r["query_embedding_ms"] > 0]

    query_embed_vals = [r["query_embedding_ms"] for r in embed_results]
    vector_search_vals = [r["vector_search_ms"] for r in embed_results]
    embed_retrieval_vals = [r["embed_retrieval_ms"] for r in embed_results]
    guardrail_vals = [r["guardrail_ms"] for r in results]
    total_vals = [r["total_pre_llm_ms"] for r in results]

    return {
        "query_embedding": percentiles(query_embed_vals),
        "vector_search": percentiles(vector_search_vals),
        "embed_retrieval": percentiles(embed_retrieval_vals),
        "guardrails": percentiles(guardrail_vals),
        "total_pre_llm": percentiles(total_vals),
    }


def write_failures(results: List[Dict[str, Any]]):
    """Write diagnostic records for failed answerable retrievals."""
    answerable = [r for r in results if r["category"] == "answerable"]
    failures = []

    for r in answerable:
        relevant = set(r.get("relevant_passage_ids", []))
        retrieved = r.get("retrieved_passage_ids", [])
        retrieved_set = set(retrieved)

        if not relevant.intersection(retrieved_set):
            # Relevant passage not in top-10 at all
            failure_type = "relevant_not_in_top10"
        elif not relevant.intersection(set(retrieved[:5])):
            # In top-10 but not top-5
            failure_type = "relevant_in_top10_not_top5"
        elif not relevant.intersection(set(retrieved[:1])):
            # In top-5 but not top-1
            failure_type = "relevant_in_top5_not_top1"
        else:
            continue  # Not a failure

        # Also flag false refusals
        if r["actual_decision"] != "proceed_to_llm":
            failure_type = f"false_refusal_{r['actual_decision']}"

        failures.append({
            "query": r["query"],
            "query_id": r["id"],
            "failure_type": failure_type,
            "expected_relevant_passage_ids": list(relevant),
            "top_10_retrieved_passage_ids": retrieved[:10],
            "top_10_scores": [r["top1_score"]] + ([r["top3_score"]] if len(retrieved) >= 3 else []) + ([r["top5_score"]] if len(retrieved) >= 5 else []) + ([r["top10_score"]] if len(retrieved) >= 10 else []),
            "retrieved_is_selected": r.get("retrieved_is_selected", [])[:10],
            "chunk_strategy": CHUNK_STRATEGY,
            "language": "hi",
            "actual_decision": r["actual_decision"],
        })

    # Also add false refusals that aren't retrieval failures
    for r in answerable:
        if r["actual_decision"] != "proceed_to_llm":
            # Check if already added
            already = any(f["query_id"] == r["id"] for f in failures)
            if not already:
                failures.append({
                    "query": r["query"],
                    "query_id": r["id"],
                    "failure_type": f"false_refusal_{r['actual_decision']}",
                    "expected_relevant_passage_ids": r.get("relevant_passage_ids", []),
                    "top_10_retrieved_passage_ids": r.get("retrieved_passage_ids", [])[:10],
                    "top_10_scores": [],
                    "retrieved_is_selected": r.get("retrieved_is_selected", [])[:10],
                    "chunk_strategy": CHUNK_STRATEGY,
                    "language": "hi",
                    "actual_decision": r["actual_decision"],
                })

    with open(FAILURES_PATH, "w", encoding="utf-8") as f:
        for rec in failures:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return failures


def run_warm_cold_diagnostic(
    embedder,
    retriever,
    safety: SafetyGuardrail,
    relevance_gate: RelevanceGate,
    insufficient_checker: InsufficientEvidenceChecker,
    queries: List[Dict[str, Any]],
    n_warm: int = 10,
) -> Dict[str, Any]:
    """Run cold + warm diagnostic on 5 representative queries."""
    # Pick 5 answerable queries
    answerable = [q for q in queries if q["category"] == "answerable"][:5]
    diagnostics = []

    for q_rec in answerable:
        # Cold run (first run after index built)
        cold_result = run_single_query(q_rec, embedder, retriever, safety, relevance_gate, insufficient_checker)
        cold_ms = cold_result["total_pre_llm_ms"]

        # Warm runs
        warm_times = []
        for _ in range(n_warm):
            warm_result = run_single_query(q_rec, embedder, retriever, safety, relevance_gate, insufficient_checker)
            warm_times.append(warm_result["total_pre_llm_ms"])

        warm_arr = np.array(warm_times)
        diagnostics.append({
            "query_id": q_rec["id"],
            "query": q_rec["query"][:80],
            "cold_ms": cold_ms,
            "warm_p50": float(np.percentile(warm_arr, 50)),
            "warm_p70": float(np.percentile(warm_arr, 70)),
            "warm_max": float(np.max(warm_arr)),
            "warm_runs": n_warm,
        })

    return {"diagnostics": diagnostics, "n_warm_runs": n_warm}


def print_report(
    results: List[Dict[str, Any]],
    retrieval_metrics: Dict[str, Any],
    guardrail_metrics: Dict[str, Any],
    latency_metrics: Dict[str, Any],
    warm_cold: Dict[str, Any],
    failures: List[Dict[str, Any]],
):
    """Print the full Phase A baseline terminal report."""
    log("")
    log("=" * 60)
    log("PHASE A — PRE-LLM BASELINE")
    log("=" * 60)
    log("")

    # Query counts
    cats = Counter(r["category"] for r in results)
    log("Queries")
    log(f"  Answerable:             {cats.get('answerable', 0)}")
    log(f"  Insufficient evidence:  {cats.get('insufficient_evidence', 0)}")
    log(f"  Off-topic:              {cats.get('off_topic', 0)}")
    log(f"  Unsafe:                 {cats.get('unsafe', 0)}")
    log(f"  Total:                  {len(results)}")
    log("")

    # Guardrails
    gm = guardrail_metrics
    log("GUARDRAILS")
    log(f"  Unsafe rejection:       {gm['unsafe_rejection_recall']*100:.1f}% ({gm['unsafe_correct']}/{gm['unsafe_total']})")
    log(f"  Off-topic rejection:    {gm['offtopic_rejection_recall']*100:.1f}% ({gm['offtopic_correct']}/{gm['offtopic_total']})")
    log(f"  Insufficient rejection: {gm['insufficient_rejection_recall']*100:.1f}% ({gm['insufficient_correct']}/{gm['insufficient_total']})")
    log(f"  False refusal:          {gm['false_refusal_rate']*100:.1f}% ({gm['false_refusals']}/{gm['answerable_total']})")
    log("")

    # Retrieval Quality
    rm = retrieval_metrics
    log("RETRIEVAL QUALITY")
    log(f"  Recall@1:               {rm.get('recall_at_1', 0)*100:.1f}%")
    log(f"  Recall@5:               {rm.get('recall_at_5', 0)*100:.1f}%")
    log(f"  Recall@10:              {rm.get('recall_at_10', 0)*100:.1f}%")
    log(f"  Hit Rate@1:             {rm.get('hit_rate_at_1', 0)*100:.1f}%")
    log(f"  Hit Rate@5:             {rm.get('hit_rate_at_5', 0)*100:.1f}%")
    log(f"  Hit Rate@10:            {rm.get('hit_rate_at_10', 0)*100:.1f}%")
    log(f"  MRR:                    {rm.get('mrr', 0):.4f}")
    log("")

    # Latency
    lm = latency_metrics
    log("LATENCY")
    log(f"  Query embedding P50:    {lm['query_embedding']['p50']:.2f} ms")
    log(f"  Query embedding P70:    {lm['query_embedding']['p70']:.2f} ms")
    log(f"  Vector search P50:      {lm['vector_search']['p50']:.2f} ms")
    log(f"  Vector search P70:      {lm['vector_search']['p70']:.2f} ms")
    log(f"  Embed+retrieval P50:    {lm['embed_retrieval']['p50']:.2f} ms")
    log(f"  Embed+retrieval P70:    {lm['embed_retrieval']['p70']:.2f} ms")
    log(f"  Embed+retrieval MAX:    {lm['embed_retrieval']['max']:.2f} ms")
    log(f"  Pre-LLM total P50:      {lm['total_pre_llm']['p50']:.2f} ms")
    log(f"  Pre-LLM total P70:      {lm['total_pre_llm']['p70']:.2f} ms")
    log(f"  Pre-LLM total MAX:      {lm['total_pre_llm']['max']:.2f} ms")
    log("")

    # LLM check
    llm_calls = sum(1 for r in results if r.get("llm_called", False))
    log("LLM")
    log(f"  Calls:                  {llm_calls} / {len(results)}")
    log("")

    # Warm/Cold
    log("WARM/COLD DIAGNOSTIC")
    for d in warm_cold.get("diagnostics", []):
        log(f"  [{d['query_id']}] cold={d['cold_ms']:.2f}ms | warm P50={d['warm_p50']:.2f}ms P70={d['warm_p70']:.2f}ms MAX={d['warm_max']:.2f}ms")
    log("")

    # Failures
    log(f"RETRIEVAL FAILURES: {len(failures)} diagnostic records written")
    for f in failures[:5]:
        log(f"  [{f['query_id']}] {f['failure_type']}")
    if len(failures) > 5:
        log(f"  ... and {len(failures) - 5} more")
    log("")

    log("=" * 60)
    log("")
    log("TUNING CANDIDATES (do NOT apply yet)")
    log("  1. CPU/thread tuning (OMP_NUM_THREADS, MKL settings)")
    log("  2. FAISS HNSW tuning (M, efSearch, efConstruction)")
    log("  3. Top-K tuning (retrieval depth vs latency)")
    log("  4. Relevance threshold calibration (current: {})".format(RELEVANCE_THRESHOLD))
    log("  5. Insufficient-evidence threshold calibration (current: {})".format(INSUFFICIENT_THRESHOLD))
    log("  6. Embedding model comparison (MiniLM vs LaBSE)")
    log("  7. Chunk strategy comparison (fixed vs semantic vs parent_child)")
    log("")
    log("=" * 60)


def main():
    log("=" * 60)
    log("PHASE A — Pre-LLM Benchmark Runner")
    log("=" * 60)
    log("")

    # Load queries
    queries = load_queries()
    log(f"Loaded {len(queries)} benchmark queries")
    assert len(queries) == 50, f"Expected 50 queries, got {len(queries)}"

    # Build pipeline
    embedder, retriever, safety, relevance_gate, insufficient_checker = build_pipeline()

    # Run benchmark
    log("\nRunning benchmark (50 queries)...")
    results = []
    for i, q in enumerate(queries):
        result = run_single_query(q, embedder, retriever, safety, relevance_gate, insufficient_checker)
        results.append(result)
        if (i + 1) % 10 == 0:
            log(f"  Processed {i + 1}/50")

    # Verify no LLM calls
    llm_calls = sum(1 for r in results if r.get("llm_called", False))
    assert llm_calls == 0, f"LLM was called {llm_calls} times! Phase A must have 0 LLM calls."

    # Compute metrics
    retrieval_metrics = compute_retrieval_metrics(results)
    guardrail_metrics = compute_guardrail_metrics(results)
    latency_metrics = compute_latency_metrics(results)

    # Write failures diagnostic
    failures = write_failures(results)

    # Run warm/cold diagnostic
    log("\nRunning warm/cold diagnostic (5 queries x 10 warm runs)...")
    warm_cold = run_warm_cold_diagnostic(
        embedder, retriever, safety, relevance_gate, insufficient_checker, queries
    )

    # Save baseline results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    baseline_output = {
        "benchmark": "phase_a_pre_llm_baseline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "max_passages": MAX_PASSAGES,
            "embedder": EMBEDDER_NAME,
            "retriever_dim": RETRIEVER_DIM,
            "hnsw_m": HNSW_M,
            "hnsw_ef_search": HNSW_EF_SEARCH,
            "chunk_strategy": CHUNK_STRATEGY,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "insufficient_threshold": INSUFFICIENT_THRESHOLD,
            "top_k": TOP_K,
        },
        "query_count": len(results),
        "llm_calls": llm_calls,
        "retrieval_metrics": retrieval_metrics,
        "guardrail_metrics": guardrail_metrics,
        "latency_metrics": latency_metrics,
        "warm_cold_diagnostic": warm_cold,
        "per_query_results": results,
    }

    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_output, f, ensure_ascii=False, indent=2)
    log(f"\nBaseline saved: {BASELINE_PATH}")
    log(f"Failures saved: {FAILURES_PATH}")

    # Print report
    print_report(results, retrieval_metrics, guardrail_metrics, latency_metrics, warm_cold, failures)


if __name__ == "__main__":
    main()
