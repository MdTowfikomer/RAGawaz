"""
Phase 7: Latency Breakdown Profiler.

Profiles component-by-component latency across >=100 warm queries and explicitly
separates cold-start (query 0/1) from warm P50/P95/P99 latencies.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever

BUNDLE_DIR = ROOT_DIR / "backend" / "data" / "multilingual_index_bundle"
QUERIES_FILE = ROOT_DIR / "benchmarks" / "experiments" / "multilingual_shootout_queries.jsonl"
REPORT_FILE = ROOT_DIR / "benchmarks" / "experiments" / "phase7_latency_profile.json"


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    return float(np.percentile(data, p))


def run_latency_profile(num_queries_target: int = 120):
    print("=" * 80)
    print("PHASE 7: BASELINE LATENCY PROFILING (301,108 VECTORS)")
    print("=" * 80)

    # 1. Load evaluation queries
    queries: List[Dict[str, Any]] = []
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # Expand query list to target count if necessary by repeating with slight variations
    all_queries = []
    while len(all_queries) < num_queries_target:
        all_queries.extend(queries)
    all_queries = all_queries[:num_queries_target]

    print(f"Loaded {len(all_queries)} queries for profiling across {len(set(q['language'] for q in all_queries))} languages.")

    # 2. Cold-Start Measurement: Component Initialization
    t0_init = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Cold Start] Initializing BAAI/bge-m3 embedder on {device} (FP16)...")
    t_embed_init0 = time.perf_counter()
    embedder = get_embedding_provider("bge_m3", device=device)
    t_embed_init_ms = (time.perf_counter() - t_embed_init0) * 1000.0

    print(f"[Cold Start] Memory-mapping FAISS-HNSW index ({BUNDLE_DIR})...")
    t_faiss_init0 = time.perf_counter()
    retriever = FAISSHNSWRetriever(dimension=1024, m=32, ef_search=64)
    retriever.load(str(BUNDLE_DIR), use_mmap=True)
    t_faiss_init_ms = (time.perf_counter() - t_faiss_init0) * 1000.0

    total_init_ms = (time.perf_counter() - t0_init) * 1000.0
    print(f"  -> Embedder init: {t_embed_init_ms:.2f} ms")
    print(f"  -> FAISS mmap init: {t_faiss_init_ms:.2f} ms")
    print(f"  -> Total startup: {total_init_ms:.2f} ms (Loaded {len(retriever.chunks_metadata)} chunks)")

    # 3. Cold Query (First Query Execution)
    first_query = all_queries[0]["query"]
    print(f"\n[Cold Query #1] Executing '{first_query[:40]}...'")
    
    t_cold_emb0 = time.perf_counter()
    cold_emb = embedder.embed_query(first_query)
    t_cold_emb_ms = (time.perf_counter() - t_cold_emb0) * 1000.0

    t_cold_search0 = time.perf_counter()
    cold_results = retriever.search(cold_emb, top_k=50)
    t_cold_search_ms = (time.perf_counter() - t_cold_search0) * 1000.0
    t_cold_total_ms = t_cold_emb_ms + t_cold_search_ms

    print(f"  -> Cold Embed latency: {t_cold_emb_ms:.2f} ms")
    print(f"  -> Cold FAISS search latency: {t_cold_search_ms:.2f} ms")
    print(f"  -> Cold Total retrieval: {t_cold_total_ms:.2f} ms")

    # 4. Warm Profiling Run (Remaining Queries)
    print(f"\n[Warm Profiling] Running {len(all_queries)-1} warm queries across languages...")
    warm_embed_latencies = []
    warm_search_latencies = []
    warm_total_latencies = []
    lang_breakdown: Dict[str, List[float]] = {}

    for i, q_item in enumerate(all_queries[1:], start=2):
        q_text = q_item["query"]
        lang = q_item["language"]
        if lang not in lang_breakdown:
            lang_breakdown[lang] = []

        # Embed stage
        t0 = time.perf_counter()
        q_emb = embedder.embed_query(q_text)
        t_emb = (time.perf_counter() - t0) * 1000.0

        # FAISS search stage
        t0 = time.perf_counter()
        results = retriever.search(q_emb, top_k=50)
        t_search = (time.perf_counter() - t0) * 1000.0

        t_total = t_emb + t_search

        warm_embed_latencies.append(t_emb)
        warm_search_latencies.append(t_search)
        warm_total_latencies.append(t_total)
        lang_breakdown[lang].append(t_total)

    # 5. Compile Statistics
    profile_results = {
        "dataset_vector_count": len(retriever.chunks_metadata),
        "total_queries_profiled": len(all_queries),
        "cold_start": {
            "embedder_init_ms": round(t_embed_init_ms, 2),
            "faiss_mmap_init_ms": round(t_faiss_init_ms, 2),
            "total_startup_ms": round(total_init_ms, 2),
            "first_query_embed_ms": round(t_cold_emb_ms, 2),
            "first_query_search_ms": round(t_cold_search_ms, 2),
            "first_query_total_ms": round(t_cold_total_ms, 2),
        },
        "warm_latency_metrics": {
            "query_embedding": {
                "p50_ms": round(percentile(warm_embed_latencies, 50), 2),
                "p95_ms": round(percentile(warm_embed_latencies, 95), 2),
                "p99_ms": round(percentile(warm_embed_latencies, 99), 2),
                "mean_ms": round(float(np.mean(warm_embed_latencies)), 2),
                "min_ms": round(float(np.min(warm_embed_latencies)), 2),
                "max_ms": round(float(np.max(warm_embed_latencies)), 2),
            },
            "faiss_search_top50": {
                "p50_ms": round(percentile(warm_search_latencies, 50), 2),
                "p95_ms": round(percentile(warm_search_latencies, 95), 2),
                "p99_ms": round(percentile(warm_search_latencies, 99), 2),
                "mean_ms": round(float(np.mean(warm_search_latencies)), 2),
                "min_ms": round(float(np.min(warm_search_latencies)), 2),
                "max_ms": round(float(np.max(warm_search_latencies)), 2),
            },
            "total_dense_retrieval": {
                "p50_ms": round(percentile(warm_total_latencies, 50), 2),
                "p95_ms": round(percentile(warm_total_latencies, 95), 2),
                "p99_ms": round(percentile(warm_total_latencies, 99), 2),
                "mean_ms": round(float(np.mean(warm_total_latencies)), 2),
                "min_ms": round(float(np.min(warm_total_latencies)), 2),
                "max_ms": round(float(np.max(warm_total_latencies)), 2),
            },
        },
        "language_p50_p95": {
            lang: {
                "count": len(lats),
                "p50_ms": round(percentile(lats, 50), 2),
                "p95_ms": round(percentile(lats, 95), 2),
                "p99_ms": round(percentile(lats, 99), 2),
            }
            for lang, lats in lang_breakdown.items()
        },
    }

    # Print summary table
    print("\n" + "=" * 80)
    print("                    PHASE 7 LATENCY PROFILE REPORT")
    print("=" * 80)
    print(f"Cold Startup Total : {total_init_ms:.2f} ms")
    print(f"Cold First Query   : {t_cold_total_ms:.2f} ms (Embed: {t_cold_emb_ms:.2f} ms, FAISS: {t_cold_search_ms:.2f} ms)")
    print("-" * 80)
    print("Stage               | P50 (ms)   | P95 (ms)   | P99 (ms)   | Mean (ms)  | Max (ms)")
    print("-" * 80)
    e = profile_results["warm_latency_metrics"]["query_embedding"]
    s = profile_results["warm_latency_metrics"]["faiss_search_top50"]
    t = profile_results["warm_latency_metrics"]["total_dense_retrieval"]
    print(f"Query Embedding     | {e['p50_ms']:>8.2f}   | {e['p95_ms']:>8.2f}   | {e['p99_ms']:>8.2f}   | {e['mean_ms']:>8.2f}   | {e['max_ms']:>8.2f}")
    print(f"FAISS Search (k=50) | {s['p50_ms']:>8.2f}   | {s['p95_ms']:>8.2f}   | {s['p99_ms']:>8.2f}   | {s['mean_ms']:>8.2f}   | {s['max_ms']:>8.2f}")
    print(f"Total Retrieval     | {t['p50_ms']:>8.2f}   | {t['p95_ms']:>8.2f}   | {t['p99_ms']:>8.2f}   | {t['mean_ms']:>8.2f}   | {t['max_ms']:>8.2f}")
    print("-" * 80)
    print("Language Breakdown (Total Retrieval P50 / P95):")
    for lang, stats in profile_results["language_p50_p95"].items():
        print(f"  {lang:<10} (n={stats['count']:>2}): P50 = {stats['p50_ms']:>5.2f} ms | P95 = {stats['p95_ms']:>5.2f} ms | P99 = {stats['p99_ms']:>5.2f} ms")
    print("=" * 80)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(profile_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved profile report to: {REPORT_FILE}")
    return profile_results


if __name__ == "__main__":
    run_latency_profile(120)
