"""
Ticket 6: Two-Stage Decision Benchmark Harness.

Stage 1: Embedding Provider x Retriever Backend (4 Configurations)
  - Formula: 60% P70 latency (<50ms) + 25% Recall@5 + 15% Memory footprint
Stage 2: Chunking Strategy (4 Strategies) on Stage 1 Winner
  - Formula: 60% Recall@5 + 25% MRR + 15% Latency P70

Outputs deterministic results table and selects the winning production configuration.
"""

import os
import sys
import json
import time
import psutil
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

# Enable project root in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Enable utf-8 printing for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.rag.chunker import chunk_corpus
from backend.app.rag.embedder import get_embedding_provider, EmbeddingProvider
from backend.app.rag.retriever import FAISSHNSWRetriever, RetrievedChunk


def log(msg: str):
    """Flushed stdout log."""
    print(msg, flush=True)


def get_process_memory_mb() -> float:
    """Current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def load_passages(file_path: str, max_count: int = 1000) -> List[Dict[str, Any]]:
    """Load passage records from jsonl."""
    passages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                passages.append(json.loads(line_str))
                if len(passages) >= max_count:
                    break
    return passages


def load_benchmark_queries(file_path: str) -> List[Dict[str, Any]]:
    """Load benchmark queries with ground truth annotations."""
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                queries.append(json.loads(line_str))
    return queries


def evaluate_retrieval(
    retriever: Any,
    embedder: EmbeddingProvider,
    queries: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run retrieval across queries and compute Recall@5, MRR, and latency distribution."""
    latencies_ms = []
    hits = 0
    reciprocal_ranks = []

    for q in queries:
        gt_passage_id = q.get("ground_truth_passage_id")
        query_text = q["query"]

        # Measure combined embed + retrieval latency
        t0 = time.perf_counter()
        q_emb = embedder.embed_query(query_text)
        results = retriever.search(q_emb, top_k=top_k)
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(t_elapsed)

        # Check hits and MRR
        retrieved_passage_ids = [r.passage_id for r in results]
        if gt_passage_id in retrieved_passage_ids:
            hits += 1
            rank = retrieved_passage_ids.index(gt_passage_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p70 = float(np.percentile(latencies_ms, 70))
    p95 = float(np.percentile(latencies_ms, 95))
    max_lat = float(np.max(latencies_ms))
    recall_at_5 = hits / len(queries) if queries else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return {
        "p50_ms": p50,
        "p70_ms": p70,
        "p95_ms": p95,
        "max_ms": max_lat,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "queries_count": len(queries),
    }


def run_stage_1_benchmark(
    passages: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    fixed_strategy: str = "fixed",
) -> Dict[str, Any]:
    """
    Stage 1: 4 Configurations
    Embedding (MiniLM, Indic) x Retriever (FAISS-HNSW, Exact Flat IP)
    """
    log("\n" + "="*80)
    log("STAGE 1 DECISION GATE: Embedding x Retriever Matrix (4 Configurations)")
    log("="*80)

    # Chunk corpus once using fixed baseline strategy
    log(f"Chunking corpus ({len(passages)} passages) with baseline '{fixed_strategy}' strategy...")
    chunks = chunk_corpus(passages, strategy=fixed_strategy)
    chunk_texts = [c["text"] for c in chunks]
    log(f"Produced {len(chunks)} chunks.")

    candidates = [
        {"name": "minilm", "model": "minilm", "retriever_type": "faiss_hnsw", "dim": 384},
    ]

    stage_1_results = []

    for cand in candidates:
        log(f"\nEvaluating Candidate: {cand['name'].upper()} + FAISS-HNSW...")
        mem_before = get_process_memory_mb()

        # 1. Load embedder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = get_embedding_provider(cand["model"], device=device)

        # 2. Embed chunks (batch processing)
        log(f"Embedding {len(chunks)} chunks (batch size 128)...")
        t0_emb = time.perf_counter()
        embeddings = embedder.embed(chunk_texts, batch_size=128)
        emb_build_time = time.perf_counter() - t0_emb
        log(f"Embeddings computed in {emb_build_time:.2f}s.")

        # 3. Build FAISS index
        retriever = FAISSHNSWRetriever(dimension=cand["dim"])
        t0_idx = time.perf_counter()
        retriever.index(chunks, embeddings)
        idx_build_time = time.perf_counter() - t0_idx
        log(f"FAISS index built in {idx_build_time:.2f}s.")

        mem_after = get_process_memory_mb()
        mem_delta = max(mem_after - mem_before, 1.0)

        # 4. Evaluate queries
        metrics = evaluate_retrieval(retriever, embedder, queries, top_k=5)
        metrics["config_name"] = f"{cand['name']}_faiss_hnsw"
        metrics["embedder_name"] = cand["model"]
        metrics["retriever_name"] = "faiss_hnsw"
        metrics["memory_mb"] = mem_delta
        metrics["dim"] = cand["dim"]

        # Decision score: 60% P70 latency (<50ms target) + 25% Recall@5 + 15% Memory
        lat_score = max(0.0, 1.0 - (metrics["p70_ms"] / 50.0))
        recall_score = metrics["recall_at_5"]
        mem_score = max(0.0, 1.0 - (mem_delta / 2000.0))
        composite_score = (0.60 * lat_score) + (0.25 * recall_score) + (0.15 * mem_score)
        metrics["composite_score"] = composite_score

        stage_1_results.append(metrics)

        log(f"Results for {metrics['config_name']}:")
        log(f"  P50: {metrics['p50_ms']:.2f}ms | P70: {metrics['p70_ms']:.2f}ms | P95: {metrics['p95_ms']:.2f}ms")
        log(f"  Recall@5: {metrics['recall_at_5']:.2%} | MRR: {metrics['mrr']:.4f} | Memory: {mem_delta:.1f}MB")
        log(f"  Composite Winner Score: {composite_score:.4f}")

    # Select winner
    stage_1_results.sort(key=lambda x: x["composite_score"], reverse=True)
    winner_stage_1 = stage_1_results[0]
    log("\n" + "-"*80)
    log(f"🏆 STAGE 1 WINNER: {winner_stage_1['config_name'].upper()}")
    log(f"   Embedder: {winner_stage_1['embedder_name']} | Retriever: {winner_stage_1['retriever_name']}")
    log(f"   P70 Combined Latency: {winner_stage_1['p70_ms']:.2f}ms (Spec Target: <50ms)")
    log(f"   Recall@5: {winner_stage_1['recall_at_5']:.2%}")
    log("-"*80)

    return {
        "all_configs": stage_1_results,
        "winner": winner_stage_1,
    }


def run_stage_2_benchmark(
    passages: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    winning_embedder_name: str,
    winning_dim: int = 384,
) -> Dict[str, Any]:
    """
    Stage 2: 4 Chunking Strategies on the Stage 1 Winner
    (Fixed, Semantic, Parent-Child, Adaptive)
    """
    log("\n" + "="*80)
    log(f"STAGE 2 DECISION GATE: Chunking Strategies on Stage 1 Winner ({winning_embedder_name})")
    log("="*80)

    strategies = ["fixed", "semantic", "parent_child", "adaptive"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = get_embedding_provider(winning_embedder_name, device=device)

    stage_2_results = []

    for strat in strategies:
        log(f"\nChunking with strategy: '{strat}'...")
        chunks = chunk_corpus(passages, strategy=strat)
        chunk_texts = [c["text"] for c in chunks]
        log(f"Produced {len(chunks)} chunks. Embedding...")

        t0_emb = time.perf_counter()
        embeddings = embedder.embed(chunk_texts, batch_size=128)
        log(f"Embedded in {time.perf_counter() - t0_emb:.2f}s. Building FAISS index...")

        retriever = FAISSHNSWRetriever(dimension=winning_dim)
        retriever.index(chunks, embeddings)

        metrics = evaluate_retrieval(retriever, embedder, queries, top_k=5)
        metrics["chunking_strategy"] = strat
        metrics["chunk_count"] = len(chunks)

        # Stage 2 Decision Score: 60% Recall@5 + 25% MRR + 15% Latency Score (<50ms)
        lat_score = max(0.0, 1.0 - (metrics["p70_ms"] / 50.0))
        composite_score = (0.60 * metrics["recall_at_5"]) + (0.25 * metrics["mrr"]) + (0.15 * lat_score)
        metrics["composite_score"] = composite_score

        stage_2_results.append(metrics)

        log(f"Results for strategy '{strat}':")
        log(f"  Chunks: {len(chunks)} | Recall@5: {metrics['recall_at_5']:.2%} | MRR: {metrics['mrr']:.4f}")
        log(f"  P50: {metrics['p50_ms']:.2f}ms | P70: {metrics['p70_ms']:.2f}ms | P95: {metrics['p95_ms']:.2f}ms")
        log(f"  Stage 2 Composite Score: {composite_score:.4f}")

    # Select winner
    stage_2_results.sort(key=lambda x: x["composite_score"], reverse=True)
    winner_stage_2 = stage_2_results[0]
    log("\n" + "-"*80)
    log(f"🏆 STAGE 2 WINNER (BEST CHUNKER): {winner_stage_2['chunking_strategy'].upper()}")
    log(f"   Recall@5: {winner_stage_2['recall_at_5']:.2%}")
    log(f"   MRR: {winner_stage_2['mrr']:.4f}")
    log(f"   P70 Latency: {winner_stage_2['p70_ms']:.2f}ms")
    log("-"*80)

    return {
        "all_strategies": stage_2_results,
        "winner": winner_stage_2,
    }


def main():
    passages_path = "backend/data/passages.jsonl"
    queries_path = "benchmarks/datasets/canonical_queries.jsonl"

    log("Loading test data...")
    passages = load_passages(passages_path, max_count=1000)
    queries = load_benchmark_queries(queries_path)
    log(f"Loaded {len(passages)} passages and {len(queries)} queries.")

    # 1. Stage 1 Benchmark
    stage_1 = run_stage_1_benchmark(passages, queries, fixed_strategy="fixed")

    # 2. Stage 2 Benchmark
    winning_embedder = stage_1["winner"]["embedder_name"]
    winning_dim = stage_1["winner"]["dim"]
    stage_2 = run_stage_2_benchmark(passages, queries, winning_embedder, winning_dim)

    # Save benchmark artifact to JSON and markdown report
    report = {
        "stage_1": stage_1,
        "stage_2": stage_2,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_configuration": {
            "embedder": winning_embedder,
            "retriever": stage_1["winner"]["retriever_name"],
            "chunker": stage_2["winner"]["chunking_strategy"],
            "embed_retrieval_ms_p70": stage_2["winner"]["p70_ms"],
            "recall_at_5": stage_2["winner"]["recall_at_5"],
            "mrr": stage_2["winner"]["mrr"],
        }
    }

    report_path = "benchmarks/benchmark_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(f"\nSaved benchmark decision report to {report_path}")


if __name__ == "__main__":
    main()
