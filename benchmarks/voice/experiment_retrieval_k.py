"""
Retrieval Experiment: Top-K Expansion (k=3, 5, 10, 20) & Rank Analysis.

Specifically analyzes:
- QID 1060361 (Barter query)
- QID 1102431 (Rachel Carson)
- QID 165349 (Delta Bangalore)
- QID 260880 (Cantaloupe)
- QID 116898 (Arbitrary)
- QID 1060359 (Basal transcription)
- Across the full 50 Canonical dataset: Recall@3, Recall@5, Recall@10, Recall@20 and search latency.
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever


def run_retrieval_experiment():
    print("=" * 85)
    print("FOCUSED RETRIEVAL EXPERIMENT: TOP-K SCALING (k=3, 5, 10, 20)")
    print("=" * 85)

    embedder = get_embedding_provider("minilm")
    retriever = FAISSHNSWRetriever()
    retriever.load(os.path.join(ROOT_DIR, "backend", "data", "faiss_cache"))

    # Load canonical queries
    canonical_file = os.path.join(ROOT_DIR, "benchmarks", "datasets", "canonical_queries.jsonl")
    queries = []
    with open(canonical_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line.strip()))

    k_values = [3, 5, 10, 20]
    results_by_k = {}

    for k in k_values:
        hits = 0
        latencies = []
        reciprocal_ranks = []
        
        for q in queries:
            gt_passage_id = q.get("ground_truth_passage_id")
            q_text = q["query"]
            
            t0 = time.perf_counter()
            q_emb = embedder.embed_query(q_text)
            chunks = retriever.search(q_emb, top_k=k)
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_elapsed)
            
            retrieved_ids = [c.passage_id for c in chunks]
            if gt_passage_id in retrieved_ids:
                hits += 1
                rank = retrieved_ids.index(gt_passage_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        recall = (hits / len(queries)) * 100.0
        mrr = float(np.mean(reciprocal_ranks))
        p50 = float(np.percentile(latencies, 50))
        p70 = float(np.percentile(latencies, 70))
        p95 = float(np.percentile(latencies, 95))

        results_by_k[k] = {
            "top_k": k,
            "recall": recall,
            "mrr": mrr,
            "latency_p50_ms": p50,
            "latency_p70_ms": p70,
            "latency_p95_ms": p95,
        }

    print(f"{'Top-K':<8} | {'Recall@K':<12} | {'MRR':<10} | {'P50 Latency':<14} | {'P70 Latency':<14}")
    print("-" * 65)
    for k, v in results_by_k.items():
        print(f"k = {v['top_k']:<4} | {v['recall']:6.2f}%     | {v['mrr']:.4f}     | {v['latency_p50_ms']:8.2f} ms     | {v['latency_p70_ms']:8.2f} ms")
    print("-" * 65)

    # Detailed inspection on specific difficult queries
    inspect_qids = [1060361, 1102431, 165349, 260880, 116898, 1060359]
    print("\nRANK & EVIDENCE INSPECTION ON HARD CASES (k=20):")
    print("=" * 85)

    for qid in inspect_qids:
        q_match = [q for q in queries if q.get("query_id") == qid]
        if not q_match:
            continue
        q = q_match[0]
        gt_id = q.get("ground_truth_passage_id")
        
        q_emb = embedder.embed_query(q["query"])
        chunks = retriever.search(q_emb, top_k=20)
        retrieved_ids = [c.passage_id for c in chunks]
        
        rank_str = f"Rank #{retrieved_ids.index(gt_id)+1}" if gt_id in retrieved_ids else "NOT in Top-20"
        top1_score = chunks[0].score if chunks else 0.0
        
        print(f"QID {qid} | '{q['query'][:35]}...'")
        print(f"  -> Ground Truth Passage: {gt_id} => {rank_str}")
        print(f"  -> Top-1 Score: {top1_score:.4f} | Chunk-1 ID: {chunks[0].passage_id}")
        if gt_id in retrieved_ids:
            gt_chunk = [c for c in chunks if c.passage_id == gt_id][0]
            print(f"  -> Target Chunk Score: {gt_chunk.score:.4f} | Text: {(gt_chunk.parent_text or gt_chunk.text)[:100]}...")
        print("-" * 85)

    out_file = os.path.join(ROOT_DIR, "benchmarks", "voice", "retrieval_k_experiment.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_by_k, f, indent=2)
    print(f"Saved retrieval experiment artifact to: {out_file}\n")


if __name__ == "__main__":
    run_retrieval_experiment()
