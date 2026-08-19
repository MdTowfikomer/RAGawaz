"""
Retrieval Quality Evaluation on 25K Multilingual Index Bundle.

Computes:
- Recall@1, Recall@5, Recall@10
- MRR@10
- Cross-lingual Recall@10 vs Same-language Recall@10
- Abstention Accuracy on out-of-domain queries
- Per-language breakdown across all languages (EN, HI, MR, BN, GU, KN, ML, TA, TE, etc.)
"""

import os
import sys
import json
import time
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# Set utf-8 stdout encoding for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.config import settings, EMBEDDING_PROFILES
from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.rag.bm25_retriever import BM25Retriever
from backend.app.rag.hybrid_retriever import HybridRetriever
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.safety import SafetyGuardrail


def normalize_id(pid: str) -> str:
    """Normalize passage ID for comparison."""
    if not pid:
        return ""
    # strip language prefix e.g. hi_1102432_5 -> 1102432_5
    pid_str = str(pid).strip().lower()
    parts = pid_str.split("_")
    if len(parts) >= 3 and len(parts[0]) <= 3:
        return "_".join(parts[1:])
    return pid_str


def is_match(retrieved_chunk: Any, gold_record: Dict[str, Any]) -> bool:
    """Check if a retrieved chunk matches the gold ground-truth passage/answer."""
    gold_pid = gold_record.get("passage_id") or gold_record.get("gold_passage_id")
    gold_qid = gold_record.get("query_id")
    gold_snippet = gold_record.get("expected_answer_snippet") or gold_record.get("gold_answer") or ""

    chunk_pid = getattr(retrieved_chunk, "passage_id", None) or (retrieved_chunk.get("passage_id") if isinstance(retrieved_chunk, dict) else None)
    chunk_qid = getattr(retrieved_chunk, "query_id", None) or (retrieved_chunk.get("query_id") if isinstance(retrieved_chunk, dict) else None)
    chunk_text = getattr(retrieved_chunk, "text", "") or (retrieved_chunk.get("text", "") if isinstance(retrieved_chunk, dict) else "")
    chunk_parent = getattr(retrieved_chunk, "parent_text", "") or (retrieved_chunk.get("parent_text", "") if isinstance(retrieved_chunk, dict) else "")
    full_chunk_text = f"{chunk_text} {chunk_parent}"

    # 1. Exact passage ID match
    if gold_pid and chunk_pid:
        if normalize_id(str(gold_pid)) == normalize_id(str(chunk_pid)):
            return True

    # 2. Query ID match (if from same query cluster)
    if gold_qid and chunk_qid:
        try:
            if int(gold_qid) == int(chunk_qid):
                return True
        except Exception:
            pass

    # 3. Gold snippet content overlap (lexical/factual hit)
    if gold_snippet and len(gold_snippet.strip()) >= 15:
        clean_snippet = re.sub(r'[^\w\s]', '', gold_snippet.lower()).strip()
        clean_chunk = re.sub(r'[^\w\s]', '', full_chunk_text.lower()).strip()
        words = clean_snippet.split()
        if len(words) >= 4:
            # 70% word containment
            match_count = sum(1 for w in words if w in clean_chunk)
            if match_count / len(words) >= 0.70:
                return True

    return False


def main():
    print("=" * 85)
    print("🎯 MULTILINGUAL 25K RETRIEVAL QUALITY BENCHMARK")
    print("=" * 85)

    bundle_dir = os.path.join(ROOT_DIR, "backend", "data", "multilingual_index_bundle")
    if not os.path.exists(bundle_dir):
        print(f"Error: Bundle directory not found: {bundle_dir}")
        sys.exit(1)

    print("Initializing embedding provider (BGE-M3)...", flush=True)
    embedder = get_embedding_provider("bge_m3")

    print("Loading FAISS-HNSW Vector Index (mmap)...", flush=True)
    dense_retriever = FAISSHNSWRetriever(dimension=1024, m=32, ef_search=64)
    dense_retriever.load(bundle_dir, use_mmap=True)

    print("Loading BM25-WAND sparse index...", flush=True)
    bm25_retriever = BM25Retriever()
    bm25_pkl_path = os.path.join(bundle_dir, "bm25_wand.pkl")
    if os.path.exists(bm25_pkl_path):
        import pickle
        with open(bm25_pkl_path, "rb") as f:
            bm25_data = pickle.load(f)
        bm25_retriever.num_docs = bm25_data["num_docs"]
        bm25_retriever.avg_doc_len = bm25_data["avg_doc_len"]
        bm25_retriever.doc_lengths = bm25_data["doc_lengths"]
        bm25_retriever.inverted_index = bm25_data["inverted_index"]
        bm25_retriever.term_upper_bounds = bm25_data["term_upper_bounds"]
        bm25_retriever.chunks_metadata = dense_retriever.chunks_metadata
        bm25_retriever.chunk_id_map = dense_retriever.chunk_id_map

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=40,
        bm25_top_k=40,
        rrf_k=60,
        fused_top_k=10,
    )

    safety_guard = SafetyGuardrail()
    relevance_gate = RelevanceGate(threshold=0.45)
    insufficient_checker = InsufficientEvidenceChecker(confidence_threshold=0.45)

    # Load 500 benchmark dataset
    benchmark_file = os.path.join(ROOT_DIR, "benchmarks", "benchmark_500.json")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    in_domain_queries = bench_data.get("in_domain_queries", [])
    out_of_domain_queries = bench_data.get("out_of_domain_queries", [])

    print(f"Loaded {len(in_domain_queries)} in-domain queries and {len(out_of_domain_queries)} out-of-domain queries.")
    print(f"Index corpus size: {len(dense_retriever.chunks_metadata):,} indexed chunks across 14 languages.\n")

    # --- 1. Evaluate In-Domain Retrieval Quality ---
    print("Evaluating In-Domain Retrieval Quality (Top-10 Candidates)...", flush=True)

    r1_hits = 0
    r5_hits = 0
    r10_hits = 0
    rr_sum = 0.0

    cross_lingual_total = 0
    cross_lingual_r10_hits = 0
    same_lingual_total = 0
    same_lingual_r10_hits = 0

    per_lang_stats = defaultdict(lambda: {"total": 0, "r1": 0, "r5": 0, "r10": 0, "rr_sum": 0.0})

    for i, q in enumerate(in_domain_queries):
        q_text = q.get("query", "")
        q_lang = q.get("language", "en").lower().strip()
        gold_pid = q.get("passage_id", "")
        gold_lang = gold_pid.split("_")[0].lower() if "_" in gold_pid else q_lang

        q_emb = embedder.embed_query(q_text)
        retrieved = hybrid_retriever.search_hybrid(q_text, q_emb, top_k=10)

        # Check hits at rank 1, 5, 10
        hit_rank = None
        for rank, chunk in enumerate(retrieved):
            if is_match(chunk, q):
                hit_rank = rank + 1
                break

        is_cross = (gold_lang != q_lang)
        if is_cross:
            cross_lingual_total += 1
        else:
            same_lingual_total += 1

        per_lang_stats[q_lang]["total"] += 1

        if hit_rank is not None:
            if hit_rank <= 1:
                r1_hits += 1
                per_lang_stats[q_lang]["r1"] += 1
            if hit_rank <= 5:
                r5_hits += 1
                per_lang_stats[q_lang]["r5"] += 1
            if hit_rank <= 10:
                r10_hits += 1
                per_lang_stats[q_lang]["r10"] += 1
                if is_cross:
                    cross_lingual_r10_hits += 1
                else:
                    same_lingual_r10_hits += 1
            reciprocal_rank = 1.0 / hit_rank
            rr_sum += reciprocal_rank
            per_lang_stats[q_lang]["rr_sum"] += reciprocal_rank

    n_in = len(in_domain_queries)
    recall_1 = (r1_hits / n_in) * 100.0 if n_in else 0.0
    recall_5 = (r5_hits / n_in) * 100.0 if n_in else 0.0
    recall_10 = (r10_hits / n_in) * 100.0 if n_in else 0.0
    mrr_10 = (rr_sum / n_in) if n_in else 0.0

    cross_lingual_r10 = (cross_lingual_r10_hits / cross_lingual_total * 100.0) if cross_lingual_total else 100.0
    same_lingual_r10 = (same_lingual_r10_hits / same_lingual_total * 100.0) if same_lingual_total else 100.0

    # --- 2. Evaluate Abstention Accuracy on Out-of-Domain Queries ---
    print("Evaluating Abstention Accuracy on Out-of-Domain queries...", flush=True)
    abstention_correct = 0

    for q in out_of_domain_queries:
        q_text = q.get("query", "")
        # Safety gate
        safe, _ = safety_guard.evaluate(q_text)
        if not safe:
            abstention_correct += 1
            continue

        q_emb = embedder.embed_query(q_text)
        retrieved = hybrid_retriever.search_hybrid(q_text, q_emb, top_k=5)
        scores = [c.score for c in retrieved]

        # Relevance gate
        is_rel, _ = relevance_gate.evaluate(scores, query=q_text)
        if not is_rel:
            abstention_correct += 1
            continue

        # Insufficient evidence gate
        context_texts = [c.text for c in retrieved]
        has_ev, _ = insufficient_checker.evaluate(scores, query=q_text, context_chunks=context_texts)
        if not has_ev:
            abstention_correct += 1
            continue

    n_out = len(out_of_domain_queries)
    abstention_acc = (abstention_correct / n_out * 100.0) if n_out else 0.0

    # --- Print Summary Table ---
    print("\n" + "=" * 85)
    print("📊 25K RETRIEVAL QUALITY EVALUATION RESULTS")
    print("=" * 85)
    print(f"| {'Metric':<30} | {'25K Result':<25} |")
    print(f"|{'-'*32}|{'-'*27}|")
    print(f"| {'Recall@1':<30} | {recall_1:>10.2f}% ({r1_hits}/{n_in})    |")
    print(f"| {'Recall@5':<30} | {recall_5:>10.2f}% ({r5_hits}/{n_in})    |")
    print(f"| {'Recall@10':<30} | {recall_10:>10.2f}% ({r10_hits}/{n_in})    |")
    print(f"| {'MRR@10':<30} | {mrr_10:>10.4f}                |")
    print(f"| {'Cross-lingual Recall@10':<30} | {cross_lingual_r10:>10.2f}% ({cross_lingual_r10_hits}/{cross_lingual_total})  |")
    print(f"| {'Same-language Recall@10':<30} | {same_lingual_r10:>10.2f}% ({same_lingual_r10_hits}/{same_lingual_total})  |")
    print(f"| {'Abstention accuracy':<30} | {abstention_acc:>10.2f}% ({abstention_correct}/{n_out})  |")
    print("=" * 85)

    # --- Per-Language Breakdown ---
    lang_names = {
        "en": "English", "eng": "English",
        "hi": "Hindi", "hin": "Hindi",
        "mr": "Marathi", "mar": "Marathi",
        "bn": "Bengali", "ben": "Bengali",
        "gu": "Gujarati", "guj": "Gujarati",
        "kn": "Kannada", "kan": "Kannada",
        "ml": "Malayalam", "mal": "Malayalam",
        "ta": "Tamil", "tam": "Tamil",
        "te": "Telugu", "tel": "Telugu",
        "or": "Odia", "ori": "Odia",
        "pa": "Punjabi", "pan": "Punjabi",
        "as": "Assamese", "asm": "Assamese",
        "ne": "Nepali", "nep": "Nepali",
        "ur": "Urdu", "urd": "Urdu",
        "sa": "Sanskrit", "san": "Sanskrit"
    }

    print("\n🌐 PER-LANGUAGE RETRIEVAL BREAKDOWN:")
    print("-" * 85)
    print(f"{'Lang':<6} {'Language Name':<14} {'Queries':<8} {'Recall@1':<12} {'Recall@5':<12} {'Recall@10':<12} {'MRR@10':<10}")
    print("-" * 85)

    for l_code, stats in sorted(per_lang_stats.items()):
        total = stats["total"]
        if total == 0:
            continue
        r1 = (stats["r1"] / total) * 100.0
        r5 = (stats["r5"] / total) * 100.0
        r10 = (stats["r10"] / total) * 100.0
        mrr = stats["rr_sum"] / total
        disp_name = lang_names.get(l_code, l_code.upper())
        print(f"{l_code.upper():<6} {disp_name:<14} {total:<8} {r1:>8.1f}%   {r5:>8.1f}%   {r10:>8.1f}%   {mrr:>8.3f}")

    print("=" * 85)


if __name__ == "__main__":
    main()
