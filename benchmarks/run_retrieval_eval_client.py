"""
Retrieval Quality Evaluation on 25K Multilingual Index via Active Backend Server.
Zero extra RAM footprint (prevents Windows MemoryError/OOM).

Metrics computed:
- Recall@1, Recall@5, Recall@10
- MRR@10
- Cross-lingual Recall@10
- Same-language Recall@10
- Abstention Accuracy on out-of-domain queries
- Language breakdown across all languages: EN, HI, MR, BN, GU, KN, ML, TA, TE, OR, PA, AS, NE, UR, SA
"""

import os
import sys
import json
import time
import re
from collections import defaultdict
from typing import List, Dict, Any
import httpx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_URL = "http://127.0.0.1:8000/api/query"


def normalize_id(pid: str) -> str:
    """Normalize passage ID for robust comparison across delimiters (: vs _)."""
    if not pid:
        return ""
    pid_str = str(pid).strip().lower().replace(":", "_")
    parts = pid_str.split("_")
    if len(parts) >= 3 and len(parts[0]) <= 3:
        return "_".join(parts[1:])
    return pid_str


def is_match(retrieved_chunk: Dict[str, Any], gold_record: Dict[str, Any]) -> bool:
    """Check if a retrieved chunk matches the gold ground-truth passage/answer."""
    gold_pid = gold_record.get("passage_id") or gold_record.get("gold_passage_id")
    gold_qid = gold_record.get("query_id")
    gold_snippet = gold_record.get("expected_answer_snippet") or gold_record.get("gold_answer") or ""

    chunk_pid = retrieved_chunk.get("passage_id") or retrieved_chunk.get("vector_id")
    chunk_qid = retrieved_chunk.get("query_id")
    chunk_text = retrieved_chunk.get("text", "")
    chunk_parent = retrieved_chunk.get("parent_text", "")
    full_chunk_text = f"{chunk_text} {chunk_parent}"

    # 1. Exact passage ID match (handling colon vs underscore delimiters)
    if gold_pid and chunk_pid:
        norm_gold = normalize_id(str(gold_pid))
        norm_chunk = normalize_id(str(chunk_pid))
        if norm_gold == norm_chunk:
            return True
        # Check canonical full ID with delimiters normalized
        if str(gold_pid).strip().lower().replace(":", "_") == str(chunk_pid).strip().lower().replace(":", "_"):
            return True

    # 2. Query ID match (cluster containment)
    if gold_pid:
        gold_str = str(gold_pid).strip().lower().replace(":", "_")
        for p in gold_str.split("_"):
            if p.isdigit() and len(p) >= 4:
                gold_qid = int(p)
                break

    if chunk_pid and not chunk_qid:
        chunk_str = str(chunk_pid).strip().lower().replace(":", "_")
        for p in chunk_str.split("_"):
            if p.isdigit() and len(p) >= 4:
                chunk_qid = int(p)
                break

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
            # 60% word containment
            match_count = sum(1 for w in words if w in clean_chunk)
            if match_count / len(words) >= 0.60:
                return True

    return False


def main():
    print("=" * 85)
    print("🎯 MULTILINGUAL 25K RETRIEVAL QUALITY BENCHMARK")
    print(f"Connecting to live backend at: {SERVER_URL}")
    print("=" * 85)

    benchmark_file = os.path.join(ROOT_DIR, "benchmarks", "benchmark_500.json")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    in_domain_queries = bench_data.get("in_domain_queries", [])
    out_of_domain_queries = bench_data.get("out_of_domain_queries", [])

    print(f"Loaded {len(in_domain_queries)} in-domain queries and {len(out_of_domain_queries)} out-of-domain queries.")

    client = httpx.Client(timeout=15.0)

    # --- 1. Evaluate In-Domain Retrieval Quality ---
    print("\nEvaluating In-Domain Retrieval Quality (Top-10 candidates per query)...", flush=True)

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

        try:
            resp = client.post(SERVER_URL, json={"query": q_text, "top_k": 10})
            if resp.status_code == 200:
                data = resp.json()
                retrieved = data.get("retrieved_chunks", [])
            else:
                retrieved = []
        except Exception as e:
            retrieved = []

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

        if (i + 1) % 50 == 0 or (i + 1) == len(in_domain_queries):
            print(f"  Processed {i+1}/{len(in_domain_queries)} in-domain queries... (Current Recall@10: {r10_hits/(i+1)*100:.1f}%)", flush=True)

    n_in = len(in_domain_queries)
    recall_1 = (r1_hits / n_in) * 100.0 if n_in else 0.0
    recall_5 = (r5_hits / n_in) * 100.0 if n_in else 0.0
    recall_10 = (r10_hits / n_in) * 100.0 if n_in else 0.0
    mrr_10 = (rr_sum / n_in) if n_in else 0.0

    cross_lingual_r10 = (cross_lingual_r10_hits / cross_lingual_total * 100.0) if cross_lingual_total else 100.0
    same_lingual_r10 = (same_lingual_r10_hits / same_lingual_total * 100.0) if same_lingual_total else 100.0

    # --- 2. Evaluate Abstention Accuracy on Out-of-Domain Queries ---
    print("\nEvaluating Abstention Accuracy on Out-of-Domain queries...", flush=True)
    abstention_correct = 0

    for i, q in enumerate(out_of_domain_queries):
        q_text = q.get("query", "")
        try:
            resp = client.post(SERVER_URL, json={"query": q_text, "top_k": 5})
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "")
                if status.startswith("refusal"):
                    abstention_correct += 1
        except Exception:
            pass

        if (i + 1) % 50 == 0 or (i + 1) == len(out_of_domain_queries):
            print(f"  Processed {i+1}/{len(out_of_domain_queries)} out-of-domain queries...", flush=True)

    n_out = len(out_of_domain_queries)
    abstention_acc = (abstention_correct / n_out * 100.0) if n_out else 0.0

    # --- Print Metric Summary Table ---
    print("\n" + "=" * 85)
    print("📊 25K RETRIEVAL QUALITY EVALUATION RESULTS")
    print("=" * 85)
    print(f"| {'Metric':<30} | {'25K Result':<30} |")
    print(f"|{'-'*32}|{'-'*32}|")
    print(f"| {'Recall@1':<30} | {recall_1:>10.2f}% ({r1_hits}/{n_in})        |")
    print(f"| {'Recall@5':<30} | {recall_5:>10.2f}% ({r5_hits}/{n_in})        |")
    print(f"| {'Recall@10':<30} | {recall_10:>10.2f}% ({r10_hits}/{n_in})        |")
    print(f"| {'MRR@10':<30} | {mrr_10:>10.4f}                    |")
    print(f"| {'Cross-lingual Recall@10':<30} | {cross_lingual_r10:>10.2f}% ({cross_lingual_r10_hits}/{cross_lingual_total})      |")
    print(f"| {'Same-language Recall@10':<30} | {same_lingual_r10:>10.2f}% ({same_lingual_r10_hits}/{same_lingual_total})      |")
    print(f"| {'Abstention accuracy':<30} | {abstention_acc:>10.2f}% ({abstention_correct}/{n_out})      |")
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
    print(f"{'Code':<6} {'Language Name':<14} {'Queries':<8} {'Recall@1':<12} {'Recall@5':<12} {'Recall@10':<12} {'MRR@10':<10}")
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
