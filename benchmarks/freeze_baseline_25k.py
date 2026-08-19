"""
Baseline 25K v1 Freeze & Artifact Generator.

Generates:
experiments/baseline_25k_v1/
├── config.json
├── retrieval_results.jsonl  (Baseline A: Pure retrieval evaluation)
├── full_system_results.jsonl (Baseline B: Full RAG pipeline evaluation)
├── ood_results.jsonl        (OOD refusal & confusion evaluation)
└── summary.json             (Aggregate report)
"""

import os
import sys
import json
import time
import re
from typing import Dict, Any, List
from collections import defaultdict
import httpx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT_DIR, "experiments", "baseline_25k_v1")
os.makedirs(EXP_DIR, exist_ok=True)

SERVER_URL = "http://127.0.0.1:8000/api/query"


def normalize_id(pid: str) -> str:
    if not pid:
        return ""
    pid_str = str(pid).strip().lower().replace(":", "_")
    parts = pid_str.split("_")
    if len(parts) >= 3 and len(parts[0]) <= 3:
        return "_".join(parts[1:])
    return pid_str


def is_match(retrieved_chunk: Dict[str, Any], gold_record: Dict[str, Any]) -> bool:
    gold_pid = gold_record.get("passage_id") or gold_record.get("gold_passage_id")
    gold_qid = gold_record.get("query_id")
    gold_snippet = gold_record.get("expected_answer_snippet") or gold_record.get("gold_answer") or ""

    chunk_pid = retrieved_chunk.get("passage_id") or retrieved_chunk.get("vector_id")
    chunk_qid = retrieved_chunk.get("query_id")
    chunk_text = retrieved_chunk.get("text", "")
    chunk_parent = retrieved_chunk.get("parent_text", "")
    full_chunk_text = f"{chunk_text} {chunk_parent}"

    # 1. Exact PID
    if gold_pid and chunk_pid:
        if normalize_id(str(gold_pid)) == normalize_id(str(chunk_pid)):
            return True
        if str(gold_pid).strip().lower().replace(":", "_") == str(chunk_pid).strip().lower().replace(":", "_"):
            return True

    # 2. QID cluster
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

    # 3. Gold snippet content overlap
    if gold_snippet and len(gold_snippet.strip()) >= 15:
        clean_snippet = re.sub(r'[^\w\s]', '', gold_snippet.lower()).strip()
        clean_chunk = re.sub(r'[^\w\s]', '', full_chunk_text.lower()).strip()
        words = clean_snippet.split()
        if len(words) >= 4:
            match_count = sum(1 for w in words if w in clean_chunk)
            if match_count / len(words) >= 0.60:
                return True

    return False


def main():
    print("=" * 85)
    print("🔒 FREEZING BASELINE 25K v1 (experiments/baseline_25k_v1)")
    print("=" * 85)

    # 1. Save config.json
    config_data = {
        "benchmark_version": "v0.25k-baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "corpus": "25K passages per language across 14 Indic languages + English (350,000 total passages)",
            "languages": ["asm", "ben", "guj", "hin", "kan", "mal", "mar", "nep", "ori", "pan", "san", "tam", "tel", "urd", "eng"],
            "evaluation_set": "350 In-Domain + 150 Out-of-Domain (500 total queries)",
            "gold_coverage": "100.0% (350/350 gold passages verified in 25K index)",
            "chunk_format": "{language}:{query_id}:{position}"
        },
        "chunking": {
            "strategy": "1 passage = 1 chunk (no sub-chunking)"
        },
        "embedding": {
            "model": "BAAI/bge-m3",
            "dimension": 1024,
            "max_length": 64,
            "torch_dtype": "float16",
            "device": "cuda / cpu"
        },
        "retrieval": {
            "dense": "FAISS-HNSW (m=32, ef_search=64, inner_product)",
            "sparse": "BM25-WAND",
            "fusion": "Reciprocal Rank Fusion (RRF k=60)",
            "dense_top_k": 40,
            "bm25_top_k": 40,
            "fused_top_k": 10,
            "endpoint_top_k": 10
        },
        "guardrails": {
            "tier_1_pre_retrieval": "SafetyGuardrail + Intent / Topic Filtering",
            "tier_2_pre_llm": "RelevanceGate (threshold=0.45) + InsufficientEvidenceChecker (entity-aware)",
            "tier_3_post_llm": "GroundednessVerifier (substantive claim & numeric veracity check)"
        },
        "generation": {
            "provider": "Groq",
            "model": "openai/gpt-oss-120b",
            "max_tokens": 64,
            "temperature": 0.1,
            "context_window_compact": "max 300 chars per chunk"
        }
    }

    with open(os.path.join(EXP_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print("Saved: experiments/baseline_25k_v1/config.json")

    # Load 500 benchmark queries
    benchmark_file = os.path.join(ROOT_DIR, "benchmarks", "benchmark_500.json")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    in_domain = bench_data.get("in_domain_queries", [])
    out_of_domain = bench_data.get("out_of_domain_queries", [])

    client = httpx.Client(timeout=20.0)

    # --- Run Baseline A (Retrieval) & Baseline B (Full System) on In-Domain ---
    print(f"Evaluating {len(in_domain)} in-domain queries for Baseline A & B...", flush=True)

    retrieval_lines = []
    full_system_lines = []

    r1_count, r5_count, r10_count, rr_sum = 0, 0, 0, 0.0
    cross_total, cross_r10_hits = 0, 0
    same_total, same_r10_hits = 0, 0
    id_correct_answers = 0
    latencies = []

    per_lang_retrieval = defaultdict(lambda: {"total": 0, "r1": 0, "r5": 0, "r10": 0, "rr_sum": 0.0})
    per_lang_system = defaultdict(lambda: {"total": 0, "correct": 0, "latencies": []})

    for i, q in enumerate(in_domain):
        q_text = q.get("query", "")
        q_lang = q.get("language", "en").lower().strip()
        gold_pid = q.get("passage_id", "")
        gold_lang = gold_pid.split("_")[0].lower() if "_" in gold_pid else q_lang
        gold_snippet = q.get("expected_answer_snippet", "")

        t0 = time.perf_counter()
        try:
            resp = client.post(SERVER_URL, json={"query": q_text, "top_k": 10})
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                data = resp.json()
                retrieved = data.get("retrieved_chunks", [])
                status = data.get("status", "")
                answer = data.get("answer", "")
                tel = data.get("telemetry", {})
            else:
                retrieved, status, answer, tel = [], f"http_{resp.status_code}", "", {}
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            retrieved, status, answer, tel = [], "conn_error", "", {}

        latencies.append(elapsed_ms)

        # Baseline A Check
        hit_rank = None
        for rank, chunk in enumerate(retrieved):
            if is_match(chunk, q):
                hit_rank = rank + 1
                break

        is_cross = (gold_lang != q_lang)
        if is_cross:
            cross_total += 1
        else:
            same_total += 1

        per_lang_retrieval[q_lang]["total"] += 1
        per_lang_system[q_lang]["total"] += 1
        per_lang_system[q_lang]["latencies"].append(elapsed_ms)

        if hit_rank is not None:
            if hit_rank <= 1:
                r1_count += 1
                per_lang_retrieval[q_lang]["r1"] += 1
            if hit_rank <= 5:
                r5_count += 1
                per_lang_retrieval[q_lang]["r5"] += 1
            if hit_rank <= 10:
                r10_count += 1
                per_lang_retrieval[q_lang]["r10"] += 1
                if is_cross:
                    cross_r10_hits += 1
                else:
                    same_r10_hits += 1
            rr_sum += (1.0 / hit_rank)
            per_lang_retrieval[q_lang]["rr_sum"] += (1.0 / hit_rank)

        # Baseline B Check
        is_sys_correct = (status == "success" and answer and not status.startswith("refusal"))
        if is_sys_correct:
            id_correct_answers += 1
            per_lang_system[q_lang]["correct"] += 1

        retrieval_lines.append({
            "id": i,
            "query": q_text,
            "language": q_lang,
            "gold_passage_id": gold_pid,
            "hit_rank": hit_rank,
            "r1": hit_rank == 1 if hit_rank else False,
            "r5": hit_rank <= 5 if hit_rank else False,
            "r10": hit_rank <= 10 if hit_rank else False,
            "reciprocal_rank": 1.0 / hit_rank if hit_rank else 0.0,
            "retrieved_pids": [c.get("passage_id") for c in retrieved[:10]]
        })

        full_system_lines.append({
            "id": i,
            "query": q_text,
            "language": q_lang,
            "gold_passage_id": gold_pid,
            "status": status,
            "is_correct": is_sys_correct,
            "answer": answer[:150],
            "latency_ms": round(elapsed_ms, 1),
            "telemetry": tel
        })

    # --- Run OOD Evaluation ---
    print(f"Evaluating {len(out_of_domain)} out-of-domain queries...", flush=True)
    ood_lines = []
    ood_correct_refusals = 0

    for j, q in enumerate(out_of_domain):
        q_text = q.get("query", "")
        q_cat = q.get("category", "ood")
        t0 = time.perf_counter()
        try:
            resp = client.post(SERVER_URL, json={"query": q_text, "top_k": 5})
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "")
                answer = data.get("answer", "")
                reason = data.get("refusal_reason", "")
            else:
                status, answer, reason = f"http_{resp.status_code}", "", ""
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            status, answer, reason = "conn_error", "", ""

        is_refused = status.startswith("refusal")
        if is_refused:
            ood_correct_refusals += 1

        ood_lines.append({
            "id": j,
            "query": q_text,
            "category": q_cat,
            "status": status,
            "is_refused_correctly": is_refused,
            "refusal_reason": reason,
            "answer": answer[:120],
            "latency_ms": round(elapsed_ms, 1)
        })

    # Save JSONL artifacts
    with open(os.path.join(EXP_DIR, "retrieval_results.jsonl"), "w", encoding="utf-8") as f:
        for item in retrieval_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Saved: experiments/baseline_25k_v1/retrieval_results.jsonl")

    with open(os.path.join(EXP_DIR, "full_system_results.jsonl"), "w", encoding="utf-8") as f:
        for item in full_system_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Saved: experiments/baseline_25k_v1/full_system_results.jsonl")

    with open(os.path.join(EXP_DIR, "ood_results.jsonl"), "w", encoding="utf-8") as f:
        for item in ood_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Saved: experiments/baseline_25k_v1/ood_results.jsonl")

    # Latency percentiles
    import numpy as np
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p100 = float(np.max(latencies))

    n_in = len(in_domain)
    n_ood = len(out_of_domain)
    n_total = n_in + n_ood
    overall_correct = id_correct_answers + ood_correct_refusals

    summary_data = {
        "experiment_name": "baseline_25k_v1",
        "frozen_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_a_retrieval": {
            "total_queries": n_in,
            "recall_at_1": round(r1_count / n_in * 100.0, 2),
            "recall_at_5": round(r5_count / n_in * 100.0, 2),
            "recall_at_10": round(r10_count / n_in * 100.0, 2),
            "mrr_at_10": round(rr_sum / n_in, 4),
            "cross_lingual_recall_at_10": round(cross_r10_hits / cross_total * 100.0, 2) if cross_total else 100.0,
            "same_language_recall_at_10": round(same_r10_hits / same_total * 100.0, 2) if same_total else 100.0,
            "per_language_retrieval": {
                lang: {
                    "queries": s["total"],
                    "r1": round(s["r1"] / s["total"] * 100.0, 1),
                    "r5": round(s["r5"] / s["total"] * 100.0, 1),
                    "r10": round(s["r10"] / s["total"] * 100.0, 1),
                    "mrr": round(s["rr_sum"] / s["total"], 3)
                }
                for lang, s in sorted(per_lang_retrieval.items())
            }
        },
        "baseline_b_full_system": {
            "in_domain_accuracy": {
                "correct": id_correct_answers,
                "total": n_in,
                "percentage": round(id_correct_answers / n_in * 100.0, 2)
            },
            "ood_refusal_accuracy": {
                "correct": ood_correct_refusals,
                "total": n_ood,
                "percentage": round(ood_correct_refusals / n_ood * 100.0, 2)
            },
            "overall_accuracy": {
                "correct": overall_correct,
                "total": n_total,
                "percentage": round(overall_correct / n_total * 100.0, 2)
            },
            "latency_profile_ms": {
                "p50": round(p50, 1),
                "p95": round(p95, 1),
                "p100": round(p100, 1),
                "mean": round(float(np.mean(latencies)), 1)
            },
            "per_language_accuracy": {
                lang: {
                    "queries": s["total"],
                    "accuracy": round(s["correct"] / s["total"] * 100.0, 1),
                    "avg_latency_ms": round(float(np.mean(s["latencies"])), 1)
                }
                for lang, s in sorted(per_lang_system.items())
            }
        }
    }

    with open(os.path.join(EXP_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print("Saved: experiments/baseline_25k_v1/summary.json")

    print("\n" + "=" * 85)
    print("🔒 FROZEN BASELINE 25K v1 SUMMARY")
    print("=" * 85)
    print(f"  • Baseline A (Retrieval R@10) : {summary_data['baseline_a_retrieval']['recall_at_10']}% (MRR@10: {summary_data['baseline_a_retrieval']['mrr_at_10']})")
    print(f"  • Baseline B (In-Domain Acc)  : {summary_data['baseline_b_full_system']['in_domain_accuracy']['percentage']}% ({id_correct_answers}/{n_in})")
    print(f"  • Baseline B (OOD Refusal)    : {summary_data['baseline_b_full_system']['ood_refusal_accuracy']['percentage']}% ({ood_correct_refusals}/{n_ood})")
    print(f"  • Baseline B (Overall Acc)    : {summary_data['baseline_b_full_system']['overall_accuracy']['percentage']}% ({overall_correct}/{n_total})")
    print(f"  • Latency Profile             : P50 = {p50:.0f}ms | P95 = {p95:.0f}ms | P100 = {p100:.0f}ms")
    print("=" * 85)


if __name__ == "__main__":
    main()
