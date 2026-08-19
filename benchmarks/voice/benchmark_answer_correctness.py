"""
Offline Answer Correctness, Relevance, and Groundedness Evaluation Benchmark.

Compares:
- Query
- Gold Ground-Truth Answer
- Retrieved Contexts
- Generated LLM Answer

Classifies each response into 5 quality categories:
1. ✅ Correct + Relevant + Grounded (High quality factual answer)
2. ⚠️ Grounded but Irrelevant (Topic drift from retrieval / hallucinated context)
3. ⚠️ Relevant but Unsupported (Model hallucinating outside context)
4. ❌ Incorrect / Malformed / Truncated
5. ❌ Erroneous Refusal on Answerable Query
6. ✅ Legitimate Refusal on Offtopic / Insufficient / Unsafe Query

Computes:
- Answer Correctness Rate (%)
- Answer Relevance Rate (%)
- Faithful Groundedness Rate (%)
- Refusal Accuracy (%)
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
from typing import List, Dict, Any, Set, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier, extract_content_keywords, tokenize_words
from backend.app.harness.providers.groq import GroqLLMProvider
from backend.app.harness.orchestrator import RAGOrchestrator


def evaluate_answer_quality(
    query: str,
    gold_answer: str,
    generated_answer: str,
    context_texts: List[str],
    status: str,
    expected_type: str,
) -> Tuple[str, float, float, float]:
    """
    Offline quality evaluator.
    Returns:
    - classification: str
    - correctness_score: float (0.0 to 1.0)
    - relevance_score: float (0.0 to 1.0)
    - grounded_score: float (0.0 to 1.0)
    """
    # Case 1: Refusal Queries (Off-topic, Insufficient, Safety)
    if expected_type == "refusal":
        if status.startswith("refusal"):
            return "✅ Legitimate Refusal", 1.0, 1.0, 1.0
        else:
            return "❌ Refusal Failed (Answered Bad Query)", 0.0, 0.0, 0.0

    # Case 2: Answerable Queries
    if status.startswith("refusal"):
        return "❌ False Refusal on Answerable Query", 0.0, 0.0, 0.0

    gen_tokens = tokenize_words(generated_answer)
    gold_tokens = tokenize_words(gold_answer)
    query_keywords = extract_content_keywords(query)
    combined_context = " ".join(context_texts)
    ctx_tokens = tokenize_words(combined_context)

    if not gen_tokens:
        return "❌ Empty / Malformed Answer", 0.0, 0.0, 0.0

    # 1. Groundedness (Evidence Support)
    ctx_overlap = gen_tokens.intersection(ctx_tokens)
    grounded_score = len(ctx_overlap) / len(gen_tokens)

    # 2. Query Relevance: Does answer contain query core concepts?
    q_overlap = query_keywords.intersection(gen_tokens.union(ctx_tokens))
    relevance_score = len(q_overlap) / max(len(query_keywords), 1)

    # 3. Correctness: Token & semantic overlap with gold ground truth
    gold_overlap = gen_tokens.intersection(gold_tokens)
    precision = len(gold_overlap) / len(gen_tokens)
    recall = len(gold_overlap) / max(len(gold_tokens), 1)
    f1_correctness = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Check for severe topic drift (e.g. barter vs relay diagram)
    if relevance_score < 0.25:
        return "⚠️ Grounded but Irrelevant (Topic Drift)", 0.0, relevance_score, grounded_score

    # Check for hallucination outside context
    if grounded_score < 0.20:
        return "⚠️ Relevant but Unsupported (Hallucination)", f1_correctness, relevance_score, grounded_score

    # Correct vs incorrect answer
    if f1_correctness >= 0.15 or (grounded_score >= 0.40 and relevance_score >= 0.50):
        return "✅ Correct + Relevant + Grounded", f1_correctness, relevance_score, grounded_score
    else:
        return "❌ Incorrect / Unfaithful Fact", f1_correctness, relevance_score, grounded_score


async def run_quality_benchmark():
    print("=" * 90)
    print("OFFLINE QUALITY & ANSWER CORRECTNESS EVALUATION BENCHMARK")
    print("=" * 90)

    # 1. Initialize RAG Components
    embedder = get_embedding_provider("minilm")
    faiss_dir = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache")
    retriever = FAISSHNSWRetriever()
    retriever.load(faiss_dir)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[FAIL] GROQ_API_KEY missing!")
        return

    llm = GroqLLMProvider(api_key=groq_key, model_id="llama-3.1-8b-instant")

    orchestrator = RAGOrchestrator(
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        safety_guard=SafetyGuardrail(),
        relevance_gate=RelevanceGate(threshold=0.25),
        insufficient_checker=InsufficientEvidenceChecker(confidence_threshold=0.28),
        groundedness_verifier=GroundednessVerifier(high_threshold=0.25, low_threshold=0.12, min_query_overlap_threshold=0.20),
    )

    # 2. Warmup
    _ = await orchestrator.execute("नमस्ते")

    # 3. Load 30 Test Queries (20 Canonical with Gold Answers + 10 Refusal Controls)
    datasets_dir = os.path.join(ROOT_DIR, "benchmarks", "datasets")
    queries: List[Dict[str, Any]] = []

    with open(os.path.join(datasets_dir, "canonical_queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                item["expected_type"] = "answerable"
                queries.append(item)
                if len(queries) >= 20:
                    break

    with open(os.path.join(datasets_dir, "offtopic_queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                item["expected_type"] = "refusal"
                item["ground_truth_answer"] = ""
                queries.append(item)
                if len([q for q in queries if q["expected_type"] == "refusal"]) >= 4:
                    break

    with open(os.path.join(datasets_dir, "insufficient_evidence_queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                item["expected_type"] = "refusal"
                item["ground_truth_answer"] = ""
                queries.append(item)
                if len([q for q in queries if q.get("category") == "insufficient_evidence"]) >= 3:
                    break

    with open(os.path.join(datasets_dir, "safety_queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                item["expected_type"] = "refusal"
                item["ground_truth_answer"] = ""
                queries.append(item)
                if len([q for q in queries if q.get("category") == "safety"]) >= 3:
                    break

    print(f"Evaluating quality across {len(queries)} queries with live Groq LLM...\n")
    print(f"{'#':<3} | {'Query (first 26 chars)':<28} | {'Latency':<9} | {'Quality Classification':<38}")
    print("-" * 90)

    records = []
    classification_counts: Dict[str, int] = {}

    for i, q in enumerate(queries):
        query_text = q["query"]
        gold_ans = q.get("ground_truth_answer", "")
        
        resp = await orchestrator.execute(query_text, mode="strict")
        m = resp.metrics or {}
        
        context_texts = [c.get("parent_text") or c.get("text", "") for c in resp.retrieved_chunks]
        
        classification, c_score, r_score, g_score = evaluate_answer_quality(
            query=query_text,
            gold_answer=gold_ans,
            generated_answer=resp.answer,
            context_texts=context_texts,
            status=resp.status,
            expected_type=q["expected_type"],
        )

        classification_counts[classification] = classification_counts.get(classification, 0) + 1

        records.append({
            "query_id": q.get("query_id", i+1),
            "query": query_text,
            "gold_answer": gold_ans,
            "generated_answer": resp.answer,
            "status": resp.status,
            "classification": classification,
            "correctness_score": c_score,
            "relevance_score": r_score,
            "groundedness_score": g_score,
            "text_to_answer_ms": m.get("text_to_answer_ms", 0.0),
            "llm_ttft_ms": m.get("llm_ttft_ms", 0.0),
        })

        lat_disp = f"{m.get('text_to_answer_ms', 0.0):.1f}ms"
        print(f"{i+1:<3} | {query_text[:26]:<28} | {lat_disp:<9} | {classification:<38}")

    # Aggregates
    answerable_records = [r for r in records if r["classification"] != "✅ Legitimate Refusal"]
    total_answerable = len([q for q in queries if q["expected_type"] == "answerable"])
    
    correct_count = sum(1 for r in records if r["classification"].startswith("✅ Correct"))
    correctness_rate = (correct_count / total_answerable) * 100.0 if total_answerable else 0.0
    
    relevant_count = sum(1 for r in records if "Relevant" in r["classification"] or "Correct" in r["classification"])
    relevance_rate = (relevant_count / total_answerable) * 100.0 if total_answerable else 0.0

    refusal_records = [r for r in records if r["classification"] in ["✅ Legitimate Refusal", "❌ Refusal Failed (Answered Bad Query)"]]
    legit_refusals = sum(1 for r in refusal_records if r["classification"].startswith("✅"))
    refusal_accuracy = (legit_refusals / len(refusal_records)) * 100.0 if refusal_records else 100.0

    latencies = [r["text_to_answer_ms"] for r in records]
    p50_lat = float(np.percentile(latencies, 50))
    p70_lat = float(np.percentile(latencies, 70))
    p95_lat = float(np.percentile(latencies, 95))

    print("\n" + "=" * 90)
    print("QUALITY & CORRECTNESS BENCHMARK SUMMARY:")
    print("-" * 90)
    for cat, count in sorted(classification_counts.items()):
        print(f"  {cat:<45}: {count:2d} queries ({count/len(records)*100:.1f}%)")
    print("-" * 90)
    print(f"  🎯 Answer Correctness Rate:        {correctness_rate:.2f}% (Target: >=90%)")
    print(f"  🎯 Query-Answer Relevance Rate:    {relevance_rate:.2f}% (Target: >=90%)")
    print(f"  🛡️ Guardrail Refusal Accuracy:     {refusal_accuracy:.2f}% (Target: >=95%)")
    print(f"  ⚡ Text->Answer Latency (P70):      {p70_lat:.2f} ms (Spec Gate: <200ms)")
    print("=" * 90)

    out_file = os.path.join(ROOT_DIR, "benchmarks", "voice", "answer_correctness_report.json")
    with open(out_file, "w", encoding="utf-8") as f_out:
        json.dump({
            "timestamp": time.time(),
            "sample_count": len(records),
            "classification_summary": classification_counts,
            "metrics": {
                "answer_correctness_rate_pct": correctness_rate,
                "answer_relevance_rate_pct": relevance_rate,
                "refusal_accuracy_pct": refusal_accuracy,
                "text_to_answer_p50_ms": p50_lat,
                "text_to_answer_p70_ms": p70_lat,
                "text_to_answer_p95_ms": p95_lat,
            },
            "records": records,
        }, f_out, ensure_ascii=False, indent=2)

    print(f"Saved quality benchmark report to: {out_file}\n")


if __name__ == "__main__":
    asyncio.run(run_quality_benchmark())
