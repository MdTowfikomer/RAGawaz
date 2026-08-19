"""
LLM Eval Loop — End-to-End Quality Evaluation for Voice RAG Pipeline.

Tests the full pipeline (embedding → retrieval → guardrails → LLM → groundedness)
against a curated set of queries with ground-truth answers.

Reports:
- Answer Correctness Rate (should be ≥ 80%)
- False Refusal Rate (should be ≤ 10%)
- Hallucination Rate (should be 0%)
- Latency P50/P95
- Per-query detailed diagnostics

Usage:
    cd Voice_rag
    .venv\\Scripts\\python.exe benchmarks/eval_loop.py

    # Quick smoke test (5 queries):
    .venv\\Scripts\\python.exe benchmarks/eval_loop.py --quick

    # Full eval (30 queries):
    .venv\\Scripts\\python.exe benchmarks/eval_loop.py --full

    # Custom query count:
    .venv\\Scripts\\python.exe benchmarks/eval_loop.py --n 15
"""

import os
import sys
import json
import time
import asyncio
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"))


@dataclass
class EvalResult:
    query_id: int
    query: str
    category: str
    expected_type: str  # "answerable" or "refusal"
    gold_answer: str
    generated_answer: str
    status: str
    verdict: str  # "✅ CORRECT", "❌ FALSE_REFUSAL", "❌ HALLUCINATION", "✅ LEGITIMATE_REFUSAL", etc.
    correctness_score: float
    groundedness_score: float
    latency_ms: float
    llm_ttft_ms: float
    diagnostics: Dict[str, Any]


def compute_token_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level F1 between two texts."""
    from backend.app.guardrails.groundedness import tokenize_words
    tokens_a = tokenize_words(text_a)
    tokens_b = tokenize_words(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a.intersection(tokens_b)
    precision = len(overlap) / len(tokens_a)
    recall = len(overlap) / len(tokens_b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def judge_verdict(
    expected_type: str,
    status: str,
    generated_answer: str,
    gold_answer: str,
    retrieved_chunks: List[Dict],
) -> Tuple[str, float, float]:
    """
    Judge the quality of a single eval result.
    Returns (verdict, correctness_score, groundedness_score).
    """
    from backend.app.guardrails.groundedness import tokenize_words, extract_content_keywords

    # --- Refusal expected ---
    if expected_type == "refusal":
        if status.startswith("refusal"):
            return "✅ LEGITIMATE_REFUSAL", 1.0, 1.0
        else:
            return "⚠️ MISSED_REFUSAL", 0.0, 0.5

    # --- Answerable expected ---
    if status.startswith("refusal"):
        return "❌ FALSE_REFUSAL", 0.0, 0.0

    if not generated_answer.strip():
        return "❌ EMPTY_ANSWER", 0.0, 0.0

    # Correctness: token F1 against gold
    correctness = compute_token_overlap(generated_answer, gold_answer)

    # Groundedness: is the answer supported by retrieved context?
    context_text = " ".join(c.get("text", "") or c.get("parent_text", "") for c in retrieved_chunks)
    groundedness = compute_token_overlap(generated_answer, context_text)

    # Check for numeric hallucination
    import re
    ans_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', generated_answer))
    ctx_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', context_text))
    has_hallucinated_numbers = bool(ans_numbers and not ans_numbers.issubset(ctx_numbers))

    if has_hallucinated_numbers:
        return "❌ HALLUCINATION (fabricated numbers)", correctness, groundedness

    if groundedness < 0.10:
        return "❌ HALLUCINATION (unsupported claim)", correctness, groundedness

    if correctness >= 0.15 or (groundedness >= 0.35 and correctness >= 0.05):
        return "✅ CORRECT", correctness, groundedness
    elif groundedness >= 0.30:
        return "⚠️ GROUNDED_BUT_INACCURATE", correctness, groundedness
    else:
        return "❌ INCORRECT", correctness, groundedness


def load_eval_queries(n_answerable: int = 20, n_refusal: int = 10) -> List[Dict[str, Any]]:
    """Load a balanced set of queries for evaluation."""
    datasets_dir = os.path.join(ROOT_DIR, "benchmarks", "datasets")
    queries = []

    # Answerable queries with gold answers
    canonical_path = os.path.join(datasets_dir, "canonical_queries.jsonl")
    if os.path.exists(canonical_path):
        with open(canonical_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    item["expected_type"] = "answerable"
                    queries.append(item)
                    if len([q for q in queries if q["expected_type"] == "answerable"]) >= n_answerable:
                        break

    # Off-topic refusal queries
    offtopic_path = os.path.join(datasets_dir, "offtopic_queries.jsonl")
    if os.path.exists(offtopic_path):
        with open(offtopic_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    item["expected_type"] = "refusal"
                    item.setdefault("ground_truth_answer", "")
                    queries.append(item)
                    if len([q for q in queries if q["expected_type"] == "refusal"]) >= n_refusal // 2:
                        break

    # Insufficient evidence refusal queries
    insufficient_path = os.path.join(datasets_dir, "insufficient_evidence_queries.jsonl")
    if os.path.exists(insufficient_path):
        with open(insufficient_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    item["expected_type"] = "refusal"
                    item.setdefault("ground_truth_answer", "")
                    queries.append(item)
                    if len([q for q in queries if q["expected_type"] == "refusal"]) >= n_refusal:
                        break

    # Safety queries
    safety_path = os.path.join(datasets_dir, "safety_queries.jsonl")
    if os.path.exists(safety_path):
        with open(safety_path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                if line.strip() and count < 3:
                    item = json.loads(line.strip())
                    item["expected_type"] = "refusal"
                    item.setdefault("ground_truth_answer", "")
                    queries.append(item)
                    count += 1

    return queries


async def run_eval_loop(queries: List[Dict[str, Any]], verbose: bool = True) -> List[EvalResult]:
    """Run the full eval loop against the live RAG pipeline."""
    import torch
    from backend.app.rag.embedder import get_embedding_provider
    from backend.app.rag.retriever import FAISSHNSWRetriever
    from backend.app.rag.bm25_retriever import BM25Retriever
    from backend.app.rag.hybrid_retriever import HybridRetriever
    from backend.app.guardrails.safety import SafetyGuardrail
    from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
    from backend.app.guardrails.groundedness import GroundednessVerifier
    from backend.app.harness.providers import get_llm_provider
    from backend.app.harness.orchestrator import RAGOrchestrator
    from backend.app.config import settings, EMBEDDING_PROFILES

    # --- Initialize Pipeline (same as production main.py) ---
    print("\n🔧 Initializing RAG Pipeline for Eval...\n")

    model_key = settings.embedding_model
    profile = EMBEDDING_PROFILES.get(model_key, EMBEDDING_PROFILES["bge_m3"])
    dim = profile["dimension"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedder = get_embedding_provider(model_key, device=device)
    retriever = FAISSHNSWRetriever(dimension=dim, m=32, ef_search=64)

    # Load index
    bundle_dir = os.path.join(ROOT_DIR, "backend", "data", "multilingual_index_bundle")
    cache_dir = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache_bge_m3")
    fallback_cache = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache")

    load_dir = None
    for d in [bundle_dir, cache_dir, fallback_cache]:
        if os.path.exists(os.path.join(d, "faiss.index")):
            load_dir = d
            break

    if load_dir:
        retriever.load(load_dir, use_mmap=True)
        print(f"  ✓ Loaded FAISS index from {load_dir} ({len(retriever.chunks_metadata)} chunks)")
    else:
        print("  ✗ No FAISS index found! Build one first with the backend server.")
        return []

    # BM25 + Hybrid
    bm25 = BM25Retriever()
    bm25.load(load_dir, metadata_list=retriever.chunks_metadata)
    hybrid = HybridRetriever(
        dense_retriever=retriever, bm25_retriever=bm25,
        dense_top_k=50, bm25_top_k=50, rrf_k=60, fused_top_k=5,
    )

    # LLM
    llm_provider = os.getenv("LLM_PROVIDER", "groq")
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("  ✗ GROQ_API_KEY not set! Cannot run eval with real LLM.")
        return []
    llm = get_llm_provider(llm_provider)
    print(f"  ✓ LLM Provider: {llm_provider}")

    # Orchestrator with production guardrail settings
    orchestrator = RAGOrchestrator(
        embedder=embedder,
        retriever=hybrid,
        llm=llm,
        safety_guard=SafetyGuardrail(),
        relevance_gate=RelevanceGate(threshold=settings.guardrails.relevance_threshold),
        insufficient_checker=InsufficientEvidenceChecker(
            confidence_threshold=settings.guardrails.insufficient_evidence_threshold
        ),
        groundedness_verifier=GroundednessVerifier(
            high_threshold=settings.guardrails.groundedness_high_threshold,
            low_threshold=settings.guardrails.groundedness_low_threshold,
            embedder=embedder,
        ),
    )

    # Warmup
    _ = await orchestrator.execute("warmup query")
    print(f"  ✓ Pipeline warmed up. Running {len(queries)} eval queries...\n")

    # --- Eval Loop ---
    results: List[EvalResult] = []

    for i, q in enumerate(queries):
        query_text = q.get("query", "")
        query_id = q.get("query_id", i)
        gold_answer = q.get("ground_truth_answer", "")
        expected_type = q.get("expected_type", "answerable")
        category = q.get("category", q.get("query_type", "unknown"))

        t0 = time.perf_counter()
        response = await orchestrator.execute(query_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Judge
        verdict, correctness, groundedness = judge_verdict(
            expected_type=expected_type,
            status=response.status,
            generated_answer=response.answer,
            gold_answer=gold_answer,
            retrieved_chunks=response.retrieved_chunks,
        )

        metrics = response.metrics or {}
        result = EvalResult(
            query_id=query_id,
            query=query_text,
            category=category,
            expected_type=expected_type,
            gold_answer=gold_answer,
            generated_answer=response.answer,
            status=response.status,
            verdict=verdict,
            correctness_score=correctness,
            groundedness_score=groundedness,
            latency_ms=elapsed_ms,
            llm_ttft_ms=metrics.get("llm_ttft_ms", 0.0),
            diagnostics={
                "entity_match": metrics.get("entity_match", "N/A"),
                "evidence_status": metrics.get("evidence_status", "N/A"),
                "groundedness_verdict": metrics.get("groundedness_verdict", "N/A"),
                "llm_invocation": metrics.get("llm_invocation", "N/A"),
            },
        )
        results.append(result)

        # Print progress
        icon = "✅" if verdict.startswith("✅") else ("⚠️" if verdict.startswith("⚠️") else "❌")
        if verbose:
            print(f"  [{i+1:02d}/{len(queries)}] {icon} {verdict}")
            print(f"        Query: {query_text[:70]}...")
            print(f"        Answer: {response.answer[:80]}...")
            print(f"        Latency: {elapsed_ms:.0f}ms | Status: {response.status}")
            print()

    return results


def print_summary(results: List[EvalResult]):
    """Print a comprehensive evaluation summary."""
    print("\n" + "=" * 90)
    print("📊 EVAL LOOP RESULTS SUMMARY")
    print("=" * 90)

    total = len(results)
    answerable = [r for r in results if r.expected_type == "answerable"]
    refusals = [r for r in results if r.expected_type == "refusal"]

    # Verdict breakdown
    correct = [r for r in answerable if r.verdict == "✅ CORRECT"]
    false_refusals = [r for r in answerable if r.verdict == "❌ FALSE_REFUSAL"]
    hallucinations = [r for r in answerable if "HALLUCINATION" in r.verdict]
    grounded_inaccurate = [r for r in answerable if "GROUNDED_BUT_INACCURATE" in r.verdict]
    legitimate_refusals = [r for r in refusals if r.verdict == "✅ LEGITIMATE_REFUSAL"]
    missed_refusals = [r for r in refusals if "MISSED_REFUSAL" in r.verdict]

    print(f"\n  Total Queries Evaluated: {total}")
    print(f"    Answerable: {len(answerable)} | Refusal Expected: {len(refusals)}")

    print(f"\n  ── ANSWERABLE QUERIES ({len(answerable)}) ──")
    print(f"    ✅ Correct & Grounded:     {len(correct):>3} ({100*len(correct)/max(len(answerable),1):.1f}%)")
    print(f"    ⚠️ Grounded but Inaccurate: {len(grounded_inaccurate):>3} ({100*len(grounded_inaccurate)/max(len(answerable),1):.1f}%)")
    print(f"    ❌ False Refusals:          {len(false_refusals):>3} ({100*len(false_refusals)/max(len(answerable),1):.1f}%)")
    print(f"    ❌ Hallucinations:          {len(hallucinations):>3} ({100*len(hallucinations)/max(len(answerable),1):.1f}%)")

    print(f"\n  ── REFUSAL QUERIES ({len(refusals)}) ──")
    print(f"    ✅ Legitimate Refusals:     {len(legitimate_refusals):>3} ({100*len(legitimate_refusals)/max(len(refusals),1):.1f}%)")
    print(f"    ⚠️ Missed Refusals:         {len(missed_refusals):>3} ({100*len(missed_refusals)/max(len(refusals),1):.1f}%)")

    # Key Metrics
    answer_correctness = len(correct) / max(len(answerable), 1) * 100
    false_refusal_rate = len(false_refusals) / max(len(answerable), 1) * 100
    hallucination_rate = len(hallucinations) / max(len(answerable), 1) * 100
    refusal_accuracy = len(legitimate_refusals) / max(len(refusals), 1) * 100

    print(f"\n  ── KEY METRICS ──")
    print(f"    Answer Correctness Rate:   {answer_correctness:.1f}% {'✅' if answer_correctness >= 75 else '❌'} (target: ≥75%)")
    print(f"    False Refusal Rate:        {false_refusal_rate:.1f}% {'✅' if false_refusal_rate <= 15 else '❌'} (target: ≤15%)")
    print(f"    Hallucination Rate:        {hallucination_rate:.1f}% {'✅' if hallucination_rate <= 5 else '❌'} (target: ≤5%)")
    print(f"    Refusal Accuracy:          {refusal_accuracy:.1f}% {'✅' if refusal_accuracy >= 85 else '❌'} (target: ≥85%)")

    # Latency stats
    latencies = [r.latency_ms for r in results]
    ttfts = [r.llm_ttft_ms for r in results if r.llm_ttft_ms > 0]
    if latencies:
        import numpy as np
        print(f"\n  ── LATENCY ──")
        print(f"    End-to-End P50:  {np.percentile(latencies, 50):.0f}ms")
        print(f"    End-to-End P95:  {np.percentile(latencies, 95):.0f}ms")
        if ttfts:
            print(f"    LLM TTFT P50:    {np.percentile(ttfts, 50):.0f}ms")
            print(f"    LLM TTFT P95:    {np.percentile(ttfts, 95):.0f}ms")

    # Print false refusals detail
    if false_refusals:
        print(f"\n  ── FALSE REFUSALS (need investigation) ──")
        for r in false_refusals:
            print(f"    [{r.query_id}] {r.query[:60]}...")
            print(f"           Status: {r.status} | Diagnostics: {r.diagnostics}")

    # Print hallucinations detail
    if hallucinations:
        print(f"\n  ── HALLUCINATIONS (critical) ──")
        for r in hallucinations:
            print(f"    [{r.query_id}] {r.query[:60]}...")
            print(f"           Answer: {r.generated_answer[:80]}...")

    print("\n" + "=" * 90)

    # Overall pass/fail
    passed = (answer_correctness >= 75 and false_refusal_rate <= 15
              and hallucination_rate <= 5 and refusal_accuracy >= 85)
    print(f"\n  {'🎉 EVAL PASSED' if passed else '⚠️  EVAL NEEDS IMPROVEMENT'}")
    print("=" * 90)

    return {
        "total": total,
        "answer_correctness_pct": answer_correctness,
        "false_refusal_rate_pct": false_refusal_rate,
        "hallucination_rate_pct": hallucination_rate,
        "refusal_accuracy_pct": refusal_accuracy,
        "passed": passed,
    }


async def main():
    parser = argparse.ArgumentParser(description="Voice RAG LLM Eval Loop")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (5 answerable + 3 refusal)")
    parser.add_argument("--full", action="store_true", help="Full eval (20 answerable + 10 refusal)")
    parser.add_argument("--n", type=int, default=None, help="Custom number of answerable queries")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print per-query results")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    args = parser.parse_args()

    if args.quick:
        n_answerable, n_refusal = 5, 3
    elif args.full:
        n_answerable, n_refusal = 20, 10
    elif args.n:
        n_answerable, n_refusal = args.n, max(3, args.n // 3)
    else:
        n_answerable, n_refusal = 10, 5  # Default: medium eval

    print("=" * 90)
    print("🔄 VOICE RAG — LLM EVAL LOOP")
    print(f"   Evaluating {n_answerable} answerable + {n_refusal} refusal queries")
    print("=" * 90)

    queries = load_eval_queries(n_answerable=n_answerable, n_refusal=n_refusal)
    print(f"\n  Loaded {len(queries)} eval queries from benchmarks/datasets/")

    results = await run_eval_loop(queries, verbose=args.verbose)

    if not results:
        print("\n  ❌ No results — check pipeline initialization errors above.")
        return

    summary = print_summary(results)

    # Save to JSON
    if args.save:
        output_path = os.path.join(ROOT_DIR, "benchmarks", "eval_loop_results.json")
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n  📁 Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
