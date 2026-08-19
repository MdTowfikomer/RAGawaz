"""
Phase 4B: Real Text-to-Answer Latency & Quality Benchmark.

Evaluates the text pipeline independently from audio using a REAL LLM (Groq LLaMA 3.1 8B / 3.3 70B).
Measures:
- query_embedding_ms
- vector_search_ms
- reranking_ms
- embed_retrieval_ms
- guardrails_ms
- llm_ttft_ms (Time to First Token)
- llm_total_ms
- grounding_ms
- text_to_answer_ms (Total time from transcript to grounded answer)

Evaluates:
- Text->Answer P50 / P70 / P95 / MAX
- LLM TTFT P50 / P70 / MAX
- Grounded answer rate (Target: >=90%)
- Refusal accuracy (Target: >=95%)
- Text->Answer P70 < 200ms Gate
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from backend.app.rag.ingest import load_passages_from_jsonl
from backend.app.rag.chunker import chunk_corpus
from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier
from backend.app.harness.providers.groq import GroqLLMProvider
from backend.app.harness.orchestrator import RAGOrchestrator, HarnessResponse


async def benchmark_text_to_answer():
    voice_dir = os.path.join(ROOT_DIR, "benchmarks", "voice")
    os.makedirs(voice_dir, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 4B: REAL TEXT-TO-ANSWER BENCHMARK (GROQ LLaMA 3.1 LIVE LLM)")
    print("=" * 80)
    
    # 1. Initialize RAG Components
    embedder = get_embedding_provider("minilm")
    faiss_dir = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache")
    retriever = FAISSHNSWRetriever()
    retriever.load(faiss_dir)
    
    # Live Groq Provider
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[FAIL] GROQ_API_KEY not found in environment!")
        return None
        
    llm = GroqLLMProvider(api_key=groq_key, model_id="llama-3.1-8b-instant")
    
    orchestrator = RAGOrchestrator(
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        safety_guard=SafetyGuardrail(),
        relevance_gate=RelevanceGate(threshold=0.25),
        insufficient_checker=InsufficientEvidenceChecker(confidence_threshold=0.28),
        groundedness_verifier=GroundednessVerifier(high_threshold=0.20, low_threshold=0.10),
    )
    
    # 2. Warmup
    print("Warming up embedder & live Groq connection...")
    _ = await orchestrator.execute("नमस्ते, भारत की राजधानी क्या है?")
    
    # 3. Load 30 deterministic test queries (20 canonical QA, 4 offtopic, 3 insufficient, 3 safety)
    datasets_dir = os.path.join(ROOT_DIR, "benchmarks", "datasets")
    
    queries: List[Dict[str, Any]] = []
    
    # 20 Canonical QA
    with open(os.path.join(datasets_dir, "canonical_queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                item["expected_type"] = "answerable"
                queries.append(item)
                if len(queries) >= 20:
                    break
                    
    # 4 Off-topic
    with open(os.path.join(datasets_dir, "offtopic_queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                item["expected_type"] = "refusal"
                queries.append(item)
                if len([q for q in queries if q["expected_type"] == "refusal"]) >= 4:
                    break

    # 3 Insufficient
    with open(os.path.join(datasets_dir, "insufficient_evidence_queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                item["expected_type"] = "refusal"
                queries.append(item)
                if len([q for q in queries if q.get("category") == "insufficient_evidence"]) >= 3:
                    break

    # 3 Safety
    with open(os.path.join(datasets_dir, "safety_queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                item["expected_type"] = "refusal"
                queries.append(item)
                if len([q for q in queries if q.get("category") == "safety"]) >= 3:
                    break
                    
    print(f"Running benchmark across {len(queries)} deterministic test queries...\n")
    print(f"{'#':<3} | {'Query (first 28 chars)':<30} | {'Embed (ms)':<10} | {'TTFT (ms)':<10} | {'Text->Ans (ms)':<14} | {'Status':<12}")
    print("-" * 85)
    
    results = []
    
    for i, q in enumerate(queries):
        query_text = q["query"]
        resp = await orchestrator.execute(query_text, mode="strict")
        m = resp.metrics or {}
        
        results.append({
            "query_id": q.get("query_id", i+1),
            "query": query_text,
            "category": q.get("category", "canonical"),
            "expected_type": q["expected_type"],
            "status": resp.status,
            "answer": resp.answer[:120],
            "groundedness_score": resp.groundedness_score,
            "query_embedding_ms": m.get("query_embedding_ms", 0.0),
            "vector_search_ms": m.get("vector_search_ms", 0.0),
            "reranking_ms": 0.0,
            "embed_retrieval_ms": m.get("embed_retrieval_ms", 0.0),
            "guardrails_ms": m.get("guardrails_ms", 0.0),
            "llm_ttft_ms": m.get("llm_ttft_ms", 0.0),
            "llm_total_ms": m.get("llm_total_ms", 0.0),
            "grounding_ms": m.get("grounding_ms", 0.0),
            "text_to_answer_ms": m.get("text_to_answer_ms", m.get("harness_ms", 0.0)),
        })
        
        q_disp = query_text[:28]
        print(f"{i+1:<3} | {q_disp:<30} | {m.get('embed_retrieval_ms', 0.0):8.2f} ms | {m.get('llm_ttft_ms', 0.0):8.2f} ms | {m.get('text_to_answer_ms', 0.0):12.2f} ms | {resp.status:<12}")

    # Compute Statistics
    all_text_ans = [r["text_to_answer_ms"] for r in results]
    answerable_res = [r for r in results if r["expected_type"] == "answerable"]
    refusal_res = [r for r in results if r["expected_type"] == "refusal"]
    
    ttfts = [r["llm_ttft_ms"] for r in answerable_res if r["llm_ttft_ms"] > 0]
    embed_retrievals = [r["embed_retrieval_ms"] for r in results if r["embed_retrieval_ms"] > 0]
    
    # Groundedness & Refusal Accuracy
    grounded_count = sum(1 for r in answerable_res if r["status"] == "success")
    groundedness_rate = (grounded_count / len(answerable_res)) * 100.0 if answerable_res else 100.0
    
    refusal_count = sum(1 for r in refusal_res if r["status"].startswith("refusal"))
    refusal_acc = (refusal_count / len(refusal_res)) * 100.0 if refusal_res else 100.0
    
    t_p50 = float(np.percentile(all_text_ans, 50))
    t_p70 = float(np.percentile(all_text_ans, 70))
    t_p95 = float(np.percentile(all_text_ans, 95))
    t_max = float(np.max(all_text_ans))
    
    ttft_p50 = float(np.percentile(ttfts, 50)) if ttfts else 0.0
    ttft_p70 = float(np.percentile(ttfts, 70)) if ttfts else 0.0
    ttft_max = float(np.max(ttfts)) if ttfts else 0.0
    
    emb_p50 = float(np.percentile(embed_retrievals, 50)) if embed_retrievals else 0.0
    emb_p70 = float(np.percentile(embed_retrievals, 70)) if embed_retrievals else 0.0

    print("-" * 85)
    print("PHASE 4B TEXT-TO-ANSWER BENCHMARK REPORT:")
    print(f"  Embed + Retrieval: P50 = {emb_p50:.2f} ms | P70 = {emb_p70:.2f} ms (Target: <50ms)")
    print(f"  Live Groq LLM TTFT: P50 = {ttft_p50:.2f} ms | P70 = {ttft_p70:.2f} ms | MAX = {ttft_max:.2f} ms")
    print(f"  Text->Answer Latency: P50 = {t_p50:.2f} ms | P70 = {t_p70:.2f} ms | P95 = {t_p95:.2f} ms | MAX = {t_max:.2f} ms")
    print(f"  Groundedness Rate: {groundedness_rate:.2f}% (Target: >=90%)")
    print(f"  Guardrail Refusal Accuracy: {refusal_acc:.2f}% (Target: >=95%)")
    print(f"  Text->Answer P70 < 200ms Gate: {'✅ PASS' if t_p70 < 200.0 else '❌ FAIL'}")
    print("=" * 85)
    
    payload = {
        "timestamp": time.time(),
        "llm_provider": "groq",
        "llm_model": "llama-3.1-8b-instant",
        "sample_count": len(results),
        "metrics": {
            "embed_retrieval_p50_ms": emb_p50,
            "embed_retrieval_p70_ms": emb_p70,
            "llm_ttft_p50_ms": ttft_p50,
            "llm_ttft_p70_ms": ttft_p70,
            "llm_ttft_max_ms": ttft_max,
            "text_to_answer_p50_ms": t_p50,
            "text_to_answer_p70_ms": t_p70,
            "text_to_answer_p95_ms": t_p95,
            "text_to_answer_max_ms": t_max,
            "groundedness_rate_pct": groundedness_rate,
            "refusal_accuracy_pct": refusal_acc,
            "gate_p70_under_200ms": t_p70 < 200.0,
        },
        "records": results,
    }
    
    out_file = os.path.join(voice_dir, "text_answer_baseline.json")
    with open(out_file, "w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, ensure_ascii=False, indent=2)
    print(f"Saved Text->Answer baseline artifact to: {out_file}\n")
    return payload


if __name__ == "__main__":
    asyncio.run(benchmark_text_to_answer())
