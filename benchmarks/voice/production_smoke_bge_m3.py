"""
Production Smoke Validation against Live RAG Stack & Groq LLM
Evaluating canonical difficult queries on active BGE-M3 (1024d) index.
"""

import os
import sys
import io

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import time
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.config import settings, EMBEDDING_PROFILES
from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier
from backend.app.harness.providers import get_llm_provider
from backend.app.harness.orchestrator import RAGOrchestrator, RAGRequest

TARGET_QUERIES = [
    {
        "name": "QID 1060361 (Barter)",
        "query": "वस्तु विनिमय में पहला क्या था",
        "gold_pid": "hi_1060361_3",
    },
    {
        "name": "Rachel Carson / Obligation to Endure",
        "query": "रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा",
        "gold_pid": "hi_1102431_4",
    },
    {
        "name": "Cantaloupe / Cucumber Mismatch",
        "query": "कैंटालूप को कितने समय तक परिपक्व होना है",
        "gold_pid": "hi_1081829_0",
    },
    {
        "name": "Basal DNA",
        "query": "बेसल डीएनए क्या है?",
        "gold_pid": "hi_1066765_0",
    },
    {
        "name": "Arbitrary",
        "query": "मनमाना का क्या अर्थ है?",
        "gold_pid": "hi_1052843_0",
    },
    {
        "name": "English -> Hindi (Corporation)",
        "query": "What is a corporation?",
        "gold_pid": "hi_1048528_0",
    },
    {
        "name": "Hinglish -> Hindi (Rachel Carson)",
        "query": "Rachel Carson ne Obligation to Endure kyu likha tha?",
        "gold_pid": "hi_1102431_4",
    },
    {
        "name": "Marathi -> Hindi (Honesty)",
        "query": "प्रामाणिकपणा किंवा सत्याची व्याख्या काय आहे?",
        "gold_pid": "hi_1047648_0",
    },
    {
        "name": "Tamil -> Hindi (Falcon Speed)",
        "query": "பால்கன் பறவை எவ்வளவு வேகமாக பறக்கும்?",
        "gold_pid": "hi_1079313_0",
    },
    {
        "name": "Bengali -> Hindi (Corporation)",
        "query": "কর্পোরেশন কি?",
        "gold_pid": "hi_1048528_0",
    },
]


async def run_smoke():
    print("=" * 90)
    print("RUNNING PRODUCTION LIVE SMOKE VALIDATION (BGE-M3 1024d)")
    print(f"Embedding Model: {settings.embedding_model} ({settings.embedding_dim}-d)")
    print(f"Retriever: {settings.retriever_backend}, top_k={settings.default_top_k}")
    print("=" * 90)

    cache_dir = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache")
    device = "cuda"

    embedder = get_embedding_provider(settings.embedding_model, device=device)
    retriever = FAISSHNSWRetriever(dimension=settings.embedding_dim, m=32, ef_search=64)
    retriever.load(cache_dir)
    print(f"Loaded active FAISS index: {retriever.index_instance.ntotal} vectors from {cache_dir}.")

    llm = get_llm_provider("groq" if os.getenv("GROQ_API_KEY") else "mock")

    orchestrator = RAGOrchestrator(
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        safety_guard=SafetyGuardrail(),
        relevance_gate=RelevanceGate(threshold=settings.guardrails.relevance_threshold),
        insufficient_checker=InsufficientEvidenceChecker(confidence_threshold=settings.guardrails.insufficient_evidence_threshold),
        groundedness_verifier=GroundednessVerifier(
            high_threshold=settings.guardrails.groundedness_high_threshold,
            low_threshold=settings.guardrails.groundedness_low_threshold,
            embedder=embedder,
        ),
    )

    results = []
    print("\nExecuting canonical difficult queries:\n")
    for item in TARGET_QUERIES:
        q_name = item["name"]
        query = item["query"]
        gold_pid = item["gold_pid"]

        t0 = time.perf_counter()
        req = RAGRequest(query=query, top_k=settings.default_top_k, mode="strict")
        resp = await orchestrator.execute(req)
        t_e2e_ms = (time.perf_counter() - t0) * 1000

        retrieved_pids = [c.get("passage_id", "") if isinstance(c, dict) else getattr(c, "passage_id", "") for c in resp.retrieved_chunks]
        gold_rank = None
        for r_i, p in enumerate(retrieved_pids, 1):
            if p == gold_pid:
                gold_rank = r_i
                break

        top_score = 0.0
        if resp.retrieved_chunks:
            c0 = resp.retrieved_chunks[0]
            top_score = c0.get("score", 0.0) if isinstance(c0, dict) else getattr(c0, "score", 0.0)

        t_ret_ms = resp.metrics.get("embed_retrieval_ms", 0.0) if isinstance(resp.metrics, dict) else getattr(resp.metrics, "embed_retrieval_ms", 0.0)

        res_item = {
            "name": q_name,
            "query": query,
            "gold_pid": gold_pid,
            "gold_rank": gold_rank,
            "status": resp.status,
            "top_score": round(float(top_score), 4),
            "retrieval_ms": round(float(t_ret_ms), 2),
            "e2e_ms": round(float(t_e2e_ms), 2),
            "answer": resp.answer[:120] + "..." if len(resp.answer) > 120 else resp.answer,
        }
        results.append(res_item)

        print(f"[{resp.status.upper():^28}] {q_name:<36} | Ret: {t_ret_ms:>5.1f}ms | E2E: {t_e2e_ms:>5.1f}ms | Top-Score: {top_score:.3f}")
        print(f"   Answer: {res_item['answer']}")
        print("-" * 90)

    # Output JSON summary
    out_path = os.path.join(ROOT_DIR, "benchmarks", "voice", "production_smoke_bge_m3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved smoke test results to: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_smoke())
