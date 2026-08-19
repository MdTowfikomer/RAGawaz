import os
import sys
import time
import asyncio
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except Exception:
    pass

from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.rag.bm25_retriever import BM25Retriever
from backend.app.rag.hybrid_retriever import HybridRetriever
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier
from backend.app.harness.providers.groq import GroqLLMProvider
from backend.app.harness.orchestrator import RAGOrchestrator


async def test_minilm_latency():
    print("Initializing MiniLM RAG Pipeline...", flush=True)
    embedder = get_embedding_provider("minilm")
    
    minilm_dir = ROOT_DIR / "backend" / "data" / "faiss_cache_minilm"
    dense_retriever = FAISSHNSWRetriever(dimension=384, m=32, ef_search=128)
    dense_retriever.load(str(minilm_dir), use_mmap=True)
    
    bm25 = BM25Retriever()
    bm25.load(str(minilm_dir), metadata_list=dense_retriever.chunks_metadata)
    
    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25,
        dense_top_k=100,
        bm25_top_k=100,
        rrf_k=60,
        fused_top_k=7,
    )
    
    llm = GroqLLMProvider()
    
    orchestrator = RAGOrchestrator(
        embedder=embedder,
        retriever=hybrid,
        llm=llm,
        safety_guard=SafetyGuardrail(),
        relevance_gate=RelevanceGate(threshold=0.25),
        insufficient_checker=InsufficientEvidenceChecker(confidence_threshold=0.28),
        groundedness_verifier=GroundednessVerifier(embedder=embedder),
    )
    
    # Warmup
    _ = embedder.embed_query("warmup")
    
    queries = [
        "कॉर्पोरेशन क्या है?",
        "स्टबहब टोल फ्री नंबर",
        "ईमानदारी या सच्चाई की परिभाषा",
    ]
    
    print("\nExecuting Queries with MiniLM Pipeline:\n")
    for q in queries:
        t0 = time.perf_counter()
        resp = await orchestrator.execute(q)
        t_tot = (time.perf_counter() - t0) * 1000.0
        m = resp.metrics
        print(f"Query: {q}")
        print(f"  ├─ Embed: {m.get('embedding_ms', 0):.1f} ms")
        print(f"  ├─ FAISS: {m.get('faiss_ms', 0):.2f} ms | BM25: {m.get('bm25_ms', 0):.2f} ms")
        print(f"  ├─ Pre-LLM Boundary: {m.get('pre_llm_total_ms', 0):.1f} ms")
        print(f"  ├─ LLM TTFT: {m.get('llm_ttft_ms', 0):.1f} ms")
        print(f"  ├─ Total Latency: {t_tot:.1f} ms")
        print(f"  └─ Answer: {resp.answer[:120]}...\n")


if __name__ == "__main__":
    asyncio.run(test_minilm_latency())
