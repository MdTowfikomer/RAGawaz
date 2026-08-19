import os
import sys
import io
import time
import json
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
from backend.app.voice.pipeline import VoiceRAGPipeline, SarvamVoiceService


async def test_voice_answerable_latency():
    print("=" * 90)
    print("TESTING VOICE PIPELINE (ANSWERABLE QUERIES + LATENCY BREAKDOWN)")
    print("=" * 90)

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
    
    voice_service = SarvamVoiceService()
    pipeline = VoiceRAGPipeline(orchestrator=orchestrator, voice_service=voice_service)
    
    # Warmup
    _ = embedder.embed_query("warmup")
    
    test_queries = [
        "मालवाहक जहाज़ के नीचे की तरफ",
        "लिंकन में अब वायुमंडलीय दबाव क्या है?",
        "क्या चिकित्सीय मारिजुआना मदत करता है?",
        "स्ट्रूथर्स शहर स्कूल जिला राज्य संख्या",
    ]
    
    for i, q in enumerate(test_queries, 1):
        print(f"\n[{i}/3] Testing Answerable Query: '{q}'")
        t_start = time.perf_counter()
        result = await pipeline.process_text_query(q)
        t_end = time.perf_counter()
        
        telemetry = result.get("telemetry", {})
        answer = result.get("answer", "")
        has_audio = bool(result.get("audio_base64"))
        
        print(f"  ├─ Status: {result.get('status')}")
        print(f"  ├─ Embedding Latency: {telemetry.get('embedding_ms', 0):.1f} ms")
        print(f"  ├─ FAISS-HNSW Vector Search: {telemetry.get('faiss_ms', 0):.2f} ms")
        print(f"  ├─ BM25 Sparse Search: {telemetry.get('bm25_ms', 0):.2f} ms")
        print(f"  ├─ Pre-LLM Boundary: {telemetry.get('pre_llm_total_ms', 0):.1f} ms (Target <200ms)")
        print(f"  ├─ LLM Time-To-First-Token (TTFT): {telemetry.get('llm_ttft_ms', 0):.1f} ms")
        print(f"  ├─ LLM Generation Total: {telemetry.get('llm_total_ms', 0):.1f} ms")
        print(f"  ├─ Grounding Verifier: {telemetry.get('grounding_ms', 0):.2f} ms")
        print(f"  ├─ Text -> Answer Total: {telemetry.get('text_to_answer_ms', 0):.1f} ms")
        print(f"  ├─ TTS Speech Synthesis: {telemetry.get('tts_first_audio_ms', 0):.1f} ms")
        print(f"  ├─ Total Voice Pipeline Latency: {telemetry.get('voice_pipeline_ms', 0):.1f} ms")
        print(f"  ├─ Audio Bytes Generated: {has_audio}")
        print(f"  └─ Answer Text: {answer[:130]}...\n")
        print("-" * 90)


if __name__ == "__main__":
    asyncio.run(test_voice_answerable_latency())
