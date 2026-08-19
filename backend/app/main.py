"""
FastAPI Backend Server for Voice RAG Application.

Provides REST and Voice endpoints for the frontend dashboard and benchmarks.
"""

import os
import sys
import json
import time
import base64
import torch
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
# Enable project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except ImportError:
    pass

from backend.app.config import settings, EMBEDDING_PROFILES
from backend.app.rag.ingest import load_passages_from_jsonl
from backend.app.rag.chunker import chunk_corpus
from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.rag.bm25_retriever import BM25Retriever
from backend.app.rag.hybrid_retriever import HybridRetriever
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier
from backend.app.harness.providers import get_llm_provider
from backend.app.harness.orchestrator import RAGOrchestrator
from backend.app.voice.pipeline import VoiceRAGPipeline, SarvamVoiceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_rag_pipeline()
    yield


app = FastAPI(title="Voice RAG Backend API", version="3.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline_instance: Optional[VoiceRAGPipeline] = None
corpus_stats: Dict[str, Any] = {}


class QueryRequest(BaseModel):
    query: str
    provider: Optional[str] = "groq"
    top_k: Optional[int] = 3


def init_rag_pipeline():
    """Initialize embedder, FAISS index with corpus, and RAG orchestrator."""
    global pipeline_instance, corpus_stats

    model_key = settings.embedding_model
    profile = EMBEDDING_PROFILES.get(model_key, EMBEDDING_PROFILES["bge_m3"])
    dim = profile["dimension"]
    model_name = profile["model_name"]

    bundle_dir = os.path.join(ROOT_DIR, "backend", "data", "multilingual_index_bundle")
    cache_dir = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache_bge_m3")
    fallback_cache = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = get_embedding_provider(model_key, device=device)
    retriever = FAISSHNSWRetriever(dimension=dim, m=32, ef_search=64)

    minilm_dir = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache_minilm")
    target_load_dir = None
    if model_key == "minilm" and os.path.exists(os.path.join(minilm_dir, "faiss.index")):
        target_load_dir = minilm_dir
    elif os.path.exists(os.path.join(bundle_dir, "faiss.index")) and os.path.exists(os.path.join(bundle_dir, "metadata.json")):
        target_load_dir = bundle_dir
    elif os.path.exists(os.path.join(cache_dir, "faiss.index")) and os.path.exists(os.path.join(cache_dir, "metadata.json")):
        target_load_dir = cache_dir
    elif os.path.exists(os.path.join(fallback_cache, "faiss.index")) and os.path.exists(os.path.join(fallback_cache, "metadata.json")):
        target_load_dir = fallback_cache

    if target_load_dir:
        # Load HNSW-Flat index (accurate scores for proper ranking)
        print(f"Loading FAISS-HNSW index from: {target_load_dir}...", flush=True)
        retriever.load(target_load_dir, use_mmap=True)

        # Load metadata: prefer pickle (3-5s) over JSON (25-30s)
        import pickle
        meta_pkl_path = os.path.join(target_load_dir, "metadata_light.pkl")
        parent_pkl_path = os.path.join(target_load_dir, "parent_texts.pkl")
        meta_json_path = os.path.join(target_load_dir, "metadata.json")
        if os.path.exists(meta_pkl_path):
            print(f"Loading metadata from pickle (fast path)...", flush=True)
            t_meta = time.time()
            with open(meta_pkl_path, "rb") as f:
                retriever.chunks_metadata = pickle.load(f)
            retriever.chunk_id_map = {c["chunk_id"]: c for c in retriever.chunks_metadata}
            # Load parent texts separately
            retriever._parent_text_store = {}
            skip_parents = os.getenv("SKIP_PARENT_TEXTS", "false").lower() == "true"
            if not skip_parents and os.path.exists(parent_pkl_path):
                with open(parent_pkl_path, "rb") as f:
                    retriever._parent_text_store = pickle.load(f)
                print(f"  Parent texts loaded ({len(retriever._parent_text_store):,} entries)", flush=True)
            elif skip_parents:
                print(f"  Skipping parent_texts.pkl (SKIP_PARENT_TEXTS=true, saving ~3.5GB RAM)", flush=True)
            print(f"  Metadata loaded in {time.time()-t_meta:.1f}s ({len(retriever.chunks_metadata):,} chunks)", flush=True)
        elif os.path.exists(meta_json_path):
            print(f"Loading metadata from JSON (slow path)...", flush=True)
            with open(meta_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                retriever.dimension = data.get("dimension", dim)
                raw_chunks = data.get("chunks", [])
                retriever._parent_text_store = {}
                retriever.chunks_metadata = []
                for c in raw_chunks:
                    pt = c.pop("parent_text", None)
                    if pt:
                        retriever._parent_text_store[c["chunk_id"]] = pt
                    retriever.chunks_metadata.append(c)
                del raw_chunks
                retriever.chunk_id_map = {c["chunk_id"]: c for c in retriever.chunks_metadata}

        print(f"[OK] Fast startup complete! Loaded {len(retriever.chunks_metadata):,} chunks across 14 languages.", flush=True)
        corpus_stats = {
            "loaded_passages": len(retriever.chunks_metadata),
            "indexed_chunks": len(retriever.chunks_metadata),
            "embedder": f"{model_name} ({dim}-d)",
            "retriever": "FAISS-HNSW (mmap, Inner Product)",
            "chunk_strategy": "minimal-context (180-220 chars)",
            "status": "cached_mmap",
        }
        print(f"[OK] Fast startup complete! Loaded {len(retriever.chunks_metadata)} chunks across 14 languages.", flush=True)
    else:
        # 2. Build index from JSONL corpus
        passages = []
        passages_path = os.path.join(ROOT_DIR, "backend", "data", "passages.jsonl")
        if os.path.exists(passages_path):
            max_p = int(os.getenv("MAX_PASSAGES", "50000"))
            print(f"Reading {max_p} passages from {passages_path}...", flush=True)
            passages = load_passages_from_jsonl(passages_path)[:max_p]

        if not passages:
            print("No local corpus file found. Initializing built-in multilingual starter knowledge base...", flush=True)

            passages = [
                {"passage_id": "en:1", "text": "Paris is the capital and most populous city of France.", "language": "en"},
                {"passage_id": "en:2", "text": "The boiling point of water is 100 degrees Celsius or 212 degrees Fahrenheit at standard atmospheric pressure.", "language": "en"},
                {"passage_id": "en:3", "text": "The capital of India is New Delhi, which serves as the seat of the government of India.", "language": "en"},
                {"passage_id": "hi:1", "text": "ताजमहल भारत के आगरा शहर में यमुना नदी के तट पर स्थित एक विश्वप्रसिद्ध संगमरमर का मकबरा है।", "language": "hi"},
                {"passage_id": "hi:2", "text": "मुगल साम्राज्य का संस्थापक जहीरुद्दीन मुहम्मद बाबर था जिसने 1526 में पानीपत के प्रथम युद्ध में विजय प्राप्त की थी।", "language": "hi"},
                {"passage_id": "hi:3", "text": "भारत की राजधानी नई दिल्ली है।", "language": "hi"},
                {"passage_id": "ta:1", "text": "சென்னை தமிழ்நாட்டின் தலைநகரமாகும். மெரினா கடற்கரை, கபாலீஸ்வரர் கோவில் ஆகியவை இங்குள்ள முக்கிய சுற்றுலா தலங்கள் ஆகும்.", "language": "ta"},
                {"passage_id": "bn:1", "text": "পশ্চিমবঙ্গের রাজধানী হলো কলকাতা। এটি হুগলি নদীর পূর্ব তীরে অবস্থিত একটি ঐতিহাসিক শহর।", "language": "bn"},
                {"passage_id": "te:1", "text": "భారతదేశ రాజధాని న్యూఢిల్లీ. హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని.", "language": "te"},
                {"passage_id": "mr:1", "text": "महाराष्ट्राची राजधानी मुंबई आहे, जे भारताची आर्थिक राजधानी म्हणून ओळखले जाते.", "language": "mr"},
                {"passage_id": "gu:1", "text": "ગાંધીનગર ગુજરાત રાજ્યનું પાટનગર છે. અમદાવાદ ગુજરાતનું સૌથી મોટું શહેર છે.", "language": "gu"},
                {"passage_id": "kn:1", "text": "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು, ಇದನ್ನು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ.", "language": "kn"},
                {"passage_id": "ml:1", "text": "കേരളത്തിന്റെ തലസ്ഥാനം തിരുവനന്തപുരം ആണ്. കൊച്ചി കേരളത്തിലെ ഒരു പ്രധാന തുറമുഖ നഗരമാണ്.", "language": "ml"},
                {"passage_id": "pa:1", "text": "ਚੰਡੀਗੜ੍ਹ ਪੰਜਾਬ ਅਤੇ ਹਰਿਆਣਾ ਦੋਵਾਂ ਰਾਜਾਂ ਦੀ ਸਾਂਝੀ ਰਾਜਧਾਨੀ ਹੈ।", "language": "pa"},
                {"passage_id": "ur:1", "text": "تاج محل بھارت کے شہر آگرہ میں واقع ہے جسے مغل شہنشاہ شاہ جہاں نے تعمیر کروایا تھا۔", "language": "ur"},
            ]

        print(f"Indexing {len(passages)} multilingual passages on {device} ({model_name})...", flush=True)
        chunks = [{"chunk_id": p["passage_id"], "passage_id": p["passage_id"], "text": p["text"], "language": p.get("language", "en"), "parent_id": p["passage_id"]} for p in passages]
        embs = embedder.embed([c["text"] for c in chunks], batch_size=32)

        retriever.index(chunks, embs)
        corpus_stats = {
            "loaded_passages": len(passages),
            "indexed_chunks": len(chunks),
            "embedder": f"{model_name} ({dim}-d)",
            "retriever": "FAISS-HNSW (Inner Product)",
            "status": "bootstrap_starter",
        }
        print(f"[OK] Bootstrap indexing complete! ({len(chunks)} chunks indexed).", flush=True)


    # Initialize BM25 Sparse Index and Hybrid RRF Retriever
    bm25_retriever = BM25Retriever()
    bm25_pkl_path = os.path.join(target_load_dir, "bm25_wand.pkl") if target_load_dir else None
    
    if bm25_pkl_path and os.path.exists(bm25_pkl_path):
        # Fast path: load pre-built BM25 from pickle (~3s vs 47s rebuild)
        import pickle
        print(f"Loading BM25-WAND index from pickle (fast path)...", flush=True)
        t_bm25 = time.time()
        with open(bm25_pkl_path, "rb") as f:
            bm25_data = pickle.load(f)
        bm25_retriever.num_docs = bm25_data["num_docs"]
        bm25_retriever.avg_doc_len = bm25_data["avg_doc_len"]
        bm25_retriever.doc_lengths = bm25_data["doc_lengths"]
        bm25_retriever.inverted_index = bm25_data["inverted_index"]
        bm25_retriever.term_upper_bounds = bm25_data["term_upper_bounds"]
        bm25_retriever.chunks_metadata = retriever.chunks_metadata
        bm25_retriever.chunk_id_map = retriever.chunk_id_map
        del bm25_data
        print(f"  BM25-WAND loaded in {time.time()-t_bm25:.1f}s ({bm25_retriever.num_docs:,} docs, {len(bm25_retriever.inverted_index):,} terms)", flush=True)
    else:
        # Slow path: build from scratch and save pickle for next time
        if target_load_dir:
            bm25_retriever.load(target_load_dir, metadata_list=retriever.chunks_metadata)
        else:
            bm25_retriever.index(retriever.chunks_metadata)
        
        # Save pickle for instant future loads
        if bm25_pkl_path:
            try:
                import pickle
                print(f"Saving BM25-WAND pickle for instant future loads...", flush=True)
                bm25_save_data = {
                    "num_docs": bm25_retriever.num_docs,
                    "avg_doc_len": bm25_retriever.avg_doc_len,
                    "doc_lengths": bm25_retriever.doc_lengths,
                    "inverted_index": bm25_retriever.inverted_index,
                    "term_upper_bounds": bm25_retriever.term_upper_bounds,
                }
                with open(bm25_pkl_path, "wb") as f:
                    pickle.dump(bm25_save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                del bm25_save_data
                bm25_size = os.path.getsize(bm25_pkl_path) / (1024 * 1024)
                print(f"  Saved bm25_wand.pkl ({bm25_size:.0f} MB) — next startup will be instant!", flush=True)
            except Exception as e:
                print(f"  Warning: Could not save BM25 pickle: {e}", flush=True)

    hybrid_retriever = HybridRetriever(
        dense_retriever=retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=30,
        bm25_top_k=30,
        rrf_k=60,
        fused_top_k=5,
    )

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    llm_provider = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "mock")
    print(f"Initializing LLM Provider: {llm_provider}...", flush=True)
    llm = get_llm_provider(llm_provider)

    orchestrator = RAGOrchestrator(
        embedder=embedder,
        retriever=hybrid_retriever,
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

    # Warmup: multiple queries across languages to resolve JIT compilation,
    # FAISS page faults, and NumPy/torch lazy-init before serving traffic
    _warmup_queries = [
        "भारत की राजधानी क्या है?",          # Hindi
        "What is the capital of India?",        # English
        "இந்தியாவின் தலைநகரம் என்ன?",       # Tamil
        "ভারতের রাজধানী কী?",                  # Bengali
    ]
    for _wq in _warmup_queries:
        _w_emb = embedder.embed_query(_wq)
        _ = hybrid_retriever.search_hybrid(_wq, _w_emb, top_k=3)
    print(f"[OK] Warmup complete ({len(_warmup_queries)} multilingual queries).", flush=True)

    voice_service = SarvamVoiceService()
    pipeline_instance = VoiceRAGPipeline(orchestrator=orchestrator, voice_service=voice_service)
    print("Voice RAG Pipeline initialized successfully.", flush=True)




@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "corpus_stats": corpus_stats,
        "spec_version": "v3.2",
    }


from fastapi.responses import FileResponse, StreamingResponse

# Mount compiled frontend dist for single-port full-stack production serving
dist_dir = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.exists(dist_dir) and os.path.exists(os.path.join(dist_dir, "index.html")):
    assets_dir = os.path.join(dist_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        file_path = os.path.join(dist_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)

        index_path = os.path.join(dist_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)

        raise HTTPException(status_code=404, detail="Not found")


@app.post("/api/query")
async def handle_query(req: QueryRequest):
    if pipeline_instance is None or pipeline_instance.orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    from backend.app.harness.orchestrator import RAGRequest
    rag_req = RAGRequest(query=req.query, top_k=req.top_k or 5)
    response = await pipeline_instance.orchestrator.execute(rag_req)


    return {
        "query": response.query,
        "answer": response.answer,
        "status": response.status,
        "refusal_reason": response.refusal_reason,
        "groundedness_score": response.groundedness_score,
        "retrieved_chunks": response.retrieved_chunks,
        "telemetry": response.metrics or {},
    }


@app.post("/api/query/stream")
async def handle_query_stream(req: QueryRequest):
    if pipeline_instance is None or pipeline_instance.orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    async def event_generator():
        async for chunk in pipeline_instance.orchestrator.execute_stream(req.query):
            event_type = chunk.get("event", "message")
            data_str = json.dumps(chunk.get("data", {}), ensure_ascii=False)
            yield f"event: {event_type}\ndata: {data_str}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



@app.post("/api/voice/stt")
async def handle_voice_stt(file: UploadFile = File(...), language_code: Optional[str] = Form("auto")):
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    audio_bytes = await file.read()
    result = await pipeline_instance.transcribe_only(
        audio_bytes,
        language_code=language_code or "auto",
        filename=file.filename or "recording.webm",
        content_type=file.content_type or "audio/webm"
    )
    return result



@app.post("/api/voice/process")
async def handle_voice_upload(file: UploadFile = File(...), language_code: Optional[str] = Form("auto")):
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    audio_bytes = await file.read()
    result = await pipeline_instance.process_voice_audio(audio_bytes, language_code=language_code or "auto")
    return result


@app.get("/api/benchmark/results")
def get_benchmark_results():
    benchmark_dir = os.path.join(ROOT_DIR, "benchmarks")
    results_path = os.path.join(benchmark_dir, "benchmark_results.json")
    if not os.path.exists(results_path):
        return {"message": "Benchmark not executed yet."}

    with open(results_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    report_files = {
        "final_benchmark_report": "final_benchmark_report.json",
        "retrieval_comparison": os.path.join("experiments", "phase8_9_10_comparison_report.json"),
        "multilingual_matrix": os.path.join("experiments", "phase6d_matrix_results.json"),
        "voice_validation": os.path.join("voice", "phase_4e_validation_report.json"),
    }
    reports = {}
    for key, relative_path in report_files.items():
        report_path = os.path.join(benchmark_dir, relative_path)
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                reports[key] = json.load(f)

    return {"summary": summary, **reports}


# Mount compiled frontend dist for single-port full-stack production serving
dist_dir = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.exists(dist_dir) and os.path.exists(os.path.join(dist_dir, "index.html")):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")
