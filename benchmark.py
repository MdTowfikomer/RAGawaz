"""Measure end-to-end retrieval latency (embed + FAISS search) against the
50ms budget defined in app/config.py.

Usage:
    python benchmark.py [n_queries]
"""
import os
import sys
import time
import statistics
import torch
from dataclasses import dataclass
from typing import List, Any, Optional

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Set Windows console utf-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.app.config import settings, EMBEDDING_PROFILES
from backend.app.rag.embedder import get_embedding_provider, EmbeddingProvider
from backend.app.rag.retriever import FAISSHNSWRetriever

LATENCY_BUDGET_MS = getattr(settings.latencies, "embed_retrieval_target_ms", 50.0)

QUERIES = [
    "What is FAISS used for?",
    "How does HNSW indexing work?",
    "What is retrieval augmented generation?",
    "Which embedding model is fast on CPU?",
    "How do you reduce RAG latency?",
    "What does efSearch control?",
    "Why normalize embeddings before indexing?",
    "What are the stages of a RAG pipeline?",
]

_embedder: Optional[EmbeddingProvider] = None
_retriever: Optional[FAISSHNSWRetriever] = None


@dataclass
class SearchResponse:
    chunks: List[Any]
    embed_ms: float
    search_ms: float
    total_ms: float


def warmup():
    """Load embedding model, FAISS index, and perform first inference warmup."""
    global _embedder, _retriever

    model_key = settings.embedding_model
    profile = EMBEDDING_PROFILES.get(model_key, EMBEDDING_PROFILES["bge_m3"])
    dim = profile["dimension"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _embedder = get_embedding_provider(model_key, device=device)
    _retriever = FAISSHNSWRetriever(dimension=dim, m=32, ef_search=64)

    bundle_dir = os.path.join(ROOT_DIR, "backend", "data", "multilingual_index_bundle")
    cache_dir = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache_bge_m3")
    fallback_cache = os.path.join(ROOT_DIR, "backend", "data", "faiss_cache")

    target_load_dir = None
    if os.path.exists(os.path.join(bundle_dir, "faiss.index")):
        target_load_dir = bundle_dir
    elif os.path.exists(os.path.join(cache_dir, "faiss.index")):
        target_load_dir = cache_dir
    elif os.path.exists(os.path.join(fallback_cache, "faiss.index")):
        target_load_dir = fallback_cache

    if target_load_dir:
        # Load with memory-mapped pages and fast pickle metadata
        _retriever.load(target_load_dir, use_mmap=True)

    # First inference warmup
    warmup_q = "Warmup query for model initialization"
    emb = _embedder.embed_query(warmup_q)
    _retriever.search(emb, top_k=2)


def search(query: str, top_k: int = 5) -> SearchResponse:
    """Execute end-to-end embed + retrieval and record timing."""
    global _embedder, _retriever
    if _embedder is None or _retriever is None:
        warmup()

    t0 = time.perf_counter()
    emb = _embedder.embed_query(query)
    t1 = time.perf_counter()

    chunks = _retriever.search(emb, top_k=top_k)
    t2 = time.perf_counter()

    embed_ms = (t1 - t0) * 1000.0
    search_ms = (t2 - t1) * 1000.0
    total_ms = (t2 - t0) * 1000.0

    return SearchResponse(
        chunks=chunks,
        embed_ms=embed_ms,
        search_ms=search_ms,
        total_ms=total_ms,
    )


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("Warming up (model load + first inference)...")
    warmup()

    total_ms, embed_ms, search_ms = [], [], []
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        resp = search(query, top_k=5)
        total_ms.append(resp.total_ms)
        embed_ms.append(resp.embed_ms)
        search_ms.append(resp.search_ms)

    print(f"\nRan {n} queries\n")
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [("embed", embed_ms), ("search", search_ms), ("total", total_ms)]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nLatency budget: {LATENCY_BUDGET_MS}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= LATENCY_BUDGET_MS:
        print("PASS: within budget")
    else:
        print("FAIL: over budget -- see README 'Tuning latency' section")
        sys.exit(1)


if __name__ == "__main__":
    main()
