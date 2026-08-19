"""
Migration script: Build and cache FAISS-HNSW index for BAAI/bge-m3 (1024d)
while preserving MiniLM cache for rollback.
"""

import os
import sys
import shutil
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.rag.ingest import load_passages_from_jsonl
from backend.app.rag.chunker import chunk_corpus
from backend.app.rag.retriever import FAISSHNSWRetriever

CORPUS_PATH = os.path.join(ROOT_DIR, "backend", "data", "passages.jsonl")
EMBS_PATH = os.path.join(ROOT_DIR, "benchmarks", "experiments", "cache", "bge_m3_full_93k_embs.npy")

DATA_DIR = os.path.join(ROOT_DIR, "backend", "data")
FAISS_CACHE_ACTIVE = os.path.join(DATA_DIR, "faiss_cache")
FAISS_CACHE_MINILM = os.path.join(DATA_DIR, "faiss_cache_minilm")
FAISS_CACHE_BGEM3 = os.path.join(DATA_DIR, "faiss_cache_bge_m3")


def migrate():
    print("=" * 80)
    print("MIGRATING FAISS VECTOR STORE TO BAAI/bge-m3 (1024 dimensions)")
    print("=" * 80)

    # 1. Preserve existing MiniLM cache for rollback
    if os.path.exists(FAISS_CACHE_ACTIVE) and not os.path.exists(FAISS_CACHE_MINILM):
        print(f"Backing up existing MiniLM cache to: {FAISS_CACHE_MINILM}...")
        shutil.copytree(FAISS_CACHE_ACTIVE, FAISS_CACHE_MINILM)
        print("  [OK] MiniLM rollback cache preserved.")

    # 2. Load 50,000 passages and generate 93,621 fixed chunks
    print(f"Loading passages from: {CORPUS_PATH}...")
    passages = load_passages_from_jsonl(CORPUS_PATH)
    chunks = chunk_corpus(passages, strategy="fixed")
    print(f"  [OK] Prepared {len(chunks)} fixed chunks.")

    # 3. Load pre-computed BGE-M3 93k embeddings
    print(f"Loading BGE-M3 embeddings from: {EMBS_PATH}...")
    embs = np.load(EMBS_PATH)
    print(f"  [OK] Loaded embeddings array shape: {embs.shape} (dtype: {embs.dtype})")

    # 4. Build FAISS-HNSW index (1024d, M=32, efSearch=64)
    print("Building FAISS IndexHNSWFlat (dimension=1024, M=32, efSearch=64)...")
    retriever = FAISSHNSWRetriever(dimension=1024, m=32, ef_search=64)
    retriever.index(chunks, embs)
    print("  [OK] Index successfully built.")

    # 5. Save to BGE-M3 dedicated cache and active cache
    print(f"Saving BGE-M3 index to: {FAISS_CACHE_BGEM3}...")
    retriever.save(FAISS_CACHE_BGEM3)

    print(f"Updating active cache at: {FAISS_CACHE_ACTIVE}...")
    retriever.save(FAISS_CACHE_ACTIVE)

    # 6. Verify index reload
    print("Verifying active index reload...")
    verify_retriever = FAISSHNSWRetriever(dimension=1024, m=32, ef_search=64)
    verify_retriever.load(FAISS_CACHE_ACTIVE)
    print(f"  [OK] Reload verified: {verify_retriever.index_instance.ntotal} vectors, {len(verify_retriever.chunks_metadata)} metadata records.")
    print("=" * 80)
    print("MIGRATION COMPLETE: Active FAISS cache is now BGE-M3 (1024-d).")
    print("=" * 80)


if __name__ == "__main__":
    migrate()
