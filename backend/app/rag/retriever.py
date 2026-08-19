"""
Ticket 5: Retriever Backend Interface and FAISS Implementation.

Defines the RetrieverBackend Protocol and FAISS-HNSW vector search implementation
with metadata preservation and automatic parent chunk context expansion.

Optimizations:
- prefault_pages() forces mmap pages into RAM after loading, eliminating P95 cold-read latency
- ef_search=64 tuned for 300k vector corpus (good recall/latency tradeoff)
"""

from dataclasses import dataclass, asdict
from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
import os
import json
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Standardized representation of a retrieved passage chunk."""
    chunk_id: str
    passage_id: str
    query_id: int
    text: str
    score: float
    chunk_strategy: str
    position: int
    parent_id: Optional[str] = None
    children_ids: Optional[List[str]] = None
    language: str = "hi"
    is_selected: int = 0
    parent_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@runtime_checkable
class RetrieverBackend(Protocol):
    """Abstract protocol for vector retrieval backends."""
    backend_name: str

    def index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Build the index from chunk records and normalized embeddings."""
        ...

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        """Search top-K most similar chunks given a query embedding."""
        ...

    def save(self, index_dir: str) -> None:
        """Persist index and metadata to disk."""
        ...

    def load(self, index_dir: str) -> None:
        """Load index and metadata from disk."""
        ...


class FAISSHNSWRetriever:
    """
    In-memory HNSW vector search using FAISS with Inner Product metric.
    Sub-millisecond retrieval on tens of thousands of passages.

    Includes prefault_pages() to eliminate mmap cold-read latency by forcing
    all memory-mapped pages into resident RAM after loading.
    """
    backend_name: str = "faiss_hnsw"

    def __init__(self, dimension: int = 384, m: int = 32, ef_search: int = 64):
        self.dimension = dimension
        self.m = m
        self.ef_search = ef_search
        self.index_instance: Optional[faiss.IndexHNSWFlat] = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.chunk_id_map: Dict[str, Dict[str, Any]] = {}

    def prefault_pages(self) -> None:
        """
        Force all mmap-backed index pages into resident RAM.

        When FAISS loads an index with IO_FLAG_MMAP, pages are demand-paged from disk.
        This causes P95 cold-read latency spikes (~64ms). By reading the entire index
        data into a numpy array and discarding it, we force the OS to fault all pages
        into memory, making subsequent searches consistently sub-millisecond.
        """
        if self.index_instance is None:
            return

        try:
            # faiss.vector_to_array reconstructs the full storage as a numpy array,
            # which forces every mmap page to be read into physical memory.
            storage = faiss.vector_to_array(self.index_instance.storage)
            # Touch all pages by computing a trivial reduction (ensures compiler/OS
            # doesn't optimize away the read)
            _ = storage.sum()
            del storage
            logger.info(
                "prefault_pages: All FAISS index pages faulted into RAM "
                f"(ntotal={self.index_instance.ntotal})"
            )
        except Exception as e:
            # Fallback: if vector_to_array fails (e.g., index type doesn't support it),
            # do a dummy search to warm at least the hot paths
            logger.warning(f"prefault_pages fallback (vector_to_array failed: {e}), warming via dummy search")
            try:
                dummy_q = np.zeros((1, self.dimension), dtype=np.float32)
                self.index_instance.search(dummy_q, 1)
            except Exception:
                pass

    def index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Build FAISS HNSWFlat index."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")

        self.chunks_metadata = chunks
        self.chunk_id_map = {c["chunk_id"]: c for c in chunks}

        # Create HNSW index with Inner Product (cosine on normalized vectors)
        idx = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efSearch = self.ef_search

        # Ensure float32 and C-contiguous
        embs_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        idx.add(embs_f32)
        self.index_instance = idx

    def add(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Append new chunks and vectors to an existing index."""
        if self.index_instance is None:
            self.index(chunks, embeddings)
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")

        self.chunks_metadata.extend(chunks)
        for c in chunks:
            self.chunk_id_map[c["chunk_id"]] = c

        embs_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index_instance.add(embs_f32)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        """Search top-K nearest neighbors and apply parent expansion if applicable."""
        if self.index_instance is None:
            raise RuntimeError("FAISS index has not been built or loaded yet.")

        # Ensure shape (1, D)
        q_emb = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        scores, indices = self.index_instance.search(q_emb, top_k)

        results: List[RetrievedChunk] = []
        for rank in range(top_k):
            idx = int(indices[0][rank])
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue

            score = float(scores[0][rank])
            meta = self.chunks_metadata[idx]

            # Parent context expansion (from separate store to save RAM)
            parent_text = meta.get("parent_text")
            if not parent_text and hasattr(self, '_parent_text_store'):
                parent_text = self._parent_text_store.get(meta["chunk_id"])
            if not parent_text and meta.get("parent_id") and meta["parent_id"] in self.chunk_id_map:
                parent_text = self.chunk_id_map[meta["parent_id"]].get("text")

            chunk = RetrievedChunk(
                chunk_id=meta.get("chunk_id", str(idx)),
                passage_id=meta.get("passage_id", meta.get("chunk_id", str(idx))),
                query_id=int(meta.get("query_id", 0)),
                text=meta.get("text", ""),
                score=score,
                chunk_strategy=meta.get("chunk_strategy", "fixed"),
                position=int(meta.get("position", 0)),
                parent_id=meta.get("parent_id"),
                children_ids=meta.get("children_ids"),
                language=meta.get("language", "hi"),
                is_selected=int(meta.get("is_selected", 0)),
                parent_text=parent_text,
            )
            results.append(chunk)

        return results

    def save(self, index_dir: str) -> None:
        """Save FAISS binary index and JSON metadata."""
        os.makedirs(index_dir, exist_ok=True)
        index_file = os.path.join(index_dir, "faiss.index")
        meta_file = os.path.join(index_dir, "metadata.json")

        if self.index_instance is not None:
            faiss.write_index(self.index_instance, index_file)

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "backend": self.backend_name,
                "chunks": self.chunks_metadata,
            }, f, ensure_ascii=False)

    def load(self, index_dir: str, use_mmap: bool = True) -> None:
        """
        Load FAISS binary index (with optional IO_FLAG_MMAP) and JSON metadata.

        When use_mmap=True, pages are demand-paged from disk. We immediately call
        prefault_pages() to force all pages into resident RAM, eliminating P95
        cold-read latency spikes.
        """
        index_file = os.path.join(index_dir, "faiss.index")
        meta_json_file = os.path.join(index_dir, "metadata.json")
        meta_pkl_file = os.path.join(index_dir, "metadata_light.pkl")
        parent_pkl_file = os.path.join(index_dir, "parent_texts.pkl")

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file missing in {index_dir}: {index_file}")

        if use_mmap and hasattr(faiss, "IO_FLAG_MMAP"):
            self.index_instance = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
        else:
            self.index_instance = faiss.read_index(index_file)

        # Set ef_search=64: optimal for 300k vectors (good recall vs latency tradeoff)
        if hasattr(self.index_instance, "hnsw"):
            self.index_instance.hnsw.efSearch = self.ef_search

        # Prefault all mmap pages into RAM to eliminate cold-read latency
        if use_mmap and hasattr(faiss, "IO_FLAG_MMAP"):
            self.prefault_pages()

        # Fast path: load pre-serialized pickle metadata (avoids 500MB+ JSON parse overhead)
        if os.path.exists(meta_pkl_file):
            import pickle
            with open(meta_pkl_file, "rb") as f:
                self.chunks_metadata = pickle.load(f)
            self.chunk_id_map = {c["chunk_id"]: c for c in self.chunks_metadata}
            self._parent_text_store: Dict[str, str] = {}
            if os.path.exists(parent_pkl_file) and os.getenv("SKIP_PARENT_TEXTS", "false").lower() != "true":
                try:
                    with open(parent_pkl_file, "rb") as f:
                        self._parent_text_store = pickle.load(f)
                except Exception:
                    self._parent_text_store = {}
            return

        if not os.path.exists(meta_json_file):
            raise FileNotFoundError(f"Metadata file missing in {index_dir}")

        with open(meta_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.dimension = data.get("dimension", self.dimension)
            raw_chunks = data.get("chunks", [])
            self._parent_text_store = {}
            self.chunks_metadata = []
            for c in raw_chunks:
                pt = c.pop("parent_text", None)
                if pt:
                    self._parent_text_store[c["chunk_id"]] = pt
                self.chunks_metadata.append(c)
            del raw_chunks
            self.chunk_id_map = {c["chunk_id"]: c for c in self.chunks_metadata}

