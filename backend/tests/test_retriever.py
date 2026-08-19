"""
Unit tests for Ticket 5: Retriever Backend Interface and FAISS Implementation
"""
import os
import pytest
import numpy as np
from backend.app.rag.retriever import (
    RetrieverBackend,
    FAISSHNSWRetriever,
    RetrievedChunk,
)


@pytest.fixture
def sample_chunks_and_embeddings():
    """Generate dummy normalized embeddings and chunks for testing."""
    np.random.seed(42)
    dim = 384
    n = 100

    raw_embs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    embeddings = raw_embs / norms

    chunks = []
    # Add a parent chunk and child chunks
    parent_id = "hi_999_parent"
    chunks.append({
        "chunk_id": parent_id,
        "passage_id": "hi_999_0",
        "query_id": 999,
        "query": "टेस्ट प्रश्न?",
        "text": "यह पूर्ण मूल गद्यांश है जो विस्तृत संदर्भ प्रदान करता है।",
        "chunk_strategy": "parent_child_parent",
        "position": 0,
        "parent_id": None,
        "children_ids": ["hi_999_child_0"],
        "is_selected": 1,
        "language": "hi",
    })

    chunks.append({
        "chunk_id": "hi_999_child_0",
        "passage_id": "hi_999_0",
        "query_id": 999,
        "query": "टेस्ट प्रश्न?",
        "text": "यह सूक्ष्म बाल गद्यांश है।",
        "chunk_strategy": "parent_child_child",
        "position": 0,
        "parent_id": parent_id,
        "children_ids": None,
        "is_selected": 1,
        "language": "hi",
    })

    # Add other chunks
    for i in range(2, n):
        chunks.append({
            "chunk_id": f"hi_{i}_fix_0",
            "passage_id": f"hi_{i}_0",
            "query_id": i,
            "query": f"प्रश्न {i}?",
            "text": f"यह गद्यांश संख्या {i} का पाठ्य है।",
            "chunk_strategy": "fixed",
            "position": 0,
            "parent_id": None,
            "children_ids": None,
            "is_selected": 0,
            "language": "hi",
        })

    return chunks, embeddings


def test_retriever_protocol_conformance():
    """Verify FAISSHNSWRetriever implements RetrieverBackend protocol."""
    retriever = FAISSHNSWRetriever(dimension=384)
    assert isinstance(retriever, RetrieverBackend)
    assert retriever.backend_name == "faiss_hnsw"


def test_faiss_index_and_search_ordering(sample_chunks_and_embeddings):
    """Test building FAISS index and searching returns Top-K ordered by score."""
    chunks, embeddings = sample_chunks_and_embeddings
    retriever = FAISSHNSWRetriever(dimension=384)
    retriever.index(chunks, embeddings)

    # Search with embedding identical to chunk[0]
    query_emb = embeddings[0]
    results = retriever.search(query_emb, top_k=5)

    assert len(results) == 5
    # First result should be chunk 0 with score close to 1.0
    assert results[0].chunk_id == chunks[0]["chunk_id"]
    assert np.isclose(results[0].score, 1.0, atol=1e-3)

    # Scores should be monotonically non-increasing
    for i in range(len(results) - 1):
        assert results[i].score >= results[i+1].score


def test_parent_expansion_attaches_parent_text(sample_chunks_and_embeddings):
    """Verify that searching a child chunk automatically expands parent text."""
    chunks, embeddings = sample_chunks_and_embeddings
    retriever = FAISSHNSWRetriever(dimension=384)
    retriever.index(chunks, embeddings)

    # Search with embedding of child chunk (idx = 1)
    query_emb = embeddings[1]
    results = retriever.search(query_emb, top_k=1)

    assert len(results) == 1
    child_res = results[0]
    assert child_res.chunk_id == "hi_999_child_0"
    assert child_res.parent_id == "hi_999_parent"
    assert child_res.parent_text is not None
    assert "विस्तृत संदर्भ" in child_res.parent_text


def test_save_and_load_faiss_index(tmp_path, sample_chunks_and_embeddings):
    """Verify saving FAISS index and loading from disk."""
    chunks, embeddings = sample_chunks_and_embeddings
    retriever = FAISSHNSWRetriever(dimension=384)
    retriever.index(chunks, embeddings)

    save_dir = str(tmp_path / "faiss_index")
    retriever.save(save_dir)

    assert os.path.exists(os.path.join(save_dir, "faiss.index"))
    assert os.path.exists(os.path.join(save_dir, "metadata.json"))

    loaded_retriever = FAISSHNSWRetriever(dimension=384)
    loaded_retriever.load(save_dir)

    query_emb = embeddings[0]
    results = loaded_retriever.search(query_emb, top_k=3)
    assert len(results) == 3
    assert results[0].chunk_id == chunks[0]["chunk_id"]
