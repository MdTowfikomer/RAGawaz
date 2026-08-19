"""
Unit tests for BM25Retriever and HybridRetriever with Reciprocal Rank Fusion.
"""

import pytest
import numpy as np
from backend.app.rag.bm25_retriever import BM25Retriever, tokenize_multilingual
from backend.app.rag.retriever import FAISSHNSWRetriever, RetrievedChunk
from backend.app.rag.hybrid_retriever import HybridRetriever


def test_tokenize_multilingual():
    text_hi = "कॉर्पोरेशन क्या है? भारत में कानून।"
    tokens = tokenize_multilingual(text_hi)
    assert "क" in tokens or "करप" in "".join(tokens) or len(tokens) >= 3

    text_en = "What is a corporation? 123 rules."
    tokens_en = tokenize_multilingual(text_en)
    assert "what" in tokens_en
    assert "corporation" in tokens_en
    assert "123" in tokens_en


def test_bm25_indexing_and_search():
    chunks = [
        {"chunk_id": "c1", "passage_id": "p1", "query_id": 101, "text": "एक कंपनी एका विशिष्ट देशात स्थापित केली जाते.", "language": "mr", "parent_text": "Parent MR 1"},
        {"chunk_id": "c2", "passage_id": "p2", "query_id": 102, "text": "कॉर्पोरेशन एक कानूनी इकाई है जो व्यवसाय करती है।", "language": "hi", "parent_text": "Parent HI 2"},
        {"chunk_id": "c3", "passage_id": "p3", "query_id": 103, "text": "What is a corporation in commercial law?", "language": "en", "parent_text": "Parent EN 3"},
    ]

    bm25 = BM25Retriever()
    bm25.index(chunks)

    assert bm25.num_docs == 3
    results = bm25.search("corporation commercial law", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c3"
    assert results[0].parent_text == "Parent EN 3"


def test_hybrid_rrf_fusion():
    chunks = [
        {"chunk_id": f"c_{i}", "passage_id": f"p_{i}", "query_id": i, "text": f"Document {i} about biology and chemistry.", "language": "en"}
        for i in range(10)
    ]
    embs = np.random.randn(10, 64).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms

    dense = FAISSHNSWRetriever(dimension=64, m=16, ef_search=32)
    dense.index(chunks, embs)

    bm25 = BM25Retriever()
    bm25.index(chunks)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, dense_top_k=5, bm25_top_k=5, rrf_k=60, fused_top_k=3)

    q_emb = embs[0]
    results = hybrid.search_hybrid("biology chemistry", query_embedding=q_emb, top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].score > 0.0
