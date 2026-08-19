"""
Unit and Integration Tests for Phase 6C: Multilingual Offline Pipeline & Index Builder.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import faiss

from backend.app.rag.multilingual_pipeline import MultilingualIndexBuilder
from backend.app.rag.retriever import FAISSHNSWRetriever


def test_multilingual_builder_artifact_generation():
    """Verify that MultilingualIndexBuilder creates valid FAISS, BM25, and metadata artifacts."""
    temp_out = Path(tempfile.mkdtemp(prefix="test_multi_bundle_"))
    try:
        builder = MultilingualIndexBuilder(
            languages=["hin"],
            max_passages_per_lang=10,
            output_dir=temp_out,
        )

        dummy_chunks = [
            {
                "chunk_id": f"hi_test_{i}_c0",
                "passage_id": f"hi_test_{i}",
                "query_id": 100 + i,
                "text": f"यह परीक्षण वाक्य संख्या {i} है।",
                "parent_text": f"यह पूरा परीक्षण संदर्भ अनुच्छेद संख्या {i} है।",
                "position": 0,
                "language": "hi",
                "is_selected": 1,
                "char_length": 30,
            }
            for i in range(20)
        ]
        dummy_lang_stats = {"hin": {"passages_ingested": 20, "chunks_generated": 20, "expansion_ratio": 1.0}}

        dummy_embeddings = np.random.randn(20, 1024).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(dummy_embeddings, axis=1, keepdims=True)
        dummy_embeddings = dummy_embeddings / norms

        manifest = builder.build_and_save_artifacts(dummy_chunks, dummy_embeddings, dummy_lang_stats)

        assert (temp_out / "faiss.index").exists()
        assert (temp_out / "metadata.json").exists()
        assert (temp_out / "bm25_vocab.json").exists()
        assert (temp_out / "manifest.json").exists()

        # Test loading via FAISSHNSWRetriever with mmap
        retriever = FAISSHNSWRetriever(dimension=1024)
        retriever.load(str(temp_out), use_mmap=True)

        assert len(retriever.chunks_metadata) == 20
        results = retriever.search(dummy_embeddings[0], top_k=3)
        assert len(results) == 3
        assert results[0].chunk_id == "hi_test_0_c0"
        assert results[0].parent_text == "यह पूरा परीक्षण संदर्भ अनुच्छेद संख्या 0 है।"

    finally:
        shutil.rmtree(temp_out, ignore_errors=True)
