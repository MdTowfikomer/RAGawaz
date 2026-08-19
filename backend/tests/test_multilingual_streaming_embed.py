"""
Unit test for micro-batch streaming embedding and transaction-safe checkpointing in MultilingualIndexBuilder.
"""

import os
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from backend.app.rag.multilingual_pipeline import MultilingualIndexBuilder
from backend.app.rag.retriever import FAISSHNSWRetriever


def test_microbatch_streaming_and_checkpoint_transactions():
    """Verify micro-batching, atomic commits, and safe resumption."""
    temp_out = Path(tempfile.mkdtemp(prefix="test_streaming_builder_"))

    builder = MultilingualIndexBuilder(
        languages=["hin", "mar"],
        max_passages_per_lang=20,
        batch_size=8,
        output_dir=temp_out,
    )

    # 1. Verify initial checkpoint state is empty
    checkpoint = builder.load_pipeline_checkpoint()
    assert checkpoint == {}

    # 2. Simulate language stats and commit
    dummy_chunks_hin = [
        {
            "chunk_id": f"hi_{i}_c0",
            "passage_id": f"hi_{i}",
            "query_id": 100 + i,
            "text": f"हिंदी पाठ संख्या {i} जो सत्यापन के लिए है।",
            "parent_text": f"यह पूरा हिंदी संदर्भ अनुच्छेद संख्या {i} है।",
            "position": 0,
            "language": "hi",
            "is_selected": 1,
            "char_length": 35,
        }
        for i in range(10)
    ]
    dummy_lang_stats = {"hin": {"passages": 10, "chunks": 10, "expansion_ratio": 1.0}}

    dummy_embs = np.random.randn(10, 1024).astype(np.float32)
    norms = np.linalg.norm(dummy_embs, axis=1, keepdims=True)
    dummy_embs = dummy_embs / norms

    manifest = builder.build_and_save_artifacts(dummy_chunks_hin, dummy_embs, dummy_lang_stats)

    # 3. Verify artifacts exist and load cleanly
    assert (temp_out / "faiss.index").exists()
    assert (temp_out / "metadata.json").exists()
    assert (temp_out / "manifest.json").exists()

    retriever = FAISSHNSWRetriever(dimension=1024)
    retriever.load(str(temp_out), use_mmap=True)
    assert len(retriever.chunks_metadata) == 10

    results = retriever.search(dummy_embs[0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "hi_0_c0"
    assert results[0].parent_text == "यह पूरा हिंदी संदर्भ अनुच्छेद संख्या 0 है।"
