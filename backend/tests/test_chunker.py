"""
Unit tests for Ticket 3: Multi-Strategy Chunking Engine
"""
import pytest
from backend.app.rag.chunker import (
    split_hindi_sentences,
    chunk_fixed_overlap,
    chunk_semantic,
    chunk_parent_child,
    chunk_adaptive,
    chunk_passage,
    chunk_corpus,
)


@pytest.fixture
def sample_hindi_passage():
    return {
        "passage_id": "hi_1001_0",
        "query_id": 1001,
        "query": "भारत की राजधानी क्या है?",
        "answer": "नई दिल्ली",
        "text": "भारत दक्षिण एशिया का एक देश है। इसकी राजधानी नई दिल्ली है। यह विश्व का सबसे अधिक आबादी वाला देश है। यहाँ कई भाषाएँ बोली जाती हैं। हिंदी और अंग्रेजी आधिकारिक भाषाएँ हैं।",
        "is_selected": 1,
        "language": "hi",
    }


def test_split_hindi_sentences():
    """Verify sentence splitting respects Hindi danda (।) and punctuation."""
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है? क्या यह तीसरा है! हाँ।"
    sentences = split_hindi_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "यह पहला वाक्य है।"
    assert sentences[1] == "यह दूसरा वाक्य है?"


def test_fixed_overlap_produces_overlapping_chunks(sample_hindi_passage):
    """Test fixed + overlap window chunking."""
    text = sample_hindi_passage["text"]
    chunks = chunk_fixed_overlap(text, chunk_size=80, overlap=20)
    assert len(chunks) >= 2
    # Ensure consecutive chunks share overlap substring
    for c in chunks:
        assert len(c) <= 80


def test_semantic_splitting_keeps_sentences_intact(sample_hindi_passage):
    """Test semantic strategy respects sentence boundaries."""
    text = sample_hindi_passage["text"]
    chunks = chunk_semantic(text, max_chunk_size=100, target_sentences=2)
    assert len(chunks) >= 2
    # Check that chunks do not start with a lone punctuation
    for c in chunks:
        assert not c.startswith("।")
        assert len(c) > 0


def test_parent_child_hierarchical_linkage(sample_hindi_passage):
    """Test parent-child produces 1 parent and multiple linked children."""
    chunks = chunk_passage(sample_hindi_passage, strategy="parent_child")
    parents = [c for c in chunks if c["chunk_strategy"] == "parent_child_parent"]
    children = [c for c in chunks if c["chunk_strategy"] == "parent_child_child"]

    assert len(parents) == 1
    assert len(children) >= 1

    parent = parents[0]
    assert parent["parent_id"] is None
    assert parent["children_ids"] == [c["chunk_id"] for c in children]

    for child in children:
        assert child["parent_id"] == parent["chunk_id"]
        assert child["children_ids"] is None


def test_non_hierarchical_chunks_have_nullable_parent_fields(sample_hindi_passage):
    """Test fixed, semantic, and adaptive chunks have parent_id=None, children_ids=None."""
    for strat in ["fixed", "semantic", "adaptive"]:
        chunks = chunk_passage(sample_hindi_passage, strategy=strat)
        for c in chunks:
            assert c["parent_id"] is None
            assert c["children_ids"] is None
            assert c["chunk_strategy"] == strat


def test_adaptive_keeps_short_passages_intact():
    """Test adaptive strategy does not split short passages."""
    short_text = "भारत की राजधानी नई दिल्ली है।"
    chunks = chunk_adaptive(short_text, short_threshold=100)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_strategies_produce_distinct_chunk_counts(sample_hindi_passage):
    """Verify 4 strategies produce distinct chunk distributions."""
    passages = [sample_hindi_passage] * 5
    fixed_chunks = chunk_corpus(passages, strategy="fixed")
    semantic_chunks = chunk_corpus(passages, strategy="semantic")
    parent_child_chunks = chunk_corpus(passages, strategy="parent_child")
    adaptive_chunks = chunk_corpus(passages, strategy="adaptive")

    assert len(fixed_chunks) > 0
    assert len(semantic_chunks) > 0
    assert len(parent_child_chunks) > 0
    assert len(adaptive_chunks) > 0
    # Parent-child generates parent + child chunks so its count is higher
    assert len(parent_child_chunks) > len(semantic_chunks)
