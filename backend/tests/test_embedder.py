"""
Unit tests for Ticket 4: Embedding Providers
"""
import pytest
import numpy as np
import torch
from backend.app.rag.embedder import (
    EmbeddingProvider,
    MiniLMEmbeddingProvider,
    get_embedding_provider,
)


@pytest.fixture(scope="module")
def minilm_provider():
    """Load MiniLM provider once for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return get_embedding_provider("minilm", device=device)


def test_embedding_provider_protocol_conformance(minilm_provider):
    """Verify provider implements the EmbeddingProvider protocol."""
    assert isinstance(minilm_provider, EmbeddingProvider)
    assert minilm_provider.dimension == 384
    assert hasattr(minilm_provider, "embed")
    assert hasattr(minilm_provider, "embed_query")


def test_minilm_embeds_hindi_text(minilm_provider):
    """Verify embedding produces correct shape and float32 dtype."""
    texts = [
        "भारत की राजधानी नई दिल्ली है।",
        "यह एक परीक्षण वाक्य है।",
    ]
    embs = minilm_provider.embed(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 384)
    assert embs.dtype == np.float32


def test_embeddings_are_finite_and_normalized(minilm_provider):
    """Verify embeddings have no NaNs and are unit-normalized (L2 norm = 1)."""
    texts = ["नमस्ते दुनिया", "मशीन लर्निंग और आवाज खोज"]
    embs = minilm_provider.embed(texts)

    # Check finite
    assert np.all(np.isfinite(embs)), "Embeddings contain NaN or Inf"

    # Check L2 norms
    norms = np.linalg.norm(embs, axis=1)
    for n in norms:
        assert np.isclose(n, 1.0, atol=1e-3), f"Expected norm ≈ 1.0, got {n}"


def test_same_text_has_near_unit_similarity(minilm_provider):
    """Verify same text embedded twice yields cosine similarity ≈ 1.0."""
    text = "दिल्ली भारत का एक प्रमुख शहर है।"
    emb1 = minilm_provider.embed_query(text)
    emb2 = minilm_provider.embed_query(text)

    sim = float(np.dot(emb1, emb2.T)[0, 0])
    assert np.isclose(sim, 1.0, atol=1e-3), f"Expected self-similarity ≈ 1.0, got {sim}"


def test_known_positive_beats_known_negative(minilm_provider):
    """Verify relative ordering: positive matching passage scores higher than unrelated text."""
    query = "कंप्यूटर क्या है?"
    positive_doc = "कंप्यूटर एक इलेक्ट्रॉनिक उपकरण है जो डेटा को संसाधित करता है।"
    negative_doc = "आम एक स्वादिष्ट और मीठा फल होता है।"

    q_emb = minilm_provider.embed_query(query)
    doc_embs = minilm_provider.embed([positive_doc, negative_doc])

    sim_pos = float(np.dot(q_emb, doc_embs[0:1].T)[0, 0])
    sim_neg = float(np.dot(q_emb, doc_embs[1:2].T)[0, 0])

    print(f"Similarity positive: {sim_pos:.4f}, negative: {sim_neg:.4f}")
    assert sim_pos > sim_neg, f"Expected pos ({sim_pos}) > neg ({sim_neg})"
