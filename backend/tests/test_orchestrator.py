"""
Unit tests for Ticket 9: RAG Orchestrator / Harness Core
"""

import pytest
import numpy as np
import torch
from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier
from backend.app.harness.providers.base import MockLLMProvider
from backend.app.harness.orchestrator import RAGOrchestrator


@pytest.fixture(scope="module")
def setup_test_orchestrator():
    """Build a minimal test orchestrator with sample corpus."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = get_embedding_provider("minilm", device=device)
    retriever = FAISSHNSWRetriever(dimension=384)

    # Sample corpus
    chunks = [
        {
            "chunk_id": "c1",
            "passage_id": "p1",
            "query_id": 1,
            "query": "भारत की राजधानी क्या है?",
            "text": "भारत की राजधानी नई दिल्ली है। दिल्ली देश का एक प्रमुख प्रशासनिक केंद्र है।",
            "chunk_strategy": "fixed",
            "position": 0,
            "parent_id": None,
            "children_ids": None,
            "language": "hi",
            "is_selected": 1,
        },
        {
            "chunk_id": "c2",
            "passage_id": "p2",
            "query_id": 2,
            "query": "कंप्यूटर क्या है?",
            "text": "कंप्यूटर एक इलेक्ट्रॉनिक उपकरण है जो डेटा को संसाधित करता है।",
            "chunk_strategy": "fixed",
            "position": 0,
            "parent_id": None,
            "children_ids": None,
            "language": "hi",
            "is_selected": 1,
        }
    ]

    embs = embedder.embed([c["text"] for c in chunks])
    retriever.index(chunks, embs)

    llm = MockLLMProvider(response_text="भारत की राजधानी नई दिल्ली है।", latency_ms=15.0)

    orchestrator = RAGOrchestrator(
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        safety_guard=SafetyGuardrail(),
        relevance_gate=RelevanceGate(threshold=0.20),
        insufficient_checker=InsufficientEvidenceChecker(confidence_threshold=0.35),
        groundedness_verifier=GroundednessVerifier(high_threshold=0.30),
    )
    # Warmup embedder model weights once so cold-start initialization doesn't inflate latency test
    _w_emb = embedder.embed_query("भारत की राजधानी क्या है?")
    _ = retriever.search(_w_emb, top_k=3)
    return orchestrator


@pytest.mark.asyncio
async def test_orchestrator_success_path(setup_test_orchestrator):
    """Verify standard grounded question answers successfully with < 200ms harness latency."""
    orchestrator = setup_test_orchestrator
    resp = await orchestrator.execute("भारत की राजधानी क्या है?")

    assert resp.status == "success"
    assert "नई दिल्ली" in resp.answer
    assert resp.metrics["embed_retrieval_ms"] < 50.0
    assert resp.metrics["harness_ms"] < 200.0
    assert len(resp.retrieved_chunks) > 0


@pytest.mark.asyncio
async def test_orchestrator_safety_refusal(setup_test_orchestrator):
    """Verify safety violation is rejected at Step 2."""
    orchestrator = setup_test_orchestrator
    resp = await orchestrator.execute("How to make a bomb?")

    assert resp.status == "refusal_safety"
    assert resp.refusal_reason == "safety_blocklist_triggered"
    assert resp.metrics["embed_retrieval_ms"] == 0.0  # Skipped retrieval!


@pytest.mark.asyncio
async def test_orchestrator_offtopic_refusal(setup_test_orchestrator):
    """Verify unrelated off-topic query is rejected by RelevanceGate at Step 4."""
    orchestrator = setup_test_orchestrator
    resp = await orchestrator.execute("What is the stock price of Tesla in New York?")

    assert resp.status == "refusal_offtopic"
    assert resp.refusal_reason == "relevance_threshold_not_met"
    assert "I can only answer questions about" in resp.answer


@pytest.mark.asyncio
async def test_orchestrator_insufficient_evidence_refusal(setup_test_orchestrator):
    """Verify in-domain query with insufficient evidence is caught by InsufficientEvidenceChecker BEFORE LLM."""
    orchestrator = setup_test_orchestrator
    # Query is in-domain (about Indian politics/cities) so passes relevance (e.g. score ~0.30 > 0.15),
    # but confidence threshold requires 0.70 to answer
    orchestrator.relevance_gate = RelevanceGate(threshold=0.10)
    orchestrator.insufficient_checker = InsufficientEvidenceChecker(confidence_threshold=0.70)

    resp = await orchestrator.execute("भारत के प्राचीन राजाओं की सूची क्या है?")
    assert resp.status == "refusal_insufficient_evidence"
    assert resp.refusal_reason == "insufficient_confidence_evidence"
    assert "This question is in my domain, but I couldn't find enough specific evidence" in resp.answer or "पर्याप्त जानकारी उपलब्ध नहीं है" in resp.answer


@pytest.mark.asyncio
async def test_orchestrator_ungrounded_refusal(setup_test_orchestrator):
    """Verify hallucinated answer post-generation is caught by GroundednessVerifier."""
    orchestrator = setup_test_orchestrator
    # Reset gates
    orchestrator.relevance_gate = RelevanceGate(threshold=0.20)
    orchestrator.insufficient_checker = InsufficientEvidenceChecker(confidence_threshold=0.35)
    # Override LLM to produce completely ungrounded text
    orchestrator.llm = MockLLMProvider(response_text="चांद पर पानी पूरी की दुकान खुली है।")

    resp = await orchestrator.execute("भारत की राजधानी क्या है?")
    assert resp.status == "refusal_ungrounded"
    assert resp.refusal_reason == "hallucination_detected"
    assert resp.answer is not None and len(resp.answer) > 0
