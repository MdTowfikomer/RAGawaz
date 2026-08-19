"""
Regression tests for Answer Correctness, Topic Drift Prevention, Cross-Lingual Verification, and Refusal Handling.
"""

import pytest
from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.guardrails.groundedness import GroundednessVerifier
from backend.app.harness.orchestrator import RAGOrchestrator
from backend.app.harness.providers.base import MockLLMProvider


@pytest.fixture
def regression_embedder():
    embedder = get_embedding_provider("minilm")
    _ = embedder.embed_query("warmup")
    return embedder


def test_hindi_query_relevant_hindi_passage_accepted(regression_embedder):
    """Verify that a native Hindi query with relevant Hindi context passes grounding."""
    verifier = GroundednessVerifier(embedder=regression_embedder)
    query = "कॉर्पोरेशन क्या है?"
    
    grounded_answer = "एक कंपनी या लोगों का समूह जो एक एकल इकाई के रूप में कार्य करता है, कॉर्पोरेशन कहलाता है।"
    context = ["निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है।"]
    
    is_grounded, method, score, msg = verifier.evaluate(grounded_answer, context, query=query)
    
    assert is_grounded
    assert score >= 0.25
    assert msg is None


def test_english_query_relevant_hindi_passage_accepted(regression_embedder):
    """Verify that an English query with relevant Hindi context passes cross-lingual grounding."""
    verifier = GroundednessVerifier(embedder=regression_embedder)
    query = "What is a corporation?"
    
    # Grounded answer in Hindi supported by context
    grounded_answer = "निगम एक कंपनी या लोगों का समूह है जो एकल कानूनी इकाई के रूप में कार्य करता है।"
    context = ["निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है।"]
    
    is_grounded, method, score, msg = verifier.evaluate(grounded_answer, context, query=query)
    
    assert is_grounded
    assert score >= 0.25
    assert msg is None


def test_hinglish_query_relevant_hindi_passage_accepted(regression_embedder):
    """Verify that a Hinglish/code-mixed query with relevant Hindi context passes cross-lingual grounding."""
    verifier = GroundednessVerifier(embedder=regression_embedder)
    query = "Corporation kya hota hai?"
    
    grounded_answer = "निगम एक कंपनी या लोगों का समूह होता है जो एक इकाई के रूप में काम करता है।"
    context = ["निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है।"]
    
    is_grounded, method, score, msg = verifier.evaluate(grounded_answer, context, query=query)
    
    assert is_grounded
    assert score >= 0.25
    assert msg is None


def test_english_query_unrelated_hindi_passage_refused(regression_embedder):
    """Verify that an English query with completely unrelated Hindi context is strictly rejected."""
    verifier = GroundednessVerifier(embedder=regression_embedder)
    query = "What is the capital of France?"
    
    # Hallucinated answer repeating irrelevant context
    answer = "रिले आरेख दिखाते हैं जब एक रिले संपर्क खुला होता है।"
    irrelevant_context = ["रिले आरेख दिखाते हैं, जब एक रिले संपर्क सामान्य रूप से खुला होता है।"]
    
    is_grounded, method, score, msg = verifier.evaluate(answer, irrelevant_context, query=query)
    
    assert not is_grounded
    assert method == "topic_drift_rejected"


def test_hinglish_query_unrelated_hindi_passage_refused(regression_embedder):
    """Verify that a Hinglish query with unrelated Hindi context is strictly rejected."""
    verifier = GroundednessVerifier(embedder=regression_embedder)
    query = "Share market me trading kaise kare?"
    
    answer = "रिले आरेख दिखाते हैं जब एक रिले संपर्क खुला होता है।"
    irrelevant_context = ["रिले आरेख दिखाते हैं, जब एक रिले संपर्क सामान्य रूप से खुला होता है।"]
    
    is_grounded, method, score, msg = verifier.evaluate(answer, irrelevant_context, query=query)
    
    assert not is_grounded
    assert method == "topic_drift_rejected"


def test_barter_relay_distractor_refused(regression_embedder):
    """Verify that barter queries matched against electrical relay distractors are strictly rejected."""
    verifier = GroundednessVerifier(embedder=regression_embedder)
    query = "किन्ना पारस्परिक आदान-प्रदान होता है और इसकी समस्याएं क्या हैं"
    
    answer = "रिले आरेख दिखाते हैं जब एक रिले संपर्क सामान्य रूप से खुला होता है।"
    irrelevant_context = ["रिले आरेख दिखाते हैं, जब एक रिले संपर्क सामान्य रूप से खुला (एन.ओ.) होता है, तो रिले के ऊर्जीकृत न होने पर एक खुला संपर्क होता है।"]
    
    is_grounded, method, score, msg = verifier.evaluate(answer, irrelevant_context, query=query)
    
    assert not is_grounded
    assert method == "topic_drift_rejected"


def test_cantaloupe_cucumber_mismatch_refused(regression_embedder):
    """
    Verify that cantaloupe queries with cucumber passages (MSMARCO translation mismatch):
    When the LLM correctly identifies insufficient evidence and produces a refusal answer,
    it should PASS THROUGH (grounded=True, method=explicit_refusal_passthrough).
    The LLM's contextual, language-matched refusal is preserved for the user.
    """
    verifier = GroundednessVerifier(embedder=regression_embedder)
    query = "कैंटालूप को कितने समय तक परिपक्व होना है"
    
    refusal_answer = "दिए गए संदर्भ में इस प्रश्न का सटीक उत्तर उपलब्ध नहीं है।"
    context = ["खीरे आम तौर पर 75-90 दिनों के बीच परिपक्वता प्राप्त करते हैं।"]
    
    is_grounded, method, score, msg = verifier.evaluate(refusal_answer, context, query=query)
    
    # LLM refusal is a legitimate, grounded response — it should pass through
    assert is_grounded
    assert method == "explicit_refusal_passthrough"
    assert score == 1.0
    assert msg is None
