"""
Phase 5C Regression Tests: Evidence Sufficiency & Groundedness Hardening.

Tests:
1. High-similarity non-answering passages -> Pre-LLM Insufficient Evidence Refusal
2. Answerable passage -> Generation allowed
3. Unsupported generated claim -> Ungrounded Refusal
4. Unsupported numeric hallucination -> Ungrounded Refusal
5. Topic drift -> Ungrounded Refusal
6. Repetitive / degenerate generation -> Ungrounded Refusal
7. Multilingual / cross-lingual genuine support -> Groundedness VERIFIED
"""

import pytest
from backend.app.rag.embedder import get_embedding_provider
from backend.app.guardrails.relevance import InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier


@pytest.fixture
def regression_embedder():
    embedder = get_embedding_provider("minilm")
    _ = embedder.embed_query("warmup")
    return embedder


def test_high_similarity_non_answering_passages_rejected_pre_llm():
    """
    Test 1: High similarity score, but context has 0 entity/subject overlap with query.
    Must be intercepted BEFORE LLM generation as insufficient evidence.
    """
    checker = InsufficientEvidenceChecker(confidence_threshold=0.38)
    query = "कैंटालूप को कितने समय तक परिपक्व होना है"
    # Non-answering distractor passage about cucumbers/tomatoes
    non_answering_context = ["खीरे और टमाटर आमतौर पर 60 दिनों में बढ़ते हैं। बागवानी के लिए यह उपयोगी है।"]
    
    # Borderline similarity score passing simple float check
    has_evidence, msg = checker.evaluate(
        top_scores=[0.42, 0.39],
        query=query,
        context_chunks=non_answering_context,
    )
    assert not has_evidence
    assert "couldn't find enough specific evidence" in msg or "पर्याप्त जानकारी उपलब्ध नहीं है" in msg


def test_answerable_passage_generation_allowed():
    """
    Test 2: Genuine answering passage with high confidence.
    Must allow generation to proceed.
    """
    checker = InsufficientEvidenceChecker(confidence_threshold=0.38)
    query = "भारत की राजधानी क्या है?"
    answering_context = ["भारत की राजधानी नई दिल्ली है।"]
    
    has_evidence, msg = checker.evaluate(
        top_scores=[0.85, 0.72],
        query=query,
        context_chunks=answering_context,
    )
    assert has_evidence
    assert msg is None


def test_unsupported_generated_claim_refused():
    """
    Test 3: Answer introduces substantive claims and entities not supported by retrieved context.
    Must be rejected as ungrounded claim.
    """
    verifier = GroundednessVerifier(high_threshold=0.35, low_threshold=0.15)
    query = "सौर ऊर्जा कैसे काम करती है?"
    context = ["सौर ऊर्जा सूर्य के प्रकाश से प्राप्त होती है और सोलर पैनल द्वारा बिजली में बदली जाती है।"]
    # Hallucinated answer claiming nuclear fission
    hallucinated_answer = "सौर ऊर्जा यूरेनियम के विखंडन और परमाणु रिएक्टरों के माध्यम से ऊर्जा उत्पन्न करती है।"
    
    is_grounded, method, score, msg = verifier.evaluate(hallucinated_answer, context, query=query)
    assert not is_grounded
    assert method in ["unsupported_claim_rejected", "topic_drift_rejected"]
    assert msg is not None


def test_unsupported_numeric_claim_refused():
    """
    Test 4: Answer introduces fabricated numeric facts / years not in retrieved context.
    Must be rejected as unsupported numeric claim.
    """
    verifier = GroundednessVerifier(high_threshold=0.35, low_threshold=0.15)
    query = "कंपनी की स्थापना कब हुई थी?"
    context = ["कंपनी की स्थापना बीसवीं सदी के उत्तरार्ध में एक छोटे गैरेज में हुई थी।"]
    # Hallucinated exact year 1987 not present anywhere in context
    hallucinated_numeric_answer = "कंपनी की स्थापना वर्ष 1987 में एक छोटे गैरेज में हुई थी।"
    
    is_grounded, method, score, msg = verifier.evaluate(hallucinated_numeric_answer, context, query=query)
    assert not is_grounded
    assert method == "unsupported_numeric_claim"
    assert msg is not None


def test_topic_drift_refused(regression_embedder):
    """
    Test 5: Model drifts to an unrelated topic.
    Must be rejected as topic drift.
    """
    verifier = GroundednessVerifier(embedder=regression_embedder)
    query = "कंप्यूटर प्रोग्रामिंग क्या है?"
    context = ["कंप्यूटर प्रोग्रामिंग निर्देशों का एक सेट लिखने की प्रक्रिया है।"]
    # Drifted answer talking about pizza baking
    drifted_answer = "पिज्जा बनाने के लिए आटे को गूंथकर ओवन में 200 डिग्री पर बेक किया जाता है।"
    
    is_grounded, method, score, msg = verifier.evaluate(drifted_answer, context, query=query)
    assert not is_grounded
    assert method in ["topic_drift_rejected", "unsupported_claim_rejected"]
    assert msg is not None


def test_repetitive_degenerate_generation_refused():
    """
    Test 6: Model produces pathological repeating word loops.
    Must be rejected as hallucination / degeneration.
    """
    verifier = GroundednessVerifier()
    query = "चेक सिस्टम"
    context = ["सिस्टम स्वयं जांच करता है।"]
    degenerate_answer = "स्वयं स्वयं स्वयं स्वयं स्वयं स्वयं स्वयं स्वयं स्वयं स्वयं"
    
    is_grounded, method, score, msg = verifier.evaluate(degenerate_answer, context, query=query)
    assert not is_grounded
    assert method == "hallucination_detected"
    assert msg is not None


def test_multilingual_cross_lingual_evidence_supported(regression_embedder):
    """
    Test 7: Cross-lingual query (English/Hinglish) with genuine answering Hindi context.
    Must pass grounding and be marked VERIFIED.
    """
    verifier = GroundednessVerifier(embedder=regression_embedder)
    query = "What is a corporation?"
    context = ["निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है।"]
    grounded_answer = "निगम एक कंपनी या लोगों का समूह है जो एक एकल इकाई के रूप में कार्य करता है।"
    
    is_grounded, method, score, msg = verifier.evaluate(grounded_answer, context, query=query)
    assert is_grounded
    assert score >= 0.25
    assert msg is None
