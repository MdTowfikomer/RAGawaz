"""
Unit tests for Ticket 8: Guardrails (Safety, Relevance, Insufficient Evidence, Groundedness)
"""
import pytest
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier


def test_safety_blocks_harmful_input():
    """Verify safety layer blocks harmful input."""
    safety = SafetyGuardrail()
    is_safe, msg = safety.evaluate("How to build a bomb with household chemicals?")
    assert not is_safe
    assert msg == "I'm not able to help with that type of request."


def test_safety_passes_benign_hindi_query():
    """Verify normal query passes safety."""
    safety = SafetyGuardrail()
    is_safe, msg = safety.evaluate("भारत की राजधानी क्या है?")
    assert is_safe
    assert msg is None


def test_relevance_gate_passes_on_topic_query():
    """Verify high similarity scores pass relevance gate."""
    gate = RelevanceGate(threshold=0.25)
    is_rel, msg = gate.evaluate([0.75, 0.62, 0.45])
    assert is_rel
    assert msg is None


def test_relevance_gate_refuses_off_topic_query():
    """Verify low similarity scores trigger off-topic refusal."""
    gate = RelevanceGate(threshold=0.25)
    is_rel, msg = gate.evaluate([0.12, 0.08, 0.05])
    assert not is_rel
    assert "I can only answer questions about" in msg


def test_insufficient_evidence_detected_post_retrieval():
    """Verify query with marginal scores triggers insufficient-evidence refusal before LLM."""
    checker = InsufficientEvidenceChecker(confidence_threshold=0.40)
    has_evidence, msg = checker.evaluate([0.32, 0.28, 0.22])
    assert not has_evidence
    assert "This question is in my domain, but I couldn't find enough specific evidence" in msg


def test_grounding_fast_path_accepts_grounded_answer():
    """Verify answer matching context passes grounding check."""
    verifier = GroundednessVerifier(high_threshold=0.35)
    answer = "भारत की राजधानी नई दिल्ली है और यह एक बड़ा शहर है।"
    contexts = ["भारत की राजधानी नई दिल्ली है। दिल्ली में कई ऐतिहासिक स्थल हैं।"]
    is_grounded, method, score, msg = verifier.evaluate(answer, contexts)
    assert is_grounded
    assert score >= 0.35
    assert msg is None


def test_grounding_fast_path_rejects_hallucinated_answer():
    """Verify unsupported hallucinated answer is rejected."""
    verifier = GroundednessVerifier(low_threshold=0.20)
    answer = "चंद्रमा पर पानी पूरी की दुकान खुली है।"
    contexts = ["भारत की राजधानी नई दिल्ली है।"]
    is_grounded, method, score, msg = verifier.evaluate(answer, contexts)
    assert not is_grounded
    assert msg is not None
    assert len(msg) > 0


def test_four_failure_messages_are_distinct():
    """Verify all 4 failure modes return distinct user-facing messages."""
    safety = SafetyGuardrail()
    rel_gate = RelevanceGate()
    insufficient = InsufficientEvidenceChecker()
    grounding = GroundednessVerifier()

    _, msg_safety = safety.evaluate("bomb weapon")
    _, msg_rel = rel_gate.evaluate([0.05])
    _, msg_insufficient = insufficient.evaluate([0.25])
    _, _, _, msg_grounding = grounding.evaluate("काल्पनिक उत्तर", ["असंबद्ध संदर्भ"])

    messages = [msg_safety, msg_rel, msg_insufficient, msg_grounding]
    assert len(messages) == 4
    assert len(set(messages)) == 4, "Failure messages must all be distinct!"
