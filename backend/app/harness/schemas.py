"""
Ticket 5 / Spec Section 5d: Pydantic Schemas for Voice RAG Harness.

Declares frozen request/response schemas, telemetry breakdowns, and chunk models:
1. RetrievedChunk
2. LatencyBreakdown
3. GroundingResult
4. RAGRequest
5. RAGResponse
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Normalized retrieval candidate chunk."""
    chunk_id: str
    text: str
    score: float
    passage_id: Optional[str] = None
    query_id: Optional[int] = None
    position: Optional[int] = None
    chunk_strategy: Optional[str] = None
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    """Per-stage telemetry latency measurements in milliseconds."""
    safety_ms: float = 0.0
    embedding_ms: float = 0.0
    faiss_ms: float = 0.0
    bm25_ms: float = 0.0
    rrf_ms: float = 0.0
    evidence_gate_ms: float = 0.0
    pre_llm_total_ms: float = 0.0
    embed_retrieval_ms: float = 0.0
    rag_pipeline_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0
    groq_network_ms: float = 0.0
    grounding_ms: float = 0.0
    text_to_answer_ms: float = 0.0
    harness_ms: float = 0.0
    guardrails_ms: float = 0.0
    tts_first_audio_ms: float = 0.0
    voice_pipeline_ms: float = 0.0
    # Boundary Decision Diagnostics
    entity_match: Optional[str] = "N/A"
    evidence_status: Optional[str] = "SUFFICIENT"
    llm_invocation: Optional[str] = "SKIPPED"
    groundedness_verdict: Optional[str] = "N/A"

    model_config = {"extra": "allow"}


class GroundingResult(BaseModel):
    """Outcome of the Grounding verification stage."""
    is_grounded: bool
    method: Literal["overlap", "llm_judge", "bypass", "unverified"] = "overlap"
    confidence: float = 1.0
    refusal_reason: Optional[str] = None


class RAGRequest(BaseModel):
    """Input contract for the RAG Orchestrator."""
    query: str
    language: str = "hi"
    contexts: Optional[List[RetrievedChunk]] = None
    max_tokens: int = 256
    mode: Literal["strict", "quality"] = "strict"
    provider: Optional[str] = "mock"
    top_k: int = 3


class RAGResponse(BaseModel):
    """Output contract for the RAG Orchestrator."""
    answer: str
    status: Literal[
        "success",
        "refusal_safety",
        "refusal_offtopic",
        "refusal_insufficient_evidence",
        "refusal_ungrounded",
        "error"
    ]
    confidence: float = 1.0
    grounded: bool = True
    grounding_method: str = "overlap"
    source_chunks: List[str] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    refusal_reason: Optional[str] = None
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    mode: Literal["strict", "quality"] = "strict"
