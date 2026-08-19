"""
Ticket 2 / Spec Section 8: Centralized System Configuration.

Contains all system-wide defaults, threshold values, timeout budgets,
and provider configurations across RAG, Guardrails, LLMs, and Voice pipelines.
"""

import os
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field


# Preserved Embedding Profiles (Active vs Rollback)
EMBEDDING_PROFILES: Dict[str, Dict[str, Any]] = {
    "bge_m3": {
        "model_key": "bge_m3",
        "model_name": "BAAI/bge-m3",
        "dimension": 1024,
        "cache_subdir": "faiss_cache_bge_m3",
        "description": "Production: High-accuracy multilingual model (1024 dimensions)",
    },
    "minilm": {
        "model_key": "minilm",
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "cache_subdir": "faiss_cache_minilm",
        "description": "Rollback: Lightweight multilingual MiniLM (384 dimensions)",
    },
}


class GuardrailThresholds(BaseModel):
    """Centralized guardrail thresholds."""
    relevance_threshold: float = Field(default=0.45, description="Min cosine similarity for query relevance")
    insufficient_evidence_threshold: float = Field(default=0.45, description="Min confidence score for sufficient evidence")
    groundedness_high_threshold: float = Field(default=0.20, description="Min token overlap for high groundedness")
    groundedness_low_threshold: float = Field(default=0.10, description="Max token overlap before hallucination refusal")


class LatencyBudgets(BaseModel):
    """Timeout budgets (in milliseconds)."""
    strict_llm_timeout_ms: int = Field(default=150, description="STRICT mode max LLM generation time")
    quality_llm_timeout_ms: int = Field(default=500, description="QUALITY mode max LLM generation time")
    embed_retrieval_target_ms: float = Field(default=50.0, description="Target for embed+retrieval combined")
    rag_pipeline_target_ms: float = Field(default=80.0, description="Target for context assembly")
    harness_target_ms: float = Field(default=200.0, description="Spec target for text to grounded answer")


class AppConfig(BaseModel):
    """Global system configuration."""
    environment: str = Field(default="production")
    default_mode: Literal["strict", "quality"] = Field(default="strict")
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "mock"))
    
    # Active Production Embedding Configuration: BAAI/bge-m3 (1024d)
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge_m3"))
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    
    # Explicit Rollback Configuration Reference: MiniLM (384d)
    rollback_embedding_model: str = Field(default="minilm")
    rollback_embedding_dim: int = Field(default=384)
    
    retriever_backend: str = Field(default="faiss_hnsw")
    default_top_k: int = Field(default=5)
    max_passages: int = Field(default_factory=lambda: int(os.getenv("MAX_PASSAGES", "50000")))
    
    # Nested configs
    guardrails: GuardrailThresholds = Field(default_factory=GuardrailThresholds)
    latencies: LatencyBudgets = Field(default_factory=LatencyBudgets)


# Singleton global config instance
settings = AppConfig()
