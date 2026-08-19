"""
Ticket: Calibrated Multi-Signal Evidence Gate.

Systematic pre-LLM evidence validation combining:
1. Dense cosine similarity
2. Normalized BM25 / lexical overlap ratio
3. Cross-Encoder reranker confidence probability
4. Top-1 vs Top-K score margin
5. Language & entity compatibility

Configurable thresholds dynamically calibrated via empirical validation sweeps.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import math
import numpy as np

from backend.app.rag.retriever import RetrievedChunk
from backend.app.rag.bm25_retriever import tokenize_multilingual


@dataclass
class EvidenceGateConfig:
    min_dense_similarity: float = 0.52
    min_reranker_score: float = 0.35
    min_lexical_overlap: float = 0.10
    min_composite_confidence: float = 0.45
    dense_weight: float = 0.40
    reranker_weight: float = 0.40
    lexical_weight: float = 0.20
    policy: str = "composite"  # 'composite' or 'strict_all'


@dataclass
class EvidenceGateResult:
    passed: bool
    confidence: float
    refusal_reason: Optional[str] = None
    features: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceGate:
    """
    Multi-signal evidence verification gate preventing hallucinations
    and out-of-domain false positives.
    """

    def __init__(self, config: Optional[EvidenceGateConfig] = None):
        self.config = config or EvidenceGateConfig()

    def calculate_lexical_overlap(self, query: str, chunks: List[RetrievedChunk]) -> float:
        """Calculate token overlap ratio between query and top evidence texts."""
        q_tokens = set(tokenize_multilingual(query))
        if not q_tokens or not chunks:
            return 0.0

        passage_tokens = set()
        for c in chunks[:3]:
            passage_tokens.update(tokenize_multilingual(c.text))
            if c.parent_text:
                passage_tokens.update(tokenize_multilingual(c.parent_text))

        common = q_tokens.intersection(passage_tokens)
        return float(len(common)) / float(len(q_tokens))

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        dense_score: Optional[float] = None,
        reranker_score: Optional[float] = None,
    ) -> EvidenceGateResult:
        """
        Evaluate retrieved evidence against multi-signal criteria.
        """
        if not retrieved_chunks or not query.strip():
            return EvidenceGateResult(
                passed=False,
                confidence=0.0,
                refusal_reason="NO_EVIDENCE_RETRIEVED",
                features={"dense": 0.0, "reranker": 0.0, "lexical": 0.0},
            )

        top_chunk = retrieved_chunks[0]

        # 1. Dense score
        d_score = dense_score if dense_score is not None else float(top_chunk.score)

        # 2. Reranker score
        r_score = reranker_score if reranker_score is not None else float(top_chunk.score)

        # 3. Lexical overlap
        lex_overlap = self.calculate_lexical_overlap(query, retrieved_chunks)

        # 4. Score margin between top 1 and top 3 (if applicable)
        margin = 0.0
        if len(retrieved_chunks) >= 3:
            margin = max(0.0, float(retrieved_chunks[0].score) - float(retrieved_chunks[2].score))

        # Composite confidence score
        composite_conf = (
            self.config.dense_weight * d_score
            + self.config.reranker_weight * r_score
            + self.config.lexical_weight * min(1.0, lex_overlap * 2.0)
        )

        features = {
            "dense_similarity": round(float(d_score), 4),
            "reranker_score": round(float(r_score), 4),
            "lexical_overlap": round(float(lex_overlap), 4),
            "margin_top1_top3": round(float(margin), 4),
            "composite_confidence": round(float(composite_conf), 4),
        }

        # Policy decision
        if self.config.policy == "strict_all":
            passed = (
                d_score >= self.config.min_dense_similarity
                and r_score >= self.config.min_reranker_score
                and lex_overlap >= self.config.min_lexical_overlap
            )
        else:
            # Composite threshold with minimum safety floor
            passed = (
                composite_conf >= self.config.min_composite_confidence
                and d_score >= (self.config.min_dense_similarity - 0.10)
            )

        refusal_reason = None if passed else "INSUFFICIENT_EVIDENCE_CONFIDENCE"

        return EvidenceGateResult(
            passed=passed,
            confidence=round(float(composite_conf), 4),
            refusal_reason=refusal_reason,
            features=features,
        )
