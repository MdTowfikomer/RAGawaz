"""
Ticket: Multilingual Cross-Encoder Reranker.

Applies deep cross-attention over (Query, Chunk) candidate pairs to compute
calibrated relevance logits. Filters dense+sparse candidate pools down to
the top 3-5 highest-precision evidence passages.
"""

import os
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder

from backend.app.rag.retriever import RetrievedChunk


class MultilingualReranker:
    """
    Multilingual Cross-Encoder Reranker using BAAI/bge-reranker-base.
    Evaluates query-candidate pairs with cross-attention on GPU/CPU.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            t0 = time.perf_counter()
            print(f"[MultilingualReranker] Loading '{self.model_name}' on {self.device}...", flush=True)
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            if self.device == "cuda":
                # Ensure model runs in FP16 to minimize VRAM footprint (< 600MB)
                try:
                    self._model.model.half()
                except Exception:
                    pass
            t_load = time.perf_counter() - t0
            print(f"[MultilingualReranker] Loaded in {t_load:.2f}s.", flush=True)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Rerank candidates using cross-attention score.
        Returns top_k chunks sorted by descending cross-encoder score.
        """
        if not candidates or not query:
            return []

        model = self._load_model()
        pairs = [(query, c.text) for c in candidates]

        with torch.inference_mode():
            scores = model.predict(
                pairs,
                batch_size=min(self.batch_size, len(pairs)),
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # Convert logits to probabilities via sigmoid for calibrated confidence
        probs = 1.0 / (1.0 + np.exp(-scores))

        reranked_chunks: List[Tuple[float, RetrievedChunk]] = []
        for prob, chunk in zip(probs, candidates):
            reranked = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                passage_id=chunk.passage_id,
                query_id=chunk.query_id,
                text=chunk.text,
                score=float(prob),  # Calibrated probability in [0, 1]
                chunk_strategy=chunk.chunk_strategy,
                position=chunk.position,
                parent_id=chunk.parent_id,
                children_ids=chunk.children_ids,
                language=chunk.language,
                is_selected=chunk.is_selected,
                parent_text=chunk.parent_text,
            )
            reranked_chunks.append((float(prob), reranked))

        # Sort by reranker probability
        reranked_chunks.sort(key=lambda x: x[0], reverse=True)

        return [item[1] for item in reranked_chunks[:top_k]]
