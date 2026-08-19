"""
Phase 6B: Minimal-Context Indic Chunker Engine.

Implements sentence-aware, minimal-context chunking (target: 180-220 characters / 1-2 sentences)
respecting Devanagari danda '।', Indic punctuation, and strict word boundaries.
"""

import os
import re
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Generator, Tuple


# Sentence delimiter regex for Indic languages (Devanagari danda ।, question mark, exclamation, semicolon, newlines, English full stop)
INDIC_SENTENCE_DELIMITERS = re.compile(r'([।?!;\n]+)')


@dataclass
class ChunkingStats:
    """Detailed summary statistics for minimal-context chunking."""
    total_passages_processed: int = 0
    total_chunks_generated: int = 0
    expansion_ratio: float = 0.0
    avg_chunk_chars: float = 0.0
    min_chunk_chars: int = 0
    max_chunk_chars: int = 0
    total_chars_in_chunks: int = 0
    length_distribution: Dict[str, int] = field(default_factory=lambda: {
        "<100": 0,
        "100-150": 0,
        "151-200": 0,
        "201-250": 0,
        ">250": 0,
    })
    elapsed_seconds: float = 0.0
    chunks_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def split_sentences_indic(text: str) -> List[str]:
    """
    Split Indic text on sentence boundaries (Devanagari danda ।, ?, !, newlines).
    Preserves delimiters on sentences and strips extraneous whitespace.
    """
    parts = INDIC_SENTENCE_DELIMITERS.split(text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = (parts[i] + parts[i+1]).strip()
        if len(sentence) >= 2:
            sentences.append(sentence)
    if len(parts) % 2 == 1 and len(parts[-1].strip()) >= 2:
        sentences.append(parts[-1].strip())
    return sentences if sentences else [text.strip()]


def split_long_sentence_on_words(sentence: str, max_chars: int = 220) -> List[str]:
    """
    Splits an unusually long sentence into chunks at word boundaries without breaking words.
    """
    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split()
    sub_chunks = []
    current_words = []
    current_len = 0

    for w in words:
        word_len = len(w) + (1 if current_words else 0)
        if current_len + word_len > max_chars and current_words:
            sub_chunks.append(" ".join(current_words))
            current_words = [w]
            current_len = len(w)
        else:
            current_words.append(w)
            current_len += word_len

    if current_words:
        sub_chunks.append(" ".join(current_words))

    return sub_chunks if sub_chunks else [sentence]


def chunk_minimal_context(
    text: str,
    target_min: int = 180,
    target_max: int = 220,
    hard_max: int = 250,
) -> List[str]:
    """
    Minimal-context chunking:
    - Groups 1-2 Indic sentences to approach the 180-220 character target window.
    - If a single sentence is in the 100-220 range, keeps it intact.
    - If grouping exceeds hard_max, closes the current chunk.
    - Never breaks words mid-string.
    """
    cleaned = text.strip()
    if len(cleaned) <= target_max:
        return [cleaned]

    sentences = split_sentences_indic(cleaned)
    if len(sentences) <= 1:
        # Fallback to word-boundary splitting
        return split_long_sentence_on_words(cleaned, max_chars=target_max)

    chunks: List[str] = []
    current_group: List[str] = []
    current_length = 0

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue

        # If individual sentence is longer than hard_max, split on word boundaries
        if len(s_clean) > hard_max:
            if current_group:
                chunks.append(" ".join(current_group).strip())
                current_group = []
                current_length = 0
            long_parts = split_long_sentence_on_words(s_clean, max_chars=target_max)
            chunks.extend(long_parts)
            continue

        added_len = len(s_clean) + (1 if current_group else 0)

        if current_length + added_len > target_max and current_group:
            # Current group is full
            chunk_str = " ".join(current_group).strip()
            if len(chunk_str) >= 5:
                chunks.append(chunk_str)
            current_group = [s_clean]
            current_length = len(s_clean)
        else:
            current_group.append(s_clean)
            current_length += added_len

            # If we've reached the target minimal window, emit chunk
            if current_length >= target_min:
                chunk_str = " ".join(current_group).strip()
                if len(chunk_str) >= 5:
                    chunks.append(chunk_str)
                current_group = []
                current_length = 0

    if current_group:
        chunk_str = " ".join(current_group).strip()
        if len(chunk_str) >= 5:
            # If the trailing chunk is small (<80 chars) and we have prior chunks, attach it to the previous chunk if total <= hard_max
            if len(chunk_str) < 80 and chunks and (len(chunks[-1]) + len(chunk_str) + 1) <= hard_max:
                chunks[-1] = f"{chunks[-1]} {chunk_str}".strip()
            else:
                chunks.append(chunk_str)

    # Final pass: merge any stray micro-chunks (<20 chars) with previous chunk
    valid_chunks = []
    for c in chunks:
        c_clean = c.strip()
        if len(c_clean) < 20 and valid_chunks and (len(valid_chunks[-1]) + len(c_clean) + 1) <= hard_max:
            valid_chunks[-1] = f"{valid_chunks[-1]} {c_clean}".strip()
        elif len(c_clean) >= 15:
            valid_chunks.append(c_clean)
        elif not valid_chunks:
            valid_chunks.append(c_clean)

    return valid_chunks if valid_chunks else [cleaned]


class MinimalContextChunker:
    """
    Streaming chunker engine that consumes passage records and outputs minimal-context chunks.
    """

    def __init__(
        self,
        target_min: int = 180,
        target_max: int = 220,
        hard_max: int = 250,
    ):
        self.target_min = target_min
        self.target_max = target_max
        self.hard_max = hard_max

    def chunk_passage(self, passage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single passage record and attach all required metadata.
        """
        passage_id = str(passage["passage_id"])
        query_id = int(passage["query_id"])
        query = str(passage.get("query", ""))
        is_selected = int(passage.get("is_selected", 0))
        text = str(passage["text"])
        language = str(passage.get("language", "hi"))

        text_chunks = chunk_minimal_context(
            text,
            target_min=self.target_min,
            target_max=self.target_max,
            hard_max=self.hard_max,
        )

        chunk_records: List[Dict[str, Any]] = []
        for idx, ct in enumerate(text_chunks):
            chunk_records.append({
                "chunk_id": f"{passage_id}_mchk_{idx}",
                "passage_id": passage_id,
                "query_id": query_id,
                "query": query,
                "text": ct,
                "chunk_strategy": "minimal_context",
                "position": idx,
                "parent_id": passage_id,
                "children_ids": None,
                "is_selected": is_selected,
                "language": language,
                "char_length": len(ct),
            })

        return chunk_records

    def chunk_stream(
        self,
        passages: Iterator[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], ChunkingStats]:
        """
        Process a stream or list of passages, optionally writing directly to output JSONL.
        """
        t_start = time.perf_counter()
        stats = ChunkingStats()
        out_handle = None

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_handle = open(out_p, "w", encoding="utf-8")

        all_chunks: List[Dict[str, Any]] = []
        lengths: List[int] = []

        for p in passages:
            stats.total_passages_processed += 1
            chunks = self.chunk_passage(p)

            for c in chunks:
                stats.total_chunks_generated += 1
                c_len = c["char_length"]
                lengths.append(c_len)
                stats.total_chars_in_chunks += c_len

                # Bucket distribution
                if c_len < 100:
                    stats.length_distribution["<100"] += 1
                elif c_len <= 150:
                    stats.length_distribution["100-150"] += 1
                elif c_len <= 200:
                    stats.length_distribution["151-200"] += 1
                elif c_len <= 250:
                    stats.length_distribution["201-250"] += 1
                else:
                    stats.length_distribution[">250"] += 1

                if out_handle:
                    out_handle.write(json.dumps(c, ensure_ascii=False) + "\n")
                else:
                    all_chunks.append(c)

        if out_handle:
            out_handle.close()

        t_end = time.perf_counter()
        stats.elapsed_seconds = t_end - t_start

        if stats.total_passages_processed > 0:
            stats.expansion_ratio = stats.total_chunks_generated / stats.total_passages_processed
        if stats.total_chunks_generated > 0:
            stats.avg_chunk_chars = stats.total_chars_in_chunks / stats.total_chunks_generated
            stats.min_chunk_chars = min(lengths) if lengths else 0
            stats.max_chunk_chars = max(lengths) if lengths else 0
        if stats.elapsed_seconds > 0:
            stats.chunks_per_second = stats.total_chunks_generated / stats.elapsed_seconds

        return all_chunks, stats
