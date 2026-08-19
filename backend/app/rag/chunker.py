"""
Ticket 3: Multi-Strategy Chunking Engine.

Implements 4 distinct, production-grade chunking strategies:
1. Fixed + Overlap (token/char window sliding)
2. Semantic / Sentence Splitting (Hindi punctuation & Devanagari danda '।' boundaries)
3. Parent-Child Hierarchical (micro-child search units linked to macro-parent context)
4. Adaptive / Structure-Aware (dynamic splitting based on length, paragraph & sentence density)
"""

import re
from typing import List, Dict, Any


HINDI_SENTENCE_DELIMITERS = re.compile(r'([।?!;\n]+)')


def split_hindi_sentences(text: str) -> List[str]:
    """Split Hindi text on sentence boundaries (Devanagari danda ।, question marks, newlines)."""
    parts = HINDI_SENTENCE_DELIMITERS.split(text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = (parts[i] + parts[i+1]).strip()
        if len(sentence) >= 2:
            sentences.append(sentence)
    if len(parts) % 2 == 1 and len(parts[-1].strip()) >= 2:
        sentences.append(parts[-1].strip())
    return sentences if sentences else [text.strip()]


def chunk_fixed_overlap(
    text: str,
    chunk_size: int = 250,
    overlap: int = 50,
) -> List[str]:
    """Strategy 1: Fixed character/token window with sliding overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= 5:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step

    return chunks if chunks else [text.strip()]


def chunk_semantic(
    text: str,
    max_chunk_size: int = 300,
    target_sentences: int = 3,
) -> List[str]:
    """Strategy 2: Semantic sentence-boundary grouping."""
    sentences = split_hindi_sentences(text)
    if len(sentences) <= 1:
        return [text]

    chunks = []
    current_sentences = []
    current_length = 0

    for s in sentences:
        if current_length + len(s) > max_chunk_size and current_sentences:
            chunk = " ".join(current_sentences).strip()
            if len(chunk) >= 5:
                chunks.append(chunk)
            current_sentences = [s]
            current_length = len(s)
        else:
            current_sentences.append(s)
            current_length += len(s)
            if len(current_sentences) >= target_sentences:
                chunk = " ".join(current_sentences).strip()
                if len(chunk) >= 5:
                    chunks.append(chunk)
                current_sentences = []
                current_length = 0

    if current_sentences:
        chunk = " ".join(current_sentences).strip()
        if len(chunk) >= 5:
            chunks.append(chunk)

    return chunks if chunks else [text.strip()]


def chunk_parent_child(
    text: str,
    child_size: int = 120,
    child_overlap: int = 20,
) -> Dict[str, Any]:
    """
    Strategy 3: Hierarchical Parent-Child chunking.
    Parent is the whole or macro-passage context (~500 chars).
    Children are micro-passages (~100-150 chars) for precise vector matching.
    """
    parent_text = text.strip()
    child_texts = chunk_fixed_overlap(text, chunk_size=child_size, overlap=child_overlap)

    return {
        "parent": parent_text,
        "children": child_texts,
    }


def chunk_adaptive(
    text: str,
    short_threshold: int = 220,
    medium_threshold: int = 600,
) -> List[str]:
    """
    Strategy 4: Adaptive Structure-Aware chunking.
    - Short passages (< short_threshold): kept intact as single chunk.
    - Medium passages (short to medium): split into 2-sentence groups.
    - Long passages (> medium_threshold): split on paragraph and structural density.
    """
    length = len(text)
    if length <= short_threshold:
        return [text]
    elif length <= medium_threshold:
        return chunk_semantic(text, max_chunk_size=250, target_sentences=2)
    else:
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= 5]
        if len(paragraphs) > 1:
            chunks = []
            for p in paragraphs:
                if len(p) <= short_threshold:
                    chunks.append(p)
                else:
                    chunks.extend(chunk_semantic(p, max_chunk_size=250, target_sentences=2))
            return chunks if chunks else [text.strip()]
        else:
            return chunk_semantic(text, max_chunk_size=300, target_sentences=3)


def chunk_passage(
    passage: Dict[str, Any],
    strategy: str = "fixed",
) -> List[Dict[str, Any]]:
    """
    Chunk a single passage record using the specified strategy.
    Attaches full metadata schema including optional parent/child linkage.
    """
    passage_id = passage["passage_id"]
    query_id = passage["query_id"]
    query = passage.get("query", "")
    is_selected = passage.get("is_selected", 0)
    text = passage["text"]
    language = passage.get("language", "hi")

    chunks_out: List[Dict[str, Any]] = []

    if strategy == "fixed":
        raw_chunks = chunk_fixed_overlap(text)
        for idx, ct in enumerate(raw_chunks):
            chunks_out.append({
                "chunk_id": f"{passage_id}_fix_{idx}",
                "passage_id": passage_id,
                "query_id": query_id,
                "query": query,
                "text": ct,
                "chunk_strategy": "fixed",
                "position": idx,
                "parent_id": None,
                "children_ids": None,
                "is_selected": is_selected,
                "language": language,
            })

    elif strategy == "semantic":
        raw_chunks = chunk_semantic(text)
        for idx, ct in enumerate(raw_chunks):
            chunks_out.append({
                "chunk_id": f"{passage_id}_sem_{idx}",
                "passage_id": passage_id,
                "query_id": query_id,
                "query": query,
                "text": ct,
                "chunk_strategy": "semantic",
                "position": idx,
                "parent_id": None,
                "children_ids": None,
                "is_selected": is_selected,
                "language": language,
            })

    elif strategy == "parent_child":
        hierarchy = chunk_parent_child(text)
        parent_id = f"{passage_id}_parent"
        child_ids = [f"{passage_id}_child_{idx}" for idx in range(len(hierarchy["children"]))]

        # Add parent chunk
        chunks_out.append({
            "chunk_id": parent_id,
            "passage_id": passage_id,
            "query_id": query_id,
            "query": query,
            "text": hierarchy["parent"],
            "chunk_strategy": "parent_child_parent",
            "position": 0,
            "parent_id": None,
            "children_ids": child_ids,
            "is_selected": is_selected,
            "language": language,
        })

        # Add child chunks
        for idx, ct in enumerate(hierarchy["children"]):
            chunks_out.append({
                "chunk_id": child_ids[idx],
                "passage_id": passage_id,
                "query_id": query_id,
                "query": query,
                "text": ct,
                "chunk_strategy": "parent_child_child",
                "position": idx,
                "parent_id": parent_id,
                "children_ids": None,
                "is_selected": is_selected,
                "language": language,
            })

    elif strategy == "adaptive":
        raw_chunks = chunk_adaptive(text)
        for idx, ct in enumerate(raw_chunks):
            chunks_out.append({
                "chunk_id": f"{passage_id}_adp_{idx}",
                "passage_id": passage_id,
                "query_id": query_id,
                "query": query,
                "text": ct,
                "chunk_strategy": "adaptive",
                "position": idx,
                "parent_id": None,
                "children_ids": None,
                "is_selected": is_selected,
                "language": language,
            })

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return chunks_out


def chunk_corpus(
    passages: List[Dict[str, Any]],
    strategy: str = "fixed",
) -> List[Dict[str, Any]]:
    """Chunk an entire corpus of passage records."""
    all_chunks = []
    for p in passages:
        all_chunks.extend(chunk_passage(p, strategy=strategy))
    return all_chunks
