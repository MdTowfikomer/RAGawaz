import re
from typing import List
from backend.app.rag.chunker import split_hindi_sentences
from backend.app.guardrails.groundedness import extract_content_keywords


def _clean_passage_text(text: str) -> str:
    """Remove web URLs, search snippet headers, and common metadata prefixes."""
    # Remove URLs
    cleaned = re.sub(r"https?://\S+", "", text)
    # Remove 'query | ' or 'title - ' artifacts
    if " | " in cleaned:
        cleaned = cleaned.split(" | ", 1)[-1]
    # Remove common forum / portal headers
    cleaned = re.sub(r"^[A-Za-z0-9\s\+]+-\s+[A-Za-z0-9\s]+-\s+Ask Me Help Desk\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(Best Answer:|Quick Answer:|Popular Posts\.|Answer:)\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_best_answer_span(query: str, retrieved_texts: List[str]) -> str:
    """
    Select the single most relevant and fact-dense sentence from the retrieved passages.
    Runs purely in memory in < 0.5ms.
    """
    if not retrieved_texts:
        return "उत्तर उपलब्ध नहीं है।"

    query_keywords = extract_content_keywords(query)
    best_sentence = ""
    best_score = -1.0

    for raw_text in retrieved_texts:
        cleaned_text = _clean_passage_text(raw_text)
        if not cleaned_text:
            continue
        sentences = split_hindi_sentences(cleaned_text)
        for sent in sentences:
            sent_clean = sent.strip()
            if len(sent_clean) < 12:
                continue
            # Skip if sentence is mostly a leftover URL fragment
            if re.search(r"\.(com|org|net|html|php)\b", sent_clean):
                continue
            sent_keywords = extract_content_keywords(sent_clean)
            if not sent_keywords:
                continue
            overlap = len(query_keywords.intersection(sent_keywords))
            score = overlap / (len(query_keywords) + 1e-5)
            # Prefer concise, information-dense sentences with highest query overlap
            if score > best_score:
                best_score = score
                best_sentence = sent_clean

    if best_sentence and best_score > 0:
        return best_sentence

    # Fallback to first clean sentence across retrieved texts
    for raw_text in retrieved_texts:
        cleaned_text = _clean_passage_text(raw_text)
        sentences = split_hindi_sentences(cleaned_text)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 15 and not re.search(r"\.(com|org|net|html)\b", s_clean):
                return s_clean

    return _clean_passage_text(retrieved_texts[0]) or "उत्तर उपलब्ध नहीं है।"

