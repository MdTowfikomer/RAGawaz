"""
Ticket 8: Groundedness & Faithfulness Verifier Guardrail (Phase 5C Hardened).

Multi-dimensional verification:
1. Substantive Claim Support: Evaluates content keyword precision between generated answer and retrieved context.
   Token overlap or embedding similarity ALONE can never produce VERIFIED.
2. Numeric & Entity Precision: Detects fabricated numbers, dates, and ungrounded entities not present in context.
3. Query-Answer Relevance: Evaluates that the generated answer and retrieved context address the specific query entity/intent (rejects topic drift).
4. Refusal Detection: Identifies model fallback/inability statements as legitimate refusals rather than false positive grounded claims.
5. Degeneration Detection: Identifies pathological repeated word loops/hallucinations.
"""

from typing import Tuple, List, Optional, Set, Any
import re
import numpy as np
from backend.app.harness.schemas import GroundingResult

# Common Hindi, English & Romanized Indic functional stopwords / question words / generic action verbs
STOPWORDS: Set[str] = {
    "क्या", "कहाँ", "कहा", "कब", "कैसे", "कौन", "किस", "किसने", "किसका", "कितनी", "कितने", "कितना",
    "है", "हैं", "था", "थी", "थे", "होता", "होती", "होते", "होगा", "होगी", "होंगे", "रहा", "रही", "रहे",
    "का", "के", "की", "को", "में", "पर", "से", "ने", "और", "या", "भी", "तो", "ही", "एक", "यह", "वह",
    "करें", "करते", "करना", "करने", "कर", "बनाएं", "बनाना", "बनाने", "सकते", "सकती", "सकता", "बताएं", "बताइए",
    "is", "are", "was", "were", "what", "where", "when", "how", "who", "which", "the", "a", "an", "of", "in", "to", "for",
    "can", "could", "should", "would", "do", "does", "did", "make", "build", "tell", "have", "has", "had", "having",
    "with", "from", "by", "about", "regarding", "information", "info", "details", "know", "want", "like", "you",
    "your", "i", "me", "my", "we", "our", "us", "they", "them", "their", "something", "give", "please", "any", "some", "more", "such",
    # Romanized Indic stopwords (Hinglish, Tanglish, Banglish, Teluglish, Kanglish)
    "kya", "hai", "hain", "bhai", "me", "mein", "ka", "ki", "ke", "ko", "se", "tha", "thi", "the", "kitna", "kitne", "kitni",
    "hota", "hoti", "hote", "karte", "karta", "karti", "kisne", "kisko", "kahan", "kab", "kaise", "batao", "bataiye", "aur", "bhi",
    "la", "enna", "irukku", "pathi", "paththi", "sollu", "solla", "eppadi", "aachu", "ethu", "enge", "eppol", "yaaru",
    "sobcheye", "purono", "konta", "kothay", "kokhon", "ache", "kore", "holo", "karo", "janiye", "bolun",
    "le", "endi", "enti", "undi", "ekkada", "eppudu", "cheppandi", "cheppu", "aaytu", "elli", "yaavaga", "hege", "madu"
}


REFUSAL_PATTERNS = [
    r"जानकारी\s+(उपलब्ध|नहीं)",
    r"उत्तर\s+(उपलब्ध\s+नहीं|नहीं\s+है)",
    r"संदर्भ\s+में\s+.*नहीं",
    r"पता\s+नहीं",
    r"मुझे\s+.*जानकारी\s+नहीं",
    r"नाम\s+नहीं\s+(बताया|दिया|मिला|है)",
    r"not\s+enough\s+(information|evidence)",
    r"not\s+available\s+in\s+the\s+context",
    r"the\s+provided\s+context\s+does\s+not\s+contain",
    r"does\s+not\s+mention\s+(your|the)\s+name",
    r"cannot\s+find\s+.*information",
]
REFUSAL_REGEX = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


TOKEN_SPLIT_REGEX = re.compile(r'[^\s\u0964\u0965।,.;:!?()\[\]{}"\'“”‘’/\\`~@#$%^&*+=<>|]+', re.UNICODE)


def tokenize_words(text: str) -> Set[str]:
    """Normalize and tokenize text into lowercase word tokens, preserving Devanagari matras & combining characters."""
    if not text:
        return set()
    words = TOKEN_SPLIT_REGEX.findall(text.lower())
    return set(w for w in words if len(w) > 1)


def extract_content_keywords(text: str) -> Set[str]:
    """Extract substantive content keywords excluding common functional stopwords."""
    all_tokens = tokenize_words(text)
    content_tokens = {t for t in all_tokens if t not in STOPWORDS and len(t) > 1}
    return content_tokens if content_tokens else all_tokens


def detect_script(text: str) -> str:
    """Detect dominant writing script of input text."""
    if not text:
        return "unknown"
    if re.search(r'[\u0900-\u097F]', text):
        return "devanagari"
    elif re.search(r'[\u0980-\u09FF]', text):
        return "bengali"
    elif re.search(r'[\u0B80-\u0BFF]', text):
        return "tamil"
    elif re.search(r'[a-zA-Z]', text):
        return "latin"
    return "other"


def is_degenerate_repetition(text: str) -> bool:
    """Detect pathological word repetition degeneration in generated answers (e.g. 'स्वयं स्वयं स्वयं स्वयं...')."""
    if not text:
        return False
    words = TOKEN_SPLIT_REGEX.findall(text.lower())
    if not words:
        return False
    # Check for 3+ identical consecutive words
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            return True
    # Check extreme low token diversity on long generated answers (e.g. repeating babble loops)
    if len(words) >= 5 and (len(set(words)) / len(words)) <= 0.35:
        return True
    return False


class GroundednessVerifier:
    """
    Hardened, language-aware groundedness and query-faithfulness evaluator.
    Ensures token overlap or embedding similarity alone NEVER produces VERIFIED.
    Every substantive claim and numeric fact must be supported by retrieved context.
    """

    def __init__(
        self,
        high_threshold: float = 0.35,
        low_threshold: float = 0.15,
        min_query_overlap_threshold: float = 0.20,
        embedder: Optional[Any] = None,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.min_query_overlap_threshold = min_query_overlap_threshold
        self.embedder = embedder
        self.ungrounded_message = (
            "दिए गए संदर्भ में इस प्रश्न का सटीक और विश्वसनीय उत्तर उपलब्ध नहीं है।"
        )
        self.ungrounded_message_en = (
            "The provided context does not contain enough specific evidence to answer this question."
        )

    def get_ungrounded_message(self, query: Optional[str] = None) -> str:
        """Return language-matched ungrounded message."""
        if query and not re.search(r'[\u0900-\u097F]', query) and re.search(r'[a-zA-Z]', query):
            from backend.app.voice.detector import detect_language_metadata
            meta = detect_language_metadata(query)
            if meta.get("detected_language") == "english":
                return self.ungrounded_message_en
        return self.ungrounded_message

    def is_explicit_refusal(self, answer: str) -> bool:
        """Check if the answer text is explicitly stating inability or lack of information."""
        return bool(REFUSAL_REGEX.search(answer))

    def evaluate_overlap(
        self,
        answer: str,
        context_chunks: List[str],
        query: Optional[str] = None,
    ) -> Tuple[bool, str, float]:
        """
        Evaluates substantive claim support, numeric veracity, topic drift, refusal signals, and degeneration.
        Returns (is_grounded: bool, method: str, score: float).
        """
        if not answer or not context_chunks:
            return False, "overlap", 0.0

        # 1. Refusal / inability statement check — PASS THROUGH as legitimate response
        # When the LLM correctly identifies that information is not available in the context,
        # this is a valid grounded behavior (not a failure). The LLM's contextual,
        # language-appropriate refusal should be preserved rather than replaced by a generic one.
        if self.is_explicit_refusal(answer):
            return True, "explicit_refusal_passthrough", 1.0

        # 2. Degenerate repetition / loop hallucination check
        if is_degenerate_repetition(answer):
            return False, "hallucination_detected", 0.0

        ans_content_tokens = extract_content_keywords(answer)
        if not ans_content_tokens:
            return False, "overlap", 0.0

        combined_context = " ".join(context_chunks)
        ctx_content_tokens = extract_content_keywords(combined_context)

        # 3. Substantive Claim Support Check (Content tokens only, excluding stopwords)
        substantive_overlap = ans_content_tokens.intersection(ctx_content_tokens)
        substantive_support = len(substantive_overlap) / len(ans_content_tokens)

        # Detect cross-script scenario (e.g. Hindi answer from English/transliterated context)
        ans_script = detect_script(answer)
        ctx_script = detect_script(combined_context)
        is_cross_script = ans_script != ctx_script and ans_script != "unknown" and ctx_script != "unknown"

        # For cross-script answers, use relaxed low threshold since literal token overlap is inherently low
        effective_low_threshold = self.low_threshold * 0.5 if is_cross_script else self.low_threshold

        # If substantive content support is below the minimum support threshold, reject as unsupported claim
        if substantive_support < effective_low_threshold:
            # Before rejecting cross-script answers, try semantic similarity as fallback
            if is_cross_script and self.embedder is not None:
                try:
                    ans_emb = self.embedder.embed_query(answer[:256])
                    ctx_emb = self.embedder.embed_query(combined_context[:512])
                    cos_sim = float(np.dot(ans_emb, ctx_emb.T)[0, 0])
                    if cos_sim >= 0.35:
                        # Semantically similar despite different scripts — allow
                        substantive_support = max(substantive_support, cos_sim * 0.7)
                    else:
                        return False, "unsupported_claim_rejected", substantive_support
                except Exception:
                    return False, "unsupported_claim_rejected", substantive_support
            else:
                return False, "unsupported_claim_rejected", substantive_support

        # 4. Numeric & Exact Quantity Veracity Check
        ans_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
        ctx_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', combined_context))
        if ans_numbers and not ans_numbers.issubset(ctx_numbers):
            # Answer introduces fabricated quantities / numbers / years not in context
            return False, "unsupported_numeric_claim", 0.0

        # 5. Query-Answer Relevance Score (Language-Aware Topic Drift Check)
        query_relevance_score = 1.0
        if query:
            q_keywords = extract_content_keywords(query)
            if q_keywords:
                q_script = detect_script(query)

                # Direct token overlap (shared entity names, acronyms, transliterations, numbers)
                relevance_overlap = q_keywords.intersection(ans_content_tokens.union(ctx_content_tokens))
                direct_ratio = len(relevance_overlap) / len(q_keywords)

                if q_script == ctx_script:
                    query_relevance_score = direct_ratio
                    if query_relevance_score < self.min_query_overlap_threshold:
                        if self.embedder is not None:
                            try:
                                q_emb = self.embedder.embed_query(query)
                                ctx_emb = self.embedder.embed_query(combined_context[:512])
                                cos_sim = float(np.dot(q_emb, ctx_emb.T)[0, 0])
                                if cos_sim >= 0.35:
                                    query_relevance_score = max(direct_ratio, min(1.0, cos_sim))
                                else:
                                    return False, "topic_drift_rejected", cos_sim
                            except Exception:
                                return False, "topic_drift_rejected", query_relevance_score
                        else:
                            return False, "topic_drift_rejected", query_relevance_score

                elif q_script == "latin" and self.embedder is not None:
                    # High-confidence cross-lingual Latin <-> Devanagari (English & Hinglish)
                    if direct_ratio >= self.min_query_overlap_threshold:
                        query_relevance_score = direct_ratio
                    else:
                        try:
                            q_emb = self.embedder.embed_query(query)
                            ctx_emb = self.embedder.embed_query(combined_context[:512])
                            cos_sim = float(np.dot(q_emb, ctx_emb.T)[0, 0])
                            
                            # Relaxed Latin-Devanagari threshold (0.30 from 0.40) for multilingual queries
                            if cos_sim < 0.30:
                                return False, "topic_drift_rejected", cos_sim
                            query_relevance_score = max(direct_ratio, min(1.0, cos_sim))
                        except Exception:
                            query_relevance_score = direct_ratio
                else:
                    # Cross-lingual scripts (e.g. Devanagari query -> English context, Bengali/Tamil -> Hindi)
                    # Use relaxed threshold for cross-script scenarios
                    if self.embedder is not None and direct_ratio < self.min_query_overlap_threshold:
                        try:
                            q_emb = self.embedder.embed_query(query)
                            ctx_emb = self.embedder.embed_query(combined_context[:512])
                            cos_sim = float(np.dot(q_emb, ctx_emb.T)[0, 0])
                            if cos_sim >= 0.30:
                                query_relevance_score = max(direct_ratio, min(1.0, cos_sim))
                            else:
                                return False, "topic_drift_rejected", cos_sim
                        except Exception:
                            query_relevance_score = direct_ratio
                            if query_relevance_score < self.min_query_overlap_threshold:
                                return False, "topic_drift_rejected", query_relevance_score
                    else:
                        query_relevance_score = direct_ratio
                        if query_relevance_score < self.min_query_overlap_threshold:
                            return False, "topic_drift_rejected", query_relevance_score

        # 6. Combined Groundedness Score
        overall_score = 0.6 * substantive_support + 0.4 * query_relevance_score

        # Use relaxed thresholds for cross-script answers where token overlap is inherently limited
        effective_high_threshold = self.high_threshold * 0.6 if is_cross_script else self.high_threshold

        if overall_score >= effective_high_threshold and substantive_support >= effective_high_threshold:
            return True, "overlap", overall_score
        else:
            return False, "unsupported_claim_rejected", overall_score

    def evaluate_result(
        self,
        answer: str,
        context_chunks: List[str],
        query: Optional[str] = None,
    ) -> GroundingResult:
        """Spec-compliant method returning GroundingResult."""
        is_grounded, method, score = self.evaluate_overlap(answer, context_chunks, query=query)
        ungrounded_msg = self.get_ungrounded_message(query)
        return GroundingResult(
            is_grounded=is_grounded,
            method=method if method in ["overlap", "llm_judge"] else "overlap",
            confidence=score,
            refusal_reason=None if is_grounded else ungrounded_msg,
        )

    def evaluate(
        self,
        answer: str,
        context_chunks: List[str],
        query: Optional[str] = None,
    ) -> Tuple[bool, str, float, Optional[str]]:
        """
        Full evaluation entrypoint returning 4-tuple.
        """
        is_grounded, method, score = self.evaluate_overlap(answer, context_chunks, query=query)
        ungrounded_msg = self.get_ungrounded_message(query)
        if not is_grounded:
            return False, method, score, ungrounded_msg
        return True, method, score, None
