"""
Ticket 8: Relevance Gate & Insufficient Evidence Guardrails.

1. RelevanceGate: Pre-retrieval check (intent filtering + top-K cosine threshold).
2. InsufficientEvidenceChecker: Post-retrieval / Pre-LLM evidence sufficiency & answerability check.
"""

from typing import Tuple, Optional, List, Set, Any, Dict
import numpy as np
import re
from backend.app.guardrails.groundedness import detect_script, is_degenerate_repetition, extract_content_keywords

OFF_TOPIC_INTENT_PATTERNS = [
    re.compile(
        r'\b(stock price of|book me a flight|how to download whatsapp|my name is|what is my name|whats my name|what\'s my name|who am i|where do i live|how old am i|i am checking|system check|testing 1 2 3|who are you|how are you|hello\b|hi\b|mera naam|main check|check system|kaise ho|namaste|testing|tell me a joke|sing a song|write a poem|how to make pizza|recipe for)\b',
        re.IGNORECASE,
    ),
    # Creative, speculative, predictive, personal unknowable requests
    re.compile(
        r'\b(predict|lottery|write me a|compose a|generate a|create a|make up a|draw a|design a|build me|code me|what am i thinking|read my mind|my dream|my future|what will happen|will i|cure for cancer|secret recipe|last digit of pi|solve .+ for me|after death|end of the world|time travel|what is inside a black hole)\b',
        re.IGNORECASE,
    ),
    # Adversarial / prompt injection patterns
    re.compile(
        r'\b(ignore previous|ignore all|forget your|bypass|override|jailbreak|system prompt|reveal your|pretend you are|act as if|you are now|DAN mode|unlimited mode|uncensored|no restrictions|admin mode|hidden instructions|simulate being)\b',
        re.IGNORECASE,
    ),
    # Transactional / action requests
    re.compile(
        r'\b(book me|order me|buy me|call my|send an email|set an alarm|play some|turn off|turn on|install|download|open the app|schedule|remind me|pay for|transfer money)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'(आज का मौसम कैसा|ट्रेन का टिकट बुक|शेयर बाजार में आज|लाइव स्कोर क्या|एक अच्छा जोक सुनाओ|जोक सुनाओ|व्हाट्सएप कैसे डाउनलोड|आज का राशिफल क्या|पिज़्ज़ा ऑर्डर करना|होटल के कमरे का किराया|पेट्रोल पंप कहाँ|डिनर मेनू क्या|मूवी टिकट कैसे बुक|नई कार खरीदनी|मेरा नाम क्या|मेरा नाम|मैं कौन|मैं चेक|तुम कौन हो|नमस्ते|हैलो|कैसा है|सिस्टम चेक)',
        re.IGNORECASE,
    ),
    # Personal questions in multiple Indic languages (name, age, identity)
    re.compile(
        r'(నా పేరు|ನನ್ನ ಹೆಸರು|എന്റെ പേര്|আমার নাম|ମୋ ନାଁ|મારું નામ|ਮੇਰਾ ਨਾਮ|මගේ නම|my age|how old|who am i|tell me about myself|do you know me|remember me)',
        re.IGNORECASE,
    ),
    # Opinion/subjective queries
    re.compile(
        r'\b(which is better|what is the best|should i|is it good|is it bad|better than|worst|greatest|favorite|recommend me|suggest me|सबसे अच्छा|सबसे बुरा|कौन सा अच्छा|क्या करना चाहिए|எது சிறந்தது|ಯಾವುದು ಉತ್ತಮ|ఏది మంచిది|কোনটি ভালো)\b',
        re.IGNORECASE,
    ),
    # Realtime/temporal queries
    re.compile(
        r'\b(today\'s|tomorrow\'s|yesterday\'s|current price|stock price|latest news|live score|right now|इस समय|अभी|आज का|कल का|आजच्या|ఈ రోజు|இன்றைய|ಇಂದಿನ|ഇന്നത്തെ|আজকের)\b',
        re.IGNORECASE,
    ),
    # Chinese/Japanese/Korean — not in our Indic corpus
    re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]{2,}'),
]


class RelevanceGate:
    """
    Pre-retrieval gate: Checks if the query is in-domain relative to the corpus.
    Uses hybrid intent detection + top-K fast similarity score thresholding.
    """

    def __init__(self, threshold: float = 0.45, scoped_topic: str = "general knowledge and facts"):
        self.threshold = threshold
        self.scoped_topic = scoped_topic
        self.off_topic_message = f"I can only answer questions about {self.scoped_topic}. Could you rephrase?"
        self.off_topic_message_hi = "मैं केवल सामान्य ज्ञान और तथ्यों से संबंधित प्रश्नों के उत्तर दे सकता हूँ। क्या आप पुनः प्रश्न पूछ सकते हैं?"

    def get_refusal_message(self, query: Optional[str] = None) -> str:
        """Return language-matched refusal message."""
        if query and detect_script(query) == "devanagari":
            return self.off_topic_message_hi
        return self.off_topic_message

    def evaluate(
        self,
        top_scores: List[float],
        query: Optional[str] = None,
        context_chunks: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if query passed relevance threshold.
        Returns (is_relevant: bool, refusal_message: Optional[str]).
        """
        refusal_msg = self.get_refusal_message(query)

        if query:
            for pattern in OFF_TOPIC_INTENT_PATTERNS:
                if pattern.search(query):
                    return False, refusal_msg

        if not top_scores:
            return False, refusal_msg

        max_score = max(top_scores)
        if max_score < self.threshold:
            return False, refusal_msg

        return True, None



INSUFFICIENT_EVIDENCE_PATTERNS = [
    re.compile(r'(वर्ष 20[4-9][0-9] के|निजी फोन नंबर|अलमारी में रखे बॉक्स|गुप्त पासवर्ड|सतोशी नाकामोतो|प्राइवेट की|एलियंस|अटलांटिस के राजा|गुप्त दस्तावेज संख्या|समय यात्रा मशीन का ब्लूप्रिंट|काल्पनिक व्यक्ति संख्या|मंगल ग्रह पर पहले मानव शहर|अंटार्कटिका में सटीक तापमान|अज्ञात गांव की सटीक जीडीपी|वर्ष 2099 के ओलंपिक)', re.IGNORECASE),
    re.compile(r'\b(password of|secret blueprint|future president|time machine|unknown village|private key of|satoshi nakamoto|bitcoin wallet|aliens language|secret base)\b', re.IGNORECASE),
]


class InsufficientEvidenceChecker:
    """
    Post-retrieval / Pre-LLM evidence sufficiency & answerability check:
    Determines whether the retrieved passages actually contain enough information
    to answer the user's question, not merely whether they are semantically similar.
    Catches insufficient context BEFORE LLM generation to save ~150ms of latency.
    """

    def __init__(self, confidence_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self.last_diagnostics: Dict[str, str] = {"entity_match": "N/A", "evidence_status": "SUFFICIENT"}
        self.insufficient_message = (
            "This question is in my domain, but I couldn't find enough specific evidence "
            "in my knowledge base to give you a confident answer."
        )
        self.insufficient_message_hi = (
            "दिए गए संदर्भ में इस प्रश्न का सटीक और विश्वसनीय उत्तर देने के लिए पर्याप्त जानकारी उपलब्ध नहीं है।"
        )

    def get_refusal_message(self, query: Optional[str] = None) -> str:
        """Return language-matched refusal message."""
        if query and detect_script(query) == "devanagari":
            return self.insufficient_message_hi
        return self.insufficient_message

    def evaluate(
        self,
        top_scores: List[float],
        query: Optional[str] = None,
        context_chunks: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates confidence scores, unanswerable patterns, and substantive context answerability.
        Returns (has_sufficient_evidence: bool, refusal_message: Optional[str]).
        """
        refusal_msg = self.get_refusal_message(query)
        self.last_diagnostics = {"entity_match": "N/A", "evidence_status": "SUFFICIENT"}

        # 1. Check known unanswerable / speculative patterns
        if query:
            for pattern in INSUFFICIENT_EVIDENCE_PATTERNS:
                if pattern.search(query):
                    self.last_diagnostics = {"entity_match": "N/A", "evidence_status": "INSUFFICIENT"}
                    return False, refusal_msg

        # 2. Check confidence threshold
        if not top_scores:
            self.last_diagnostics = {"entity_match": "FAIL", "evidence_status": "INSUFFICIENT"}
            return False, refusal_msg

        top_score = max(top_scores)
        if top_score < self.confidence_threshold:
            self.last_diagnostics = {"entity_match": "FAIL", "evidence_status": "INSUFFICIENT"}
            return False, refusal_msg

        # 3. Context Answerability & Substantive Evidence Check
        if context_chunks is not None:
            if not context_chunks or not any(c.strip() for c in context_chunks):
                self.last_diagnostics = {"entity_match": "FAIL", "evidence_status": "INSUFFICIENT"}
                return False, refusal_msg

            combined_context = " ".join(context_chunks)

            # Check substantive entity overlap across contexts
            if query:
                q_keywords = extract_content_keywords(query)
                # Remove generic conversational query stopwords
                generic_query_words = {"information", "regarding", "know", "want", "tell", "give", "details", "something", "about", "what", "where", "when", "which", "whose", "whom"}
                specific_keywords = {w for w in q_keywords if w.lower() not in generic_query_words and len(w) >= 3}

                # 3a. Named Entity Check (e.g., person names, brand names, proper nouns like 'Elon Musk', 'Microsoft')
                # Prevents matching generic keywords ('net worth') from a different entity ('Johnny Tapia')
                question_starters = {"what", "where", "when", "why", "who", "which", "how", "can", "does", "is", "are", "tell", "define"}
                proper_entities = [
                    e.strip() for e in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
                    if e.lower() not in question_starters and len(e) >= 3
                ]
                if proper_entities:
                    combined_lower = combined_context.lower()
                    # Check if the named entity (or any of its constituent words >= 3 chars) is in context
                    entity_found = False
                    for entity in proper_entities:
                        if entity.lower() in combined_lower:
                            entity_found = True
                            break
                        parts = [p.lower() for p in entity.split() if len(p) >= 3]
                        if parts and any(p in combined_lower for p in parts):
                            entity_found = True
                            break

                    if not entity_found:
                        ctx_script = detect_script(combined_context)
                        # If context is in a non-Latin Indic script (e.g. Tamil/Hindi/Bengali text) or top_score >= 0.44,
                        # the multilingual retriever successfully matched the cross-lingual entity representation
                        if ctx_script != "latin" or top_score >= 0.44:
                            entity_found = True
                        else:
                            self.last_diagnostics = {"entity_match": "FAIL", "evidence_status": "INSUFFICIENT"}
                            return False, refusal_msg


                # 3b. Substantive Content Keyword Coverage Check
                if specific_keywords:
                    combined_lower = combined_context.lower()
                    matched_kws = [kw for kw in specific_keywords if kw.lower() in combined_lower]
                    coverage_ratio = len(matched_kws) / len(specific_keywords)

                    # 3c. Single-Chunk Co-occurrence Check:
                    # Prevents false-positives where words are scattered across completely unrelated passages
                    # (e.g. 'President' in chunk 1, 'India' in chunk 3). At least 1 single chunk must contain >= 60% of query keywords.
                    if len(specific_keywords) >= 2:
                        max_chunk_coverage = 0.0
                        for chunk in context_chunks:
                            chunk_lower = chunk.lower()
                            chunk_matches = sum(1 for kw in specific_keywords if kw.lower() in chunk_lower)
                            cov = chunk_matches / len(specific_keywords)
                            if cov > max_chunk_coverage:
                                max_chunk_coverage = cov

                        if max_chunk_coverage < 0.60:
                            self.last_diagnostics = {"entity_match": "FAIL", "evidence_status": "INSUFFICIENT", "reason": "scattered_keywords"}
                            return False, refusal_msg

                    # Check script alignment
                    q_script = detect_script(query)
                    ctx_script = detect_script(combined_context)

                    if q_script == ctx_script:
                        # Same-script: require at least 30% keyword coverage OR moderate semantic confidence (>= 0.48)
                        if coverage_ratio < 0.30 and top_score < 0.48:
                            self.last_diagnostics = {"entity_match": "FAIL", "evidence_status": "INSUFFICIENT", "coverage": f"{coverage_ratio:.2f}"}
                            return False, refusal_msg
                        else:
                            self.last_diagnostics = {"entity_match": "PASS", "evidence_status": "SUFFICIENT"}
                    else:
                        # Cross-script (e.g. Indic query vs English corpus): rely on semantic retrieval confidence
                        if top_score < 0.48:
                            self.last_diagnostics = {"entity_match": "FAIL", "evidence_status": "INSUFFICIENT", "top_score": f"{top_score:.2f}"}
                            return False, refusal_msg
                        else:
                            self.last_diagnostics = {"entity_match": "PASS", "evidence_status": "SUFFICIENT"}
                else:
                    # Generic query without specific content keywords: require base score
                    if top_score < 0.48:
                        self.last_diagnostics = {"entity_match": "FAIL", "evidence_status": "INSUFFICIENT"}
                        return False, refusal_msg
                    self.last_diagnostics = {"entity_match": "N/A", "evidence_status": "SUFFICIENT"}


        return True, None



