"""
Ticket 9: RAG Orchestrator / Harness Core.

Orchestrates the entire sub-200ms grounded voice-RAG pipeline:
1. Normalization
2. Safety Guardrail
3. Embedding + FAISS Retrieval
4. Relevance Gate
5. Insufficient Evidence Gate (caught PRE-generation to save latency)
6. Context Assembly
7. LLM Streaming / Completion with Extractive Fallback
8. Two-Tier Groundedness Verification

Emits rich telemetry matching the frozen spec:
- embed_retrieval_ms (<50ms)
- rag_pipeline_ms (<80ms)
- harness_ms (<200ms)
"""

import re
import time
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple, Union

from backend.app.rag.embedder import EmbeddingProvider
from backend.app.rag.retriever import RetrieverBackend, RetrievedChunk
from backend.app.rag.extractive import extract_best_answer_span
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier
from backend.app.harness.providers.base import LLMProvider
from backend.app.harness.schemas import RAGRequest, RAGResponse, LatencyBreakdown


@dataclass
class HarnessMetrics:
    """Comprehensive latency and telemetry metrics across every discrete boundary."""
    safety_ms: float = 0.0
    embedding_ms: float = 0.0
    faiss_ms: float = 0.0
    bm25_ms: float = 0.0
    rrf_ms: float = 0.0
    evidence_gate_ms: float = 0.0
    pre_llm_total_ms: float = 0.0
    query_embedding_ms: float = 0.0  # legacy alias
    vector_search_ms: float = 0.0    # legacy alias
    reranking_ms: float = 0.0
    embed_retrieval_ms: float = 0.0
    rag_pipeline_ms: float = 0.0
    guardrails_ms: float = 0.0
    groq_network_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0
    grounding_ms: float = 0.0
    text_to_answer_ms: float = 0.0
    harness_ms: float = 0.0
    
    # Boundary Decision Diagnostics
    entity_match: str = "N/A"          # 'PASS', 'FAIL', 'N/A'
    evidence_status: str = "SUFFICIENT" # 'SUFFICIENT', 'INSUFFICIENT'
    llm_invocation: str = "SKIPPED"    # 'EXECUTED', 'SKIPPED'
    groundedness_verdict: str = "N/A"  # 'VERIFIED', 'UNVERIFIED', 'SKIPPED'


@dataclass
class HarnessResponse:
    """Standardized response from RAG Harness."""
    query: str
    answer: str
    status: str  # 'success', 'refusal_safety', 'refusal_offtopic', 'refusal_insufficient_evidence', 'refusal_ungrounded', 'fallback'
    retrieved_chunks: List[Dict[str, Any]]
    refusal_reason: Optional[str] = None
    groundedness_score: float = 1.0
    metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.metrics is None:
            data["metrics"] = {}
        return data

    def to_rag_response(self, mode: str = "strict") -> RAGResponse:
        """Convert to Pydantic RAGResponse matching frozen schema."""
        lat = LatencyBreakdown(**(self.metrics or {}))
        return RAGResponse(
            answer=self.answer,
            status=self.status if self.status in [
                "success", "refusal_safety", "refusal_offtopic",
                "refusal_insufficient_evidence", "refusal_ungrounded", "error"
            ] else "error",
            confidence=self.groundedness_score,
            grounded=self.status == "success",
            grounding_method="overlap",
            source_chunks=[c.get("text", "") for c in self.retrieved_chunks],
            retrieved_chunks=[RetrievedChunk(**c) for c in self.retrieved_chunks],
            refusal_reason=self.refusal_reason,
            latency=lat,
            mode=mode if mode in ["strict", "quality"] else "strict",
        )


class RAGOrchestrator:
    """Core RAG Harness Orchestrator."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        retriever: RetrieverBackend,
        llm: LLMProvider,
        safety_guard: Optional[SafetyGuardrail] = None,
        relevance_gate: Optional[RelevanceGate] = None,
        insufficient_checker: Optional[InsufficientEvidenceChecker] = None,
        groundedness_verifier: Optional[GroundednessVerifier] = None,
        top_k: int = 3,
        llm_timeout_ms: int = 1500,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm
        self.safety_guard = safety_guard or SafetyGuardrail()
        self.relevance_gate = relevance_gate or RelevanceGate()
        self.insufficient_checker = insufficient_checker or InsufficientEvidenceChecker()
        self.groundedness_verifier = groundedness_verifier or GroundednessVerifier(embedder=self.embedder)
        if self.groundedness_verifier.embedder is None:
            self.groundedness_verifier.embedder = self.embedder
        self.top_k = top_k
        self.llm_timeout_ms = llm_timeout_ms

    def normalize_query(self, query: str) -> str:
        """Lightweight string normalization (< 1ms)."""
        return " ".join(query.strip().split())

    def check_conversational_intent(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Fast (<0.5ms) conversational intent interceptor for audio-checks, greetings, and introductions.
        Prevents unnecessary MS-MARCO search for queries like 'am I audible?' or 'hello'.
        Returns (response_text, detected_lang) or None.
        """
        from backend.app.voice.detector import detect_language_metadata
        lang_meta = detect_language_metadata(query)
        lang = lang_meta.get("detected_language", "english")
        q_lower = query.lower().strip()

        # 1. Audio check / Mic test intent
        audio_check_patterns = [
            r"am\s+i\s+(audible|clear|heard)",
            r"can\s+you\s+hear\s+me",
            r"are\s+you\s+listening",
            r"sound\s+check",
            r"mic\s+test",
            r"testing\s+(1|one)",
            r"आवाज़\s+(आ\s+रही|सुन\s+सकते|सुनाई\s+दे\s+रहा)",
            r"क्या\s+आप\s+मुझे\s+सुन",
            r"കേൾക്കാമോ",
            r"கேட்கிறதா",
            r"వినిపిస్తుందా",
            r"ऐकू\s+येत\s+आहे",
            r"শুনতে\s+পাচ্ছ",
        ]
        if any(re.search(pat, q_lower) for pat in audio_check_patterns):
            responses = {
                "english": "Yes, I can hear you clearly! I'm Vaani — ask me any question and I'll find the answer for you.",
                "hindi": "हाँ, मैं आपको स्पष्ट सुन सकती हूँ! मैं वाणी हूँ — मुझसे कोई भी प्रश्न पूछें।",
                "tamil": "ஆம், உங்கள் குரல் தெளிவாக கேட்கிறது! நான் வாணி — எந்த கேள்வியும் கேளுங்கள்.",
                "telugu": "అవును, మీ వాయిస్ స్పష్టంగా వినిపిస్తోంది! నేను వాణి — ఏదైనా ప్రశ్న అడగండి.",
                "marathi": "होय, तुमचा आवाज स्पष्ट ऐकू येत आहे! मी वाणी आहे — कोणताही प्रश्न विचारा.",
                "bengali": "হ্যাঁ, আপনার কথা স্পষ্টভাবে শোনা যাচ্ছে! আমি বাণী — যেকোনো প্রশ্ন জিজ্ঞাসা করুন।",
                "gujarati": "હા, હું તમને સ્પષ્ટ સાંભળી શકું છું! હું વાણી છું — કોઈપણ પ્રશ્ન પૂછો.",
                "kannada": "ಹೌದು, ನಿಮ್ಮ ಧ್ವನಿ ಸ್ಪಷ್ಟವಾಗಿ ಕೇಳಿಸುತ್ತಿದೆ! ನಾನು ವಾಣಿ — ಯಾವುದೇ ಪ್ರಶ್ನೆ ಕೇಳಿ.",
                "malayalam": "അതെ, നിങ്ങളുടെ ശബ്ദം വ്യക്തമായി കേൾക്കുന്നുണ്ട്! ഞാൻ വാണി — ഏതെങ്കിലും ചോദ്യം ചോദിക്കൂ.",
            }
            return responses.get(lang, responses["english"]), lang

        # 2. Greeting / Hello intent
        greeting_patterns = [
            r"^(hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening)\b",
            r"^(नमस्ते|नमस्कार|हैलो|प्रणाम)\b",
            r"^(வணக்கம்)\b",
            r"^(నమస్కారం)\b",
            r"^(নমস্কার|হ্যালো)\b",
            r"^(നമസ്കാരം)\b",
            r"^(namaste|kaise\s+ho|kya\s+haal\s+hai)\b",
        ]
        if any(re.search(pat, q_lower) for pat in greeting_patterns):
            responses = {
                "english": "Hello! I'm Vaani, your Multilingual Voice Knowledge Assistant. Ask me any factual question — I'll find the answer from my knowledge base for you.",
                "hindi": "नमस्ते! मैं वाणी हूँ, आपकी बहुभाषी वॉयस ज्ञान सहायक। मुझसे कोई भी तथ्यात्मक प्रश्न पूछें — मैं अपने ज्ञान आधार से उत्तर खोजकर दूँगी।",
                "tamil": "வணக்கம்! நான் வாணி, உங்கள் பன்மொழி குரல் அறிவு உதவியாளர். எந்த உண்மையான கேள்வியையும் கேளுங்கள் — நான் பதில் கண்டறிவேன்.",
                "telugu": "నమస్కారం! నేను వాణి, మీ బహుభాషా వాయిస్ జ్ఞాన సహాయకురాలు. ఏదైనా వాస్తవిక ప్రశ్న అడగండి — నేను సమాధానం కనుగొంటాను.",
                "marathi": "नमस्कार! मी वाणी आहे, तुमची बहुभाषिक व्हॉइस ज्ञान सहाय्यक. कोणताही तथ्यात्मक प्रश्न विचारा — मी उत्तर शोधून देईन.",
                "bengali": "নমস্কার! আমি বাণী, আপনার বহুভাষী ভয়েস জ্ঞান সহকারী। যেকোনো তথ্যমূলক প্রশ্ন জিজ্ঞাসা করুন — আমি উত্তর খুঁজে দেব।",
                "gujarati": "નમસ્તે! હું વાણી છું, તમારી બહુભાષી વોઇસ જ્ઞાન સહાયક. કોઈપણ તથ્યાત્મક પ્રશ્ન પૂછો — હું જવાબ શોધી આપીશ.",
                "kannada": "ನಮಸ್ಕಾರ! ನಾನು ವಾಣಿ, ನಿಮ್ಮ ಬಹುಭಾಷಾ ಧ್ವನಿ ಜ್ಞಾನ ಸಹಾಯಕಿ. ಯಾವುದೇ ವಾಸ್ತವಿಕ ಪ್ರಶ್ನೆ ಕೇಳಿ — ನಾನು ಉತ್ತರ ಹುಡುಕುತ್ತೇನೆ.",
                "malayalam": "നമസ്കാരം! ഞാൻ വാണി, നിങ്ങളുടെ ബഹുഭാഷാ വോയ്സ് വിജ്ഞാന സഹായി. ഏതെങ്കിലും വസ്തുതാപരമായ ചോദ്യം ചോദിക്കൂ — ഞാൻ ഉത്തരം കണ്ടെത്തും.",
            }
            return responses.get(lang, responses["english"]), lang

        return None

    def build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> Tuple[str, str]:
        """Construct grounded prompt with strict language matching and anti-hallucination rules."""
        from backend.app.voice.detector import detect_language_metadata
        lang_meta = detect_language_metadata(query)
        detected_lang = lang_meta.get("detected_language", "english")
        lang_display = lang_meta.get("language_display", "English")

        from backend.app.rag.extractive import _clean_passage_text
        context_blocks = []
        for i, c in enumerate(chunks):
            raw_text = c.parent_text if c.parent_text else c.text
            clean_text = _clean_passage_text(raw_text)
            # Compact text to 300 chars to maximize prefill speed
            final_text = (clean_text if clean_text else raw_text)[:300].strip()
            context_blocks.append(f"[{i+1}] {final_text}")

        context_str = "\n\n".join(context_blocks)


        if detected_lang == "hinglish":
            lang_instruction = "Hinglish (Conversational Hindi in Roman/Latin script, or clear natural English/Hindi)"
            target_script_rule = "Write the answer using Latin / Roman alphabet (Hinglish). Do NOT output in Odia, Tamil, Malayalam, Bengali, Telugu, Kannada, or Devanagari script."
        elif detected_lang == "english":
            lang_instruction = "English"
            target_script_rule = "Write the answer strictly in English using standard Latin alphabet. The retrieved Context contains factual definitions/passages in Indic languages (e.g. Odia, Punjabi, Hindi, Bengali, etc.). Translate the core factual definition/answer found in the Context directly into clear, natural English."
        elif detected_lang == "hindi":
            lang_instruction = "Hindi (हिन्दी)"
            target_script_rule = "Write the answer strictly in Hindi using Devanagari script (हिन्दी)."
        elif detected_lang == "tamil":
            lang_instruction = "Tamil (தமிழ்)"
            target_script_rule = "Write the answer strictly in Tamil using Tamil script (தமிழ்)."
        elif detected_lang == "marathi":
            lang_instruction = "Marathi (मराठी)"
            target_script_rule = "Write the answer strictly in Marathi using Devanagari script (मराठी)."
        elif detected_lang == "bengali":
            lang_instruction = "Bengali (বাংলা)"
            target_script_rule = "Write the answer strictly in Bengali using Bengali script (বাংলা)."
        elif detected_lang == "telugu":
            lang_instruction = "Telugu (తెలుగు)"
            target_script_rule = "Write the answer strictly in Telugu using Telugu script (తెలుగు)."
        elif detected_lang == "kannada":
            lang_instruction = "Kannada (ಕನ್ನಡ)"
            target_script_rule = "Write the answer strictly in Kannada using Kannada script (ಕನ್ನಡ)."
        elif detected_lang == "malayalam":
            lang_instruction = "Malayalam (മലയാളം)"
            target_script_rule = "Write the answer strictly in Malayalam using Malayalam script (മലയാളം)."
        elif detected_lang == "gujarati":
            lang_instruction = "Gujarati (ગુજરાતી)"
            target_script_rule = "Write the answer strictly in Gujarati using Gujarati script (ગુજરાતી)."
        else:
            lang_instruction = lang_display
            target_script_rule = f"Write the answer in {lang_display} using the correct corresponding script."

        system_prompt = (
            f"You are a factual Q&A assistant. Answer questions using ONLY the Context below.\n"
            f"Language: {lang_instruction}. {target_script_rule}\n\n"
            f"STRICT RULES:\n"
            f"1. Read ALL context passages. Choose the one that BEST answers the specific question.\n"
            f"2. Start your response with the DIRECT answer to the question.\n"
            f"3. Do NOT copy-paste the first passage blindly. Pick the MOST RELEVANT one.\n"
            f"4. Keep it to 1-2 sentences. Synthesize, don't dump raw text.\n"
            f"5. If NONE of the passages actually answer the question, say: 'This information is not available in my knowledge base.'\n"
            f"6. No invented facts, numbers, or names not present in the Context."
        )

        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        return prompt, system_prompt

    async def execute(
        self,
        request_or_query: Union[str, RAGRequest],
        mode: Optional[str] = None,
    ) -> HarnessResponse:
        """Execute full RAG pipeline and return HarnessResponse."""
        t_start = time.perf_counter()
        metrics = HarnessMetrics()

        if isinstance(request_or_query, RAGRequest):
            raw_query = request_or_query.query
            exec_mode = mode or request_or_query.mode
            top_k = request_or_query.top_k or self.top_k
            max_tokens = request_or_query.max_tokens or 64
        else:
            raw_query = request_or_query
            exec_mode = mode or "strict"
            top_k = self.top_k
            max_tokens = 64


        # Enforce mode timeouts (strict: 2500ms LLM cloud budget; quality: 5000ms)
        timeout_ms = 2500 if exec_mode == "strict" else 5000

        # Step 1: Normalize
        query = self.normalize_query(raw_query)

        # Fast Conversational / Audio-check interceptor (<0.5ms)
        conv_res = self.check_conversational_intent(query)
        if conv_res:
            conv_answer, _ = conv_res
            metrics.text_to_answer_ms = (time.perf_counter() - t_start) * 1000.0
            metrics.harness_ms = metrics.text_to_answer_ms
            metrics.llm_invocation = "SKIPPED"
            metrics.evidence_status = "SUFFICIENT"
            metrics.groundedness_verdict = "VERIFIED"
            return HarnessResponse(
                query=query,
                answer=conv_answer,
                status="success",
                retrieved_chunks=[],
                refusal_reason=None,
                groundedness_score=1.0,
                metrics=asdict(metrics),
            )

        # Step 2a: Query Sanitization (Strip adversarial / prompt-injection wrappers to uncover underlying question)
        from backend.app.guardrails.query_sanitizer import QuerySanitizer
        sanitized_query, was_wrapped = QuerySanitizer.sanitize(query)
        effective_query = sanitized_query if was_wrapped else query

        # Step 2: Safety Guardrail
        t_guard_0 = time.perf_counter()
        is_safe, safety_msg = self.safety_guard.evaluate(effective_query)
        metrics.safety_ms = round((time.perf_counter() - t_guard_0) * 1000.0, 2)
        metrics.guardrails_ms += metrics.safety_ms

        if not is_safe:
            metrics.pre_llm_total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            metrics.text_to_answer_ms = metrics.pre_llm_total_ms
            metrics.harness_ms = metrics.text_to_answer_ms
            metrics.llm_invocation = "SKIPPED"
            metrics.evidence_status = "INSUFFICIENT"
            metrics.groundedness_verdict = "SKIPPED"
            return HarnessResponse(
                query=query,
                answer=safety_msg,
                status="refusal_safety",
                retrieved_chunks=[],
                refusal_reason="safety_blocklist_triggered",
                metrics=asdict(metrics),
            )

        # Step 2b: Pre-Retrieval Intent and Domain Classifier (< 1ms fast path)
        from backend.app.guardrails.intent_classifier import IntentDomainClassifier
        is_in_domain, ood_cat, ood_msg = IntentDomainClassifier.classify(effective_query)
        if not is_in_domain:
            metrics.pre_llm_total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            metrics.text_to_answer_ms = metrics.pre_llm_total_ms
            metrics.harness_ms = metrics.text_to_answer_ms
            metrics.llm_invocation = "SKIPPED"
            metrics.evidence_status = "INSUFFICIENT"
            metrics.groundedness_verdict = "SKIPPED"
            return HarnessResponse(
                query=query,
                answer=ood_msg or "I can only answer questions about factual domain knowledge.",
                status="refusal_offtopic",
                retrieved_chunks=[],
                refusal_reason=f"ood_{ood_cat}",
                metrics=asdict(metrics),
            )

        # Step 3: Embed + Retrieve (supports both Dense FAISS and Hybrid RRF)

        t_emb_0 = time.perf_counter()
        q_emb = await asyncio.to_thread(self.embedder.embed_query, effective_query)
        t_emb_end = time.perf_counter()
        metrics.embedding_ms = round((t_emb_end - t_emb_0) * 1000.0, 2)
        metrics.query_embedding_ms = metrics.embedding_ms

        t_search_0 = time.perf_counter()
        if hasattr(self.retriever, "search_hybrid"):
            from backend.app.voice.detector import detect_language_metadata
            _lang_meta = detect_language_metadata(effective_query)
            _detected_lang = _lang_meta.get("detected_language", None)
            retrieved_chunks = await asyncio.to_thread(self.retriever.search_hybrid, effective_query, q_emb, top_k, _detected_lang)
            if hasattr(self.retriever, "last_timings"):
                metrics.faiss_ms = self.retriever.last_timings.get("faiss_ms", 0.0)
                metrics.bm25_ms = self.retriever.last_timings.get("bm25_ms", 0.0)
                metrics.rrf_ms = self.retriever.last_timings.get("rrf_ms", 0.0)
        else:
            retrieved_chunks = await asyncio.to_thread(self.retriever.search, q_emb, top_k)
            metrics.faiss_ms = round((time.perf_counter() - t_search_0) * 1000.0, 2)
        t_search_end = time.perf_counter()
        metrics.vector_search_ms = round((t_search_end - t_search_0) * 1000.0, 2)
        metrics.embed_retrieval_ms = round(metrics.embedding_ms + metrics.vector_search_ms, 2)
        metrics.rag_pipeline_ms = round((t_search_end - t_start) * 1000.0, 2)

        scores = [c.score for c in retrieved_chunks]
        context_texts = [c.text for c in retrieved_chunks]

        # Step 4: Relevance Gate (with substantive context answerability check)
        t_gate_0 = time.perf_counter()
        is_relevant, offtopic_msg = self.relevance_gate.evaluate(scores, query=effective_query, context_chunks=context_texts)


        if not is_relevant:
            metrics.evidence_gate_ms = round((time.perf_counter() - t_gate_0) * 1000.0, 2)
            metrics.guardrails_ms += metrics.evidence_gate_ms
            metrics.pre_llm_total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            metrics.text_to_answer_ms = metrics.pre_llm_total_ms
            metrics.harness_ms = metrics.text_to_answer_ms
            metrics.entity_match = "FAIL"
            metrics.evidence_status = "INSUFFICIENT"
            metrics.llm_invocation = "SKIPPED"
            metrics.groundedness_verdict = "SKIPPED"
            return HarnessResponse(
                query=query,
                answer=offtopic_msg,
                status="refusal_offtopic",
                retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                refusal_reason="relevance_threshold_not_met",
                metrics=asdict(metrics),
            )

        # Step 5: Insufficient Evidence Gate (PRE-LLM Refusal!)
        context_texts = [c.parent_text if c.parent_text else c.text for c in retrieved_chunks]
        has_evidence, insufficient_msg = self.insufficient_checker.evaluate(
            scores, query=effective_query, context_chunks=context_texts
        )
        metrics.evidence_gate_ms = round((time.perf_counter() - t_gate_0) * 1000.0, 2)
        metrics.guardrails_ms += metrics.evidence_gate_ms
        metrics.pre_llm_total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        if hasattr(self.insufficient_checker, "last_diagnostics"):
            metrics.entity_match = self.insufficient_checker.last_diagnostics.get("entity_match", "N/A")
            metrics.evidence_status = self.insufficient_checker.last_diagnostics.get("evidence_status", "SUFFICIENT")

        if not has_evidence:
            metrics.text_to_answer_ms = metrics.pre_llm_total_ms
            metrics.harness_ms = metrics.text_to_answer_ms
            metrics.llm_invocation = "SKIPPED"
            metrics.groundedness_verdict = "SKIPPED"
            return HarnessResponse(
                query=query,
                answer=insufficient_msg,
                status="refusal_insufficient_evidence",
                retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                refusal_reason="insufficient_confidence_evidence",
                metrics=asdict(metrics),
            )

        # Step 6: Always use LLM synthesis for quality answers.
        # Extractive bypass disabled — LLM reads all chunks and synthesizes the best answer.
        # Falls back to extractive only on LLM timeout (handled in the except block below).

        # Step 7: Assemble Prompt & Call LLM
        metrics.llm_invocation = "EXECUTED"
        prompt, system_prompt = self.build_prompt(query, retrieved_chunks)
        t_llm_0 = time.perf_counter()
        answer_parts: List[str] = []
        first_token_recorded = False

        try:
            async for token in self.llm.generate_stream(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                timeout_ms=timeout_ms,
            ):
                if not first_token_recorded:
                    metrics.llm_ttft_ms = round((time.perf_counter() - t_llm_0) * 1000.0, 2)
                    first_token_recorded = True
                answer_parts.append(token)

            t_llm_end = time.perf_counter()
            metrics.llm_total_ms = round((t_llm_end - t_llm_0) * 1000.0, 2)
            if metrics.llm_ttft_ms == 0.0:
                metrics.llm_ttft_ms = metrics.llm_total_ms
            answer = "".join(answer_parts).strip()
            if not answer:
                answer = extract_best_answer_span(query, context_texts) if context_texts else "उत्तर उपलब्ध नहीं है।"
        except Exception:
            # Circuit breaker / extractive fallback
            answer = extract_best_answer_span(query, context_texts) if context_texts else "उत्तर उपलब्ध नहीं है।"
            metrics.llm_total_ms = round((time.perf_counter() - t_llm_0) * 1000.0, 2)
            if metrics.llm_ttft_ms == 0.0:
                metrics.llm_ttft_ms = metrics.llm_total_ms

        # Step 7: Groundedness & Relevance Verification (Post-LLM)
        t_guard_3 = time.perf_counter()
        context_texts = [c.parent_text if c.parent_text else c.text for c in retrieved_chunks]
        is_grounded, method, ground_score, ungrounded_msg = self.groundedness_verifier.evaluate(
            answer, context_texts, query=effective_query
        )

        t_guard_3_end = time.perf_counter()
        metrics.grounding_ms = round((t_guard_3_end - t_guard_3) * 1000.0, 2)
        metrics.guardrails_ms = round(metrics.guardrails_ms + metrics.grounding_ms, 2)

        metrics.text_to_answer_ms = round((t_guard_3_end - t_start) * 1000.0, 2)
        metrics.harness_ms = metrics.text_to_answer_ms
        metrics.groundedness_verdict = "VERIFIED" if is_grounded else "UNVERIFIED"

        if not is_grounded:
            refusal_reason = (
                "inability_stated" if method == "explicit_refusal"
                else ("topic_drift_rejected" if method == "topic_drift_rejected" else "hallucination_detected")
            )
            status = (
                "refusal_insufficient_evidence"
                if method in ["explicit_refusal", "topic_drift_rejected"]
                else "refusal_ungrounded"
            )
            return HarnessResponse(
                query=query,
                answer=ungrounded_msg,
                status=status,
                retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                refusal_reason=refusal_reason,
                groundedness_score=ground_score,
                metrics=asdict(metrics),
            )

        # LLM refusal passthrough: the LLM correctly identified the context doesn't answer the question.
        # Return the LLM's natural language refusal (preserving language match) with refusal status.
        if method == "explicit_refusal_passthrough":
            return HarnessResponse(
                query=query,
                answer=answer,
                status="refusal_insufficient_evidence",
                retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
                refusal_reason="llm_identified_insufficient_evidence",
                groundedness_score=ground_score,
                metrics=asdict(metrics),
            )

        return HarnessResponse(
            query=query,
            answer=answer,
            status="success",
            retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
            groundedness_score=ground_score,
            metrics=asdict(metrics),
        )

    async def execute_stream(
        self,
        request_or_query: Union[str, RAGRequest],
        mode: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream RAG pipeline execution events:
        - status: RETRIEVING
        - status: GENERATING (with retrieved chunks & initial telemetry)
        - token: streaming LLM delta tokens
        - done: final HarnessResponse
        - refusal: early guardrail / ungrounded rejection
        """
        t_start = time.perf_counter()
        if isinstance(request_or_query, RAGRequest):
            query = request_or_query.query
            max_tokens = request_or_query.max_tokens or 64
            top_k = request_or_query.top_k
            exec_mode = request_or_query.mode or mode or "strict"
        else:
            query = str(request_or_query)
            max_tokens = 64
            top_k = 3
            exec_mode = mode or "strict"


        timeout_ms = 2500
        metrics = HarnessMetrics()

        # Step 1: Fast Conversational / Audio-Check Interceptor (< 0.5ms)
        conv_res = self.check_conversational_intent(query)
        if conv_res:
            conv_answer, _ = conv_res
            metrics.text_to_answer_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            metrics.harness_ms = metrics.text_to_answer_ms
            metrics.llm_invocation = "SKIPPED"
            metrics.evidence_status = "SUFFICIENT"
            metrics.groundedness_verdict = "VERIFIED"
            yield {
                "event": "done",
                "data": {
                    "query": query,
                    "answer": conv_answer,
                    "status": "success",
                    "retrieved_chunks": [],
                    "groundedness_score": 1.0,
                    "metrics": asdict(metrics),
                },
            }
            return

        yield {"event": "status", "data": {"state": "RETRIEVING", "message": "Normalizing query & evaluating safety..."}}

        # Step 2: Safety Guardrail
        t_guard_0 = time.perf_counter()
        is_safe, safety_msg = self.safety_guard.evaluate(query)
        metrics.safety_ms = round((time.perf_counter() - t_guard_0) * 1000.0, 2)
        metrics.guardrails_ms += metrics.safety_ms

        if not is_safe:
            metrics.pre_llm_total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            metrics.text_to_answer_ms = metrics.pre_llm_total_ms
            metrics.harness_ms = metrics.text_to_answer_ms
            metrics.llm_invocation = "SKIPPED"
            metrics.evidence_status = "INSUFFICIENT"
            metrics.groundedness_verdict = "SKIPPED"
            yield {
                "event": "refusal",
                "data": {
                    "query": query,
                    "status": "refusal_safety",
                    "answer": safety_msg,
                    "refusal_reason": "safety_blocklist_triggered",
                    "retrieved_chunks": [],
                    "metrics": asdict(metrics),
                },
            }
            return

        # Step 3: Embed + Retrieve
        t_emb_0 = time.perf_counter()
        q_emb = await asyncio.to_thread(self.embedder.embed_query, query)
        t_emb_end = time.perf_counter()
        metrics.embedding_ms = round((t_emb_end - t_emb_0) * 1000.0, 2)
        metrics.query_embedding_ms = metrics.embedding_ms

        t_search_0 = time.perf_counter()
        if hasattr(self.retriever, "search_hybrid"):
            from backend.app.voice.detector import detect_language_metadata
            _lang_meta = detect_language_metadata(query)
            _detected_lang = _lang_meta.get("detected_language", None)
            retrieved_chunks = await asyncio.to_thread(self.retriever.search_hybrid, query, q_emb, top_k, _detected_lang)
            if hasattr(self.retriever, "last_timings"):
                metrics.faiss_ms = self.retriever.last_timings.get("faiss_ms", 0.0)
                metrics.bm25_ms = self.retriever.last_timings.get("bm25_ms", 0.0)
                metrics.rrf_ms = self.retriever.last_timings.get("rrf_ms", 0.0)
        else:
            retrieved_chunks = await asyncio.to_thread(self.retriever.search, q_emb, top_k)
            metrics.faiss_ms = round((time.perf_counter() - t_search_0) * 1000.0, 2)
        t_search_end = time.perf_counter()
        metrics.vector_search_ms = round((t_search_end - t_search_0) * 1000.0, 2)
        metrics.embed_retrieval_ms = round(metrics.embedding_ms + metrics.vector_search_ms, 2)
        metrics.rag_pipeline_ms = round((t_search_end - t_start) * 1000.0, 2)

        scores = [c.score for c in retrieved_chunks]

        # Step 4: Relevance Gate
        t_gate_0 = time.perf_counter()
        is_relevant, offtopic_msg = self.relevance_gate.evaluate(scores, query=query)

        if not is_relevant:
            metrics.evidence_gate_ms = round((time.perf_counter() - t_gate_0) * 1000.0, 2)
            metrics.guardrails_ms += metrics.evidence_gate_ms
            metrics.pre_llm_total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            metrics.text_to_answer_ms = metrics.pre_llm_total_ms
            metrics.harness_ms = metrics.text_to_answer_ms
            metrics.entity_match = "FAIL"
            metrics.evidence_status = "INSUFFICIENT"
            metrics.llm_invocation = "SKIPPED"
            metrics.groundedness_verdict = "SKIPPED"
            yield {
                "event": "refusal",
                "data": {
                    "query": query,
                    "status": "refusal_offtopic",
                    "answer": offtopic_msg,
                    "refusal_reason": "relevance_threshold_not_met",
                    "retrieved_chunks": [c.to_dict() for c in retrieved_chunks],
                    "metrics": asdict(metrics),
                },
            }
            return

        # Step 5: Insufficient Evidence Gate
        context_texts = [c.parent_text if c.parent_text else c.text for c in retrieved_chunks]
        has_evidence, insufficient_msg = self.insufficient_checker.evaluate(
            scores, query=query, context_chunks=context_texts
        )
        metrics.evidence_gate_ms = round((time.perf_counter() - t_gate_0) * 1000.0, 2)
        metrics.guardrails_ms += metrics.evidence_gate_ms
        metrics.pre_llm_total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        if hasattr(self.insufficient_checker, "last_diagnostics"):
            metrics.entity_match = self.insufficient_checker.last_diagnostics.get("entity_match", "N/A")
            metrics.evidence_status = self.insufficient_checker.last_diagnostics.get("evidence_status", "SUFFICIENT")

        if not has_evidence:
            metrics.text_to_answer_ms = metrics.pre_llm_total_ms
            metrics.harness_ms = metrics.text_to_answer_ms
            metrics.llm_invocation = "SKIPPED"
            metrics.groundedness_verdict = "SKIPPED"
            yield {
                "event": "refusal",
                "data": {
                    "query": query,
                    "status": "refusal_insufficient_evidence",
                    "answer": insufficient_msg,
                    "refusal_reason": "insufficient_confidence_evidence",
                    "retrieved_chunks": [c.to_dict() for c in retrieved_chunks],
                    "metrics": asdict(metrics),
                },
            }
            return

        # Step 6: Stream LLM Tokens
        metrics.llm_invocation = "EXECUTED"
        yield {
            "event": "status",
            "data": {
                "state": "GENERATING",
                "message": "Generating grounded answer...",
                "retrieved_chunks": [c.to_dict() for c in retrieved_chunks],
                "metrics": asdict(metrics),
            },
        }

        prompt, system_prompt = self.build_prompt(query, retrieved_chunks)
        t_llm_0 = time.perf_counter()
        answer_parts: List[str] = []
        first_token_recorded = False

        try:
            async for token in self.llm.generate_stream(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                timeout_ms=timeout_ms,
            ):
                if not first_token_recorded:
                    metrics.llm_ttft_ms = round((time.perf_counter() - t_llm_0) * 1000.0, 2)
                    first_token_recorded = True
                answer_parts.append(token)
                yield {"event": "token", "data": {"delta": token}}

            t_llm_end = time.perf_counter()
            metrics.llm_total_ms = round((t_llm_end - t_llm_0) * 1000.0, 2)
            if metrics.llm_ttft_ms == 0.0:
                metrics.llm_ttft_ms = metrics.llm_total_ms
            answer = "".join(answer_parts).strip()
            if not answer:
                answer = retrieved_chunks[0].text if retrieved_chunks else "उत्तर उपलब्ध नहीं है।"
        except Exception:
            answer = retrieved_chunks[0].text if retrieved_chunks else "उत्तर उपलब्ध नहीं है।"
            metrics.llm_total_ms = round((time.perf_counter() - t_llm_0) * 1000.0, 2)
            if metrics.llm_ttft_ms == 0.0:
                metrics.llm_ttft_ms = metrics.llm_total_ms
            yield {"event": "token", "data": {"delta": answer}}

        # Step 7: Groundedness Verification
        t_guard_3 = time.perf_counter()
        context_texts = [c.parent_text if c.parent_text else c.text for c in retrieved_chunks]
        is_grounded, method, ground_score, ungrounded_msg = self.groundedness_verifier.evaluate(
            answer, context_texts, query=query
        )
        t_guard_3_end = time.perf_counter()
        metrics.grounding_ms = round((t_guard_3_end - t_guard_3) * 1000.0, 2)
        metrics.guardrails_ms = round(metrics.guardrails_ms + metrics.grounding_ms, 2)
        metrics.text_to_answer_ms = round((t_guard_3_end - t_start) * 1000.0, 2)
        metrics.harness_ms = metrics.text_to_answer_ms
        metrics.groundedness_verdict = "VERIFIED" if is_grounded else "UNVERIFIED"

        if not is_grounded:
            refusal_reason = (
                "inability_stated" if method == "explicit_refusal"
                else ("topic_drift_rejected" if method == "topic_drift_rejected" else "hallucination_detected")
            )
            status = (
                "refusal_insufficient_evidence"
                if method in ["explicit_refusal", "topic_drift_rejected"]
                else "refusal_ungrounded"
            )
            yield {
                "event": "refusal",
                "data": {
                    "query": query,
                    "status": status,
                    "answer": ungrounded_msg,
                    "refusal_reason": refusal_reason,
                    "groundedness_score": ground_score,
                    "retrieved_chunks": [c.to_dict() for c in retrieved_chunks],
                    "metrics": asdict(metrics),
                },
            }
            return

        # LLM refusal passthrough: preserve the LLM's natural language refusal
        if method == "explicit_refusal_passthrough":
            yield {
                "event": "complete",
                "data": {
                    "query": query,
                    "status": "refusal_insufficient_evidence",
                    "answer": answer,
                    "refusal_reason": "llm_identified_insufficient_evidence",
                    "groundedness_score": ground_score,
                    "retrieved_chunks": [c.to_dict() for c in retrieved_chunks],
                    "metrics": asdict(metrics),
                },
            }
            return

        yield {
            "event": "complete",
            "data": {
                "query": query,
                "status": "success",
                "answer": answer,
                "groundedness_score": ground_score,
                "retrieved_chunks": [c.to_dict() for c in retrieved_chunks],
                "metrics": asdict(metrics),
            },
        }

