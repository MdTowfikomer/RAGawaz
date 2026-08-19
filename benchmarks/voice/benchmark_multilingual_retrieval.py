"""
Phase 4D: Multilingual & Cross-Lingual Retrieval Validation Benchmark.

Evaluates cross-lingual retrieval against the Hindi corpus across:
1. Hindi (hi)
2. English (en) - Cross-lingual en->hi
3. Hinglish (hi-Latn / code-mixed) - Cross-lingual code-mixed->hi
4. Bengali (bn) - Cross-lingual bn->hi
5. Marathi (mr) - Cross-lingual mr->hi
6. Tamil (ta) - Cross-lingual ta->hi

Measures:
- Recall@5
- MRR
- Embed + Retrieval Latency (P50, P70)
- Answer Correctness Rate
- False-Positive Hallucination / Drift Rate
- Guardrail Refusal Accuracy
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.guardrails.groundedness import GroundednessVerifier
from backend.app.harness.orchestrator import RAGOrchestrator
from backend.app.harness.providers.groq import GroqLLMProvider


# Standard parallel multilingual evaluation dataset with verified ground truth passage IDs
MULTILINGUAL_TEST_SET = [
    # 1. Corporation
    {
        "concept": "corporation",
        "gt_passage_id": "hi_1102432_5",
        "queries": {
            "hi": "कॉर्पोरेशन क्या है?",
            "en": "What is a corporation?",
            "hinglish": "Corporation kya hota hai?",
            "bn": "কর্পোরেশন কি?",
            "mr": "महानगरपालिका / कॉर्पोरेशन म्हणजे काय?",
            "ta": "கார்ப்பரேஷன் என்றால் என்ன?",
        },
        "gold_keywords": ["कंपनी", "समूह", "इकाई", "अधिकार", "company", "group", "entity"],
    },
    # 2. Rachel Carson
    {
        "concept": "rachel_carson",
        "gt_passage_id": "hi_1102431_4",
        "queries": {
            "hi": "रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा",
            "en": "Why did Rachel Carson write The Obligation to Endure?",
            "hinglish": "Rachel Carson ne Obligation to Endure kyu likha tha?",
            "bn": "র‍্যাচেল কারসন কেন দ্য অবলিগেশন টু এনডিউর লিখেছিলেন?",
            "mr": "रेचल कार्सन यांनी द ऑब्लिगेशन टू एंड्योर का लिहिले?",
            "ta": "ரேச்சல் கார்சன் ஏன் தி ஆப்ளிகேஷன் டு என்டூர் எழுதினார்?",
        },
        "gold_keywords": ["कार्सन", "पर्यावरण", "रसायनों", "कीटनाशक", "carson", "endure", "environment"],
    },
    # 3. Honest/Truthfulness
    {
        "concept": "honesty_definition",
        "gt_passage_id": "hi_205107_2",
        "queries": {
            "hi": "ईमानदारी या सच्चाई की परिभाषा क्या है?",
            "en": "What is the definition of honesty or truthfulness?",
            "hinglish": "Honesty aur truthfulness ki definition kya hai?",
            "bn": "সততা বা সত্যবাদিতার সংজ্ঞা কি?",
            "mr": "प्रामाणिकपणा किंवा सत्याची व्याख्या काय आहे?",
            "ta": "நேர்மை அல்லது உண்மையின் வரையறை என்ன?",
        },
        "gold_keywords": ["ईमानदार", "सच्चाई", "सद्गुण", "नैतिक", "चरित्र", "truth", "moral"],
    },
    # 4. Falcon speed
    {
        "concept": "falcon_speed",
        "gt_passage_id": "hi_300122_1",
        "queries": {
            "hi": "बाज़ कितनी तेजी से यात्रा कर सकते हैं?",
            "en": "How fast can falcons travel?",
            "hinglish": "Falcon kitni speed se fly kar sakta hai?",
            "bn": "বাজপাখি কত দ্রুত উড়তে পারে?",
            "mr": "ससाणा किती वेगाने उडू शकतो?",
            "ta": "பால்கன் பறவை எவ்வளவு வேகமாக பறக்கும்?",
        },
        "gold_keywords": ["मील", "घंटा", "200", "गति", "झपट्टा", "speed", "mph", "fast"],
    },
    # 5. StubHub phone number
    {
        "concept": "stubhub_number",
        "gt_passage_id": "hi_233826_0",
        "queries": {
            "hi": "स्टबहब टोल फ्री नंबर क्या है?",
            "en": "What is the StubHub toll free phone number?",
            "hinglish": "StubHub ka customer care toll free number kya hai?",
            "bn": "স্টাবহাব টোল ফ্রি নম্বর কি?",
            "mr": "स्टबहबचा टोल फ्री नंबर काय आहे?",
            "ta": "ஸ்டப்ஹப் இலவச தொலைபேசி எண் என்ன?",
        },
        "gold_keywords": ["1-866", "866-788-2482", "नंबर", "फ़ोन", "number", "toll-free"],
    },
    # 6. Climatology study
    {
        "concept": "climatology_study",
        "gt_passage_id": "hi_1090355_7",
        "queries": {
            "hi": "जलवायु मौसम का अध्ययन क्या कहलाता है?",
            "en": "What is the study of climate and weather?",
            "hinglish": "Climate aur weather ki study ko kya bolte hain?",
            "bn": "জলবায়ু এবং আবহাওয়ার অধ্যয়নকে কী বলা হয়?",
            "mr": "हवामान आणि वातावरणाचा अभ्यास काय मानला जातो?",
            "ta": "காலநிலை மற்றும் வானிலை பற்றிய ஆய்வு என்ன?",
        },
        "gold_keywords": ["जलवायु", "मौसम", "विज्ञान", "क्लाइमेटोलॉजी", "climatology", "study"],
    },
    # 7. Delta Airlines Bangalore
    {
        "concept": "delta_bangalore",
        "gt_passage_id": "hi_165349_0",
        "queries": {
            "hi": "क्या डेल्टा बैंगलोर के लिए उड़ान भरता है?",
            "en": "Does Delta fly to Bangalore?",
            "hinglish": "Kya Delta Airlines Bangalore ke liye fly karti hai?",
            "bn": "ডেল্টা কি ব্যাঙ্গালোরে বিমান চালায়?",
            "mr": "डेल्टा बंगळुरूसाठी उड्डाण करते का?",
            "ta": "டெல்டா பெங்களூருக்கு பறக்கிறதா?",
        },
        "gold_keywords": ["डेल्टा", "उड़ान", "बैंगलोर", "पेरिस", "delta", "bangalore", "flight"],
    },
    # 8. Stye causes
    {
        "concept": "stye_cause",
        "gt_passage_id": "hi_1102396_1",
        "queries": {
            "hi": "स्टाई का कारण क्या होता है?",
            "en": "What causes a stye in the eye?",
            "hinglish": "Aankh me stye kyu hota hai aur causes kya hain?",
            "bn": "চোখে অঞ্জনি বা স্টাই কেন হয়?",
            "mr": "डोळ्यातील रांजणवाडी कशामुळे होते?",
            "ta": "கண் கட்டி எதனால் ஏற்படுகிறது?",
        },
        "gold_keywords": ["जीवाणु", "संक्रमण", "पलक", "ग्रंथि", "staphylococcus", "bacteria", "infection"],
    },
]

# Unanswerable Out-Of-Domain test queries in each language to verify guardrail refusal accuracy
UNANSWERABLE_TEST_SET = [
    {
        "hi": "घर पर परमाणु रिएक्टर कैसे बनाएं?",
        "en": "How to build a nuclear reactor at home?",
        "hinglish": "Ghar par nuclear reactor kaise banaye?",
        "bn": "বাড়িতে পারমাণবিক চুল্লি কীভাবে তৈরি করবেন?",
        "mr": "घरी अणुभट्टी कशी तयार करावी?",
        "ta": "வீட்டில் அணு உலை உருவாக்குவது எப்படி?",
    },
    {
        "hi": "मंगल ग्रह पर आज का तापमान और मौसम कैसा है?",
        "en": "What is the live weather and temperature on Mars today?",
        "hinglish": "Mars planet par live weather aur temperature kaisa hai?",
        "bn": "মঙ্গল গ্রহে আজকের আবহাওয়া এবং তাপমাত্রা কেমন?",
        "mr": "मंगळावर आजचे हवामान आणि तापमान कसे आहे?",
        "ta": "செவ்வாய் கிரகத்தில் இன்றைய வெப்பநிலை என்ன?",
    },
]


async def evaluate_multilingual():
    print("=" * 90)
    print("PHASE 4D: MULTILINGUAL & CROSS-LINGUAL RETRIEVAL & QUALITY BENCHMARK")
    print("=" * 90)

    embedder = get_embedding_provider("minilm")
    retriever = FAISSHNSWRetriever()
    retriever.load(os.path.join(ROOT_DIR, "backend", "data", "faiss_cache"))

    groq_key = os.getenv("GROQ_API_KEY")
    llm = GroqLLMProvider(api_key=groq_key, model_id="llama-3.1-8b-instant")
    verifier = GroundednessVerifier(embedder=embedder)

    orchestrator = RAGOrchestrator(
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        groundedness_verifier=verifier,
        top_k=5,
    )

    languages = ["hi", "en", "hinglish", "bn", "mr", "ta"]
    lang_labels = {
        "hi": "Hindi (Native)",
        "en": "English (Cross-Lingual)",
        "hinglish": "Hinglish (Code-Mixed)",
        "bn": "Bengali (Indic Cross-Lingual)",
        "mr": "Marathi (Indic Cross-Lingual)",
        "ta": "Tamil (Dravidian Cross-Lingual)",
    }

    summary_report = {}

    for lang in languages:
        print(f"\nEvaluating Category: {lang_labels[lang]}...")
        print("-" * 90)
        
        recall_hits = 0
        rr_list = []
        embed_retrieval_latencies = []
        correct_answers = 0
        hallucinations = 0
        refusals_on_unanswerables = 0
        
        text_to_answer_latencies = []
        verification_latencies = []
        false_refusals = 0

        # 1. Answerable Query Evaluation
        for item in MULTILINGUAL_TEST_SET:
            q_text = item["queries"][lang]
            gt_prefix = item["gt_passage_id"].rsplit("_", 1)[0]
            
            # Measure Embed + Retrieval
            t0 = time.perf_counter()
            q_emb = embedder.embed_query(q_text)
            chunks = retriever.search(q_emb, top_k=5)
            retrieval_time_ms = (time.perf_counter() - t0) * 1000.0
            embed_retrieval_latencies.append(retrieval_time_ms)
            
            retrieved_prefixes = [c.passage_id.rsplit("_", 1)[0] for c in chunks]
            has_relevant_chunk = gt_prefix in retrieved_prefixes
            if has_relevant_chunk:
                recall_hits += 1
                rank = retrieved_prefixes.index(gt_prefix) + 1
                rr_list.append(1.0 / rank)
            else:
                rr_list.append(0.0)

            # End-to-End Orchestrator evaluation
            resp = await orchestrator.execute(q_text)
            if resp.metrics:
                text_to_answer_latencies.append(resp.metrics.get("text_to_answer_ms", 0.0))
                verification_latencies.append(resp.metrics.get("grounding_ms", 0.0))
            
            # Correctness spot-check
            if resp.status == "success":
                ans_text = resp.answer.lower()
                has_key = any(k.lower() in ans_text for k in item["gold_keywords"])
                if has_key and resp.groundedness_score >= 0.25:
                    correct_answers += 1
                else:
                    hallucinations += 1
            elif resp.status.startswith("refusal"):
                if has_relevant_chunk:
                    false_refusals += 1

        # 2. Unanswerable Out-Of-Domain Guardrail Evaluation
        for unans in UNANSWERABLE_TEST_SET:
            q_unans = unans[lang]
            resp_unans = await orchestrator.execute(q_unans)
            if resp_unans.status.startswith("refusal"):
                refusals_on_unanswerables += 1

        n_ans = len(MULTILINGUAL_TEST_SET)
        n_unans = len(UNANSWERABLE_TEST_SET)

        recall_at_5 = (recall_hits / n_ans) * 100.0
        mrr = float(np.mean(rr_list))
        lat_p50 = float(np.percentile(embed_retrieval_latencies, 50))
        lat_p70 = float(np.percentile(embed_retrieval_latencies, 70))
        verif_p70 = float(np.percentile(verification_latencies, 70)) if verification_latencies else 0.0
        t2a_p70 = float(np.percentile(text_to_answer_latencies, 70)) if text_to_answer_latencies else 0.0
        correctness_rate = (correct_answers / n_ans) * 100.0
        false_refusal_rate = (false_refusals / n_ans) * 100.0
        hallucination_rate = (hallucinations / n_ans) * 100.0
        refusal_acc = (refusals_on_unanswerables / n_unans) * 100.0

        summary_report[lang] = {
            "label": lang_labels[lang],
            "recall_at_5": recall_at_5,
            "mrr": mrr,
            "latency_p50_ms": lat_p50,
            "latency_p70_ms": lat_p70,
            "verification_latency_p70_ms": verif_p70,
            "text_to_answer_p70_ms": t2a_p70,
            "correctness_rate": correctness_rate,
            "false_refusal_rate": false_refusal_rate,
            "hallucination_rate": hallucination_rate,
            "guardrail_refusal_acc": refusal_acc,
        }

        print(f"  Recall@5:                {recall_at_5:6.2f}%")
        print(f"  MRR:                     {mrr:6.4f}")
        print(f"  Embed+Retrieval (P70):   {lat_p70:6.2f} ms")
        print(f"  Verification Lat (P70):  {verif_p70:6.2f} ms")
        print(f"  Text->Answer Lat (P70):  {t2a_p70:6.2f} ms")
        print(f"  Answer Correctness:      {correctness_rate:6.2f}%")
        print(f"  False Refusal Rate:      {false_refusal_rate:6.2f}%")
        print(f"  Hallucination / Drift:   {hallucination_rate:6.2f}%")
        print(f"  Guardrail Refusal Acc:   {refusal_acc:6.2f}%")

    print("\n" + "=" * 105)
    print("PHASE 4D: MULTILINGUAL VERIFICATION BENCHMARK SUMMARY (TOP_K = 5):")
    print("=" * 105)
    header = f"{'Language / Category':<28} | {'Recall@5':<9} | {'MRR':<7} | {'Ret P70':<9} | {'Verif P70':<10} | {'T2A P70':<9} | {'Correct':<8} | {'FalseRef':<9} | {'Halluc':<7} | {'RefusalAcc':<10}"
    print(header)
    print("-" * len(header))
    for k, v in summary_report.items():
        row = f"{v['label']:<28} | {v['recall_at_5']:5.1f}%    | {v['mrr']:.4f} | {v['latency_p70_ms']:6.1f} ms | {v['verification_latency_p70_ms']:6.2f} ms  | {v['text_to_answer_p70_ms']:6.1f} ms | {v['correctness_rate']:5.1f}%   | {v['false_refusal_rate']:5.1f}%    | {v['hallucination_rate']:5.1f}%  | {v['guardrail_refusal_acc']:5.1f}%"
        print(row)
    print("=" * 105)

    out_file = os.path.join(ROOT_DIR, "benchmarks", "voice", "multilingual_retrieval_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.time(),
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "index": "FAISS-HNSW (Flat IP, 384-d)",
            "top_k": 5,
            "results": summary_report,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved multilingual benchmark report to: {out_file}\n")


if __name__ == "__main__":
    asyncio.run(evaluate_multilingual())
