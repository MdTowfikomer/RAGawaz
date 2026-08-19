import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.rag.embedder import get_embedding_provider
from backend.app.rag.retriever import FAISSHNSWRetriever
from backend.app.guardrails.relevance import RelevanceGate, InsufficientEvidenceChecker
from backend.app.guardrails.groundedness import GroundednessVerifier

def analyze():
    report_file = os.path.join(ROOT_DIR, "benchmarks", "voice", "answer_correctness_report.json")
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    false_refusals = [r for r in report["records"] if r["classification"] == "❌ False Refusal on Answerable Query"]
    print(f"Analyzing {len(false_refusals)} False Refusal Cases:\n")

    embedder = get_embedding_provider("minilm")
    retriever = FAISSHNSWRetriever()
    retriever.load(os.path.join(ROOT_DIR, "backend", "data", "faiss_cache"))
    relevance_gate = RelevanceGate(threshold=0.25)
    insufficient_checker = InsufficientEvidenceChecker(confidence_threshold=0.28)
    groundedness_verifier = GroundednessVerifier(high_threshold=0.25, low_threshold=0.12, min_query_overlap_threshold=0.20)

    for i, r in enumerate(false_refusals):
        qid = r["query_id"]
        q_text = r["query"]
        gold_ans = r["gold_answer"]
        print("=" * 90)
        print(f"CASE {i+1} / {len(false_refusals)}: Query ID {qid}")
        print(f"Query:        '{q_text}'")
        print(f"Gold Answer:  '{gold_ans}'")

        # Retrieval check
        q_emb = embedder.embed_query(q_text)
        chunks = retriever.search(q_emb, top_k=3)
        scores = [c.score for c in chunks]

        print("\nTop 3 Retrieved Chunks:")
        for j, c in enumerate(chunks):
            print(f"  [{j+1}] Score={c.score:.4f} | PassageID={c.passage_id}")
            text_prev = (c.parent_text or c.text)[:160].replace("\n", " ")
            print(f"      Text: {text_prev}...")

        # Guardrail decisions
        is_rel, rel_msg = relevance_gate.evaluate(scores, query=q_text)
        has_evi, evi_msg = insufficient_checker.evaluate(scores, query=q_text)
        
        print("\nGuardrail Diagnostics:")
        print(f"  - Relevance Gate:      {'PASS' if is_rel else 'REFUSED (' + str(rel_msg) + ')'}")
        print(f"  - Insufficient Gate:   {'PASS' if has_evi else 'REFUSED (' + str(evi_msg) + ')'}")
        
        gen_ans = r["generated_answer"]
        ctx_texts = [c.parent_text or c.text for c in chunks]
        is_grd, method, g_score, grd_msg = groundedness_verifier.evaluate(gen_ans, ctx_texts, query=q_text)
        print(f"  - Grounding Verifier:  {'PASS' if is_grd else 'REFUSED (method=' + method + ', score=' + str(g_score) + ')'}")
        print(f"  - Generated Answer:    '{gen_ans}'")
        print("=" * 90 + "\n")

if __name__ == "__main__":
    analyze()
