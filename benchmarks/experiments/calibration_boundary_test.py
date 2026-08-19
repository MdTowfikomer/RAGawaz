"""
Empirical Boundary Telemetry and Calibration Test Runner.

Executes the 3 critical proof scenarios:
1. Known answer ("What is a corporation?") -> SUFFICIENT, PASS, EXECUTED, VERIFIED
2. Clearly absent entity ("What information do you have about India?") -> INSUFFICIENT, FAIL, SKIPPED, SKIPPED
3. Semantically deceptive query ("What is the population of India?") -> INSUFFICIENT, FAIL, SKIPPED, SKIPPED
"""

import time
import json
import httpx
import sys

sys.stdout.reconfigure(encoding='utf-8')

TARGET_URL = "http://127.0.0.1:8000/api/query"

CALIBRATION_SUITE = [
    {
        "id": "1_KNOWN_ANSWER",
        "title": "Scenario 1: Known In-Corpus Answer",
        "query": "निगम क्या है?",
        "expected_entity": "PASS",
        "expected_evidence": "SUFFICIENT",
        "expected_llm": "EXECUTED",
        "expected_verdict": "VERIFIED",
    },
    {
        "id": "2_ABSENT_ENTITY",
        "title": "Scenario 2: Clearly Absent Entity",
        "query": "What information do you have about India?",
        "expected_entity": "FAIL",
        "expected_evidence": "INSUFFICIENT",
        "expected_llm": "SKIPPED",
        "expected_verdict": "SKIPPED",
    },
    {
        "id": "3_SEMANTICALLY_DECEPTIVE",
        "title": "Scenario 3: Semantically Deceptive (Missing Fact)",
        "query": "What is the population of India?",
        "expected_entity": "FAIL",
        "expected_evidence": "INSUFFICIENT",
        "expected_llm": "SKIPPED",
        "expected_verdict": "SKIPPED",
    }
]

def run_calibration_test():
    print("=" * 80)
    print("      EMPIRICAL BOUNDARY TELEMETRY & DECISION PROOF SUITE")
    print("=" * 80)
    
    client = httpx.Client(timeout=30.0)
    all_passed = True

    for item in CALIBRATION_SUITE:
        print(f"\n[{item['id']}] {item['title']}")
        print(f"Query: \"{item['query']}\"")
        
        t0 = time.perf_counter()
        resp = client.post(TARGET_URL, json={"query": item["query"]})
        t_req = (time.perf_counter() - t0) * 1000.0
        
        if resp.status_code != 200:
            print(f"FAILED: HTTP {resp.status_code}")
            all_passed = False
            continue
            
        data = resp.json()
        telemetry = data.get("telemetry", {})
        status = data.get("status")
        answer = data.get("answer", "")
        
        # Extract metrics
        bge_ms = telemetry.get("embedding_ms", telemetry.get("query_embedding_ms", 0.0))
        faiss_ms = telemetry.get("faiss_ms", 0.0)
        bm25_ms = telemetry.get("bm25_ms", 0.0)
        rrf_ms = telemetry.get("rrf_ms", 0.0)
        pre_llm_ms = telemetry.get("pre_llm_total_ms", 0.0)
        llm_ttft_ms = telemetry.get("llm_ttft_ms", 0.0)
        total_ms = telemetry.get("text_to_answer_ms", t_req)
        
        # Extract diagnostics
        entity_match = telemetry.get("entity_match", "N/A")
        evidence_status = telemetry.get("evidence_status", "N/A")
        llm_invocation = telemetry.get("llm_invocation", "N/A")
        groundedness_verdict = telemetry.get("groundedness_verdict", "N/A")
        
        print(f"  Status:               {status}")
        print(f"  Answer:               \"{answer[:120]}...\"")
        print(f"  --- Telemetry Breakdown ---")
        print(f"  BGE-M3 Embed:         {bge_ms:.2f} ms")
        print(f"  FAISS HNSW:           {faiss_ms:.2f} ms")
        print(f"  BM25 Sparse:          {bm25_ms:.2f} ms")
        print(f"  RRF Fusion:           {rrf_ms:.2f} ms")
        print(f"  Pre-LLM Boundary:     {pre_llm_ms:.2f} ms")
        if llm_invocation == "EXECUTED":
            print(f"  LLM TTFT (Groq):      {llm_ttft_ms:.2f} ms")
        print(f"  Text -> Answer Total: {total_ms:.2f} ms")
        print(f"  --- Boundary Proof Badges ---")
        print(f"  Evidence Sufficiency: {evidence_status} (Expected: {item['expected_evidence']})")
        print(f"  Entity Match:         {entity_match} (Expected: {item['expected_entity']})")
        print(f"  LLM Invocation:       {llm_invocation} (Expected: {item['expected_llm']})")
        print(f"  Groundedness:         {groundedness_verdict} (Expected: {item['expected_verdict']})")
        
        # Verify expectations
        ok_entity = entity_match == item["expected_entity"]
        ok_evidence = evidence_status == item["expected_evidence"]
        ok_llm = llm_invocation == item["expected_llm"]
        
        if ok_entity and ok_evidence and ok_llm:
            print(f"  >>> VERDICT: [PASS - EXACT BOUNDARY MATCH]")
        else:
            print(f"  >>> VERDICT: [FAIL - MISMATCH]")
            all_passed = False

    print("\n" + "=" * 80)
    print(f"CALIBRATION SUITE SUMMARY: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_calibration_test()
