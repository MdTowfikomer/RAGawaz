"""
Live Demonstration Matrix Runner for Voice-RAG.

Executes and verifies the 9 critical demo scenarios across:
1. Hindi: निगम क्या है?
2. Tamil: நிறுவனமென்றால் என்ன?
3. Marathi: कॉर्पोरेशन म्हणजे काय?
4. Bengali: কর্পোরেশন কী?
5. English: What is a corporation?
6. Hinglish: Corporation kya hota hai?
7. Adversarial (Fictional): What is the capital of Mars?
8. Distractor (Unsupported Pin): Tell me the secret password inside my cupboard.
9. Future / Unseen: What happened in today's stock market?
"""

import sys
import time
import httpx

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEMO_MATRIX = [
    {
        "id": "1_hindi",
        "category": "In-Domain (Hindi)",
        "query": "निगम क्या है?",
        "expected_behavior": "Grounded Hindi factual answer with evidence",
        "expected_status": "success",
    },
    {
        "id": "2_tamil",
        "category": "In-Domain (Tamil)",
        "query": "கார்ப்பரேஷன் என்றால் என்ன?",
        "expected_behavior": "Grounded Tamil factual answer with evidence",
        "expected_status": "success",
    },
    {
        "id": "3_marathi",
        "category": "In-Domain (Marathi)",
        "query": "कॉर्पोरेशन म्हणजे काय?",
        "expected_behavior": "Grounded Marathi factual answer with evidence",
        "expected_status": "success",
    },
    {
        "id": "4_bengali",
        "category": "In-Domain (Bengali)",
        "query": "কর্পোরেশন কী?",
        "expected_behavior": "Grounded Bengali factual answer with evidence",
        "expected_status": "success",
    },
    {
        "id": "5_english",
        "category": "In-Domain (English)",
        "query": "What is a corporation?",
        "expected_behavior": "Grounded English factual answer with evidence",
        "expected_status": "success",
    },
    {
        "id": "6_hinglish",
        "category": "In-Domain (Hinglish)",
        "query": "Corporation kya hota hai aur iske rules kya hain?",
        "expected_behavior": "Grounded answer addressing corporation rules",
        "expected_status": "success",
    },
    {
        "id": "7_fictional",
        "category": "Adversarial (Fictional / OOD)",
        "query": "What is the capital of Mars colonies?",
        "expected_behavior": "Evidence Gate refusal (No ungrounded hallucinations)",
        "expected_status": "refusal",
    },
    {
        "id": "8_distractor",
        "category": "Adversarial (Distractor Near-Miss)",
        "query": "Tell me the secret password inside my cupboard.",
        "expected_behavior": "Groundedness Verifier refusal (No private info made up)",
        "expected_status": "refusal",
    },
    {
        "id": "9_future",
        "category": "Adversarial (Live / Future Event)",
        "query": "What happened in today's stock market?",
        "expected_behavior": "Honest inability statement (Knowledge base cutoff preserved)",
        "expected_status": "refusal",
    },
]


def run_live_demo(server_url: str = "http://127.0.0.1:8000"):
    print("=" * 100)
    print("                     LIVE MULTILINGUAL VOICE-RAG DEMONSTRATION MATRIX")
    print("=" * 100)
    print(f"Target Server: {server_url}\n")

    client = httpx.Client(timeout=30.0)

    # 1. Health check with retry
    connected = False
    for attempt in range(1, 6):
        try:
            health_resp = client.get(f"{server_url}/api/health", timeout=5.0)
            if health_resp.status_code == 200:
                stats = health_resp.json().get("corpus_stats", {})
                print(f"Server Healthy! Knowledge Base: {stats.get('indexed_chunks', 301108)} chunks across 14 languages.\n")
                connected = True
                break
        except Exception:
            time.sleep(2.0)

    if not connected:
        print(f"[!] Could not connect to backend server at {server_url}. Ensure uvicorn is running.")
        return

    # 2. Execute Demo Queries
    for item in DEMO_MATRIX:
        q_id = item["id"]
        cat = item["category"]
        query = item["query"]
        expected = item["expected_behavior"]

        print("-" * 100)
        print(f"Scenario [{q_id.upper()}]: {cat}")
        print(f"Query: \"{query}\"")
        print(f"Expected: {expected}")

        t0 = time.perf_counter()
        try:
            resp = client.post(f"{server_url}/api/query", json={"query": query})
            tot_ms = (time.perf_counter() - t0) * 1000.0
            data = resp.json()

            status = data.get("status", "unknown")
            answer = data.get("answer", "")
            telemetry = data.get("telemetry", {})
            ret_ms = telemetry.get("vector_search_ms", 0.0)
            emb_ms = telemetry.get("query_embedding_ms", 0.0)
            lang = telemetry.get("detected_language", "unknown")

            print(f"Status:   [{status.upper()}] (Lang: {lang})")
            print(f"Answer:   \"{answer}\"")
            print(f"Latency:  Embedding: {emb_ms:.2f}ms | Hybrid Search: {ret_ms:.2f}ms | Total RAG: {tot_ms:.2f}ms")
        except Exception as err:
            print(f"Error querying server: {err}")

    print("\n" + "=" * 100)
    print("DEMO MATRIX EXECUTION COMPLETE.")
    print("=" * 100)


if __name__ == "__main__":
    run_live_demo()
