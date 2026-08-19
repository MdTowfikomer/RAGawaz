"""
Investigation of Failed In-Domain Queries on 25K Multilingual Index.

For every failed query (where gold passage is not retrieved in Top-10 or not answered correctly), records:
- query
- language
- gold passage ID and snippet
- top-1, top-5, top-10 retrieved chunks
- embedding score, BM25 rank, RRF rank
- LLM answer / status
- failure type classification (A-F)
"""

import os
import sys
import json
import time
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import httpx



if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_URL = "http://127.0.0.1:8000/api/query"


def normalize_id(pid: str) -> str:
    if not pid:
        return ""
    pid_str = str(pid).strip().lower().replace(":", "_")
    parts = pid_str.split("_")
    if len(parts) >= 3 and len(parts[0]) <= 3:
        return "_".join(parts[1:])
    return pid_str


def is_match(retrieved_chunk: Dict[str, Any], gold_record: Dict[str, Any]) -> bool:
    gold_pid = gold_record.get("passage_id") or gold_record.get("gold_passage_id")
    gold_qid = gold_record.get("query_id")
    gold_snippet = gold_record.get("expected_answer_snippet") or gold_record.get("gold_answer") or ""

    chunk_pid = retrieved_chunk.get("passage_id") or retrieved_chunk.get("vector_id")
    chunk_qid = retrieved_chunk.get("query_id")
    chunk_text = retrieved_chunk.get("text", "")
    chunk_parent = retrieved_chunk.get("parent_text", "")
    full_chunk_text = f"{chunk_text} {chunk_parent}"

    # 1. Exact PID
    if gold_pid and chunk_pid:
        if normalize_id(str(gold_pid)) == normalize_id(str(chunk_pid)):
            return True
        if str(gold_pid).strip().lower().replace(":", "_") == str(chunk_pid).strip().lower().replace(":", "_"):
            return True

    # 2. QID cluster
    if gold_pid:
        gold_str = str(gold_pid).strip().lower().replace(":", "_")
        for p in gold_str.split("_"):
            if p.isdigit() and len(p) >= 4:
                gold_qid = int(p)
                break

    if chunk_pid and not chunk_qid:
        chunk_str = str(chunk_pid).strip().lower().replace(":", "_")
        for p in chunk_str.split("_"):
            if p.isdigit() and len(p) >= 4:
                chunk_qid = int(p)
                break

    if gold_qid and chunk_qid:
        try:
            if int(gold_qid) == int(chunk_qid):
                return True
        except Exception:
            pass

    # 3. Content overlap
    if gold_snippet and len(gold_snippet.strip()) >= 15:
        clean_snippet = re.sub(r'[^\w\s]', '', gold_snippet.lower()).strip()
        clean_chunk = re.sub(r'[^\w\s]', '', full_chunk_text.lower()).strip()
        words = clean_snippet.split()
        if len(words) >= 4:
            match_count = sum(1 for w in words if w in clean_chunk)
            if match_count / len(words) >= 0.60:
                return True

    return False


def classify_failure(
    query_record: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    status: str,
    answer: str,
    hit_rank: Any
) -> Tuple[str, str]:
    """
    Classify failures into A-F:
    A. Retrieval failure (target passage not in top-10)
    B. Correct passage retrieved but LLM failed / hallucinated
    C. Correct passage retrieved but grounding rejected
    D. Wrong-language retrieval (script/language mismatch with query)
    E. Answer extraction failure (empty / malformed)
    F. Dataset/gold ambiguity
    """
    q_lang = query_record.get("language", "en").lower()
    gold_pid = query_record.get("passage_id", "")
    gold_snippet = query_record.get("expected_answer_snippet", "")

    if hit_rank is None:
        # Check if top retrieved chunks are in completely different language
        if retrieved_chunks:
            ret_langs = [c.get("language", "") for c in retrieved_chunks[:3] if c.get("language")]
            if ret_langs and all(l != q_lang and l not in ["en", "eng"] for l in ret_langs):
                return "D", "Wrong-language retrieval (dominant chunks in non-target script/lang)"
        return "A", "Retrieval failure (Gold passage not in Top-10)"

    # Hit was found in top-10, but pipeline failed
    if status.startswith("refusal"):
        return "C", f"Correct passage retrieved at rank #{hit_rank}, but rejected by guardrails ({status})"

    if not answer or len(answer.strip()) < 5:
        return "E", "Answer extraction failure (Empty or truncated generation)"

    # Check if answer contradicts gold
    if gold_snippet and len(gold_snippet.strip()) >= 10:
        words = set(re.findall(r'\w+', gold_snippet.lower()))
        ans_words = set(re.findall(r'\w+', answer.lower()))
        if len(words) >= 3 and len(words.intersection(ans_words)) == 0:
            return "B", "Correct passage retrieved, but LLM generated inaccurate/divergent response"

    return "F", "Dataset / gold ambiguity or partial match"


def main():
    print("=" * 85)
    print("🔍 INVESTIGATING IN-DOMAIN QUERY FAILURES (25K Corpus)")
    print("=" * 85)

    benchmark_file = os.path.join(ROOT_DIR, "benchmarks", "benchmark_500.json")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    in_domain_queries = bench_data.get("in_domain_queries", [])
    client = httpx.Client(timeout=15.0)

    failures = []
    category_counts = defaultdict(int)

    for i, q in enumerate(in_domain_queries):
        q_text = q.get("query", "")
        q_lang = q.get("language", "en")
        gold_pid = q.get("passage_id", "")
        gold_snippet = q.get("expected_answer_snippet", "")

        try:
            resp = client.post(SERVER_URL, json={"query": q_text, "top_k": 10})
            if resp.status_code == 200:
                data = resp.json()
                retrieved = data.get("retrieved_chunks", [])
                status = data.get("status", "")
                answer = data.get("answer", "")
                tel = data.get("telemetry", {})
            else:
                retrieved = []
                status = f"http_{resp.status_code}"
                answer = ""
                tel = {}
        except Exception as e:
            retrieved = []
            status = "connection_error"
            answer = ""
            tel = {}

        hit_rank = None
        for rank, chunk in enumerate(retrieved):
            if is_match(chunk, q):
                hit_rank = rank + 1
                break

        # A failure is either a retrieval miss (hit_rank is None) OR a status failure
        is_success = (hit_rank is not None and status == "success")

        if not is_success:
            cat_code, cat_desc = classify_failure(q, retrieved, status, answer, hit_rank)
            category_counts[cat_code] += 1

            top1_str = f"[{retrieved[0].get('passage_id', 'N/A')}] (score: {retrieved[0].get('score', 0):.3f}) {retrieved[0].get('text', '')[:60]}" if len(retrieved) >= 1 else "None"
            top5_str = f"[{retrieved[4].get('passage_id', 'N/A')}] (score: {retrieved[4].get('score', 0):.3f}) {retrieved[4].get('text', '')[:60]}" if len(retrieved) >= 5 else "None"
            top10_str = f"[{retrieved[9].get('passage_id', 'N/A')}] (score: {retrieved[9].get('score', 0):.3f}) {retrieved[9].get('text', '')[:60]}" if len(retrieved) >= 10 else "None"

            failures.append({
                "id": i,
                "query": q_text,
                "language": q_lang,
                "gold_passage_id": gold_pid,
                "gold_snippet": gold_snippet[:100],
                "hit_rank": hit_rank,
                "status": status,
                "answer": answer[:120] if answer else "N/A",
                "top_1": top1_str,
                "top_5": top5_str,
                "top_10": top10_str,
                "embedding_score": retrieved[0].get("score", 0.0) if retrieved else 0.0,
                "category_code": cat_code,
                "category_description": cat_desc,
            })

    print(f"\nTotal in-domain failures: {len(failures)} / {len(in_domain_queries)} ({len(failures)/len(in_domain_queries)*100:.2f}%)")
    print("=" * 85)

    print("\n📊 FAILURE CLASSIFICATION SUMMARY:")
    print("-" * 85)
    category_labels = {
        "A": "Retrieval failure (Target not in Top-10)",
        "B": "Correct passage retrieved but LLM failed",
        "C": "Correct passage retrieved but grounding rejected",
        "D": "Wrong-language retrieval",
        "E": "Answer extraction failure",
        "F": "Dataset / gold ambiguity"
    }

    for code in ["A", "B", "C", "D", "E", "F"]:
        count = category_counts.get(code, 0)
        pct = (count / len(failures) * 100.0) if failures else 0.0
        print(f"  [{code}] {category_labels[code]:<52}: {count:>2} ({pct:>5.1f}%)")

    print("=" * 85)

    print("\n🔍 DETAILED BREAKDOWN OF INDIVIDUAL FAILURES:")
    print("-" * 85)

    for f_idx, fail in enumerate(failures, 1):
        print(f"#{f_idx:02d} [{fail['language'].upper()}] Query: {fail['query']}")
        print(f"    Gold Passage ID : {fail['gold_passage_id']}")
        print(f"    Gold Snippet    : {fail['gold_snippet']}")
        print(f"    Hit Rank        : {fail['hit_rank'] if fail['hit_rank'] else 'NOT IN TOP-10'}")
        print(f"    Status / Answer : {fail['status']} | Ans: {fail['answer']}")
        print(f"    Top-1 Chunk     : {fail['top_1']}")
        print(f"    Failure Type    : [{fail['category_code']}] {fail['category_description']}")
        print()

    # Save detailed report to disk
    out_json = os.path.join(ROOT_DIR, "benchmarks", "in_domain_failures_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"total_queries": len(in_domain_queries), "failure_count": len(failures), "summary": dict(category_counts), "failures": failures}, f, indent=2, ensure_ascii=False)
    print(f"Full failure report saved to: {out_json}")


if __name__ == "__main__":
    main()
