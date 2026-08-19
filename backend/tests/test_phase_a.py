"""
Phase A — Pre-LLM Benchmark Harness Tests.

Verifies:
1. phase_a_query_file_is_valid
2. exactly 50 queries exist
3. category counts are 20/10/10/10
4. answerable queries have relevant_passage_ids
5. no answerable query references a passage outside the indexed corpus
6. benchmark runner produces results JSON
7. llm_called is false for every query
8. results contain all required telemetry fields
9. benchmark is deterministic with the same seed
"""

import json
import os
import sys
import subprocess
from collections import Counter

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

QUERIES_PATH = os.path.join(ROOT_DIR, "benchmarks", "phase_a", "phase_a_queries.jsonl")
BASELINE_PATH = os.path.join(ROOT_DIR, "benchmarks", "phase_a", "results", "baseline_results.json")
CORPUS_PATH = os.path.join(ROOT_DIR, "backend", "data", "passages.jsonl")
GENERATE_SCRIPT = os.path.join(ROOT_DIR, "benchmarks", "phase_a", "generate_queries.py")


def load_queries():
    """Load all queries from phase_a_queries.jsonl."""
    queries = []
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            ls = line.strip()
            if ls:
                queries.append(json.loads(ls))
    return queries


def load_corpus_passage_ids():
    """Load all passage_ids from the corpus."""
    pids = set()
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            ls = line.strip()
            if not ls:
                continue
            rec = json.loads(ls)
            pids.add(rec["passage_id"])
    return pids


# ─── Test 1: Query file is valid JSONL ────────────────────────────────────────
def test_phase_a_query_file_is_valid():
    """Verify phase_a_queries.jsonl exists and contains valid JSON lines."""
    assert os.path.exists(QUERIES_PATH), f"Query file not found: {QUERIES_PATH}"
    queries = load_queries()
    assert len(queries) > 0, "Query file is empty"
    # Every record should have required fields
    required_fields = {"id", "query", "category", "expected_decision"}
    for q in queries:
        missing = required_fields - set(q.keys())
        assert not missing, f"Query {q.get('id', '?')} missing fields: {missing}"


# ─── Test 2: Exactly 50 queries ──────────────────────────────────────────────
def test_exactly_50_queries_exist():
    """Verify exactly 50 queries in the benchmark dataset."""
    queries = load_queries()
    assert len(queries) == 50, f"Expected 50 queries, got {len(queries)}"


# ─── Test 3: Category counts are 20/10/10/10 ─────────────────────────────────
def test_category_counts_are_correct():
    """Verify category distribution: 20 answerable, 10 insufficient, 10 off-topic, 10 unsafe."""
    queries = load_queries()
    counts = Counter(q["category"] for q in queries)
    assert counts["answerable"] == 20, f"Answerable: expected 20, got {counts.get('answerable', 0)}"
    assert counts["insufficient_evidence"] == 10, f"Insufficient: expected 10, got {counts.get('insufficient_evidence', 0)}"
    assert counts["off_topic"] == 10, f"Off-topic: expected 10, got {counts.get('off_topic', 0)}"
    assert counts["unsafe"] == 10, f"Unsafe: expected 10, got {counts.get('unsafe', 0)}"


# ─── Test 4: Answerable queries have relevant_passage_ids ─────────────────────
def test_answerable_queries_have_relevant_passage_ids():
    """Verify all answerable queries contain non-empty relevant_passage_ids."""
    queries = load_queries()
    answerable = [q for q in queries if q["category"] == "answerable"]
    assert len(answerable) == 20

    for q in answerable:
        assert "relevant_passage_ids" in q, f"Query {q['id']} missing relevant_passage_ids"
        pids = q["relevant_passage_ids"]
        assert isinstance(pids, list), f"Query {q['id']} relevant_passage_ids is not a list"
        assert len(pids) >= 1, f"Query {q['id']} has empty relevant_passage_ids"


# ─── Test 5: No answerable query references a passage outside the corpus ──────
def test_no_answerable_passage_outside_corpus():
    """Verify all relevant_passage_ids actually exist in the indexed 50k corpus."""
    queries = load_queries()
    answerable = [q for q in queries if q["category"] == "answerable"]
    corpus_pids = load_corpus_passage_ids()

    for q in answerable:
        for pid in q["relevant_passage_ids"]:
            assert pid in corpus_pids, (
                f"Query {q['id']} references passage {pid} "
                f"which is NOT in the corpus ({len(corpus_pids)} passages loaded)"
            )


# ─── Test 6: Benchmark runner produces results JSON ───────────────────────────
def test_benchmark_runner_produces_results_json():
    """Verify baseline_results.json exists and is valid JSON (run benchmark if missing)."""
    if not os.path.exists(BASELINE_PATH):
        pytest.skip("Baseline results not yet generated. Run: python benchmarks/phase_a/run_phase_a.py")

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "benchmark" in data
    assert data["benchmark"] == "phase_a_pre_llm_baseline"
    assert "per_query_results" in data
    assert "retrieval_metrics" in data
    assert "guardrail_metrics" in data
    assert "latency_metrics" in data
    assert data["query_count"] == 50


# ─── Test 7: llm_called is false for every query ─────────────────────────────
def test_llm_called_is_false_for_every_query():
    """Verify llm_called is false for all 50 queries in results."""
    if not os.path.exists(BASELINE_PATH):
        pytest.skip("Baseline results not yet generated. Run: python benchmarks/phase_a/run_phase_a.py")

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_query = data["per_query_results"]
    assert len(per_query) == 50

    for r in per_query:
        assert r["llm_called"] is False, (
            f"Query {r['id']} has llm_called=True! Phase A must never call LLM."
        )

    assert data["llm_calls"] == 0


# ─── Test 8: Results contain all required telemetry fields ────────────────────
def test_results_contain_all_required_telemetry_fields():
    """Verify each per-query result has all specified metric fields."""
    if not os.path.exists(BASELINE_PATH):
        pytest.skip("Baseline results not yet generated. Run: python benchmarks/phase_a/run_phase_a.py")

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_fields = {
        "id", "category", "query", "expected_decision", "actual_decision",
        "query_embedding_ms", "vector_search_ms", "reranking_ms",
        "embed_retrieval_ms", "guardrail_ms", "total_pre_llm_ms",
        "llm_called", "top1_score", "top3_score", "top5_score", "top10_score",
        "retrieved_passage_ids", "retrieved_is_selected", "relevant_passage_ids",
    }

    for r in data["per_query_results"]:
        missing = required_fields - set(r.keys())
        assert not missing, f"Query {r.get('id', '?')} missing telemetry fields: {missing}"


# ─── Test 9: Benchmark is deterministic with the same seed ────────────────────
def test_benchmark_is_deterministic_with_same_seed():
    """Verify regenerating queries with seed=42 produces identical output."""
    # Load current queries
    queries_before = load_queries()

    # Regenerate
    result = subprocess.run(
        [sys.executable, GENERATE_SCRIPT],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
        encoding="utf-8",
    )
    assert result.returncode == 0, f"Generator failed: {result.stderr}"

    # Load regenerated queries
    queries_after = load_queries()

    assert len(queries_before) == len(queries_after), "Query count changed after regeneration"

    for q_before, q_after in zip(queries_before, queries_after):
        assert q_before["id"] == q_after["id"], f"ID mismatch: {q_before['id']} vs {q_after['id']}"
        assert q_before["query"] == q_after["query"], f"Query text mismatch for {q_before['id']}"
        assert q_before["category"] == q_after["category"], f"Category mismatch for {q_before['id']}"
        if "relevant_passage_ids" in q_before:
            assert q_before["relevant_passage_ids"] == q_after["relevant_passage_ids"], (
                f"Passage IDs mismatch for {q_before['id']}"
            )
