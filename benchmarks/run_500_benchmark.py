"""
Run 500-query benchmark against live server and produce detailed report.
Usage: python benchmarks/run_500_benchmark.py
"""
import sys, json, time, os
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import httpx

BENCHMARK_FILE = "benchmarks/benchmark_500.json"
REPORT_FILE = "benchmarks/benchmark_500_results.json"
SERVER_URL = "http://127.0.0.1:8000/api/query"

print("Loading benchmark...")
with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

in_domain = benchmark["in_domain_queries"]
out_of_domain = benchmark["out_of_domain_queries"]
total = len(in_domain) + len(out_of_domain)
print(f"Loaded {total} queries ({len(in_domain)} in-domain + {len(out_of_domain)} out-of-domain)")
print(f"Server: {SERVER_URL}")
print("=" * 80)

# Results storage
results = []
errors = []

# Run in-domain queries
print(f"\n[1/2] Running {len(in_domain)} IN-DOMAIN queries...")
in_domain_correct = 0
in_domain_times = []
lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "times": []})

for i, q in enumerate(in_domain):
    try:
        t0 = time.perf_counter()
        r = httpx.post(SERVER_URL, json={"query": q["query"], "top_k": 5}, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        d = r.json()
        
        status = d["status"]
        is_correct = status == "success"
        tel = d.get("telemetry", {})
        
        result = {
            "id": i,
            "type": "in_domain",
            "query": q["query"],
            "language": q["language"],
            "category": q["category"],
            "status": status,
            "correct": is_correct,
            "total_ms": round(elapsed, 1),
            "embed_ms": tel.get("embedding_ms", 0),
            "faiss_ms": tel.get("faiss_ms", 0),
            "bm25_ms": tel.get("bm25_ms", 0),
            "llm_ttft_ms": tel.get("llm_ttft_ms", 0),
            "answer": d.get("answer", "")[:150] if is_correct else None,
            "refusal_reason": d.get("refusal_reason") if not is_correct else None,
        }
        results.append(result)
        
        if is_correct:
            in_domain_correct += 1
            in_domain_times.append(elapsed)
        
        lang_results[q["language"]]["total"] += 1
        if is_correct:
            lang_results[q["language"]]["correct"] += 1
            lang_results[q["language"]]["times"].append(elapsed)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(in_domain)} ({100*(i+1)/len(in_domain):.0f}%) | Accuracy so far: {in_domain_correct}/{i+1}")
    
    except Exception as e:
        errors.append({"id": i, "query": q["query"], "error": str(e)})
        results.append({"id": i, "type": "in_domain", "query": q["query"], "status": "error", "correct": False})

print(f"\n  IN-DOMAIN COMPLETE: {in_domain_correct}/{len(in_domain)} ({100*in_domain_correct/len(in_domain):.1f}%)")

# Run out-of-domain queries
print(f"\n[2/2] Running {len(out_of_domain)} OUT-OF-DOMAIN queries...")
ood_correct = 0
ood_times = []
category_results = defaultdict(lambda: {"correct": 0, "total": 0})

for i, q in enumerate(out_of_domain):
    try:
        t0 = time.perf_counter()
        r = httpx.post(SERVER_URL, json={"query": q["query"], "top_k": 5}, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        d = r.json()
        
        status = d["status"]
        is_correct = status != "success"  # Should NOT answer
        
        result = {
            "id": len(in_domain) + i,
            "type": "out_of_domain",
            "query": q["query"],
            "language": q.get("language", "unknown"),
            "category": q["category"],
            "status": status,
            "correct": is_correct,
            "total_ms": round(elapsed, 1),
            "reason": q.get("reason"),
            "answer_if_wrong": d.get("answer", "")[:100] if not is_correct else None,
        }
        results.append(result)
        
        if is_correct:
            ood_correct += 1
            ood_times.append(elapsed)
        
        category_results[q["category"]]["total"] += 1
        if is_correct:
            category_results[q["category"]]["correct"] += 1
        
        if (i + 1) % 30 == 0:
            print(f"  Progress: {i+1}/{len(out_of_domain)} ({100*(i+1)/len(out_of_domain):.0f}%) | Refusal rate: {ood_correct}/{i+1}")
    
    except Exception as e:
        errors.append({"id": len(in_domain) + i, "query": q["query"], "error": str(e)})
        results.append({"id": len(in_domain) + i, "type": "out_of_domain", "query": q["query"], "status": "error", "correct": True})

print(f"\n  OUT-OF-DOMAIN COMPLETE: {ood_correct}/{len(out_of_domain)} ({100*ood_correct/len(out_of_domain):.1f}%)")

# ============================================================
# GENERATE REPORT
# ============================================================
print(f"\n{'='*80}")
print(f"BENCHMARK RESULTS SUMMARY")
print(f"{'='*80}")

overall_correct = in_domain_correct + ood_correct
overall_total = len(in_domain) + len(out_of_domain)
print(f"\nOVERALL ACCURACY: {overall_correct}/{overall_total} ({100*overall_correct/overall_total:.1f}%)")
print(f"  In-domain (should answer):  {in_domain_correct}/{len(in_domain)} ({100*in_domain_correct/len(in_domain):.1f}%)")
print(f"  Out-of-domain (should refuse): {ood_correct}/{len(out_of_domain)} ({100*ood_correct/len(out_of_domain):.1f}%)")

if in_domain_times:
    sorted_times = sorted(in_domain_times)
    p50 = sorted_times[len(sorted_times)//2]
    p95 = sorted_times[int(len(sorted_times)*0.95)]
    print(f"\nIN-DOMAIN LATENCY (successful answers only):")
    print(f"  P50: {p50:.0f}ms | P95: {p95:.0f}ms | Min: {min(in_domain_times):.0f}ms | Max: {max(in_domain_times):.0f}ms | Mean: {sum(in_domain_times)/len(in_domain_times):.0f}ms")

if ood_times:
    sorted_ood = sorted(ood_times)
    print(f"\nOUT-OF-DOMAIN LATENCY (successful refusals only):")
    print(f"  P50: {sorted_ood[len(sorted_ood)//2]:.0f}ms | Mean: {sum(ood_times)/len(ood_times):.0f}ms")

print(f"\nPER-LANGUAGE RECALL (in-domain):")
for lang in sorted(lang_results.keys()):
    lr = lang_results[lang]
    pct = 100 * lr["correct"] / lr["total"] if lr["total"] > 0 else 0
    avg_t = sum(lr["times"])/len(lr["times"]) if lr["times"] else 0
    print(f"  {lang:>4}: {lr['correct']:>2}/{lr['total']:>2} ({pct:>5.1f}%) | Avg: {avg_t:.0f}ms")

print(f"\nOUT-OF-DOMAIN BY CATEGORY:")
for cat in sorted(category_results.keys()):
    cr = category_results[cat]
    pct = 100 * cr["correct"] / cr["total"] if cr["total"] > 0 else 0
    print(f"  {cat:<30}: {cr['correct']:>2}/{cr['total']:>2} ({pct:>5.1f}%)")

if errors:
    print(f"\nERRORS: {len(errors)} queries failed to execute")

# Save detailed report
report = {
    "summary": {
        "overall_accuracy": round(100*overall_correct/overall_total, 1),
        "in_domain_accuracy": round(100*in_domain_correct/len(in_domain), 1),
        "out_of_domain_accuracy": round(100*ood_correct/len(out_of_domain), 1),
        "in_domain_latency_p50_ms": round(sorted_times[len(sorted_times)//2], 1) if in_domain_times else 0,
        "in_domain_latency_p95_ms": round(sorted_times[int(len(sorted_times)*0.95)], 1) if in_domain_times else 0,
        "total_queries": overall_total,
        "errors": len(errors),
    },
    "per_language": {lang: {"correct": lr["correct"], "total": lr["total"], "accuracy_pct": round(100*lr["correct"]/lr["total"], 1)} for lang, lr in lang_results.items()},
    "per_category": {cat: {"correct": cr["correct"], "total": cr["total"], "accuracy_pct": round(100*cr["correct"]/cr["total"], 1)} for cat, cr in category_results.items()},
    "failures": [r for r in results if not r.get("correct", True)],
    "all_results": results,
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\nDetailed report saved to: {REPORT_FILE}")
print(f"{'='*80}")
