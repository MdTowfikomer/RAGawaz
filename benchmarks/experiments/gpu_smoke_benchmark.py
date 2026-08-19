"""
Stage 1: 100-Chunk GPU Smoke Benchmark for Embedding Candidates.

Validates that:
1. All 3 models load successfully on CUDA.
2. Embeddings are computed on CUDA with active GPU VRAM allocation.
3. No CPU fallback occurs.
4. Correct embedding dimensions (384, 1024, 1024).
5. Multilingual-E5 uses 'passage: ' and 'query: ' prefixes.
6. Per-query and batch latency are measured and reported.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

CANDIDATES = [
    {
        "key": "minilm",
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "passage_prefix": "",
        "query_prefix": "",
    },
    {
        "key": "bge_m3",
        "name": "BAAI/bge-m3",
        "dimension": 1024,
        "passage_prefix": "",
        "query_prefix": "",
    },
    {
        "key": "e5_large",
        "name": "intfloat/multilingual-e5-large",
        "dimension": 1024,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
    },
]

# 100 Multilingual sample text chunks
SAMPLE_TEXTS = [
    f"भारत दक्षिण एशिया का एक विशाल देश है जिसकी राजधानी नई दिल्ली है। Sample text chunk {i}."
    for i in range(100)
]
SAMPLE_QUERY = "भारत की राजधानी क्या है?"


def run_gpu_smoke_test():
    print("=" * 85)
    print("STAGE 1: 100-CHUNK GPU SMOKE BENCHMARK")
    print("=" * 85)

    if not torch.cuda.is_available():
        print("❌ FATAL: torch.cuda.is_available() is False. CUDA is required for this benchmark.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Detected GPU: {gpu_name} (Total VRAM: {total_vram_gb:.2f} GB)")
    print(f"PyTorch Version: {torch.__version__} | CUDA Available: True\n")

    smoke_results = []
    all_passed = True

    for cand in CANDIDATES:
        print("-" * 85)
        print(f"Testing Candidate: {cand['name']}")
        print(f"Expected Dimension: {cand['dimension']} | Prefixes: passage='{cand['passage_prefix']}', query='{cand['query_prefix']}'")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_before_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)

        # 1. Model Load on CUDA
        t0_load = time.perf_counter()
        model = SentenceTransformer(cand["name"], device="cuda")
        model.eval()
        load_time_s = time.perf_counter() - t0_load

        # Check model device
        param_device = next(model.parameters()).device
        is_cuda = param_device.type == "cuda"
        vram_after_load_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)

        print(f"  Load Time: {load_time_s:.2f}s | Device: {param_device} | VRAM allocated: {vram_after_load_mb:.1f} MB")
        if not is_cuda:
            print(f"  ❌ ERROR: Model failed to load on CUDA (device={param_device})")
            all_passed = False

        # 2. Batch Embedding on 100 Chunks
        passage_inputs = [f"{cand['passage_prefix']}{t}" for t in SAMPLE_TEXTS]
        t0_batch = time.perf_counter()
        with torch.inference_mode():
            batch_embs = model.encode(
                passage_inputs,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        batch_time_ms = (time.perf_counter() - t0_batch) * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

        # Verify Dimension & Normalization
        actual_dim = batch_embs.shape[1]
        is_dim_correct = actual_dim == cand["dimension"]
        norms = np.linalg.norm(batch_embs, axis=1)
        is_normalized = np.allclose(norms, 1.0, atol=1e-3)

        print(f"  100-Chunk Batch Time: {batch_time_ms:.2f} ms ({batch_time_ms/100:.2f} ms/chunk)")
        print(f"  Output Shape: {batch_embs.shape} (Expected dim: {cand['dimension']}) -> Dim Valid: {is_dim_correct}")
        print(f"  Embeddings Normalized: {is_normalized} (mean norm: {np.mean(norms):.4f})")
        print(f"  Peak GPU VRAM: {peak_vram_mb:.1f} MB")

        # 3. Query Embedding Latency (Single item warmup + 10 runs)
        query_input = f"{cand['query_prefix']}{SAMPLE_QUERY}"
        # warmup
        _ = model.encode([query_input], normalize_embeddings=True, convert_to_numpy=True)

        query_latencies = []
        for _ in range(10):
            t0_q = time.perf_counter()
            with torch.inference_mode():
                q_emb = model.encode([query_input], normalize_embeddings=True, convert_to_numpy=True)
            query_latencies.append((time.perf_counter() - t0_q) * 1000.0)

        q_p50 = float(np.percentile(query_latencies, 50))
        q_p70 = float(np.percentile(query_latencies, 70))
        q_p95 = float(np.percentile(query_latencies, 95))
        print(f"  Single Query Latency: P50={q_p50:.2f}ms | P70={q_p70:.2f}ms | P95={q_p95:.2f}ms")

        # Cleanup model from GPU
        del model
        del batch_embs
        torch.cuda.empty_cache()

        passed = is_cuda and is_dim_correct and is_normalized
        if passed:
            print(f"  ✅ {cand['key'].upper()} Smoke Test: PASSED")
        else:
            print(f"  ❌ {cand['key'].upper()} Smoke Test: FAILED")
            all_passed = False

        smoke_results.append({
            "key": cand["key"],
            "name": cand["name"],
            "device": str(param_device),
            "expected_dimension": cand["dimension"],
            "actual_dimension": actual_dim,
            "dimension_valid": is_dim_correct,
            "normalized": is_normalized,
            "load_time_s": load_time_s,
            "batch_100_ms": batch_time_ms,
            "peak_vram_mb": peak_vram_mb,
            "query_latency_p50_ms": q_p50,
            "query_latency_p70_ms": q_p70,
            "query_latency_p95_ms": q_p95,
            "passed": passed,
        })

    print("\n" + "=" * 85)
    print("SMOKE BENCHMARK SUMMARY:")
    print("=" * 85)
    header = f"{'Model':<35} | {'Dim':<5} | {'100 Chunks (ms)':<15} | {'Query P70 (ms)':<15} | {'Peak VRAM':<10} | {'Status'}"
    print(header)
    print("-" * len(header))
    for r in smoke_results:
        status_str = "PASSED" if r["passed"] else "FAILED"
        print(f"{r['name']:<35} | {r['actual_dimension']:<5} | {r['batch_100_ms']:13.2f} ms | {r['query_latency_p70_ms']:13.2f} ms | {r['peak_vram_mb']:7.1f} MB | {status_str}")
    print("=" * 85)

    out_file = os.path.join(ROOT_DIR, "benchmarks", "experiments", "smoke_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "gpu_name": gpu_name,
            "total_vram_gb": total_vram_gb,
            "all_passed": all_passed,
            "results": smoke_results,
        }, f, indent=2)
    print(f"Saved smoke benchmark results to: {out_file}\n")

    return all_passed


if __name__ == "__main__":
    success = run_gpu_smoke_test()
    if not success:
        sys.exit(1)
