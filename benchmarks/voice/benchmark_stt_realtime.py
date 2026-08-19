"""
Phase 4C: Sarvam Real-Time Streaming STT Benchmark.

Evaluates:
- WebSocket Streaming vs Resilient REST Batch Fallback
- First-Partial Latency (stt_first_partial_ms)
- Final Transcript Latency (stt_final_transcript_ms)
- Total STT Latency (total_stt_ms)
- Word Error / Overlap Accuracy against ground truth

Evaluated on 20 real Hindi audio recordings.
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

from backend.app.voice.pipeline import SarvamVoiceService


async def benchmark_realtime_stt():
    print("=" * 85)
    print("PHASE 4C: SARVAM REAL-TIME STREAMING & RESILIENT BATCH STT BENCHMARK")
    print("=" * 85)

    voice_service = SarvamVoiceService()
    audio_dir = os.path.join(ROOT_DIR, "benchmarks", "datasets", "audio_files")
    queries_file = os.path.join(ROOT_DIR, "benchmarks", "datasets", "audio_queries.jsonl")

    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()][:20]

    print(f"{'#':<3} | {'Audio File':<20} | {'Mode':<18} | {'Final Lat (ms)':<15} | {'Transcript Spot-Check':<28}")
    print("-" * 88)

    results = []

    for i, q in enumerate(queries):
        wav_path = os.path.join(audio_dir, q["audio_file"])
        if not os.path.exists(wav_path):
            continue

        with open(wav_path, "rb") as f_wav:
            audio_bytes = f_wav.read()

        stt_res = await voice_service.transcribe_stream_realtime(audio_bytes, language_code="hi-IN")
        
        expected_clean = q["query"].split("(")[0].strip()
        actual_transcript = stt_res["transcript"]
        
        # Word overlap accuracy spot check
        exp_words = set(expected_clean.split())
        act_words = set(actual_transcript.split())
        overlap = len(exp_words.intersection(act_words)) / max(len(exp_words), 1)

        results.append({
            "query_id": q["query_id"],
            "audio_file": q["audio_file"],
            "mode": stt_res["mode"],
            "expected": expected_clean,
            "transcript": actual_transcript,
            "accuracy_overlap": overlap,
            "first_partial_ms": stt_res.get("first_partial_ms"),
            "final_transcript_ms": stt_res["final_transcript_ms"],
            "total_stt_ms": stt_res["total_stt_ms"],
        })

        print(f"{i+1:<3} | {q['audio_file']:<20} | {stt_res['mode']:<18} | {stt_res['total_stt_ms']:9.2f} ms     | {actual_transcript[:25]:<28}")

    latencies = [r["total_stt_ms"] for r in results]
    accuracies = [r["accuracy_overlap"] for r in results]

    p50 = float(np.percentile(latencies, 50))
    p70 = float(np.percentile(latencies, 70))
    p95 = float(np.percentile(latencies, 95))
    max_lat = float(np.max(latencies))
    avg_acc = float(np.mean(accuracies)) * 100.0

    print("-" * 88)
    print("PHASE 4C STT BENCHMARK REPORT:")
    print(f"  Primary Mode:               {results[0]['mode']}")
    print(f"  STT Latency P50:            {p50:.2f} ms")
    print(f"  STT Latency P70:            {p70:.2f} ms")
    print(f"  STT Latency P95:            {p95:.2f} ms")
    print(f"  STT Latency MAX:            {max_lat:.2f} ms")
    print(f"  Word Accuracy Spot-Check:   {avg_acc:.2f}%")
    print("=" * 88)

    out_file = os.path.join(ROOT_DIR, "benchmarks", "voice", "stt_realtime_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f_out:
        json.dump({
            "timestamp": time.time(),
            "sample_count": len(results),
            "metrics": {
                "p50_ms": p50,
                "p70_ms": p70,
                "p95_ms": p95,
                "max_ms": max_lat,
                "word_accuracy_pct": avg_acc,
            },
            "records": results,
        }, f_out, ensure_ascii=False, indent=2)

    print(f"Saved STT real-time benchmark artifact to: {out_file}\n")


if __name__ == "__main__":
    asyncio.run(benchmark_realtime_stt())
