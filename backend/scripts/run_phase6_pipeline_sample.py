"""
Phase 6A & 6B Controlled Execution Runner.

Executes streaming ingestion (Phase 6A) on real hinval.parquet and
minimal-context chunking (Phase 6B) on the extracted passages, collecting
exact empirical metrics and exporting reports.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.rag.stream_ingest import StreamIngestor, StreamIngestionConfig
from backend.app.rag.minimal_chunker import MinimalContextChunker


def run_controlled_test(sample_target: int = 5000):
    print("=" * 80)
    print(f"PHASE 6A & 6B CONTROLLED VALIDATION RUN (Target: {sample_target} Passages)")
    print("=" * 80)

    stream_data_dir = os.path.join(ROOT_DIR, "backend", "data", "passages_stream")
    chunks_data_dir = os.path.join(ROOT_DIR, "backend", "data", "chunks_stream")
    os.makedirs(stream_data_dir, exist_ok=True)
    os.makedirs(chunks_data_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Phase 6A Streaming Ingestion
    # -------------------------------------------------------------
    print("\n[Step 1/3] Executing Phase 6A Streaming Ingestion from hinval.parquet...")
    t0_ingest = time.perf_counter()

    ingest_config = StreamIngestionConfig(
        output_dir=stream_data_dir,
        shard_size=100000,
        min_length=20,
        max_passages=sample_target,
    )
    ingestor = StreamIngestor(ingest_config)
    ingest_stats = ingestor.stream_passages()
    t_ingest = time.perf_counter() - t0_ingest

    print(f"  [OK] Scanned rows: {ingest_stats.total_rows_scanned:,}")
    print(f"  [OK] Extracted raw passages: {ingest_stats.total_raw_passages_extracted:,}")
    print(f"  [OK] Valid unique passages saved: {ingest_stats.valid_unique_passages_saved:,}")
    print(f"  [OK] Duplicates filtered: {ingest_stats.duplicates_skipped:,}")
    print(f"  [OK] Short/invalid skipped: {ingest_stats.short_or_invalid_skipped:,}")
    print(f"  [OK] Shards created: {ingest_stats.shards_written}")
    print(f"  [OK] Ingestion time: {t_ingest:.2f}s ({ingest_stats.valid_unique_passages_saved / max(t_ingest, 0.001):.1f} passages/s)")

    # -------------------------------------------------------------
    # Step 2: Phase 6B Minimal-Context Chunking
    # -------------------------------------------------------------
    print("\n[Step 2/3] Executing Phase 6B Minimal-Context Sentence Chunking...")
    t0_chunk = time.perf_counter()

    chunker = MinimalContextChunker(target_min=180, target_max=220, hard_max=250)

    # Stream from the generated shard
    shard_file = os.path.join(stream_data_dir, "passages_shard_0000.jsonl")
    output_chunks_file = os.path.join(chunks_data_dir, "chunks_shard_0000.jsonl")

    def passage_generator():
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line.strip())

    _, chunk_stats = chunker.chunk_stream(passage_generator(), output_path=output_chunks_file)
    t_chunk = time.perf_counter() - t0_chunk

    print(f"  [OK] Passages processed: {chunk_stats.total_passages_processed:,}")
    print(f"  [OK] Chunks generated: {chunk_stats.total_chunks_generated:,}")
    print(f"  [OK] Expansion ratio: {chunk_stats.expansion_ratio:.3f}x")
    print(f"  [OK] Average chunk length: {chunk_stats.avg_chunk_chars:.1f} chars")
    print(f"  [OK] Min/Max chunk length: {chunk_stats.min_chunk_chars} / {chunk_stats.max_chunk_chars} chars")
    print(f"  [OK] Length distribution: {chunk_stats.length_distribution}")
    print(f"  [OK] Chunking throughput: {chunk_stats.total_chunks_generated / max(t_chunk, 0.001):.1f} chunks/s")

    # -------------------------------------------------------------
    # Step 3: Export Machine-Readable Manifest and Report
    # -------------------------------------------------------------
    print("\n[Step 3/3] Exporting Phase 6 Dataset Manifest and Statistics Report...")
    resources_dir = os.path.join(ROOT_DIR, "resources")
    os.makedirs(resources_dir, exist_ok=True)

    manifest_data = {
        "manifest_version": "1.0",
        "phase": "6A_6B_validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": "ai4bharat/MSMARCO-XI",
        "split_evaluated": "validation/hinval.parquet",
        "ingestion_stats": ingest_stats.to_dict(),
        "chunking_stats": chunk_stats.to_dict(),
        "projections": {
            "full_hindi_val_passages": 979410,
            "full_hindi_val_estimated_chunks": int(979410 * chunk_stats.expansion_ratio),
            "full_hindi_total_passages": 9059410,
            "full_hindi_total_estimated_chunks": int(9059410 * chunk_stats.expansion_ratio),
            "estimated_disk_mb_per_100k_chunks": round(os.path.getsize(output_chunks_file) / (chunk_stats.total_chunks_generated / 100000) / (1024 * 1024), 2) if chunk_stats.total_chunks_generated > 0 else 0,
        }
    }

    manifest_path = os.path.join(resources_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Manifest saved to: {manifest_path}")
    print("=" * 80)
    print("PHASE 6A & 6B VALIDATION RUN COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_controlled_test(sample_target=5000)
