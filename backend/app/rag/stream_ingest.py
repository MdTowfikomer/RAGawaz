"""
Phase 6A: Full-Dataset Streaming Ingestion Engine.

Streams, validates, cleans, deduplicates, and shards Indic passages from
MSMARCO-XI Parquet files with bounded RAM footprint and resumable checkpoints.
"""

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Generator, Iterator
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


# --- Text Cleaning Utilities ---

# Precompiled patterns for fast cleaning
_MULTI_WHITESPACE_RE = re.compile(r'[ \t]+')
_MULTI_NEWLINE_RE = re.compile(r'\n{3,}')
_ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]')
_REPEATED_SENTENCE_RE = re.compile(r'(.{30,}?[।.!?])\s*\1', re.DOTALL)
_HTML_ENTITY_RE = re.compile(r'&[a-zA-Z]+;|&#\d+;')
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def clean_passage_text(text: str) -> str:
    """
    Clean and normalize a passage text for better chunking and embedding quality.
    
    Handles:
    - Zero-width Unicode characters (translation artifacts)
    - Multiple whitespace / extra newlines
    - HTML entities
    - Control characters
    - Repeated sentences (machine translation duplicates)
    - Leading/trailing whitespace per line
    """
    if not text:
        return ""
    
    # 1. Remove zero-width Unicode chars (common in Indic MT output)
    text = _ZERO_WIDTH_RE.sub('', text)
    
    # 2. Remove control characters (except newline, tab)
    text = _CONTROL_CHAR_RE.sub('', text)
    
    # 3. Replace HTML entities with space
    text = _HTML_ENTITY_RE.sub(' ', text)
    
    # 4. Normalize whitespace: collapse multiple spaces/tabs to single space
    text = _MULTI_WHITESPACE_RE.sub(' ', text)
    
    # 5. Collapse excessive newlines (3+ → 2)
    text = _MULTI_NEWLINE_RE.sub('\n\n', text)
    
    # 6. Strip each line individually
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    # 7. Remove repeated sentences (translation artifact: same sentence appearing twice)
    text = _REPEATED_SENTENCE_RE.sub(r'\1', text)
    
    # 8. Final strip
    text = text.strip()
    
    return text


@dataclass
class IngestionStats:
    """Detailed telemetry and counting stats for dataset ingestion."""
    source_file: str
    language: str
    total_row_groups: int = 0
    row_groups_processed: int = 0
    total_rows_scanned: int = 0
    total_raw_passages_extracted: int = 0
    valid_unique_passages_saved: int = 0
    duplicates_skipped: int = 0
    short_or_invalid_skipped: int = 0
    corrupt_rows_skipped: int = 0
    shards_written: int = 0
    start_time_iso: str = ""
    end_time_iso: str = ""
    elapsed_seconds: float = 0.0
    passages_per_second: float = 0.0
    shard_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StreamIngestionConfig:
    """Configuration for streaming dataset ingestion."""
    source_file_path: Optional[str] = None
    hf_repo_id: str = "ai4bharat/MSMARCO-XI"
    hf_subpath: str = "validation/hinval.parquet"
    output_dir: str = "backend/data/passages_stream"
    shard_size: int = 100000  # Number of passages per JSONL shard
    min_length: int = 20      # Minimum characters for valid passage
    max_passages: Optional[int] = None  # None for full ingestion, or integer for controlled run
    checkpoint_interval_rg: int = 1     # Save checkpoint after every N row groups
    resume_from_checkpoint: bool = True
    language_override: Optional[str] = None


class StreamIngestor:
    """
    Streaming ingestion engine that processes Parquet files row-group by row-group
    without loading the entire dataset into memory.
    """

    def __init__(self, config: StreamIngestionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "ingestion_checkpoint.json"
        self.manifest_path = self.output_dir / "dataset_manifest.json"

        self.seen_hashes: Set[str] = set()
        self.current_shard_idx: int = 0
        self.current_shard_count: int = 0
        self.current_shard_handle = None
        self.current_shard_path: Optional[Path] = None

        self.stats = IngestionStats(
            source_file="",
            language=config.language_override or "hi",
            start_time_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def resolve_source_file(self) -> str:
        """Resolve local file path or download via huggingface_hub."""
        if self.config.source_file_path and os.path.exists(self.config.source_file_path):
            return self.config.source_file_path

        return hf_hub_download(
            repo_id=self.config.hf_repo_id,
            filename=self.config.hf_subpath,
            repo_type="dataset",
        )

    def detect_language(self, filename: str) -> str:
        """Derive language ISO code from filename if not overridden."""
        if self.config.language_override:
            return self.config.language_override
        fn = filename.lower()
        if "hin" in fn or "hi" in fn:
            return "hi"
        elif "ben" in fn or "bn" in fn:
            return "bn"
        elif "mar" in fn or "mr" in fn:
            return "mr"
        elif "tam" in fn or "ta" in fn:
            return "ta"
        elif "tel" in fn or "te" in fn:
            return "te"
        elif "kan" in fn or "kn" in fn:
            return "kn"
        elif "mal" in fn or "ml" in fn:
            return "ml"
        elif "guj" in fn or "gu" in fn:
            return "gu"
        elif "pan" in fn or "pa" in fn:
            return "pa"
        elif "urd" in fn or "ur" in fn:
            return "ur"
        return "hi"

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists and valid."""
        if self.checkpoint_path.exists() and self.config.resume_from_checkpoint:
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_checkpoint(self, last_rg_idx: int) -> None:
        """Persist resumable checkpoint state to disk."""
        data = {
            "last_row_group_idx": last_rg_idx,
            "source_file": self.stats.source_file,
            "stats": self.stats.to_dict(),
            "seen_hashes_count": len(self.seen_hashes),
            "current_shard_idx": self.current_shard_idx,
            "current_shard_count": self.current_shard_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _get_or_create_shard_handle(self):
        """Rotate to new shard file when current shard exceeds shard_size."""
        if self.current_shard_handle is None or self.current_shard_count >= self.config.shard_size:
            if self.current_shard_handle is not None:
                self.current_shard_handle.close()
                self.current_shard_idx += 1

            shard_name = f"passages_shard_{self.current_shard_idx:04d}.jsonl"
            self.current_shard_path = self.output_dir / shard_name
            self.stats.shard_paths.append(str(self.current_shard_path))
            self.current_shard_handle = open(self.current_shard_path, "a", encoding="utf-8")
            self.current_shard_count = 0
            self.stats.shards_written = self.current_shard_idx + 1

        return self.current_shard_handle

    def stream_passages(self) -> IngestionStats:
        """
        Execute full streaming extraction with bounded memory and checkpointing.
        """
        t_start = time.perf_counter()
        source_path = self.resolve_source_file()
        self.stats.source_file = source_path
        self.stats.language = self.detect_language(source_path)

        parquet_file = pq.ParquetFile(source_path)
        self.stats.total_row_groups = parquet_file.metadata.num_row_groups

        checkpoint = self.load_checkpoint()
        start_rg = checkpoint.get("last_row_group_idx", -1) + 1 if checkpoint else 0

        if start_rg > 0:
            print(f"[Phase 6A] Resuming from checkpoint: starting at row group {start_rg}/{self.stats.total_row_groups}")
            # Restore counts from checkpoint if resuming
            prev_stats = checkpoint.get("stats", {})
            self.stats.total_rows_scanned = prev_stats.get("total_rows_scanned", 0)
            self.stats.total_raw_passages_extracted = prev_stats.get("total_raw_passages_extracted", 0)
            self.stats.valid_unique_passages_saved = prev_stats.get("valid_unique_passages_saved", 0)
            self.stats.duplicates_skipped = prev_stats.get("duplicates_skipped", 0)
            self.stats.short_or_invalid_skipped = prev_stats.get("short_or_invalid_skipped", 0)
            self.current_shard_idx = checkpoint.get("current_shard_idx", 0)
            self.current_shard_count = checkpoint.get("current_shard_count", 0)

        for rg_idx in range(start_rg, self.stats.total_row_groups):
            try:
                table = parquet_file.read_row_group(rg_idx)
                df = table.to_pandas()
            except Exception as e:
                print(f"[Phase 6A] Warning: Failed reading row group {rg_idx}: {e}")
                self.stats.corrupt_rows_skipped += 1
                continue

            for _, row in df.iterrows():
                self.stats.total_rows_scanned += 1
                try:
                    query_id = int(row["query_id"])
                    query = str(row["query"]).strip() if row.get("query") is not None else ""
                    answer = str(row["Answer"]).strip() if row.get("Answer") is not None else ""
                    query_type = str(row.get("query_type", "UNKNOWN"))

                    passages_dict = row.get("passages")
                    if not isinstance(passages_dict, dict):
                        self.stats.short_or_invalid_skipped += 1
                        continue

                    translated_passages = passages_dict.get("Translated_passages")
                    is_selected_list = passages_dict.get("is_selected")

                    if translated_passages is None or not hasattr(translated_passages, "__iter__"):
                        self.stats.short_or_invalid_skipped += 1
                        continue

                    # Process translated passages (Indic language)
                    for pos, text in enumerate(translated_passages):
                        if text is None:
                            self.stats.short_or_invalid_skipped += 1
                            continue

                        self.stats.total_raw_passages_extracted += 1
                        cleaned_text = clean_passage_text(str(text))

                        if len(cleaned_text) < self.config.min_length:
                            self.stats.short_or_invalid_skipped += 1
                            continue

                        # Deterministic SHA-256 Deduplication
                        text_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
                        if text_hash in self.seen_hashes:
                            self.stats.duplicates_skipped += 1
                            continue

                        self.seen_hashes.add(text_hash)

                        is_selected = 0
                        if is_selected_list is not None and pos < len(is_selected_list):
                            is_selected = int(is_selected_list[pos])

                        passage_record = {
                            "passage_id": f"{self.stats.language}_{query_id}_{pos}",
                            "query_id": query_id,
                            "query": query,
                            "answer": answer,
                            "query_type": query_type,
                            "text": cleaned_text,
                            "is_selected": is_selected,
                            "position": pos,
                            "language": self.stats.language,
                            "char_length": len(cleaned_text),
                            "sha256": text_hash,
                        }

                        # Write to disk shard
                        handle = self._get_or_create_shard_handle()
                        handle.write(json.dumps(passage_record, ensure_ascii=False) + "\n")
                        self.current_shard_count += 1
                        self.stats.valid_unique_passages_saved += 1

                        if self.config.max_passages and self.stats.valid_unique_passages_saved >= self.config.max_passages:
                            break

                    # Process English passages from same row (safe, separate loop)
                    # Only runs if we haven't hit the passage limit yet
                    if not (self.config.max_passages and self.stats.valid_unique_passages_saved >= self.config.max_passages):
                        try:
                            english_passages = passages_dict.get("English_passages")
                            eng_query = ""
                            eng_answer = ""
                            if row.get("Eng_Query") is not None:
                                eng_query = str(row["Eng_Query"]).strip()
                            if row.get("Eng_Answer") is not None:
                                eng_answer = str(row["Eng_Answer"]).strip()

                            if english_passages is not None and hasattr(english_passages, "__iter__"):
                                for pos, eng_text in enumerate(english_passages):
                                    if eng_text is None:
                                        continue

                                    cleaned_eng = clean_passage_text(str(eng_text))
                                    if len(cleaned_eng) < self.config.min_length:
                                        continue

                                    eng_hash = hashlib.sha256(cleaned_eng.encode("utf-8")).hexdigest()
                                    if eng_hash in self.seen_hashes:
                                        continue

                                    self.seen_hashes.add(eng_hash)
                                    self.stats.total_raw_passages_extracted += 1

                                    is_selected_eng = 0
                                    if is_selected_list is not None and pos < len(is_selected_list):
                                        is_selected_eng = int(is_selected_list[pos])

                                    eng_record = {
                                        "passage_id": f"eng_{query_id}_{pos}",
                                        "query_id": query_id,
                                        "query": eng_query if eng_query else query,
                                        "answer": eng_answer if eng_answer else answer,
                                        "query_type": query_type,
                                        "text": cleaned_eng,
                                        "is_selected": is_selected_eng,
                                        "position": pos,
                                        "language": "eng",
                                        "char_length": len(cleaned_eng),
                                        "sha256": eng_hash,
                                    }

                                    handle = self._get_or_create_shard_handle()
                                    handle.write(json.dumps(eng_record, ensure_ascii=False) + "\n")
                                    self.current_shard_count += 1
                                    self.stats.valid_unique_passages_saved += 1

                                    if self.config.max_passages and self.stats.valid_unique_passages_saved >= self.config.max_passages:
                                        break
                        except Exception:
                            pass  # Skip English extraction for this row if anything fails

                except Exception as row_err:
                    self.stats.corrupt_rows_skipped += 1
                    continue

                if self.config.max_passages and self.stats.valid_unique_passages_saved >= self.config.max_passages:
                    break

            self.stats.row_groups_processed += 1

            if (rg_idx + 1) % self.config.checkpoint_interval_rg == 0:
                self.save_checkpoint(rg_idx)

            if self.config.max_passages and self.stats.valid_unique_passages_saved >= self.config.max_passages:
                print(f"[Phase 6A] Target max passages ({self.config.max_passages}) reached.")
                break

        if self.current_shard_handle is not None:
            self.current_shard_handle.close()
            self.current_shard_handle = None

        t_end = time.perf_counter()
        self.stats.elapsed_seconds = t_end - t_start
        self.stats.end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.stats.elapsed_seconds > 0:
            self.stats.passages_per_second = self.stats.valid_unique_passages_saved / self.stats.elapsed_seconds

        # Save final checkpoint & manifest
        self.save_checkpoint(self.stats.total_row_groups - 1)
        self.save_manifest()

        return self.stats

    def save_manifest(self) -> None:
        """Persist machine-readable dataset manifest."""
        manifest = {
            "manifest_version": "1.0",
            "phase": "6A_streaming_ingestion",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "source_file": self.stats.source_file,
                "language": self.stats.language,
                "output_dir": str(self.output_dir),
                "shard_size": self.config.shard_size,
                "min_length": self.config.min_length,
                "max_passages": self.config.max_passages,
            },
            "statistics": self.stats.to_dict(),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


def run_stream_ingest(
    source_file: Optional[str] = None,
    output_dir: str = "backend/data/passages_stream",
    max_passages: Optional[int] = None,
    shard_size: int = 100000,
) -> IngestionStats:
    """Entrypoint function to run Phase 6A streaming ingestion."""
    config = StreamIngestionConfig(
        source_file_path=source_file,
        output_dir=output_dir,
        max_passages=max_passages,
        shard_size=shard_size,
    )
    ingestor = StreamIngestor(config)
    return ingestor.stream_passages()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_stream_ingest(max_passages=1000)
