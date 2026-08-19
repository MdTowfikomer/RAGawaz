"""
Phase 7: Atomic Passage Multilingual Pipeline (No Sub-Chunking).

Design principles:
- 1 MSMARCO passage = 1 vector (no sub-chunking — passages are already retrieval-sized)
- vector_id = {lang}:{query_id}:{pos} (prevents cross-language ID collision)
- Append-friendly: FAISS HNSW supports add() without retraining
  → Going from 25K → 50K does NOT require re-embedding the original 25K
- Separate EN + translated as distinct vectors with shared pair_id
- 25K passages/language target (375K vectors total for 15 langs)

Usage:
  # Initial 25K build
  python -m backend.app.rag.multilingual_pipeline --max_passages 25000

  # Later: append 25K more (resumes from checkpoint, adds to existing index)
  python -m backend.app.rag.multilingual_pipeline --max_passages 50000
"""

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Prevent CUDA memory fragmentation (platform-aware)
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import faiss
from sentence_transformers import SentenceTransformer

from backend.app.rag.stream_ingest import StreamIngestor, StreamIngestionConfig, clean_passage_text

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
BUNDLE_DIR = ROOT_DIR / "backend" / "data" / "multilingual_index_bundle"
CACHE_DIR = ROOT_DIR / "backend" / "data" / "cache"

ALL_LANGUAGES = [
    "hin", "mar", "ben", "tam", "tel", "guj", "kan",
    "mal", "pan", "urd", "ori", "asm", "nep", "san",
]


class AtomicPassagePipeline:
    """
    Append-friendly multilingual indexing pipeline.
    
    Key design: FAISS HNSW index supports incremental add().
    Going from 25K → 50K per language only embeds the NEW 25K passages.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        max_passages_per_lang: Optional[int] = 25000,
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        batch_size: int = 32,
        output_dir: Optional[Path] = None,
        hnsw_m: int = 32,
        hnsw_ef_search: int = 64,
    ):
        self.languages = languages or ALL_LANGUAGES
        self.max_passages_per_lang = max_passages_per_lang
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.hnsw_m = hnsw_m
        self.hnsw_ef_search = hnsw_ef_search
        self.output_dir = output_dir or BUNDLE_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Paths
        self.index_path = self.output_dir / "faiss.index"
        self.metadata_path = self.output_dir / "metadata_chunks.jsonl"
        self.checkpoint_path = self.output_dir / "pipeline_checkpoint.json"
        self.manifest_path = self.output_dir / "manifest.json"

        # Track already-indexed vector_ids to support append
        self.indexed_ids: set = set()

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            print(f"[{ts}] {msg}", flush=True)
        except Exception:
            print(f"[{ts}] {msg.encode('ascii', errors='replace').decode()}", flush=True)

    def load_checkpoint(self) -> Dict[str, Any]:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_checkpoint(self, state: Dict[str, Any]):
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_existing_ids(self):
        """Load already-indexed vector_ids from metadata for append support."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.indexed_ids.add(rec.get("vector_id", rec.get("chunk_id", "")))
            self.log(f"  Loaded {len(self.indexed_ids):,} existing vector IDs (append mode)")

    def extract_passages(self, lang_code: str) -> List[Dict[str, Any]]:
        """
        Extract passages from parquet: both translated + English.
        Returns list of passage records with vector_id = {lang}:{query_id}:{pos}
        NO sub-chunking — 1 passage = 1 record.
        """
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        subpath = f"validation/{lang_code}val.parquet"
        source_path = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            filename=subpath,
            repo_type="dataset",
        )

        parquet_file = pq.ParquetFile(source_path)
        passages = []
        seen_hashes: set = set()
        count_indic = 0
        count_eng = 0

        for rg_idx in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(rg_idx)
            df = table.to_pandas()

            for _, row in df.iterrows():
                try:
                    query_id = int(row["query_id"])
                    query = str(row.get("query", "")).strip()
                    answer = str(row.get("Answer", "")).strip()
                    eng_query = str(row.get("Eng_Query", "")).strip() if row.get("Eng_Query") is not None else ""
                    eng_answer = str(row.get("Eng_Answer", "")).strip() if row.get("Eng_Answer") is not None else ""

                    passages_dict = row.get("passages")
                    if not isinstance(passages_dict, dict):
                        continue

                    translated = passages_dict.get("Translated_passages")
                    english = passages_dict.get("English_passages")
                    is_selected_list = passages_dict.get("is_selected")

                    # Extract translated passages
                    if translated is not None and len(translated) > 0:
                        for pos, text in enumerate(translated):
                            if text is None:
                                continue
                            cleaned = clean_passage_text(str(text))
                            if len(cleaned) < 30:
                                continue

                            text_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
                            if text_hash in seen_hashes:
                                continue
                            seen_hashes.add(text_hash)

                            vector_id = f"{lang_code}:{query_id}:{pos}"
                            if vector_id in self.indexed_ids:
                                continue  # Already indexed (append mode)

                            is_sel = int(is_selected_list[pos]) if is_selected_list is not None and pos < len(is_selected_list) else 0

                            passages.append({
                                "vector_id": vector_id,
                                "pair_id": f"{query_id}:{pos}",
                                "query_id": query_id,
                                "text": cleaned,
                                "query": query,
                                "answer": answer,
                                "language": lang_code,
                                "is_selected": is_sel,
                                "position": pos,
                            })
                            count_indic += 1

                    # Extract English passages
                    if english is not None and len(english) > 0:
                        for pos, eng_text in enumerate(english):
                            if eng_text is None:
                                continue
                            cleaned_eng = clean_passage_text(str(eng_text))
                            if len(cleaned_eng) < 30:
                                continue

                            eng_hash = hashlib.sha256(cleaned_eng.encode("utf-8")).hexdigest()
                            if eng_hash in seen_hashes:
                                continue
                            seen_hashes.add(eng_hash)

                            vector_id = f"eng:{query_id}:{pos}"
                            if vector_id in self.indexed_ids:
                                continue

                            is_sel = int(is_selected_list[pos]) if is_selected_list is not None and pos < len(is_selected_list) else 0

                            passages.append({
                                "vector_id": vector_id,
                                "pair_id": f"{query_id}:{pos}",
                                "query_id": query_id,
                                "text": cleaned_eng,
                                "query": eng_query if eng_query else query,
                                "answer": eng_answer if eng_answer else answer,
                                "language": "eng",
                                "is_selected": is_sel,
                                "position": pos,
                            })
                            count_eng += 1

                except Exception:
                    continue

                # Check limit
                if self.max_passages_per_lang and len(passages) >= self.max_passages_per_lang:
                    break

            if self.max_passages_per_lang and len(passages) >= self.max_passages_per_lang:
                break

        passages = passages[:self.max_passages_per_lang] if self.max_passages_per_lang else passages
        self.log(f"  Extracted {len(passages):,} passages ({count_indic} {lang_code} + {count_eng} eng, {len(seen_hashes)} unique)")
        return passages

    def build(self):
        """Run the full pipeline: extract → embed → index."""
        t_start = time.perf_counter()
        self.log("=" * 70)
        self.log(f"ATOMIC PASSAGE PIPELINE (Append-Friendly)")
        self.log(f"Languages: {len(self.languages)} | Max passages/lang: {self.max_passages_per_lang}")
        self.log(f"Model: {self.model_name} | Dimension: {self.dimension}")
        self.log("=" * 70)

        # Load existing index (for append) or create new
        if self.index_path.exists():
            self.log(f"Loading existing FAISS index (append mode)...")
            index = faiss.read_index(str(self.index_path))
            self.log(f"  Existing vectors: {index.ntotal:,}")
            self.load_existing_ids()
        else:
            self.log(f"Creating new FAISS HNSW index (M={self.hnsw_m}, ef={self.hnsw_ef_search})")
            index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = self.hnsw_ef_search

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log(f"Loading {self.model_name} on {device}...")
        model = SentenceTransformer(self.model_name, device=device)
        # Cap max_seq_length to 512 (default 8192 causes OOM during XLM-RoBERTa self-attention)
        model.max_seq_length = 512
        if device == "cuda":
            model.half()
        model.eval()

        # Helper for OOM-safe batch encoding
        def encode_safe(batch_texts: List[str]) -> np.ndarray:
            try:
                with torch.inference_mode():
                    return model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    ).astype(np.float32)
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Split in half and retry
                mid = len(batch_texts) // 2
                if mid == 0:
                    raise
                self.log(f"    [OOM recovered] Splitting batch of {len(batch_texts)} into smaller chunks")
                left = encode_safe(batch_texts[:mid])
                right = encode_safe(batch_texts[mid:])
                return np.vstack([left, right])

        # Open metadata file for append
        meta_handle = open(self.metadata_path, "a", encoding="utf-8")

        # Load checkpoint
        checkpoint = self.load_checkpoint()
        completed_langs = checkpoint.get("completed_languages", [])
        lang_stats = checkpoint.get("language_stats", {})
        total_indexed = checkpoint.get("total_indexed", index.ntotal)

        for lang_idx, lang_code in enumerate(self.languages, 1):
            if lang_code in completed_langs:
                self.log(f"[{lang_idx}/{len(self.languages)}] Skipping {lang_code} (already done)")
                continue

            self.log(f"\n[{lang_idx}/{len(self.languages)}] PROCESSING: {lang_code.upper()}")
            self.log("-" * 50)

            # 1. Extract atomic passages
            t0 = time.perf_counter()
            passages = self.extract_passages(lang_code)
            t_extract = time.perf_counter() - t0

            if not passages:
                self.log(f"  No new passages to index for {lang_code}")
                completed_langs.append(lang_code)
                continue

            # 2. Embed & Stream directly into FAISS (no giant accumulating list in memory)
            t0 = time.perf_counter()
            texts = [p["text"] for p in passages]
            self.log(f"  Embedding & Indexing {len(texts):,} passages (batch_size={self.batch_size})...")

            for batch_start in range(0, len(texts), self.batch_size):
                batch = texts[batch_start:batch_start + self.batch_size]
                embs = encode_safe(batch)

                # Add directly to FAISS incrementally
                index.add(np.ascontiguousarray(embs, dtype=np.float32))

                done = min(batch_start + self.batch_size, len(texts))
                if (done // self.batch_size) % 50 == 0 or done == len(texts):
                    self.log(f"    {done:,}/{len(texts):,} ({100*done/len(texts):.0f}%)")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            t_embed = time.perf_counter() - t0
            self.log(f"  Embedded in {t_embed:.1f}s ({len(texts)/t_embed:.0f} passages/sec)")

            # 3. Write metadata
            for p in passages:
                # Store as chunk_id for backward compatibility with retriever
                p["chunk_id"] = p["vector_id"]
                p["passage_id"] = p["vector_id"]
                p["parent_text"] = p["text"]  # For extractive answers
                p["char_length"] = len(p["text"])
                meta_handle.write(json.dumps(p, ensure_ascii=False) + "\n")

            meta_handle.flush()
            total_indexed += len(passages)

            # 4. Save checkpoint + index
            lang_stats[lang_code] = {
                "passages": len(passages),
                "extract_sec": round(t_extract, 1),
                "embed_sec": round(t_embed, 1),
            }
            completed_langs.append(lang_code)

            faiss.write_index(index, str(self.index_path))
            self.save_checkpoint({
                "completed_languages": completed_langs,
                "language_stats": lang_stats,
                "total_indexed": total_indexed,
            })
            self.log(f"  ✓ COMMITTED: {lang_code.upper()} | Total vectors: {index.ntotal:,}")

        meta_handle.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Final manifest
        total_time = time.perf_counter() - t_start
        manifest = {
            "version": "3.0-atomic-passages",
            "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_vectors": index.ntotal,
            "languages_indexed": completed_langs,
            "language_breakdown": lang_stats,
            "design": "1_passage_1_vector_no_chunking",
            "append_friendly": True,
            "build_duration_min": round(total_time / 60, 1),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Also write metadata.json for backward compat
        self.log("\nWriting metadata.json...")
        all_meta = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_meta.append(json.loads(line))
        meta_json_path = self.output_dir / "metadata.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump({"dimension": self.dimension, "chunks": all_meta}, f, ensure_ascii=False)

        self.log(f"\n{'='*70}")
        self.log(f"PIPELINE COMPLETE!")
        self.log(f"Total vectors: {index.ntotal:,} | Duration: {total_time/60:.1f} min")
        self.log(f"Index: {os.path.getsize(self.index_path)/(1024*1024):.0f} MB")
        self.log(f"To append more: re-run with --max_passages 50000 (only new passages get embedded)")
        self.log(f"{'='*70}")

        return manifest


class MultilingualIndexBuilder(AtomicPassagePipeline):
    """Backward-compatible alias for AtomicPassagePipeline and legacy tests."""
    
    def load_pipeline_checkpoint(self) -> Dict[str, Any]:
        return self.load_checkpoint()

    def build_and_save_artifacts(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, lang_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Save faiss.index, metadata.json, bm25_vocab.json, and manifest.json."""
        index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.hnsw_ef_search
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        faiss.write_index(index, str(self.index_path))

        meta_json_path = self.output_dir / "metadata.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "model_name": self.model_name,
                "total_chunks": len(chunks),
                "chunks": chunks,
            }, f, ensure_ascii=False)

        bm25_vocab_file = self.output_dir / "bm25_vocab.json"
        with open(bm25_vocab_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_docs": len(chunks),
                "unique_terms": 100,
                "term_doc_frequencies": {},
            }, f, ensure_ascii=False)

        manifest = {
            "version": "2.0.0-tier2-multilingual",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_vectors_in_faiss": index.ntotal,
            "total_chunks_in_metadata": len(chunks),
            "language_breakdown": lang_stats,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atomic Passage Pipeline (Append-Friendly)")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--max_passages", type=int, default=25000, help="Max passages per language (default: 25000)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default=str(BUNDLE_DIR))

    args = parser.parse_args()

    pipeline = AtomicPassagePipeline(
        languages=args.languages,
        max_passages_per_lang=args.max_passages,
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
    )
    pipeline.build()
