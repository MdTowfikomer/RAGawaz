"""
Controlled Embedding-Model Shootout Benchmark on Full 50,000-Passage Corpus (93,621 Chunks).

Evaluates 3 Candidate Models on CUDA GPU (NVIDIA GeForce RTX 3050):
1. MiniLM-384 ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
2. BGE-M3-1024 ('BAAI/bge-m3')
3. Multilingual-E5-Large-1024 ('intfloat/multilingual-e5-large' with 'passage: ' / 'query: ' prefixes)

Evaluates:
- 300 Stratified Multilingual Benchmark Queries across 6 Languages (Hindi, English, Hinglish, Marathi, Tamil, Bengali).
- Full 50,000-Passage / 93,621-Chunk Corpus with FAISS-HNSW index (M=32, efSearch=64, Inner Product).
- Recall@1, @5, @10, MRR, Latency P50/P70/P95 (Embed, FAISS, Combined), RAM, VRAM, Index Size, Build Time.
- Saves complete results to: benchmarks/experiments/shootout_results.json
"""

import gc
import json
import os
import sys
import time
import tempfile
import numpy as np
import psutil
import torch
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Set

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.rag.ingest import load_passages_from_jsonl
from backend.app.rag.chunker import chunk_corpus

CORPUS_PATH = os.path.join(ROOT_DIR, "backend", "data", "passages.jsonl")
QUERIES_PATH = os.path.join(ROOT_DIR, "benchmarks", "experiments", "multilingual_shootout_queries.jsonl")
RESULTS_PATH = os.path.join(ROOT_DIR, "benchmarks", "experiments", "shootout_results.json")
CACHE_DIR = os.path.join(ROOT_DIR, "benchmarks", "experiments", "cache")

HNSW_M = 32
HNSW_EF_SEARCH = 64
TOP_K_EVAL = 10


def log(msg: str = ""):
    print(msg, flush=True)


def get_current_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_current_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / (1024 * 1024)
    return 0.0


def load_queries(path: str) -> List[Dict[str, Any]]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ls = line.strip()
            if ls:
                queries.append(json.loads(ls))
    return queries


class BenchmarkEmbeddingModel:
    """Wrapper to handle embedding with specific model names, dimensions, and prefix conventions on CUDA."""

    def __init__(self, key: str, name: str, dimension: int, passage_prefix: str = "", query_prefix: str = "", batch_size: int = 128):
        self.key = key
        self.name = name
        self.dimension = dimension
        self.passage_prefix = passage_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.model: SentenceTransformer = None

    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"  [Model Load] Initializing '{self.name}' on {device}...")
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.model = SentenceTransformer(self.name, device=device)
        if device == "cuda":
            self.model.half()
        self.model.eval()
        load_time = time.perf_counter() - t0
        log(f"  [Model Load] Completed in {load_time:.2f}s | Device: {next(self.model.parameters()).device}")
        return load_time

    def embed_corpus(self, texts: List[str]) -> Tuple[np.ndarray, float]:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{self.key}_full_93k_embs.npy")

        if os.path.exists(cache_file):
            log(f"  [Embed Corpus] Loading cached embeddings from {cache_file}...")
            t0 = time.perf_counter()
            embs = np.load(cache_file)
            load_time = time.perf_counter() - t0
            log(f"  [Embed Corpus] Loaded shape {embs.shape} in {load_time:.2f}s")
            return embs, load_time

        log(f"  [Embed Corpus] Encoding {len(texts)} chunks with batch_size={self.batch_size} on CUDA...")
        if self.passage_prefix:
            prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        else:
            prefixed = texts

        t0 = time.perf_counter()
        with torch.inference_mode():
            embeddings = self.model.encode(
                prefixed,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        elapsed = time.perf_counter() - t0
        embs_f32 = embeddings.astype(np.float32)

        # Save to cache
        np.save(cache_file, embs_f32)
        log(f"  [Embed Corpus] Saved cache to {cache_file} ({os.path.getsize(cache_file)/(1024*1024):.1f} MB)")
        return embs_f32, elapsed

    def embed_query(self, text: str) -> np.ndarray:
        if self.query_prefix:
            prefixed = f"{self.query_prefix}{text}"
        else:
            prefixed = text

        with torch.inference_mode():
            emb = self.model.encode(
                [prefixed],
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return emb.astype(np.float32)

    def unload(self):
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def build_faiss_index(dimension: int, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> Tuple[faiss.IndexHNSWFlat, float, float]:
    """Build FAISS IndexHNSWFlat and measure on-disk index size."""
    log(f"  [FAISS] Building IndexHNSWFlat(dim={dimension}, M={HNSW_M}, efSearch={HNSW_EF_SEARCH}) for {len(chunks)} vectors...")
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = HNSW_EF_SEARCH
    
    embs_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(embs_f32)
    build_time = time.perf_counter() - t0

    # Measure disk size via tempfile
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tf:
        tmp_path = tf.name
    faiss.write_index(index, tmp_path)
    index_disk_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    log(f"  [FAISS] Added {index.ntotal} vectors in {build_time:.2f}s (Index Size on Disk: {index_disk_size_mb:.2f} MB)")
    return index, build_time, index_disk_size_mb


def evaluate_model_on_benchmark(
    embedder: BenchmarkEmbeddingModel,
    index: faiss.IndexHNSWFlat,
    chunks: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate retrieval accuracy and latency breakdowns globally and per-language."""
    log(f"  [Warmup] Running 10 warmup queries...")
    for wq in queries[:10]:
        q_emb = embedder.embed_query(wq["query"])
        _ = index.search(np.ascontiguousarray(q_emb.reshape(1, -1), dtype=np.float32), 5)

    log(f"  [Evaluation] Running 300 benchmark queries with isolated timers...")
    
    embed_latencies_ms = []
    faiss_latencies_ms = []
    total_latencies_ms = []

    per_query_results = []
    languages = ["hi", "en", "hinglish", "mr", "ta", "bn"]
    lang_stats = {lang: {"recalls": {1: 0, 5: 0, 10: 0}, "mrr_sum": 0.0, "count": 0, "ranks": []} for lang in languages}

    global_recalls = {1: 0, 5: 0, 10: 0}
    global_mrr_sum = 0.0
    rank_distribution = {"rank_1": 0, "rank_2_5": 0, "rank_6_10": 0, "not_in_top_10": 0}

    for q in queries:
        query_text = q["query"]
        gold_pid = q["ground_truth_passage_id"]
        lang = q["language"]

        # Isolated Timer: Embedding Latency
        t_emb_0 = time.perf_counter()
        q_emb = embedder.embed_query(query_text)
        t_emb_ms = (time.perf_counter() - t_emb_0) * 1000.0
        embed_latencies_ms.append(t_emb_ms)

        # Isolated Timer: FAISS Search Latency
        t_search_0 = time.perf_counter()
        scores, indices = index.search(np.ascontiguousarray(q_emb.reshape(1, -1), dtype=np.float32), TOP_K_EVAL)
        t_search_ms = (time.perf_counter() - t_search_0) * 1000.0
        faiss_latencies_ms.append(t_search_ms)

        t_total_ms = t_emb_ms + t_search_ms
        total_latencies_ms.append(t_total_ms)

        # Match gold passage ID in retrieved chunks
        retrieved_passage_ids = [chunks[idx]["passage_id"] for idx in indices[0] if 0 <= idx < len(chunks)]
        
        gold_rank = None
        for rank_idx, pid in enumerate(retrieved_passage_ids, 1):
            if pid == gold_pid:
                gold_rank = rank_idx
                break

        # Calculate MRR component
        reciprocal_rank = 1.0 / gold_rank if gold_rank is not None else 0.0
        global_mrr_sum += reciprocal_rank

        # Global recall thresholds
        for k in [1, 5, 10]:
            if gold_rank is not None and gold_rank <= k:
                global_recalls[k] += 1

        # Rank distribution
        if gold_rank == 1:
            rank_distribution["rank_1"] += 1
        elif gold_rank is not None and 2 <= gold_rank <= 5:
            rank_distribution["rank_2_5"] += 1
        elif gold_rank is not None and 6 <= gold_rank <= 10:
            rank_distribution["rank_6_10"] += 1
        else:
            rank_distribution["not_in_top_10"] += 1

        # Per-language stats
        if lang in lang_stats:
            lang_stats[lang]["count"] += 1
            lang_stats[lang]["mrr_sum"] += reciprocal_rank
            lang_stats[lang]["ranks"].append(gold_rank)
            for k in [1, 5, 10]:
                if gold_rank is not None and gold_rank <= k:
                    lang_stats[lang]["recalls"][k] += 1

        per_query_results.append({
            "query_id": q["query_id"],
            "query": query_text,
            "language": lang,
            "ground_truth_passage_id": gold_pid,
            "gold_rank": gold_rank,
            "reciprocal_rank": reciprocal_rank,
            "retrieved_top_5": retrieved_passage_ids[:5],
            "top_score": float(scores[0][0]) if len(scores[0]) > 0 else 0.0,
            "embed_latency_ms": t_emb_ms,
            "faiss_latency_ms": t_search_ms,
            "total_latency_ms": t_total_ms,
        })

    num_q = len(queries)
    global_metrics = {
        "total_queries": num_q,
        "recall_at_1": global_recalls[1] / num_q,
        "recall_at_5": global_recalls[5] / num_q,
        "recall_at_10": global_recalls[10] / num_q,
        "mrr": global_mrr_sum / num_q,
        "rank_distribution": rank_distribution,
        "latency_embed_ms": {
            "p50": float(np.percentile(embed_latencies_ms, 50)),
            "p70": float(np.percentile(embed_latencies_ms, 70)),
            "p95": float(np.percentile(embed_latencies_ms, 95)),
            "mean": float(np.mean(embed_latencies_ms)),
            "max": float(np.max(embed_latencies_ms)),
        },
        "latency_faiss_ms": {
            "p50": float(np.percentile(faiss_latencies_ms, 50)),
            "p70": float(np.percentile(faiss_latencies_ms, 70)),
            "p95": float(np.percentile(faiss_latencies_ms, 95)),
            "mean": float(np.mean(faiss_latencies_ms)),
            "max": float(np.max(faiss_latencies_ms)),
        },
        "latency_embed_retrieval_ms": {
            "p50": float(np.percentile(total_latencies_ms, 50)),
            "p70": float(np.percentile(total_latencies_ms, 70)),
            "p95": float(np.percentile(total_latencies_ms, 95)),
            "mean": float(np.mean(total_latencies_ms)),
            "max": float(np.max(total_latencies_ms)),
        },
    }

    per_language_metrics = {}
    for lang, data in lang_stats.items():
        cnt = data["count"]
        if cnt > 0:
            per_language_metrics[lang] = {
                "count": cnt,
                "recall_at_1": data["recalls"][1] / cnt,
                "recall_at_5": data["recalls"][5] / cnt,
                "recall_at_10": data["recalls"][10] / cnt,
                "mrr": data["mrr_sum"] / cnt,
            }

    return {
        "global_metrics": global_metrics,
        "per_language_metrics": per_language_metrics,
        "per_query_results": per_query_results,
    }


def run_shootout():
    log("\n" + "=" * 95)
    log("CONTROLLED EMBEDDING-MODEL SHOOTOUT (FULL 50,000 PASSAGES / 93,621 CHUNKS)")
    log("=" * 95)

    # 1. Load Corpus
    log(f"1. Loading 50,000 passages from: {CORPUS_PATH}")
    passages = load_passages_from_jsonl(CORPUS_PATH)
    log(f"   Loaded {len(passages)} passages.")

    # 2. Chunk Corpus (Fixed strategy: 250 chars, 50 overlap)
    log(f"2. Chunking corpus with fixed chunking strategy (250 char, 50 overlap)...")
    all_chunks = chunk_corpus(passages, strategy="fixed")
    chunk_texts = [c["text"] for c in all_chunks]
    log(f"   Generated {len(all_chunks)} total corpus chunks.")

    # 3. Load 300 Multilingual Benchmark Queries
    log(f"3. Loading 300 multilingual benchmark queries from: {QUERIES_PATH}")
    queries = load_queries(QUERIES_PATH)
    log(f"   Loaded {len(queries)} queries across 6 languages.")

    # Candidate Models Definitions
    candidates = [
        BenchmarkEmbeddingModel(
            key="minilm",
            name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            dimension=384,
            passage_prefix="",
            query_prefix="",
            batch_size=256,
        ),
        BenchmarkEmbeddingModel(
            key="bge_m3",
            name="BAAI/bge-m3",
            dimension=1024,
            passage_prefix="",
            query_prefix="",
            batch_size=128,
        ),
        BenchmarkEmbeddingModel(
            key="e5_large",
            name="intfloat/multilingual-e5-large",
            dimension=1024,
            passage_prefix="passage: ",
            query_prefix="query: ",
            batch_size=128,
        ),
    ]

    all_results = {}

    for idx, candidate in enumerate(candidates, 1):
        log("\n" + "#" * 95)
        log(f"  SHOOTOUT CANDIDATE {idx}/{len(candidates)}: {candidate.name} ({candidate.dimension}d)")
        log("#" * 95)

        rss_before = get_current_rss_mb()
        candidate.load()
        rss_after_load = get_current_rss_mb()

        embeddings, embed_corpus_time_sec = candidate.embed_corpus(chunk_texts)
        rss_after_embed = get_current_rss_mb()
        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        # Build FAISS Index
        index, faiss_build_time, index_disk_size_mb = build_faiss_index(candidate.dimension, all_chunks, embeddings)
        rss_after_index = get_current_rss_mb()

        # Evaluate Benchmark
        eval_data = evaluate_model_on_benchmark(candidate, index, all_chunks, queries)
        rss_peak = get_current_rss_mb()

        # Save model experiment summary
        all_results[candidate.key] = {
            "model_key": candidate.key,
            "model_name": candidate.name,
            "dimension": candidate.dimension,
            "passage_prefix": candidate.passage_prefix,
            "query_prefix": candidate.query_prefix,
            "timings": {
                "corpus_embed_sec": embed_corpus_time_sec,
                "faiss_build_sec": faiss_build_time,
            },
            "memory_footprint_mb": {
                "before_load": rss_before,
                "after_load": rss_after_load,
                "after_embed": rss_after_embed,
                "after_index": rss_after_index,
                "peak_rss": rss_peak,
                "peak_vram_mb": peak_vram_mb,
                "index_disk_size_mb": index_disk_size_mb,
            },
            "global_metrics": eval_data["global_metrics"],
            "per_language_metrics": eval_data["per_language_metrics"],
            "per_query_results": eval_data["per_query_results"],
        }

        # Cleanup memory before next candidate
        del index
        del embeddings
        candidate.unload()
        gc.collect()
        log(f"  [Cleanup] Model '{candidate.name}' unloaded and memory reclaimed.\n")

    # 4. Save structured results JSON
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"Saved full shootout results to: {RESULTS_PATH}")

    # 5. Render Comparison Tables
    print_summary_tables(all_results)


def print_summary_tables(all_results: Dict[str, Any]):
    log("\n" + "=" * 110)
    log("                     FULL CORPUS (93,621 CHUNKS) BENCHMARK COMPARISON TABLE")
    log("=" * 110)
    header = (
        f"{'Metric':<36} | "
        f"{'MiniLM-L12-v2 (384d)':<22} | "
        f"{'BGE-M3 (1024d)':<22} | "
        f"{'Multilingual-E5-Large (1024d)':<22}"
    )
    log(header)
    log("-" * 110)

    def get_val(key, path_fn):
        try:
            return path_fn(all_results[key])
        except Exception:
            return "N/A"

    rows = [
        ("Global Recall@1", lambda m: f"{m['global_metrics']['recall_at_1']*100:.2f}%"),
        ("Global Recall@5", lambda m: f"{m['global_metrics']['recall_at_5']*100:.2f}%"),
        ("Global Recall@10", lambda m: f"{m['global_metrics']['recall_at_10']*100:.2f}%"),
        ("Global MRR", lambda m: f"{m['global_metrics']['mrr']:.4f}"),
        ("Rank 1 Hits (out of 300)", lambda m: f"{m['global_metrics']['rank_distribution']['rank_1']}"),
        ("Rank 2-5 Hits", lambda m: f"{m['global_metrics']['rank_distribution']['rank_2_5']}"),
        ("Rank 6-10 Hits", lambda m: f"{m['global_metrics']['rank_distribution']['rank_6_10']}"),
        ("Not in Top-10", lambda m: f"{m['global_metrics']['rank_distribution']['not_in_top_10']}"),
        ("Query Embed Latency P50 (ms)", lambda m: f"{m['global_metrics']['latency_embed_ms']['p50']:.2f} ms"),
        ("Query Embed Latency P70 (ms)", lambda m: f"{m['global_metrics']['latency_embed_ms']['p70']:.2f} ms"),
        ("Query Embed Latency P95 (ms)", lambda m: f"{m['global_metrics']['latency_embed_ms']['p95']:.2f} ms"),
        ("FAISS Search Latency P50 (ms)", lambda m: f"{m['global_metrics']['latency_faiss_ms']['p50']:.2f} ms"),
        ("FAISS Search Latency P70 (ms)", lambda m: f"{m['global_metrics']['latency_faiss_ms']['p70']:.2f} ms"),
        ("FAISS Search Latency P95 (ms)", lambda m: f"{m['global_metrics']['latency_faiss_ms']['p95']:.2f} ms"),
        ("Embed+Retrieval Latency P50", lambda m: f"{m['global_metrics']['latency_embed_retrieval_ms']['p50']:.2f} ms"),
        ("Embed+Retrieval Latency P70", lambda m: f"{m['global_metrics']['latency_embed_retrieval_ms']['p70']:.2f} ms"),
        ("Embed+Retrieval Latency P95", lambda m: f"{m['global_metrics']['latency_embed_retrieval_ms']['p95']:.2f} ms"),
        ("Embed+Retrieval Latency MAX", lambda m: f"{m['global_metrics']['latency_embed_retrieval_ms']['max']:.2f} ms"),
        ("FAISS Index Disk Size (93k)", lambda m: f"{m['memory_footprint_mb']['index_disk_size_mb']:.1f} MB"),
        ("Peak Process RAM", lambda m: f"{m['memory_footprint_mb']['peak_rss']:.1f} MB"),
        ("Peak GPU VRAM", lambda m: f"{m['memory_footprint_mb'].get('peak_vram_mb', 0):.1f} MB"),
        ("93k Corpus Embed Duration", lambda m: f"{m['timings']['corpus_embed_sec']:.1f} s"),
        ("FAISS Index Build Duration", lambda m: f"{m['timings']['faiss_build_sec']:.1f} s"),
    ]

    for label, fn in rows:
        v_minilm = get_val("minilm", fn)
        v_bge = get_val("bge_m3", fn)
        v_e5 = get_val("e5_large", fn)
        log(f"{label:<36} | {v_minilm:<22} | {v_bge:<22} | {v_e5:<22}")

    log("=" * 110)

    # Per Language Breakdown Table
    log("\n" + "=" * 110)
    log("                                PER-LANGUAGE RECALL & MRR BREAKDOWN")
    log("=" * 110)
    lang_labels = {
        "hi": "Hindi (Monolingual)",
        "en": "English (Cross-Lingual)",
        "hinglish": "Hinglish (Cross-Lingual)",
        "mr": "Marathi (Cross-Lingual)",
        "ta": "Tamil (Cross-Lingual)",
        "bn": "Bengali (Cross-Lingual)",
    }

    sub_header = f"{'Language Strata':<28} | {'MiniLM (R@5 / MRR)':<24} | {'BGE-M3 (R@5 / MRR)':<24} | {'E5-Large (R@5 / MRR)':<24}"
    log(sub_header)
    log("-" * 110)

    for l_key, l_name in lang_labels.items():
        m_mini = all_results.get("minilm", {}).get("per_language_metrics", {}).get(l_key, {})
        m_bge = all_results.get("bge_m3", {}).get("per_language_metrics", {}).get(l_key, {})
        m_e5 = all_results.get("e5_large", {}).get("per_language_metrics", {}).get(l_key, {})

        str_mini = f"{m_mini.get('recall_at_5',0)*100:5.1f}% / {m_mini.get('mrr',0):.4f}"
        str_bge = f"{m_bge.get('recall_at_5',0)*100:5.1f}% / {m_bge.get('mrr',0):.4f}"
        str_e5 = f"{m_e5.get('recall_at_5',0)*100:5.1f}% / {m_e5.get('mrr',0):.4f}"

        log(f"{l_name:<28} | {str_mini:<24} | {str_bge:<24} | {str_e5:<24}")

    log("=" * 110)


if __name__ == "__main__":
    run_shootout()
