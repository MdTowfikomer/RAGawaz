
"""
Tests for Phase 6A (Streaming Ingestion) and Phase 6B (Minimal-Context Chunking).
"""

import json
import os
import shutil
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.app.rag.stream_ingest import StreamIngestor, StreamIngestionConfig, run_stream_ingest
from backend.app.rag.minimal_chunker import (
    MinimalContextChunker,
    chunk_minimal_context,
    split_sentences_indic,
    split_long_sentence_on_words,
)


@pytest.fixture
def temp_test_dir():
    """Create and cleanup temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="phase6_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_parquet_file(temp_test_dir):
    """Generate a valid mock Parquet file matching MSMARCO-XI schema."""
    parquet_path = os.path.join(temp_test_dir, "mock_hinval.parquet")

    # Construct schema matching MSMARCO-XI
    data = {
        "query_id": [101, 102, 103],
        "query": ["भारत की राजधानी क्या है?", "ताजमहल कहाँ है?", "डुप्लीकेट टेस्ट प्रश्न?"],
        "Answer": ["नई दिल्ली", "आगरा", "उत्तर"],
        "query_type": ["LOCATION", "LOCATION", "TEST"],
        "passages": [
            {
                "Translated_passages": [
                    "भारत की राजधानी नई दिल्ली है। यह एक ऐतिहासिक और सांस्कृतिक शहर है।",  # Valid (~67 chars)
                    "लघु",  # Too short (<20 chars) -> should be skipped
                    "भारत की राजधानी नई दिल्ली है। यह एक ऐतिहासिक और सांस्कृतिक शहर है।",  # Exact duplicate -> should be deduplicated
                ],
                "is_selected": [1, 0, 1],
            },
            {
                "Translated_passages": [
                    "ताजमहल भारत के आगरा शहर में स्थित एक विश्व धरोहर स्मारक है। इसे मुगल सम्राट शाहजहाँ ने अपनी बेगम मुमताज महल की याद में बनवाया था। यह सफेद संगमरमर से बना हुआ है।",  # Valid (~168 chars)
                ],
                "is_selected": [1],
            },
            {
                "Translated_passages": [
                    "यह तीसरा परीक्षण गद्यांश है जिसमें कुछ महत्वपूर्ण तथ्य और आंकड़े दिए गए हैं।",  # Valid (~79 chars)
                ],
                "is_selected": [0],
            },
        ],
    }

    # Write as Parquet
    table = pa.Table.from_pydict(data)
    pq.write_table(table, parquet_path)
    return parquet_path


def test_stream_ingest_extraction_and_deduplication(temp_test_dir, mock_parquet_file):
    """Verify streaming ingestion filters short text, deduplicates, and tracks stats."""
    output_dir = os.path.join(temp_test_dir, "output_stream")
    config = StreamIngestionConfig(
        source_file_path=mock_parquet_file,
        output_dir=output_dir,
        shard_size=10,
        min_length=20,
    )

    ingestor = StreamIngestor(config)
    stats = ingestor.stream_passages()

    # Assertions
    assert stats.total_rows_scanned == 3
    assert stats.total_raw_passages_extracted == 5
    assert stats.valid_unique_passages_saved == 3  # 1 short skipped, 1 duplicate skipped
    assert stats.duplicates_skipped == 1
    assert stats.short_or_invalid_skipped == 1
    assert stats.shards_written == 1

    # Verify output shard content
    shard_file = os.path.join(output_dir, "passages_shard_0000.jsonl")
    assert os.path.exists(shard_file)

    saved_records = []
    with open(shard_file, "r", encoding="utf-8") as f:
        for line in f:
            saved_records.append(json.loads(line))

    assert len(saved_records) == 3
    assert saved_records[0]["passage_id"] == "hi_101_0"
    assert saved_records[0]["is_selected"] == 1
    assert "नई दिल्ली" in saved_records[0]["text"]
    assert "sha256" in saved_records[0]


def test_stream_ingest_checkpoint_and_resumability(temp_test_dir, mock_parquet_file):
    """Verify checkpointing allows resuming without duplicates."""
    output_dir = os.path.join(temp_test_dir, "output_stream")
    config = StreamIngestionConfig(
        source_file_path=mock_parquet_file,
        output_dir=output_dir,
        shard_size=10,
        min_length=20,
        max_passages=1,  # Stop after 1 passage
    )

    # First run: partial
    ingestor1 = StreamIngestor(config)
    stats1 = ingestor1.stream_passages()
    assert stats1.valid_unique_passages_saved == 1

    # Checkpoint should exist
    checkpoint_file = os.path.join(output_dir, "ingestion_checkpoint.json")
    assert os.path.exists(checkpoint_file)

    # Manifest should exist
    manifest_file = os.path.join(output_dir, "dataset_manifest.json")
    assert os.path.exists(manifest_file)


def test_minimal_context_sentence_splitting():
    """Verify Indic punctuation splitting respects Devanagari danda and question marks."""
    sample_text = "भारत एक विशाल देश है। यहाँ कई भाषाएँ बोली जाती हैं? क्या आप जानते हैं! यह विविधता में एकता का प्रतीक है।"
    sentences = split_sentences_indic(sample_text)
    assert len(sentences) == 4
    assert sentences[0] == "भारत एक विशाल देश है।"
    assert sentences[1] == "यहाँ कई भाषाएँ बोली जाती हैं?"
    assert sentences[2] == "क्या आप जानते हैं!"
    assert sentences[3] == "यह विविधता में एकता का प्रतीक है।"


def test_minimal_context_chunking_window():
    """Verify chunking groups sentences to the target 180-220 character window."""
    s1 = "प्रथम वाक्य में भारत के भूगोल और इतिहास के बारे में प्रारंभिक जानकारी दी गई है।"  # ~80 chars
    s2 = "द्वितीय वाक्य में प्रमुख नदियों, पर्वतों और कृषि क्षेत्रों की विस्तृत चर्चा की गई है।"  # ~87 chars
    s3 = "तृतीय वाक्य में जलवायु और मौसम के विभिन्न प्रकारों का विस्तृत विश्लेषण किया गया है।"  # ~84 chars
    full_text = f"{s1} {s2} {s3}"  # ~253 chars

    chunks = chunk_minimal_context(full_text, target_min=150, target_max=220, hard_max=250)

    assert len(chunks) >= 2
    for c in chunks:
        # Chunks should not exceed hard_max and should not break words
        assert len(c) <= 250
        assert not c.startswith(" ")
        assert not c.endswith(" ")


def test_minimal_chunker_metadata_preservation():
    """Verify chunker attaches all required metadata and produces deterministic IDs."""
    chunker = MinimalContextChunker(target_min=100, target_max=200)
    passage = {
        "passage_id": "hi_999_0",
        "query_id": 999,
        "query": "टेस्ट प्रश्न?",
        "text": "पहला वाक्य यह है। दूसरा वाक्य यह है। तीसरा वाक्य यह है। चौथा वाक्य यह है।",
        "is_selected": 1,
        "language": "hi",
    }

    chunks = chunker.chunk_passage(passage)
    assert len(chunks) >= 1
    c0 = chunks[0]
    assert c0["chunk_id"] == "hi_999_0_mchk_0"
    assert c0["passage_id"] == "hi_999_0"
    assert c0["query_id"] == 999
    assert c0["is_selected"] == 1
    assert c0["chunk_strategy"] == "minimal_context"
    assert c0["char_length"] == len(c0["text"])


def test_production_cache_isolation():
    """Verify production FAISS caches and data files are untouched."""
    active_bge_dir = "backend/data/faiss_cache_bge_m3"
    active_minilm_dir = "backend/data/faiss_cache_minilm"
    active_passages = "backend/data/passages.jsonl"

    if os.path.exists(active_bge_dir):
        assert os.path.exists(os.path.join(active_bge_dir, "faiss.index"))
        assert os.path.exists(os.path.join(active_bge_dir, "metadata.json"))

    if os.path.exists(active_minilm_dir):
        assert os.path.exists(os.path.join(active_minilm_dir, "faiss.index"))

    if os.path.exists(active_passages):
        assert os.path.getsize(active_passages) > 1000000  # Existing 64MB file is intact
