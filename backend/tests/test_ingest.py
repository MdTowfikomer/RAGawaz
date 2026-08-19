"""
Unit tests for Ticket 2: Ingestion Pipeline
"""
import os
import json
import pytest
from pathlib import Path
from backend.app.rag.ingest import (
    get_dataset_path,
    extract_passages_from_parquet,
    save_passages_to_jsonl,
)


@pytest.fixture(scope="module")
def parquet_file_path():
    """Ensure local parquet file path is accessible."""
    path = get_dataset_path()
    assert os.path.exists(path), f"Parquet file does not exist at {path}"
    return path


def test_flatten_produces_valid_sample_passages(parquet_file_path):
    """Test extracting a small sample of 100 passages."""
    passages = extract_passages_from_parquet(parquet_file_path, target_count=100)
    assert len(passages) == 100, f"Expected 100 passages, got {len(passages)}"
    
    # Verify metadata fields
    first = passages[0]
    required_keys = {"passage_id", "query_id", "query", "answer", "text", "is_selected", "position", "language"}
    assert required_keys.issubset(first.keys()), f"Missing keys in passage: {required_keys - set(first.keys())}"
    assert first["language"] == "hi"
    assert len(first["text"]) >= 20, "Passage text too short"


def test_deduplication_removes_identical_text(parquet_file_path):
    """Verify that all extracted passages have unique text."""
    passages = extract_passages_from_parquet(parquet_file_path, target_count=200)
    texts = [p["text"] for p in passages]
    assert len(texts) == len(set(texts)), "Found duplicate passages in extracted output"


def test_save_and_load_jsonl(tmp_path, parquet_file_path):
    """Test saving passages to JSONL and reading them back."""
    passages = extract_passages_from_parquet(parquet_file_path, target_count=50)
    temp_file = str(tmp_path / "test_passages.jsonl")
    
    save_passages_to_jsonl(passages, temp_file)
    assert os.path.exists(temp_file)
    
    loaded = []
    with open(temp_file, "r", encoding="utf-8") as f:
        for line in f:
            loaded.append(json.loads(line))
            
    assert len(loaded) == 50
    assert loaded[0]["passage_id"] == passages[0]["passage_id"]
    assert loaded[0]["text"] == passages[0]["text"]
