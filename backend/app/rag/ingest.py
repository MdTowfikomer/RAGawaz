"""
Ticket 2: Dataset Ingestion and Passage Extraction Engine.

Flattens, cleans, deduplicates, and extracts exactly N passages
from the cached MSMARCO-XI Hindi dataset into a structured JSONL corpus.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def get_dataset_path() -> str:
    """Get the local path to the cached hinval.parquet file."""
    return hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename="validation/hinval.parquet",
        repo_type="dataset",
    )


def extract_passages_from_parquet(
    parquet_path: str,
    target_count: int = 50000,
    min_length: int = 20,
) -> List[Dict[str, Any]]:
    """
    Extract, flatten, and deduplicate passages from the Hindi MSMARCO-XI dataset.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    seen_hashes = set()
    extracted_passages: List[Dict[str, Any]] = []

    for rg_idx in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(rg_idx)
        df = table.to_pandas()

        for _, row in df.iterrows():
            query_id = int(row["query_id"])
            query = str(row["query"]).strip() if row.get("query") is not None else ""
            answer = str(row["Answer"]).strip() if row.get("Answer") is not None else ""
            query_type = str(row.get("query_type", "UNKNOWN"))

            passages_dict = row.get("passages")
            if not isinstance(passages_dict, dict):
                continue

            translated_passages = passages_dict.get("Translated_passages")
            is_selected_list = passages_dict.get("is_selected")

            if translated_passages is None:
                continue

            for pos, text in enumerate(translated_passages):
                if text is None:
                    continue

                cleaned_text = str(text).strip()
                if len(cleaned_text) < min_length:
                    continue

                # Hash deduplication
                text_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
                if text_hash in seen_hashes:
                    continue

                seen_hashes.add(text_hash)
                is_selected = 0
                if is_selected_list is not None and pos < len(is_selected_list):
                    is_selected = int(is_selected_list[pos])

                passage_record = {
                    "passage_id": f"hi_{query_id}_{pos}",
                    "query_id": query_id,
                    "query": query,
                    "answer": answer,
                    "query_type": query_type,
                    "text": cleaned_text,
                    "is_selected": is_selected,
                    "position": pos,
                    "language": "hi",
                }

                extracted_passages.append(passage_record)

                if len(extracted_passages) >= target_count:
                    return extracted_passages

    return extracted_passages


def save_passages_to_jsonl(passages: List[Dict[str, Any]], output_path: str) -> None:
    """Save passage records to a JSONL file."""
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load_passages_from_jsonl(file_path: str, max_count: int = 50000) -> List[Dict[str, Any]]:
    """Load passage records from JSONL file."""
    passages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                passages.append(json.loads(line_str))
                if len(passages) >= max_count:
                    break
    return passages


def run_ingest(target_count: int = 50000, output_path: str = "backend/data/passages.jsonl") -> str:
    """Main execution function for dataset ingestion."""
    print(f"Loading dataset parquet path...")
    parquet_path = get_dataset_path()
    print(f"Parquet file found: {parquet_path}")

    print(f"Extracting exactly {target_count} deduplicated passages...")
    passages = extract_passages_from_parquet(parquet_path, target_count=target_count)

    print(f"Total extracted unique passages: {len(passages)}")
    save_passages_to_jsonl(passages, output_path)
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    run_ingest(50000, "backend/data/passages.jsonl")
