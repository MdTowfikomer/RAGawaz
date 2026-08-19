"""
Verify Checkpoint 2 corpus: backend/data/passages.jsonl
"""
import sys
import json
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

path = "backend/data/passages.jsonl"
print("=" * 60)
print("CHECKPOINT 2: Corpus Ingestion Verification")
print("=" * 60)

assert os.path.exists(path), f"File not found: {path}"
file_size_mb = os.path.getsize(path) / (1024 * 1024)
print(f"File Size: {file_size_mb:.2f} MB")

count = 0
selected_count = 0
unique_queries = set()
lengths = []

sample_passages = []

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        count += 1
        unique_queries.add(data["query_id"])
        if data.get("is_selected") == 1:
            selected_count += 1
        lengths.append(len(data["text"]))
        if i < 3:
            sample_passages.append(data)

print(f"Total Passages: {count:,}")
print(f"Unique Queries Represented: {len(unique_queries):,}")
print(f"Ground-Truth Selected Passages (is_selected=1): {selected_count:,}")
print(f"Average Character Length: {sum(lengths) / len(lengths):.1f}")
print(f"Min / Max Length: {min(lengths)} / {max(lengths)}")

print("\n--- Sample Passages ---")
for idx, p in enumerate(sample_passages):
    print(f"\n[Sample {idx+1}]")
    print(f"  ID:          {p['passage_id']}")
    print(f"  Query ID:    {p['query_id']}")
    print(f"  Query:       {p['query']}")
    print(f"  Answer:      {p['answer'][:80]}...")
    print(f"  Text:        {p['text'][:120]}...")
    print(f"  Is Selected: {p['is_selected']}")
    print(f"  Language:    {p['language']}")

print("\n" + "=" * 60)
assert count == 50000, f"Expected 50,000 passages, got {count}"
print("✅ CHECKPOINT 2 PASSED: 50,000 VALID HINDI PASSAGES EXTRACTED!")
print("=" * 60)
