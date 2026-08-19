"""
T1: Dataset Schema Validation Script for ai4bharat/MSMARCO-XI (Hindi 'hintrain.parquet')
"""
import sys
import json
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from datasets import load_dataset

print("=" * 60)
print("T1: MSMARCO-XI Hindi (hintrain.parquet) Schema Validation")
print("=" * 60)

# Step 1: Download a streaming sample directly from hintrain.parquet
print("\n[1/4] Streaming sample from hintrain.parquet (5 rows)...")
ds = load_dataset(
    "parquet",
    data_files={"train": "hf://datasets/ai4bharat/MSMARCO-XI/train/hintrain.parquet"},
    split="train",
    streaming=True,
)
sample = list(ds.take(5))
print(f"  ✅ Successfully loaded {len(sample)} rows")

# Step 2: Print top-level field names
print("\n[2/4] Top-level field names:")
row = sample[0]
for key, val in row.items():
    val_type = type(val).__name__
    if isinstance(val, dict):
        print(f"  {key} (dict): keys={list(val.keys())}")
    elif isinstance(val, list):
        print(f"  {key} (list): len={len(val)}, item_type={type(val[0]).__name__ if val else 'empty'}")
    else:
        preview = str(val)[:80] + "..." if len(str(val)) > 80 else str(val)
        print(f"  {key} ({val_type}): {preview}")

# Step 3: Inspect passages structure
print("\n[3/4] Passages structure details:")
passages = row.get("passages", {})
if isinstance(passages, dict):
    for pk, pv in passages.items():
        if isinstance(pv, list):
            print(f"  passages['{pk}'] (list of len {len(pv)}):")
            if pv:
                print(f"    Item 0 preview: {str(pv[0])[:120]}...")
else:
    print(f"  passages type: {type(passages)}")

# Step 4: Sample query & answers & translated passages
print("\n[4/4] Sample Query and Translated Passage:")
print(f"  Query: {row.get('query')}")
answers = row.get("answers", [])
print(f"  Answers: {answers}")
if isinstance(passages, dict) and "Translated_passages" in passages:
    print(f"  Total passages in row 0: {len(passages['Translated_passages'])}")
    for idx, p in enumerate(passages['Translated_passages'][:2]):
        is_sel = passages.get('is_selected', [])[idx] if idx < len(passages.get('is_selected', [])) else None
        print(f"  Passage [{idx}] (is_selected={is_sel}): {p[:120]}...")

# Step 5: Passage count analysis
print("\n[SUMMARY] Passage counts across 5 sample rows:")
total_passages = 0
for i, r in enumerate(sample):
    p = r.get("passages", {})
    if isinstance(p, dict) and "Translated_passages" in p:
        count = len(p["Translated_passages"])
    elif isinstance(p, list):
        count = len(p)
    else:
        count = 0
    total_passages += count
    print(f"  Row {i}: {count} passages")

print(f"\n  Average passages per row: {total_passages / len(sample):.1f}")
print("=" * 60)
print("CHECKPOINT 1 VALIDATION COMPLETE!")
print("=" * 60)
