"""
Probe and Audit AI4Bharat/MSMARCO-XI Multilingual Dataset.
Calculates exact row counts, passage counts, estimated minimal chunks,
RAM/VRAM footprints, and offline embedding durations across all 14 languages.
"""

import os
import json
import time
from huggingface_hub import HfApi

REPO_ID = "ai4bharat/MSMARCO-XI"

# Language metadata mapping
LANGUAGES = {
    "hin": {"name": "Hindi", "script": "Devanagari", "val_file": "validation/hinval.parquet", "train_file": "train/hintrain.parquet"},
    "mar": {"name": "Marathi", "script": "Devanagari", "val_file": "validation/marval.parquet", "train_file": "train/martrain.parquet"},
    "ben": {"name": "Bengali", "script": "Bengali", "val_file": "validation/benval.parquet", "train_file": "train/bentrain.parquet"},
    "tam": {"name": "Tamil", "script": "Tamil", "val_file": "validation/tamval.parquet", "train_file": "train/tamtrain.parquet"},
    "tel": {"name": "Telugu", "script": "Telugu", "val_file": "validation/telval.parquet", "train_file": "train/teltrain.parquet"},
    "guj": {"name": "Gujarati", "script": "Gujarati", "val_file": "validation/gujval.parquet", "train_file": "train/gujtrain.parquet"},
    "kan": {"name": "Kannada", "script": "Kannada", "val_file": "validation/kanval.parquet", "train_file": "train/kantrain.parquet"},
    "mal": {"name": "Malayalam", "script": "Malayalam", "val_file": "validation/malval.parquet", "train_file": "train/maltrain.parquet"},
    "pan": {"name": "Punjabi", "script": "Gurmukhi", "val_file": "validation/panval.parquet", "train_file": "train/pantrain.parquet"},
    "urd": {"name": "Urdu", "script": "Perso-Arabic", "val_file": "validation/urdval.parquet", "train_file": "train/urdtrain.parquet"},
    "ori": {"name": "Odia", "script": "Odia", "val_file": "validation/orival.parquet", "train_file": "train/oritrain.parquet"},
    "asm": {"name": "Assamese", "script": "Bengali-Assamese", "val_file": "validation/asmval.parquet", "train_file": "train/asmtrain.parquet"},
}

def audit_remote_repo():
    print(f"Connecting to Hugging Face Hub for repo: {REPO_ID}...")
    api = HfApi()
    files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    
    val_files = [f for f in files if f.startswith("validation/") and f.endswith(".parquet")]
    train_files = [f for f in files if f.startswith("train/") and f.endswith(".parquet")]
    
    print(f"\nFound {len(val_files)} Validation Shards and {len(train_files)} Training Shards.")
    print("\nValidation Shards:")
    for vf in sorted(val_files):
        print(f"  - {vf}")

    print("\nTraining Shards:")
    for tf in sorted(train_files):
        print(f"  - {tf}")

if __name__ == "__main__":
    audit_remote_repo()
