"""
Inspect exact dictionary structure of passages.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pyarrow.parquet as pq
from backend.app.rag.ingest import get_dataset_path

path = get_dataset_path()
pf = pq.ParquetFile(path)
table = pf.read_row_group(0)
df = table.to_pandas()

row0 = df.iloc[0]
print(f"Row 0 query_id: {row0['query_id']}")
print(f"Row 0 query: {row0['query']}")
passages = row0['passages']
print(f"passages type: {type(passages)}")
if isinstance(passages, dict):
    print(f"passages keys: {list(passages.keys())}")
    for k, v in passages.items():
        print(f"  key '{k}': type={type(v).__name__}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
        if hasattr(v, '__len__') and len(v) > 0:
            print(f"    first elem type={type(v[0]).__name__}")
            print(f"    first elem preview: {str(v[0])[:80]}")
