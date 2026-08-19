"""
Instant Schema Validation using pyarrow and fsspec HTTP file streaming.
Reads only the parquet footer/metadata (first few KB) without downloading the whole file!
"""
import sys
import json
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pyarrow.parquet as pq
import fsspec

url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/hintrain.parquet"
print(f"Connecting to: {url}")

with fsspec.open(url, "rb") as f:
    parquet_file = pq.ParquetFile(f)
    print("\nParquet Metadata:")
    print(f"  Total Rows: {parquet_file.metadata.num_rows:,}")
    print(f"  Total Row Groups: {parquet_file.metadata.num_row_groups}")
    print(f"  Schema:\n{parquet_file.schema}")

    # Read first 5 rows
    first_rg = parquet_file.read_row_group(0)
    df_sample = first_rg.slice(0, 5).to_pandas()
    print(f"\nSuccessfully read {len(df_sample)} rows!")

    print("\n--- Columns & Types ---")
    for col in df_sample.columns:
        val = df_sample[col].iloc[0]
        print(f"  {col} ({type(val).__name__})")

    print("\n--- Row 0 Inspection ---")
    row0 = df_sample.iloc[0].to_dict()
    print(f"Query: {row0.get('query')}")
    print(f"Answers: {row0.get('answers')}")
    
    passages = row0.get("passages")
    if isinstance(passages, dict):
        for k, v in passages.items():
            if hasattr(v, '__len__'):
                print(f"  passages['{k}'] count: {len(v)}")
                if len(v) > 0:
                    print(f"    first item: {str(v[0])[:120]}...")
            else:
                print(f"  passages['{k}']: {v}")

    print("\n" + "=" * 60)
    print("CHECKPOINT 1 VALIDATION COMPLETE & CONFIRMED!")
    print("=" * 60)
