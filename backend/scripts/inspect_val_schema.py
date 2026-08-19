"""
Download validation/hinval.parquet directly using huggingface_hub.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

print("Downloading hinval.parquet...")
path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="validation/hinval.parquet",
    repo_type="dataset"
)
print(f"Downloaded to: {path}")

parquet_file = pq.ParquetFile(path)
print(f"\nTotal validation rows: {parquet_file.metadata.num_rows:,}")
print(f"Total row groups: {parquet_file.metadata.num_row_groups}")
print(f"\nSchema:\n{parquet_file.schema}")

table = parquet_file.read_row_group(0)
df = table.slice(0, 5).to_pandas()
print("\n--- Columns ---")
for c in df.columns:
    print(f"  {c}: {type(df[c].iloc[0]).__name__}")

print("\n--- Row 0 Content ---")
r0 = df.iloc[0].to_dict()
for k, v in r0.items():
    if isinstance(v, dict):
        print(f"{k}:")
        for subk, subv in v.items():
            if hasattr(subv, '__len__'):
                print(f"  {subk} (len {len(subv)}): {str(subv[0])[:120] if len(subv)>0 else 'empty'}")
    else:
        print(f"{k}: {str(v)[:120]}")
