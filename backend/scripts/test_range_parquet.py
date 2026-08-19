"""
Read Parquet metadata via HTTP Range request without downloading the full file.
"""
import sys
import io
import struct
import requests
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
print(f"Fetching footer from: {url}")

# 1. Fetch the last 64KB of the file
headers = {"Range": "bytes=-65536"}
resp = requests.get(url, headers=headers, allow_redirects=True)
print(f"Response status: {resp.status_code}, content length: {len(resp.content)} bytes")

if resp.status_code in [200, 206]:
    # Check PAR1 magic bytes at end
    data = resp.content
    if data[-4:] == b"PAR1":
        print("Valid Parquet footer found!")
        footer_len = struct.unpack("<I", data[-8:-4])[0]
        print(f"Footer length: {footer_len} bytes")
        
        # Parse metadata
        meta_bytes = data[-8-footer_len : -8]
        # Use pyarrow to read schema
        bio = io.BytesIO(data)
        # We can read with pyarrow
        try:
            pf = pq.ParquetFile(bio)
            print("\nParquet Schema:")
            print(pf.schema)
            print(f"\nNum rows: {pf.metadata.num_rows}")
            print(f"Num row groups: {pf.metadata.num_row_groups}")
        except Exception as e:
            print(f"Pyarrow parse on buffer: {e}")
