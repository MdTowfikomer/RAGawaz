"""
List all repository files in ai4bharat/MSMARCO-XI using huggingface_hub.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files(repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset")
print(f"Total repo files: {len(files)}")
print("\nFirst 30 files in repo:")
for f in files[:30]:
    print(f"  {f}")

print("\nFiles with 'hin' or 'hi' in filename:")
for f in files:
    if "hin" in f.lower() or "hi" in f.lower():
        print(f"  {f}")
