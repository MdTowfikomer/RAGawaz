"""
Railway deployment entry point.
Downloads FAISS index from HuggingFace Dataset, then starts FastAPI server.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def download_index():
    """Download index files from HuggingFace Dataset if not present."""
    from huggingface_hub import hf_hub_download

    DATASET_REPO = os.getenv("INDEX_DATASET_REPO", "towfikomer/voice-rag-index")
    LOCAL_DIR = "backend/data/multilingual_index_bundle"
    os.makedirs(LOCAL_DIR, exist_ok=True)

    files = ["faiss.index", "metadata_light.pkl", "bm25_wand.pkl", "manifest.json"]

    for filename in files:
        local_path = os.path.join(LOCAL_DIR, filename)
        if os.path.exists(local_path):
            print(f"  [SKIP] {filename} exists")
            continue
        print(f"  [DOWNLOAD] {filename}...")
        try:
            hf_hub_download(
                repo_id=DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"  [OK] {filename}")
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("RAGawaz — Starting up")
    print("=" * 50)

    # Step 1: Download index
    print("\n[1/2] Downloading index...")
    download_index()

    # Step 2: Start server
    port = os.environ.get("PORT", "8080")
    print(f"\n[2/2] Starting FastAPI on port {port}...")

    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=int(port))
