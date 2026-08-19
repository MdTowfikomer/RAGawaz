import os
import sys
import modal

# 1. Define the Modal App
app = modal.App("ragawaz")

# 2. Define Modal Volume for persistent storage
volume = modal.Volume.from_name("ragawaz-storage", create_if_missing=True)

# 3. Construct Container Image with PyTorch, CUDA, Audio & Data Processing dependencies
rag_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "torch>=2.2.0",
        "sentence-transformers>=2.5.0",
        "faiss-cpu>=1.7.4",
        "bm25s>=0.1.10",
        "groq>=0.5.0",
        "pydantic>=2.6.0",
        "numpy>=1.24.0",
        "httpx>=0.27.0",
        "transformers>=4.38.0",
        "python-dotenv>=1.0.0",
        "python-multipart>=0.0.9",
        "pyarrow>=14.0.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0"
    )
    .add_local_dir("backend/app", remote_path="/root/backend/app")
    .add_local_dir("frontend/dist", remote_path="/root/frontend/dist")
    .add_local_dir("benchmarks", remote_path="/root/benchmarks")
)


# 4. Define ASGI FastAPI Web Application Entrypoint
@app.function(
    image=rag_image,
    gpu="A10G",  # Options: "T4", "A10G", "L4", "A100" (Autoscales to 0 GPUs when idle!)
    volumes={"/root/backend/data": volume},
    secrets=[modal.Secret.from_dotenv()], # Automatically imports GROQ_API_KEY from .env
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    sys.path.insert(0, "/root")
    os.chdir("/root")
    
    from backend.app.main import app as main_fastapi_app, init_rag_pipeline

    print("⚡ Initializing Voice RAG Pipeline inside Modal GPU container...")
    init_rag_pipeline()
    print("🚀 Voice RAG Pipeline Ready on Modal GPU!")
    
    return main_fastapi_app
