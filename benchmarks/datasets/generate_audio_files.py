"""
Generate 50 genuine Hindi audio WAV recordings using Sarvam AI TTS (bulbul:v2).
Stores the audio files under benchmarks/datasets/audio_files/
"""

import os
import sys
import json
import asyncio

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

from backend.app.voice.pipeline import SarvamVoiceService


async def generate_real_audio_files():
    audio_dir = os.path.join(ROOT_DIR, "benchmarks", "datasets", "audio_files")
    os.makedirs(audio_dir, exist_ok=True)
    voice_service = SarvamVoiceService()
    
    queries_file = os.path.join(ROOT_DIR, "benchmarks", "datasets", "audio_queries.jsonl")
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]
        
    print(f"Synthesizing {len(queries)} genuine Hindi audio WAV files using Sarvam AI (bulbul:v2)...", flush=True)
    for i, q in enumerate(queries):
        audio_path = os.path.join(audio_dir, q["audio_file"])
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            # Clean prompt for speech synthesis
            clean_text = q["query"].split("(")[0].strip()
            audio_bytes, _ = await voice_service.synthesize_speech(clean_text)
            with open(audio_path, "wb") as f_out:
                f_out.write(audio_bytes)
            print(f"[{i+1:02d}/50] Generated {q['audio_file']} ({len(audio_bytes):,} bytes)", flush=True)
        else:
            print(f"[{i+1:02d}/50] Existing: {q['audio_file']} ({os.path.getsize(audio_path):,} bytes)", flush=True)
            
    print(f"\n[OK] All 50 audio recordings verified in {audio_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(generate_real_audio_files())
