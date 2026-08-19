"""
Diagnostic Script to test Sarvam Realtime WebSocket STT Handshake & Authentication Variations.
"""

import os
import sys
import json
import asyncio
import websockets
import httpx

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY", "")

MODELS = ["saaras:v3-realtime", "saaras:v4", "saarika:flash", "saarika:v2.5", "saaras:v3"]
ENDPOINTS = [
    "wss://api.sarvam.ai/streaming-speech-to-text",
    "wss://api.sarvam.ai/speech-to-text/streaming",
    "wss://api.sarvam.ai/streaming-transcription",
    "wss://api.sarvam.ai/speech-to-text-translate",
]

async def test_variation(name: str, url: str, headers: dict):
    print(f"\n--- Testing: {name} ---")
    print(f"URL: {url}")
    print(f"Headers: {list(headers.keys())}")
    try:
        async with asyncio.timeout(3.0):
            async with websockets.connect(url, additional_headers=headers) as ws:
                print(f"✅ SUCCESS! Connected to WebSocket!")
                # Send sample audio
                sample_file = os.path.join(ROOT_DIR, "benchmarks", "datasets", "audio_files", "sample_audio_01.wav")
                if os.path.exists(sample_file):
                    with open(sample_file, "rb") as f:
                        data = f.read()
                    await ws.send(data)
                    resp = await ws.recv()
                    print(f"✅ Received response: {resp}")
                return True
    except websockets.InvalidStatusCode as e:
        print(f"❌ Handshake Rejected: HTTP {e.status_code}")
        return False
    except asyncio.TimeoutError:
        print("❌ Connection Timeout (3s)")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__} - {e}")
        return False

async def main():
    print("=" * 80)
    print("SARVAM REALTIME WEBSOCKET STT DIAGNOSTICS")
    print(f"API Key present: {bool(API_KEY)} (Length: {len(API_KEY)})")
    print("=" * 80)

    # 1. Header authentication across models
    for model in MODELS:
        url = f"wss://api.sarvam.ai/streaming-speech-to-text?model={model}&language_code=hi-IN"
        headers = {"api-subscription-key": API_KEY}
        await test_variation(f"Header Auth with model={model}", url, headers)

    # 2. Query param authentication
    for model in ["saaras:v3-realtime", "saaras:v3"]:
        url = f"wss://api.sarvam.ai/streaming-speech-to-text?api-subscription-key={API_KEY}&model={model}&language_code=hi-IN"
        await test_variation(f"Query Param Auth with model={model}", url, {})

    # 3. Alternate endpoints
    for ep in ENDPOINTS[1:]:
        url = f"{ep}?model=saaras:v3-realtime&language_code=hi-IN"
        headers = {"api-subscription-key": API_KEY}
        await test_variation(f"Alternate Endpoint: {ep}", url, headers)

    # 4. Bearer Token vs api-subscription-key
    url = "wss://api.sarvam.ai/streaming-speech-to-text?model=saaras:v3-realtime&language_code=hi-IN"
    await test_variation("Authorization: Bearer <key>", url, {"Authorization": f"Bearer {API_KEY}"})
    await test_variation("x-api-key header", url, {"x-api-key": API_KEY})

if __name__ == "__main__":
    asyncio.run(main())
