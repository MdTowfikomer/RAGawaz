"""
Ticket 7: Groq LLM Provider Implementation.

Uses OpenAI-compatible AsyncOpenAI or httpx client to communicate with Groq's high-speed LPU inference API.
Default model: 'llama-3.3-70b-versatile' or 'llama-3.1-8b-instant'.
"""

import os
from typing import AsyncIterator, Optional, List, Dict, Any
import json
import httpx
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from backend.app.harness.providers.base import LLMProvider




class GroqLLMProvider:
    """High-speed Groq LPU inference provider (< 150ms TTFT)."""
    provider_name: str = "groq"

    def __init__(
        self,
        model_id: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
    ):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

        self.base_url = base_url
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=30, keepalive_expiry=120.0),
            timeout=10.0,
        )

    async def generate(self, prompt: str, max_tokens: int = 64) -> AsyncIterator[str]:
        """Frozen spec interface: stream generated tokens."""
        async for token in self.generate_stream(prompt=prompt, max_tokens=max_tokens):
            yield token

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 64,
        timeout_ms: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens from Groq API."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        timeout_sec = (timeout_ms / 1000.0) if timeout_ms else 8.0
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": min(max_tokens, 80),
            "temperature": 0.1,
            "stream": True,
        }


        async with self._client.stream("POST", f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=timeout_sec) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                    try:
                        data = json.loads(line[6:])
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                    except Exception:
                        continue

    async def generate_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        timeout_ms: Optional[int] = None,
    ) -> str:
        """Fetch complete generation from Groq API."""
        tokens = []
        async for t in self.generate_stream(prompt, system_prompt, max_tokens, timeout_ms):
            tokens.append(t)
        return "".join(tokens).strip()
