"""
Ticket 7: Sarvam LLM Provider Implementation.

Uses Sarvam AI API for Indic-native text generation.
Default model: 'sarvam-m' or 'sarvam-2b'.
"""

import os
import json
import httpx
from typing import AsyncIterator, Optional
from backend.app.harness.providers.base import LLMProvider


class SarvamLLMProvider:
    """Sarvam AI Indic LLM provider."""
    provider_name: str = "sarvam"

    def __init__(
        self,
        model_id: str = "sarvam-2b",
        api_key: Optional[str] = None,
        base_url: str = "https://api.sarvam.ai/v1",
    ):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0),
            timeout=10.0,
        )

    async def generate(self, prompt: str, max_tokens: int = 256) -> AsyncIterator[str]:
        """Frozen spec interface: stream generated tokens."""
        async for token in self.generate_stream(prompt=prompt, max_tokens=max_tokens):
            yield token

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        timeout_ms: Optional[int] = None,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is not set.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        timeout_sec = (timeout_ms / 1000.0) if timeout_ms else 10.0
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        resp = await self._client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        yield content

    async def generate_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        timeout_ms: Optional[int] = None,
    ) -> str:
        tokens = []
        async for t in self.generate_stream(prompt, system_prompt, max_tokens, timeout_ms):
            tokens.append(t)
        return "".join(tokens).strip()
