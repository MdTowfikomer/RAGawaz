"""
Ticket 7: LLM Provider Interface and Base Protocols.

Defines the LLMProvider Protocol and MockLLMProvider for fast testing
and extractive fallback handling.
"""

from typing import Protocol, AsyncIterator, Optional, List, Dict, Any
import asyncio
import time


class LLMProvider(Protocol):
    """Abstract protocol for streaming & complete LLM generation."""
    provider_name: str
    model_id: str

    async def generate(self, prompt: str, max_tokens: int = 256) -> AsyncIterator[str]:
        """Frozen spec interface: stream generated tokens."""
        ...

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        timeout_ms: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream generation tokens as they arrive."""
        ...

    async def generate_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        timeout_ms: Optional[int] = None,
    ) -> str:
        """Generate full completion text within specified timeout."""
        ...


class MockLLMProvider:
    """Mock LLM provider for unit testing, offline development, and circuit breaker simulation."""
    provider_name: str = "mock"
    model_id: str = "mock-v1"

    def __init__(self, response_text: str = "यह एक परीक्षण उत्तर है।", latency_ms: float = 10.0):
        self.response_text = response_text
        self.latency_ms = latency_ms

    async def generate(self, prompt: str, max_tokens: int = 256) -> AsyncIterator[str]:
        async for token in self.generate_stream(prompt, max_tokens=max_tokens):
            yield token

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        timeout_ms: Optional[int] = None,
    ) -> AsyncIterator[str]:
        words = self.response_text.split(" ")
        delay = (self.latency_ms / 1000.0) / max(len(words), 1)
        for w in words:
            await asyncio.sleep(delay)
            yield w + " "

    async def generate_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        timeout_ms: Optional[int] = None,
    ) -> str:
        await asyncio.sleep(self.latency_ms / 1000.0)
        if timeout_ms and self.latency_ms > timeout_ms:
            raise asyncio.TimeoutError("Mock LLM generation timed out")
        return self.response_text
