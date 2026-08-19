"""
LLM Provider Factory.
"""

from typing import Optional
from backend.app.harness.providers.base import LLMProvider, MockLLMProvider
from backend.app.harness.providers.groq import GroqLLMProvider
from backend.app.harness.providers.cerebras import CerebrasLLMProvider
from backend.app.harness.providers.sarvam_llm import SarvamLLMProvider


def get_llm_provider(name: str = "mock", model_id: Optional[str] = None, **kwargs) -> LLMProvider:
    """Factory to get configured LLM provider."""
    name_clean = name.lower().strip()
    if name_clean == "groq":
        return GroqLLMProvider(model_id=model_id or "openai/gpt-oss-20b", **kwargs)
    elif name_clean == "cerebras":
        return CerebrasLLMProvider(model_id=model_id or "llama3.1-8b", **kwargs)
    elif name_clean == "sarvam":
        return SarvamLLMProvider(model_id=model_id or "sarvam-2b", **kwargs)
    elif name_clean == "mock":
        return MockLLMProvider(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {name}. Options: 'groq', 'cerebras', 'sarvam', 'mock'")
