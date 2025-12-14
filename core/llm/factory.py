from __future__ import annotations

from core.llm.provider import LLMProvider
from core.llm.providers.gemini import GeminiProvider
from core.llm.providers.openai_compatible import OpenAICompatibleProvider


def create_llm_provider(config) -> LLMProvider:
    provider = (getattr(config, "llm_provider", "openai") or "openai").lower().strip()

    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleProvider(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )

    if provider in {"gemini"}:
        return GeminiProvider(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider!r}. "
        "Supported: openai, openai_compatible (DeepSeek/Grok/Ollama via base_url), gemini."
    )
