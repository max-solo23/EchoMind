from __future__ import annotations

from typing import TYPE_CHECKING

from core.llm.providers.gemini import GeminiProvider
from core.llm.providers.openai_compatible import OpenAICompatibleProvider
from core.llm.providers.openai_responses import OpenAIResponsesProvider


if TYPE_CHECKING:
    from collections.abc import Callable

    from core.llm.provider import LLMProvider


def create_llm_provider(
    config, reasoning_effort: Callable[[], str | None] | None = None
) -> LLMProvider:
    """reasoning_effort: read per call for OpenAI itself; defaults to LLM_REASONING_EFFORT."""
    provider = (getattr(config, "llm_provider", "openai") or "openai").lower().strip()
    static_effort = getattr(config, "llm_reasoning_effort", None)

    # OpenAI itself goes through /v1/responses (tools + any reasoning effort); compatible
    # providers behind a base_url only speak /v1/chat/completions.
    if provider == "openai" and not config.llm_base_url:
        return OpenAIResponsesProvider(
            api_key=config.llm_api_key,
            reasoning_effort=reasoning_effort or (lambda: static_effort),
        )

    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleProvider(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            reasoning_effort=static_effort,
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
