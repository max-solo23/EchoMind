import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai import RateLimitError

from core.llm.providers.openai_compatible import OpenAICompatibleProvider


async def test_retry_on_rate_limit():
    provider = OpenAICompatibleProvider(api_key="test-key")

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].message = AsyncMock(
        role="assistant",
        content="Hello!",
        tool_calls=None
    )

    provider._async_client.chat.completions.create = AsyncMock(
        side_effect=[
            RateLimitError("Rate limit exceeded", response=AsyncMock(), body=None),
            RateLimitError("Rate limit exceeded", response=AsyncMock(), body=None),
            mock_response,
        ]
    )

    result = await provider.complete(
        model="gpt-5.2-nano",
        messages=[{"role": "user", "content": "test"}],
    )

    assert result.message.content == "Hello!"
    assert provider._async_client.chat.completions.create.call_count == 3


async def test_gives_up_after_max_retries():
    provider = OpenAICompatibleProvider(api_key="test-key")

    provider._async_client.chat.completions.create = AsyncMock(
        side_effect=RateLimitError("Rate limit exceeded", response=AsyncMock(), body=None)
    )

    with pytest.raises(RateLimitError):
        await provider.complete(
            model="gpt-5.2-nano",
            messages=[{"role": "user", "content": "test"}],
        )

    assert provider._async_client.chat.completions.create.call_count == 3
