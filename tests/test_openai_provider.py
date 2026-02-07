import pytest
from unittest.mock import Mock, patch
from openai import RateLimitError

from core.llm.providers.openai_compatible import OpenAICompatibleProvider


def test_retry_on_rate_limit():
    provider = OpenAICompatibleProvider(api_key="test-key")

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].message = Mock(
        role="assistant",
        content="Hello!",
        tool_calls=None
    )

    provider._client.chat.completions.create = Mock(
        side_effect=[
            RateLimitError("Rate limit exceeded", response=Mock(), body=None),
            RateLimitError("Rate limit exceeded", response=Mock(), body=None),
            mock_response,
        ]
    )

    result = provider.complete(
        model="gpt-5.2-nano",
        messages=[{"role": "user", "content": "test"}],
    )

    assert result.message.content == "Hello!"
    assert provider._client.chat.completions.create.call_count == 3


def test_gives_up_after_max_retries():
    provider = OpenAICompatibleProvider(api_key="test-key")

    provider._client.chat.completions.create = Mock(
        side_effect=RateLimitError("Rate limit exceeded", response=Mock(), body=None)
    )

    with pytest.raises(RateLimitError):
        provider.complete(
            model="gpt-5.2-nano",
            messages=[{"role": "user", "content": "test"}],
        )

    assert provider._client.chat.completions.create.call_count == 3
