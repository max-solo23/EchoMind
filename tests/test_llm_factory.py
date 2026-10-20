from unittest.mock import MagicMock, patch

import pytest

from core.llm.factory import create_llm_provider


class TestCreateLLMProvider:
    @patch("core.llm.factory.OpenAIResponsesProvider")
    def test_openai_without_base_url_uses_responses_api(self, mock_provider_class):
        mock_config = MagicMock()
        mock_config.llm_provider = "openai"
        mock_config.llm_api_key = "test_key"
        mock_config.llm_base_url = None
        mock_config.llm_reasoning_effort = "low"

        create_llm_provider(mock_config)

        kwargs = mock_provider_class.call_args.kwargs
        assert kwargs["api_key"] == "test_key"
        assert kwargs["reasoning_effort"]() == "low"

    @patch("core.llm.factory.OpenAIResponsesProvider")
    def test_reasoning_effort_source_overrides_env(self, mock_provider_class):
        mock_config = MagicMock()
        mock_config.llm_provider = "openai"
        mock_config.llm_base_url = None
        mock_config.llm_reasoning_effort = "low"

        create_llm_provider(mock_config, reasoning_effort=lambda: "high")

        assert mock_provider_class.call_args.kwargs["reasoning_effort"]() == "high"

    @patch("core.llm.factory.OpenAICompatibleProvider")
    def test_openai_with_base_url_stays_on_chat_completions(self, mock_provider_class):
        mock_config = MagicMock()
        mock_config.llm_provider = "openai"
        mock_config.llm_api_key = "test_key"
        mock_config.llm_base_url = "https://api.deepseek.com"
        mock_config.llm_reasoning_effort = "low"

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once_with(
            api_key="test_key", base_url="https://api.deepseek.com", reasoning_effort="low"
        )

    @patch("core.llm.factory.OpenAICompatibleProvider")
    def test_creates_openai_compatible_provider(self, mock_provider_class):
        mock_config = MagicMock()
        mock_config.llm_provider = "openai-compatible"
        mock_config.llm_api_key = "test_key"
        mock_config.llm_base_url = "https://api.deepseek.com"

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once()

    @patch("core.llm.factory.GeminiProvider")
    def test_creates_gemini_provider(self, mock_provider_class):
        mock_config = MagicMock()
        mock_config.llm_provider = "gemini"
        mock_config.llm_api_key = "gemini_key"
        mock_config.llm_base_url = None

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once_with(api_key="gemini_key", base_url=None)

    def test_raises_for_unsupported_provider(self):
        mock_config = MagicMock()
        mock_config.llm_provider = "unsupported_provider"

        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(mock_config)

        assert "Unsupported LLM_PROVIDER" in str(exc_info.value)
        assert "unsupported_provider" in str(exc_info.value)

    @patch("core.llm.factory.OpenAIResponsesProvider")
    def test_handles_none_provider_defaults_to_openai(self, mock_provider_class):
        mock_config = MagicMock()
        mock_config.llm_provider = None
        mock_config.llm_api_key = "key"
        mock_config.llm_base_url = None

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once()

    @patch("core.llm.factory.OpenAICompatibleProvider")
    def test_handles_openai_underscore_compatible(self, mock_provider_class):
        mock_config = MagicMock()
        mock_config.llm_provider = "openai_compatible"
        mock_config.llm_api_key = "key"
        mock_config.llm_base_url = "https://api.example.com"

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once()
