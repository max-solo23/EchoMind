"""Tests for LLM provider factory."""

from unittest.mock import MagicMock, patch

import pytest

from core.llm.factory import create_llm_provider


class TestCreateLLMProvider:
    """Test LLM provider factory."""

    @patch("core.llm.factory.OpenAICompatibleProvider")
    def test_creates_openai_provider(self, mock_provider_class):
        """
        Verify OpenAI provider created for 'openai' config.

        Why: Default provider. Most common use case.
        """
        mock_config = MagicMock()
        mock_config.llm_provider = "openai"
        mock_config.llm_api_key = "test_key"
        mock_config.llm_base_url = None

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once_with(api_key="test_key", base_url=None)

    @patch("core.llm.factory.OpenAICompatibleProvider")
    def test_creates_openai_compatible_provider(self, mock_provider_class):
        """
        Verify OpenAI-compatible provider for DeepSeek/Grok/Ollama.

        Why: Allows using alternative providers with OpenAI API format.
        """
        mock_config = MagicMock()
        mock_config.llm_provider = "openai-compatible"
        mock_config.llm_api_key = "test_key"
        mock_config.llm_base_url = "https://api.deepseek.com"

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once()

    @patch("core.llm.factory.GeminiProvider")
    def test_creates_gemini_provider(self, mock_provider_class):
        """
        Verify Gemini provider created for 'gemini' config.

        Why: Alternative provider with different API.
        """
        mock_config = MagicMock()
        mock_config.llm_provider = "gemini"
        mock_config.llm_api_key = "gemini_key"
        mock_config.llm_base_url = None

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once_with(api_key="gemini_key", base_url=None)

    def test_raises_for_unsupported_provider(self):
        """
        Verify ValueError raised for unknown providers.

        Why: Clear error helps debugging config issues.
        """
        mock_config = MagicMock()
        mock_config.llm_provider = "unsupported_provider"

        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(mock_config)

        assert "Unsupported LLM_PROVIDER" in str(exc_info.value)
        assert "unsupported_provider" in str(exc_info.value)

    @patch("core.llm.factory.OpenAICompatibleProvider")
    def test_handles_none_provider_defaults_to_openai(self, mock_provider_class):
        """
        Verify None provider defaults to OpenAI.

        Why: Graceful default when config not set.
        """
        mock_config = MagicMock()
        mock_config.llm_provider = None
        mock_config.llm_api_key = "key"
        mock_config.llm_base_url = None

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once()

    @patch("core.llm.factory.OpenAICompatibleProvider")
    def test_handles_openai_underscore_compatible(self, mock_provider_class):
        """
        Verify openai_compatible (with underscore) creates OpenAI provider.

        Why: Config might use underscore variant. Both should work.
        """
        mock_config = MagicMock()
        mock_config.llm_provider = "openai_compatible"
        mock_config.llm_api_key = "key"
        mock_config.llm_base_url = "https://api.example.com"

        create_llm_provider(mock_config)

        mock_provider_class.assert_called_once()
