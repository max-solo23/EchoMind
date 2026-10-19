from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from api import dependencies
from core.chat import Chat
from core.persona import Persona
from services.cache_service import CacheService
from services.conversation_logger import ConversationLogger
from services.push_over import PushOver


def _config(**overrides):
    base = {
        "llm_provider": "openai",
        "llm_api_key": "k",
        "llm_base_url": None,
        "llm_model": "m",
        "pushover_token": None,
        "pushover_user": None,
        "persona_name": "T",
        "persona_file": "persona.yaml",
        "database_url": None,
    }
    return SimpleNamespace(**{**base, **overrides})


class FakeProvider:
    def __init__(self, tools: bool):
        self._tools = tools

    @property
    def capabilities(self):
        return {"tools": self._tools}


class TestIsDatabaseConfigured:
    def test_false_without_url(self):
        with patch.object(dependencies, "get_config", return_value=_config()):
            assert dependencies.is_database_configured() is False

    def test_true_with_url(self):
        config = _config(database_url="postgresql://x")

        with patch.object(dependencies, "get_config", return_value=config):
            assert dependencies.is_database_configured() is True


class TestGetDbSession:
    async def test_raises_without_database(self):
        with (
            patch.object(dependencies, "get_config", return_value=_config()),
            pytest.raises(RuntimeError, match="Database not configured"),
        ):
            await anext(dependencies.get_db_session())

    async def test_yields_session_when_configured(self):
        marker = object()

        @asynccontextmanager
        async def fake_session(_config):
            yield marker

        with (
            patch.object(dependencies, "get_config", return_value=_config(database_url="pg://x")),
            patch.object(dependencies, "get_session", fake_session),
        ):
            sessions = [s async for s in dependencies.get_db_session()]

        assert sessions == [marker]


class TestGetConversationLogger:
    async def test_wires_cache_service_with_caching_enabled(self):
        persona = SimpleNamespace(content_hash=lambda: "hash1234")

        with patch.object(dependencies, "get_persona", return_value=persona):
            logger = await dependencies.get_conversation_logger(object())

        assert isinstance(logger, ConversationLogger)
        assert logger.enable_caching is True
        assert isinstance(logger.cache_service, CacheService)


class TestGetChatService:
    def _build(self, config, provider, persona_file):
        persona = Persona("Test", persona_file)
        with (
            patch.object(dependencies, "get_config", return_value=config),
            patch.object(dependencies, "create_llm_provider", return_value=provider),
            patch.object(dependencies, "get_persona", return_value=persona),
        ):
            return dependencies.get_chat_service.__wrapped__()

    def test_warns_and_disables_tools_when_provider_lacks_them(self, capsys, temp_persona_file):
        config = _config(llm_provider="gemini")

        chat = self._build(config, FakeProvider(tools=False), temp_persona_file)

        assert isinstance(chat, Chat)
        assert chat.supports_tools is False
        assert chat.llm_tools.message_app is None
        assert "does not support tool calling" in capsys.readouterr().out

    def test_builds_pushover_when_configured(self, capsys, temp_persona_file):
        config = _config(pushover_token="tok", pushover_user="usr")

        chat = self._build(config, FakeProvider(tools=True), temp_persona_file)

        assert chat.supports_tools is True
        assert isinstance(chat.llm_tools.message_app, PushOver)
        assert capsys.readouterr().out == ""
