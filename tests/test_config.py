import pytest


def test_config_has_llm_provider(config):
    assert config.llm_provider == "openai"


def test_config_has_llm_model(config):
    assert config.llm_model == "gpt-5-mini"


def test_config_has_api_key(config):
    assert config.api_key
    assert isinstance(config.api_key, str)


def test_config_has_allowed_origins(config):
    assert len(config.allowed_origins) > 0
    assert isinstance(config.allowed_origins, list)


def test_config_raises_error_without_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("config.load_dotenv", lambda **kwargs: None)

    from config import Config

    with pytest.raises(RuntimeError) as exc_info:
        Config.from_env()
    assert "LLM credentials not configured" in str(exc_info.value)
