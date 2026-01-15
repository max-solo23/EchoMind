def test_config_has_llm_provider(config):
    """Test that config loads LLM provider from environment."""
    assert config.llm_provider == "openai"


def test_config_has_llm_model(config):
    """Test that config loads LLM model from environment."""
    assert config.llm_model == "gpt-5-mini"


def test_config_has_api_key(config):
    """Test that config loads API key from environment."""
    assert config.api_key
    assert isinstance(config.api_key, str)


def test_config_has_allowed_origins(config):
    """Test that config loads allowed origins."""
    assert len(config.allowed_origins) > 0
    assert isinstance(config.allowed_origins, list)
