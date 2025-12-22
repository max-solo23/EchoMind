from functools import lru_cache
from typing import Optional

from config import Config
from Chat import Chat
from Me import Me
from Tools import Tools
from EvaluatorAgent import EvaluatorAgent
from PushOver import PushOver
from core.llm import create_llm_provider


@lru_cache()
def get_config() -> Config:
    """
    Get singleton Config instance.
    Uses lru_cache to ensure config is loaded once and reused.
    """
    return Config.from_env()

@lru_cache()
def get_chat_service() -> Chat:
    """
    Get singleton Chat service with all dependencies wired up.

    Uses lru_cache to ensure service is created once and reused.
    Follows same dependency injection pattern as app.py.
    """
    config = get_config()
    llm_provider = create_llm_provider(config)

    # Warn about provider limitations
    if not llm_provider.capabilities.get("tools", False):
        print(
            f"Warning: {config.llm_provider} does not support tool calling. "
            "record_user_details and record_unknown_question will be disabled.",
            flush=True,
        )

    if config.use_evaluator and not llm_provider.capabilities.get("structured_output", False):
        print(
            f"Warning: {config.llm_provider} does not support structured outputs. "
            "Evaluator will use JSON fallback mode.",
            flush=True,
        )

    me = Me(config.persona_name, config.persona_file)
    evaluator: Optional[EvaluatorAgent] = None

    if config.use_evaluator:
        evaluator = EvaluatorAgent(me, llm_provider, config.llm_model)
    pushover: Optional[PushOver] = None

    if config.pushover_token and config.pushover_user:
        pushover = PushOver(config.pushover_token, config.pushover_user)
    
    tools = Tools(pushover)

    return Chat(me, llm_provider, config.llm_model, tools, evaluator)
