from functools import lru_cache
from typing import Optional

from config import Config
from Chat import Chat
from Me import Me
from Tools import Tools
from EvaluatorAgent import EvaluatorAgent
from PushOver import PushOver
from openai import OpenAI


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
    openai_client = OpenAI(api_key=config.openai_api_key)
    me = Me(config.persona_name, config.persona_file)
    evaluator: Optional[EvaluatorAgent] = None

    if config.use_evaluator:
        evaluator = EvaluatorAgent(me, openai_client, config.openai_model)
    pushover: Optional[PushOver] = None

    if config.pushover_token and config.pushover_user:
        pushover = PushOver(config.pushover_token, config.pushover_user)
    
    tools = Tools(pushover)

    return Chat(me, openai_client, config.openai_model, tools, evaluator)
