from __future__ import annotations

import gradio as gr

from Chat import Chat
from config import Config
from core.llm import create_llm_provider
from EvaluatorAgent import EvaluatorAgent
from Me import Me
from PushOver import PushOver
from Tools import Tools


def build_pushover_client(token: str | None, user: str | None) -> PushOver | None:
    """Return a PushOver client when credentials are present, otherwise None."""
    if token and user:
        return PushOver(token, user)
    print(
        "Pushover credentials not found; disabling notification tools.",
        flush=True,
    )
    return None


config = Config.from_env()
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

evaluator = None
if config.use_evaluator:
    evaluator = EvaluatorAgent(me, llm_provider, config.llm_model)
    print("Evaluator enabled")
else:
    print("Evaluator disabled")

pushover_client = build_pushover_client(config.pushover_token, config.pushover_user)
tools = Tools(pushover_client)
chat = Chat(me, llm_provider, config.llm_model, tools, evaluator)

gr.ChatInterface(chat.chat, type="messages").launch()  # type: ignore[call-arg]
