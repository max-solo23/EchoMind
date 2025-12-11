from __future__ import annotations
from config import Config
import gradio as gr
from openai import OpenAI
from Chat import Chat
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
openai_client = OpenAI()

me = Me(config.persona_name, config.persona_file)

evaluator = None
if config.use_evaluator:
    evaluator = EvaluatorAgent(me, openai_client, config.openai_model)
    print("Evaluator enabled")
else:
    print("Evaluator disabled")
    
pushover_client = build_pushover_client(config.pushover_token, config.pushover_user)
tools = Tools(pushover_client)
chat = Chat(me, openai_client, config.openai_model, tools, evaluator)

gr.ChatInterface(chat.chat, type="messages").launch()
