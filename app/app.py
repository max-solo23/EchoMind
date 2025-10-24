from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


sys.path.append(str(Path(__file__).resolve().parent))
from Chat import Chat
from EvaluatorAgent import EvaluatorAgent
from Me import Me
from PushOver import PushOver
from Tools import Tools


REQUIRED_OPENAI_ENV_VARS = ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")


def ensure_openai_credentials() -> None:
    """Fail fast if no API key is available for the OpenAI client."""
    if any(os.getenv(var) for var in REQUIRED_OPENAI_ENV_VARS):
        return
    missing = ", ".join(REQUIRED_OPENAI_ENV_VARS)
    raise RuntimeError(
        "OpenAI credentials are not configured. "
        f"Set one of the following environment variables: {missing}."
    )


def build_pushover_client(token: str | None, user: str | None) -> PushOver | None:
    """Return a PushOver client when credentials are present, otherwise None."""
    if token and user:
        return PushOver(token, user)
    print(
        "Pushover credentials not found; disabling notification tools.",
        flush=True,
    )
    return None


load_dotenv(override=True)
ensure_openai_credentials()

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
openai_client = OpenAI()
model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

me = Me("Maksym", "persona.yaml")
evaluator = EvaluatorAgent(me, openai_client, model)
pushover_client = build_pushover_client(pushover_token, pushover_user)
tools = Tools(pushover_client)
chat = Chat(me, openai_client, model, tools, evaluator)

gr.ChatInterface(chat.chat, type="messages").launch()
