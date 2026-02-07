import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import SimpleNamespace

from core.llm.provider import LLMProvider


try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        RateLimitError,
    )
except ImportError:

    class APIError(Exception):  # type: ignore[no-redef]
        pass

    class APITimeoutError(APIError):  # type: ignore[no-redef]
        pass

    class RateLimitError(APIError):  # type: ignore[no-redef]
        pass

    class APIConnectionError(APIError):  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)


MIN_MESSAGE_LENGTH = 2
MIN_LETTER_RATIO = 0.3
MESSAGE_PREVIEW_LENGTH = 50
SSE_KICKSTART_BUFFER_SIZE = 2048


class InvalidMessageError(Exception):
    pass


ERROR_HANDLERS: dict[type, tuple[str, str]] = {
    RateLimitError: (
        "I'm experiencing high demand right now. Please try again in a moment.",
        "rate_limit",
    ),
    APITimeoutError: (
        "I'm taking longer than expected to respond. Please try again.",
        "api_timeout",
    ),
    APIConnectionError: (
        "I'm having trouble connecting to my AI service. Please try again shortly.",
        "connection_error",
    ),
    APIError: (
        "I encountered an API issue. Please try again.",
        "api_error",
    ),
}

DEFAULT_ERROR_MESSAGE = (
    "I encountered an unexpected issue. Please try again or rephrase your question."
)


@dataclass(frozen=True)
class SSEEvent:
    delta: str | None = None
    metadata: dict | None = None

    def encode(self) -> bytes:
        payload = {"delta": self.delta, "metadata": self.metadata}
        return f"data: {json.dumps(payload)}\n\n".encode()


def _build_messages(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> list[dict]:
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_message}]
    )


def _handle_llm_error(error: Exception, context: str = "") -> tuple[str, str]:
    context_suffix = f" ({context})" if context else ""

    for error_type, (message, code) in ERROR_HANDLERS.items():
        if isinstance(error, error_type):
            logger.error(f"LLM {code} error{context_suffix}: {error}")
            return message, code

    logger.error(f"Unexpected error in chat{context_suffix}: {type(error).__name__} - {error}")
    return DEFAULT_ERROR_MESSAGE, "unknown_error"


def _create_tool_call_object(tool_call_dict: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_call_dict["id"],
        type=tool_call_dict["type"],
        function=SimpleNamespace(
            name=tool_call_dict["function"]["name"],
            arguments=tool_call_dict["function"]["arguments"],
        ),
    )


class Chat:
    def __init__(
        self,
        persona,
        llm: LLMProvider,
        llm_model: str,
        llm_tools,
    ):
        self.llm = llm
        self.llm_model = llm_model
        self.llm_tools = llm_tools
        self.persona = persona
        self.supports_tools = llm.capabilities.get("tools", False)

    @staticmethod
    def _is_valid_message(message: str) -> bool:
        cleaned = message.strip()

        if len(cleaned) < MIN_MESSAGE_LENGTH:
            return False

        no_spaces = cleaned.replace(" ", "")
        if not no_spaces:
            return False

        letter_pattern = (
            r"[a-zA-ZàèéìòùáéíóúäëïöüāēīōūаеёиоуыэюяґєіїÀÈÉÌÒÙÁÉÍÓÚÄËÏÖÜĀĒĪŌŪАЕЁИОУЫЭЮЯҐЄІЇ]"
        )
        letters = len(re.findall(letter_pattern, no_spaces))
        total = len(no_spaces)

        return total <= 0 or (letters / total) >= MIN_LETTER_RATIO

    def _get_tools(self) -> list[dict] | None:
        return self.llm_tools.tools if self.supports_tools else None

    def _validate_message(self, message: str) -> None:
        if not self._is_valid_message(message):
            logger.warning(f"Invalid message rejected: {message[:MESSAGE_PREVIEW_LENGTH]}...")
            raise InvalidMessageError(
                "Your message appears to be incomplete or invalid. "
                "Please send a clear question or message."
            )

    def _process_tool_calls(
        self,
        tool_calls,
        messages: list[dict],
        assistant_content: str | None,
    ) -> None:
        results = self.llm_tools.handle_tool_call(tool_calls)
        messages.append(
            {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
        )
        messages.extend(results)

    async def chat(self, message: str, history: list[dict]) -> str:
        self._validate_message(message)

        messages = _build_messages(self.persona.system_prompt, history, message)

        try:
            return await self._run_completion_loop(messages)
        except Exception as error:
            user_message, _ = _handle_llm_error(error)
            return user_message

    async def _run_completion_loop(self, messages: list[dict]) -> str:
        tools = self._get_tools()

        while True:
            response = await self.llm.complete(model=self.llm_model, messages=messages, tools=tools)

            if response.finish_reason != "tool_calls":
                return response.message.content or ""

            self._process_tool_calls(
                response.message.tool_calls,
                messages,
                response.message.content,
            )

    async def chat_stream(self, message: str, history: list[dict]) -> AsyncGenerator[bytes, None]:
        messages = _build_messages(self.persona.system_prompt, history, message)

        try:
            yield (":" + (" " * SSE_KICKSTART_BUFFER_SIZE) + "\n\n").encode("utf-8")

            async for event in self._run_stream_loop(messages):
                yield event

            yield SSEEvent(metadata={"done": True}).encode()

        except Exception as error:
            user_message, error_code = _handle_llm_error(error, "streaming")
            yield SSEEvent(metadata={"error": str(error), "code": error_code}).encode()

    async def _run_stream_loop(self, messages: list[dict]) -> AsyncGenerator[bytes, None]:
        tools = self._get_tools()

        while True:
            tool_calls_accumulator: list[dict] = []
            finish_reason: str | None = None

            async for delta in self.llm.stream(
                model=self.llm_model, messages=messages, tools=tools
            ):
                if delta.content:
                    yield SSEEvent(delta=delta.content).encode()

                if delta.tool_calls:
                    self._accumulate_tool_calls(delta.tool_calls, tool_calls_accumulator)

                if delta.finish_reason is not None:
                    finish_reason = delta.finish_reason

            if finish_reason != "tool_calls":
                return

            async for event in self._execute_stream_tool_calls(tool_calls_accumulator, messages):
                yield event
                if b'"status": "failed"' in event:
                    return

    def _accumulate_tool_calls(
        self,
        tool_calls,
        accumulator: list[dict],
    ) -> None:
        for tool_call in tool_calls:
            while len(accumulator) <= tool_call.index:
                accumulator.append(
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )

            if tool_call.id:
                accumulator[tool_call.index]["id"] = tool_call.id
            if tool_call.name:
                accumulator[tool_call.index]["function"]["name"] = tool_call.name
            if tool_call.arguments:
                accumulator[tool_call.index]["function"]["arguments"] += tool_call.arguments

    async def _execute_stream_tool_calls(
        self,
        tool_calls: list[dict],
        messages: list[dict],
    ) -> AsyncGenerator[bytes, None]:
        for tc in tool_calls:
            tool_name = tc["function"]["name"]

            yield SSEEvent(metadata={"tool_call": tool_name, "status": "executing"}).encode()

            try:
                tool_call_obj = _create_tool_call_object(tc)
                results = self.llm_tools.handle_tool_call([tool_call_obj])

                yield SSEEvent(metadata={"tool_call": tool_name, "status": "success"}).encode()

                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
                )
                messages.extend(results)

            except Exception as e:
                yield SSEEvent(
                    metadata={
                        "tool_call": tool_name,
                        "status": "failed",
                        "error": str(e),
                    }
                ).encode()
                return
