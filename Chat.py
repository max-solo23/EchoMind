import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

from core.llm.provider import LLMProvider


if TYPE_CHECKING:
    from EvaluatorAgent import EvaluatorAgent


try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        RateLimitError,
    )
except ImportError:

    class APIError(Exception):
        """Base exception for API-related errors."""

        pass

    class APITimeoutError(APIError):
        """Exception raised when API request times out."""

        pass

    class RateLimitError(APIError):
        """Exception raised when rate limit is exceeded."""

        pass

    class APIConnectionError(APIError):
        """Exception raised when there's a connection error to the API."""

        pass


logger = logging.getLogger(__name__)


class InvalidMessageError(Exception):
    """Raised when a message fails validation."""

    pass


# Error code mappings for consistent error handling
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
    """Represents a Server-Sent Event."""

    delta: str | None = None
    metadata: dict | None = None

    def encode(self) -> bytes:
        """Encode the event as SSE-formatted bytes."""
        payload = {"delta": self.delta, "metadata": self.metadata}
        return f"data: {json.dumps(payload)}\n\n".encode()


def _build_messages(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> list[dict]:
    """Build the message list for LLM completion."""
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_message}]
    )


def _handle_llm_error(error: Exception, context: str = "") -> tuple[str, str]:
    """
    Get appropriate error message and code for an LLM error.

    Returns:
        Tuple of (user_message, error_code)
    """
    context_suffix = f" ({context})" if context else ""

    for error_type, (message, code) in ERROR_HANDLERS.items():
        if isinstance(error, error_type):
            logger.error(f"LLM {code} error{context_suffix}: {error}")
            return message, code

    logger.error(f"Unexpected error in chat{context_suffix}: {type(error).__name__} - {error}")
    return DEFAULT_ERROR_MESSAGE, "unknown_error"


def _create_tool_call_object(tool_call_dict: dict) -> SimpleNamespace:
    """Convert a tool call dictionary to the object format expected by handle_tool_call."""
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
        person,
        llm: LLMProvider,
        llm_model: str,
        llm_tools,
        evaluator_llm: "EvaluatorAgent | None" = None,
    ):
        self.llm = llm
        self.llm_model = llm_model
        self.llm_tools = llm_tools
        self.person = person
        self.evaluator_llm = evaluator_llm
        self.supports_tools = llm.capabilities.get("tools", False)

    @staticmethod
    def _is_valid_message(message: str) -> bool:
        """
        Validate message to filter gibberish and nonsense input.

        Returns False if:
        - Message is too short (< 3 characters)
        - Message has < 30% alphabetic characters (keyboard mashing)
        - Message is only special characters/numbers

        Args:
            message: User message to validate

        Returns:
            True if message appears valid, False otherwise
        """
        cleaned = message.strip()

        if len(cleaned) < 3:
            return False

        no_spaces = cleaned.replace(" ", "")
        if not no_spaces:
            return False

        # Count alphabetic characters (supports multiple languages)
        # Pattern includes Latin, Cyrillic, and common accented characters
        letter_pattern = (
            r"[a-zA-ZàèéìòùáéíóúäëïöüāēīōūаеёиоуыэюяґєіїÀÈÉÌÒÙÁÉÍÓÚÄËÏÖÜĀĒĪŌŪАЕЁИОУЫЭЮЯҐЄІЇ]"
        )
        letters = len(re.findall(letter_pattern, no_spaces))
        total = len(no_spaces)

        return total <= 0 or (letters / total) >= 0.3

    def _get_tools(self) -> list[dict] | None:
        """Get tools list if provider supports them."""
        return self.llm_tools.tools if self.supports_tools else None

    def _validate_message(self, message: str) -> None:
        """Validate message and raise InvalidMessageError if invalid."""
        if not self._is_valid_message(message):
            logger.warning(f"Invalid message rejected: {message[:50]}...")
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
        """Process tool calls and append results to messages."""
        results = self.llm_tools.handle_tool_call(tool_calls)
        messages.append(
            {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
        )
        messages.extend(results)

    def _evaluate_and_rerun(
        self,
        reply: str,
        message: str,
        history: list[dict],
    ) -> str:
        """Evaluate reply and rerun if needed. Returns final reply."""
        if not self.evaluator_llm:
            return reply

        evaluation = self.evaluator_llm.evaluate(reply, message, history)

        if evaluation.is_acceptable:
            logger.debug("Passed evaluation - returning reply")
            return reply

        logger.debug("Failed evaluation - rerunning")
        logger.debug(f"Feedback: {evaluation.feedback}")
        return self.rerun(
            reply, message, history, evaluation.feedback, self.person.system_prompt
        )

    def chat(self, message: str, history: list[dict]) -> str:
        """
        Process a chat message with validation and error handling.

        Args:
            message: User message
            history: Conversation history

        Returns:
            Bot response string

        Raises:
            InvalidMessageError: If message fails validation
        """
        self._validate_message(message)

        messages = _build_messages(self.person.system_prompt, history, message)

        try:
            reply = self._run_completion_loop(messages)
            return self._evaluate_and_rerun(reply, message, history)
        except Exception as error:
            user_message, _ = _handle_llm_error(error)
            return user_message

    def _run_completion_loop(self, messages: list[dict]) -> str:
        """Run the completion loop, handling tool calls until done."""
        tools = self._get_tools()

        while True:
            response = self.llm.complete(model=self.llm_model, messages=messages, tools=tools)

            if response.finish_reason != "tool_calls":
                return response.message.content or ""

            self._process_tool_calls(
                response.message.tool_calls,
                messages,
                response.message.content,
            )

    def rerun(
        self,
        reply: str,
        message: str,
        history: list[dict],
        feedback: str,
        system_prompt: str,
    ) -> str:
        """Rerun completion with feedback from failed evaluation."""
        updated_system_prompt = (
            f"{system_prompt}\n\n"
            "## Previous answer rejected\n"
            "You just tried to reply, but the quality control rejected your reply\n"
            f"## Your attempted answer:\n{reply}\n\n"
            f"## Reason for rejection:\n{feedback}\n\n"
        )
        messages = _build_messages(updated_system_prompt, history, message)
        response = self.llm.complete(model=self.llm_model, messages=messages)
        return response.message.content or ""

    async def chat_stream(self, message: str, history: list[dict]) -> AsyncGenerator[bytes, None]:
        """
        Stream chat responses as SSE events.

        Yields SSE-formatted events: data: {"delta": ..., "metadata": ...}\\n\\n
        Skips evaluator for streaming mode.
        """
        messages = _build_messages(self.person.system_prompt, history, message)

        try:
            # Kick-start streaming for proxies/browsers that buffer small chunks.
            yield (":" + (" " * 2048) + "\n\n").encode("utf-8")

            async for event in self._run_stream_loop(messages):
                yield event

            yield SSEEvent(metadata={"done": True}).encode()

        except Exception as error:
            user_message, error_code = _handle_llm_error(error, "streaming")
            yield SSEEvent(metadata={"error": str(error), "code": error_code}).encode()

    async def _run_stream_loop(self, messages: list[dict]) -> AsyncGenerator[bytes, None]:
        """Run the streaming loop, handling tool calls until done."""
        tools = self._get_tools()

        while True:
            tool_calls_accumulator: list[dict] = []
            finish_reason: str | None = None

            async for delta in self.llm.stream(model=self.llm_model, messages=messages, tools=tools):
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
                # Check if we got an error event (tool execution failed)
                if b'"status": "failed"' in event:
                    return

    def _accumulate_tool_calls(
        self,
        tool_calls,
        accumulator: list[dict],
    ) -> None:
        """Accumulate streaming tool call deltas into complete tool calls."""
        for tool_call in tool_calls:
            while len(accumulator) <= tool_call.index:
                accumulator.append({
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })

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
        """Execute tool calls during streaming and yield status events."""
        for tc in tool_calls:
            tool_name = tc["function"]["name"]

            yield SSEEvent(
                metadata={"tool_call": tool_name, "status": "executing"}
            ).encode()

            try:
                tool_call_obj = _create_tool_call_object(tc)
                results = self.llm_tools.handle_tool_call([tool_call_obj])

                yield SSEEvent(
                    metadata={"tool_call": tool_name, "status": "success"}
                ).encode()

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                })
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
