import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Optional

from core.llm.provider import LLMProvider


try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        RateLimitError,
    )
except ImportError:
    # Fallback if openai package structure is different
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


class Chat:
    def __init__(
        self,
        person,
        llm: LLMProvider,
        llm_model: str,
        llm_tools,
        evaluator_llm: Optional["EvaluatorAgent"] = None,  # noqa: F821
    ):
        self.llm = llm
        self.llm_model = llm_model
        self.llm_tools = llm_tools
        self.person = person
        self.evaluator_llm = evaluator_llm

        # Check if provider supports tools
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

        # Too short
        if len(cleaned) < 3:
            return False

        # Remove whitespace for analysis
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

        # At least 30% should be letters
        return total <= 0 or (letters / total) >= 0.3

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
        # Validate message
        if not self._is_valid_message(message):
            logger.warning(f"Invalid message rejected: {message[:50]}...")
            raise InvalidMessageError(
                "Your message appears to be incomplete or invalid. "
                "Please send a clear question or message."
            )

        person_system_prompt = self.person.system_prompt

        messages = (
            [{"role": "system", "content": person_system_prompt}]
            + history
            + [{"role": "user", "content": message}]
        )

        try:
            done = False
            while not done:
                # Only pass tools if provider supports them
                tools = self.llm_tools.tools if self.supports_tools else None
                response = self.llm.complete(model=self.llm_model, messages=messages, tools=tools)
                finish_reason = response.finish_reason
                msg = response.message

                if finish_reason == "tool_calls":
                    tool_calls = msg.tool_calls
                    results = self.llm_tools.handle_tool_call(tool_calls)
                    messages.append(
                        {"role": "assistant", "content": msg.content, "tool_calls": tool_calls}
                    )
                    messages.extend(results)
                else:
                    done = True

            reply = msg.content or ""
            if self.evaluator_llm:
                evaluation = self.evaluator_llm.evaluate(reply, message, history)

                if evaluation.is_acceptable:
                    logger.debug("Passed evaluation - returning reply")
                else:
                    logger.debug("Failed evaluation - rerunning")
                    logger.debug(f"Feedback: {evaluation.feedback}")
                    reply = self.rerun(
                        reply, message, history, evaluation.feedback, self.person.system_prompt
                    )
            return reply

        except RateLimitError as e:
            logger.error(f"LLM rate limit error: {e}")
            return "I'm experiencing high demand right now. Please try again in a moment."
        except APITimeoutError as e:
            logger.error(f"LLM timeout error: {e}")
            return "I'm taking longer than expected to respond. Please try again."
        except APIConnectionError as e:
            logger.error(f"LLM connection error: {e}")
            return "I'm having trouble connecting to my AI service. Please try again shortly."
        except APIError as e:
            logger.error(f"LLM API error: {e}")
            return "I encountered an API issue. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error in chat: {type(e).__name__} - {e}")
            return "I encountered an unexpected issue. Please try again or rephrase your question."

    def rerun(
        self, reply: str, message: str, history: list[dict], feedback: str, system_prompt: str
    ) -> str:
        updated_system_prompt = (
            system_prompt
            + f"\n\n## Previous answer rejected\nYou just tried to reply, but the \
        quality control rejected your reply\n ## Your attempted answer:\n{reply}\n\n ## Reason \
        for rejection:\n{feedback}\n\n"
        )
        messages = (
            [{"role": "system", "content": updated_system_prompt}]
            + history
            + [{"role": "user", "content": message}]
        )
        response = self.llm.complete(model=self.llm_model, messages=messages)
        return response.message.content or ""

    async def chat_stream(self, message: str, history: list[dict]) -> AsyncGenerator[bytes, None]:
        """
        Stream chat responses as SSE events.

        Yields SSE-formatted events: data: {"delta": ..., "metadata": ...}\\n\\n
        Skips evaluator for streaming mode.
        """
        person_system_prompt = self.person.system_prompt

        messages = (
            [{"role": "system", "content": person_system_prompt}]
            + history
            + [{"role": "user", "content": message}]
        )

        try:
            # Kick-start streaming for proxies/browsers that buffer small chunks.
            yield (":" + (" " * 2048) + "\n\n").encode("utf-8")

            done = False
            while not done:
                tool_calls_accumulator = []
                finish_reason: str | None = None

                # Only pass tools if provider supports them
                tools = self.llm_tools.tools if self.supports_tools else None
                async for delta in self.llm.stream(
                    model=self.llm_model, messages=messages, tools=tools
                ):
                    if delta.content:
                        event = {"delta": delta.content, "metadata": None}
                        yield f"data: {json.dumps(event)}\n\n".encode()

                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if len(tool_calls_accumulator) <= tool_call.index:
                                tool_calls_accumulator.append(
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )

                            if tool_call.name:
                                tool_calls_accumulator[tool_call.index]["function"]["name"] = (
                                    tool_call.name
                                )
                            if tool_call.arguments:
                                tool_calls_accumulator[tool_call.index]["function"][
                                    "arguments"
                                ] += tool_call.arguments

                    if delta.finish_reason is not None:
                        finish_reason = delta.finish_reason

                if finish_reason == "tool_calls":
                    # Execute tool calls
                    for tc in tool_calls_accumulator:
                        tool_name = tc["function"]["name"]

                        # Yield tool call start event
                        event = {
                            "delta": None,
                            "metadata": {"tool_call": tool_name, "status": "executing"},
                        }
                        yield f"data: {json.dumps(event)}\n\n".encode()

                        try:
                            # Execute tool
                            from types import SimpleNamespace

                            tool_call_obj = SimpleNamespace(
                                id=tc["id"],
                                type=tc["type"],
                                function=SimpleNamespace(
                                    name=tc["function"]["name"],
                                    arguments=tc["function"]["arguments"],
                                ),
                            )
                            results = self.llm_tools.handle_tool_call([tool_call_obj])

                            # Yield tool call success event
                            event = {
                                "delta": None,
                                "metadata": {"tool_call": tool_name, "status": "success"},
                            }
                            yield f"data: {json.dumps(event)}\n\n".encode()

                            # Add tool results to messages for next iteration
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": tool_calls_accumulator,
                                }
                            )
                            messages.extend(results)

                        except Exception as e:
                            # Yield tool call failure event
                            event = {
                                "delta": None,
                                "metadata": {
                                    "tool_call": tool_name,
                                    "status": "failed",
                                    "error": str(e),
                                },
                            }
                            yield f"data: {json.dumps(event)}\n\n".encode()
                            done = True
                            break
                else:
                    done = True

            # Yield completion event
            event = {"delta": None, "metadata": {"done": True}}
            yield f"data: {json.dumps(event)}\n\n".encode()

        except RateLimitError as e:
            logger.error(f"LLM rate limit error (streaming): {e}")
            event = {"delta": None, "metadata": {"error": str(e), "code": "rate_limit"}}
            yield f"data: {json.dumps(event)}\n\n".encode()
        except APITimeoutError as e:
            logger.error(f"LLM timeout error (streaming): {e}")
            event = {"delta": None, "metadata": {"error": str(e), "code": "api_timeout"}}
            yield f"data: {json.dumps(event)}\n\n".encode()
        except APIConnectionError as e:
            logger.error(f"LLM connection error (streaming): {e}")
            event = {"delta": None, "metadata": {"error": str(e), "code": "connection_error"}}
            yield f"data: {json.dumps(event)}\n\n".encode()
        except APIError as e:
            logger.error(f"LLM API error (streaming): {e}")
            event = {"delta": None, "metadata": {"error": str(e), "code": "api_error"}}
            yield f"data: {json.dumps(event)}\n\n".encode()
        except Exception as e:
            logger.error(f"Unexpected error in streaming: {type(e).__name__} - {e}")
            event = {"delta": None, "metadata": {"error": str(e), "code": "unknown_error"}}
            yield f"data: {json.dumps(event)}\n\n".encode()
