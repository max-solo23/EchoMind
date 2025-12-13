from core.interfaces import LLMProvider
from typing import Optional, Any, AsyncGenerator
import json
import logging
import inspect


logger = logging.getLogger(__name__)

class Chat:
    def __init__(
            self, 
            person, 
            llm: LLMProvider, 
            llm_model: str, 
            llm_tools, 
            evaluator_llm: Optional["EvaluatorAgent"] = None,
            streaming_llm: Any | None = None,
        ):
        self.llm = llm
        self.streaming_llm = streaming_llm or llm
        self.llm_model = llm_model
        self.llm_tools = llm_tools
        self.person = person
        self.evaluator_llm = evaluator_llm

    def chat(self, message: str, history: list[dict]) -> str:
        person_system_prompt = self.person.system_prompt

        messages = [
            {"role": "system", "content": person_system_prompt}
                           ] + history + [
            {"role": "user", "content": message}
        ]

        done = False
        while not done:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=self.llm_tools.tools
            )

            finish_reason = response.choices[0].finish_reason
            msg = response.choices[0].message

            if finish_reason == "tool_calls":
                tool_calls = msg.tool_calls
                results = self.llm_tools.handle_tool_call(tool_calls)
                messages.append(msg)
                messages.extend(results)
            else:
                done = True

        reply = msg.content
        if self.evaluator_llm:
            evaluation = self.evaluator_llm.evaluate(reply, message, history)

            if evaluation.is_acceptable:
                print("Passed evaluation - returning reply")
            else:
                print("Failed evaluation - returning reply")
                print(message)
                print(evaluation.feedback)
                reply = self.rerun(reply, message, history, evaluation.feedback, self.person.system_prompt)
        return reply

    def rerun(self, reply: str, message: str, history: list[dict], feedback: str, system_prompt: str) -> str:
        updated_system_prompt = system_prompt + f"\n\n## Previous answer rejected\nYou just tried to reply, but the \
        quality control rejected your reply\n ## Your attempted answer:\n{reply}\n\n ## Reason \
        for rejection:\n{feedback}\n\n"
        messages = [
            {"role": "system", "content": updated_system_prompt}
                   ] + history + [
            {"role": "user", "content": message}
        ]
        response = self.llm.chat.completions.create(model=self.llm_model, messages=messages)
        return response.choices[0].message.content

    async def chat_stream(self, message: str, history: list[dict]) -> AsyncGenerator[bytes, None]:
        """
        Stream chat responses as SSE events.

        Yields SSE-formatted events: data: {"delta": ..., "metadata": ...}\\n\\n
        Skips evaluator for streaming mode.
        """
        person_system_prompt = self.person.system_prompt

        messages = [
            {"role": "system", "content": person_system_prompt}
        ] + history + [
            {"role": "user", "content": message}
        ]

        try:
            # Kick-start streaming for proxies/browsers that buffer small chunks.
            yield (":" + (" " * 2048) + "\n\n").encode("utf-8")

            done = False
            while not done:
                maybe_stream = self.streaming_llm.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    tools=self.llm_tools.tools,
                    stream=True
                )
                stream = await maybe_stream if inspect.isawaitable(maybe_stream) else maybe_stream

                tool_calls_accumulator = []
                finish_reason = None

                if hasattr(stream, "__aiter__"):
                    async for chunk in stream:
                        choice = chunk.choices[0]
                        finish_reason = choice.finish_reason
                        delta = choice.delta

                        # Handle token content
                        if delta.content:
                            event = {"delta": delta.content, "metadata": None}
                            yield f"data: {json.dumps(event)}\n\n".encode("utf-8")

                        # Handle tool calls
                        if delta.tool_calls:
                            for tool_call in delta.tool_calls:
                                # Accumulate tool call chunks
                                if len(tool_calls_accumulator) <= tool_call.index:
                                    tool_calls_accumulator.append({
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })

                                if tool_call.function.name:
                                    tool_calls_accumulator[tool_call.index]["function"]["name"] = tool_call.function.name
                                if tool_call.function.arguments:
                                    tool_calls_accumulator[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
                else:
                    for chunk in stream:
                        choice = chunk.choices[0]
                        finish_reason = choice.finish_reason
                        delta = choice.delta

                        # Handle token content
                        if delta.content:
                            event = {"delta": delta.content, "metadata": None}
                            yield f"data: {json.dumps(event)}\n\n".encode("utf-8")

                        # Handle tool calls
                        if delta.tool_calls:
                            for tool_call in delta.tool_calls:
                                # Accumulate tool call chunks
                                if len(tool_calls_accumulator) <= tool_call.index:
                                    tool_calls_accumulator.append({
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })

                                if tool_call.function.name:
                                    tool_calls_accumulator[tool_call.index]["function"]["name"] = tool_call.function.name
                                if tool_call.function.arguments:
                                    tool_calls_accumulator[tool_call.index]["function"]["arguments"] += tool_call.function.arguments

                if finish_reason == "tool_calls":
                    # Execute tool calls
                    for tc in tool_calls_accumulator:
                        tool_name = tc["function"]["name"]

                        # Yield tool call start event
                        event = {"delta": None, "metadata": {"tool_call": tool_name, "status": "executing"}}
                        yield f"data: {json.dumps(event)}\n\n".encode("utf-8")

                        try:
                            # Execute tool
                            from types import SimpleNamespace
                            tool_call_obj = SimpleNamespace(
                                id=tc["id"],
                                type=tc["type"],
                                function=SimpleNamespace(
                                    name=tc["function"]["name"],
                                    arguments=tc["function"]["arguments"]
                                )
                            )
                            results = self.llm_tools.handle_tool_call([tool_call_obj])

                            # Yield tool call success event
                            event = {"delta": None, "metadata": {"tool_call": tool_name, "status": "success"}}
                            yield f"data: {json.dumps(event)}\n\n".encode("utf-8")

                            # Add tool results to messages for next iteration
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls_accumulator
                            })
                            messages.extend(results)

                        except Exception as e:
                            # Yield tool call failure event
                            event = {
                                "delta": None,
                                "metadata": {"tool_call": tool_name, "status": "failed", "error": str(e)}
                            }
                            yield f"data: {json.dumps(event)}\n\n".encode("utf-8")
                            done = True
                            break
                else:
                    done = True

            # Yield completion event
            event = {"delta": None, "metadata": {"done": True}}
            yield f"data: {json.dumps(event)}\n\n".encode("utf-8")


        except Exception as e:
            # Yield error event
            error_code = "api_error"
            if "timeout" in str(e).lower():
                error_code = "api_timeout"
            elif "rate" in str(e).lower():
                error_code = "rate_limit"

            event = {"delta": None, "metadata": {"error": str(e), "code": error_code}}
            yield f"data: {json.dumps(event)}\n\n".encode("utf-8")
