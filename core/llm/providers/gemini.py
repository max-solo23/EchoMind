from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx

from core.llm.provider import LLMProvider
from core.llm.types import CompletionMessage, CompletionResponse, StreamDelta


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        self._timeout = timeout_s
        self._client = httpx.Client(timeout=timeout_s)
        self._async_client = httpx.AsyncClient(timeout=timeout_s)

    def _to_gemini(self, messages: list[dict]) -> tuple[dict | None, list[dict]]:
        system_parts: list[str] = []
        contents: list[dict] = []

        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if not content:
                continue

            if role == "system":
                system_parts.append(str(content))
                continue

            gemini_role = "user" if role == "user" else "model"
            contents.append({"role": gemini_role, "parts": [{"text": str(content)}]})

        system_instruction = None
        if system_parts:
            system_instruction = {"parts": [{"text": "\n\n".join(system_parts)}]}

        return system_instruction, contents

    def _extract_text(self, response_json: dict) -> str:
        candidates = response_json.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts: list[str] = []
        for p in parts:
            if "text" in p and p["text"] is not None:
                texts.append(str(p["text"]))
        return "".join(texts)

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> CompletionResponse:
        if tools:
            raise NotImplementedError("GeminiProvider tool/function calling not implemented yet.")

        system_instruction, contents = self._to_gemini(messages)
        body: dict[str, Any] = {"contents": contents}
        if system_instruction:
            body["system_instruction"] = system_instruction

        url = f"{self._base_url}/v1beta/models/{model}:generateContent"
        resp = self._client.post(url, params={"key": self._api_key}, json=body)
        resp.raise_for_status()
        data = resp.json()
        text = self._extract_text(data)
        return CompletionResponse(
            finish_reason=None,
            message=CompletionMessage(role="assistant", content=text, tool_calls=None),
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        if tools:
            raise NotImplementedError("GeminiProvider tool/function calling not implemented yet.")

        system_instruction, contents = self._to_gemini(messages)
        body: dict[str, Any] = {"contents": contents}
        if system_instruction:
            body["system_instruction"] = system_instruction

        url = f"{self._base_url}/v1beta/models/{model}:streamGenerateContent"

        async with self._async_client.stream(
            "POST",
            url,
            params={"key": self._api_key},
            json=body,
        ) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = self._extract_text(payload)
                if text:
                    yield StreamDelta(content=text, tool_calls=None, finish_reason=None)

        yield StreamDelta(content=None, tool_calls=None, finish_reason="stop")

    def parse(
        self,
        *,
        model: str,
        messages: list[dict],
        response_format: Any,
    ) -> Any:
        raise NotImplementedError("GeminiProvider does not support structured outputs")

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "tools": False,
            "streaming": True,
            "structured_output": False,
        }
