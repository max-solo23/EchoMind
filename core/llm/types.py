from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass(frozen=True)
class StreamDelta:
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class CompletionMessage:
    role: Role
    content: str | None
    tool_calls: Any | None = None


@dataclass(frozen=True)
class CompletionResponse:
    finish_reason: str | None
    message: CompletionMessage
