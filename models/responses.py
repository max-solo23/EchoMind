from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    reply: str = Field(...)


class ErrorResponse(BaseModel):
    detail: str = Field(...)


class HealthResponse(BaseModel):
    status: str = Field(...)
    version: str = Field(...)


class StreamEvent(BaseModel):
    delta: str | None = Field(None)
    metadata: dict | None = Field(None)
