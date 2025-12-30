from pydantic import BaseModel, Field
from typing import Literal, Optional


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    reply: str = Field(..., description="Chatbot's reply")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error description")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


# Streaming response models

class StreamEvent(BaseModel):
    """
    SSE event model for streaming chat responses.

    Format: {"delta": <string|null>, "metadata": <object|null>}
    """
    delta: str | None = Field(None, description="Text content to append")
    metadata: dict | None = Field(None, description="Event metadata")
