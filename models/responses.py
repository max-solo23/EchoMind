from pydantic import BaseModel, Field


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
