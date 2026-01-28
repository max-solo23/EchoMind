from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What are your main skills?",
                "history": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi! I'm Maksym..."},
                ],
            }
        }
    }
