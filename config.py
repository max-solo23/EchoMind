from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv


@dataclass
class Config:
    openai_api_key: str
    openai_model: str
    pushover_token: Optional[str]
    pushover_user: Optional[str]
    persona_name: str
    persona_file: str
    use_evaluator: bool = False
    api_key: str = ""
    allowed_origins: list[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(override=True)

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError(
                "OpenAI credentials not configured. "
                "Set OPENAI_API_KEY."
            )

        # Parse allowed origins
        origins_str = os.getenv("ALLOWED_ORIGINS", "")
        allowed_origins = [o.strip() for o in origins_str.split(",") if o.strip()]

        return cls(
            openai_api_key=openai_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
            pushover_token=os.getenv("PUSHOVER_TOKEN"),
            pushover_user=os.getenv("PUSHOVER_USER"),
            persona_name="Maksym",
            persona_file="persona.yaml",
            use_evaluator=os.getenv("USE_EVALUATOR", "false").lower() == "true",
            api_key=os.getenv("API_KEY", ""),
            allowed_origins=allowed_origins
        )
