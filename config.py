from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv


@dataclass
class Config:
    llm_provider: str
    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    pushover_token: Optional[str]
    pushover_user: Optional[str]
    persona_name: str
    persona_file: str
    use_evaluator: bool = False
    api_key: str = ""
    allowed_origins: list[str] = None

    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = []

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(override=True)

        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        llm_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not llm_key:
            raise RuntimeError(
                "LLM credentials not configured. "
                "Set LLM_API_KEY (or OPENAI_API_KEY for the OpenAI provider)."
            )

        # Parse allowed origins
        origins_str = os.getenv("ALLOWED_ORIGINS", "")
        allowed_origins = [o.strip() for o in origins_str.split(",") if o.strip()]

        return cls(
            llm_provider=llm_provider,
            llm_api_key=llm_key,
            llm_base_url=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            llm_model=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.2-nano"),
            pushover_token=os.getenv("PUSHOVER_TOKEN"),
            pushover_user=os.getenv("PUSHOVER_USER"),
            persona_name="Maksym",
            persona_file="persona.yaml",
            use_evaluator=os.getenv("USE_EVALUATOR", "false").lower() == "true",
            api_key=os.getenv("API_KEY", ""),
            allowed_origins=allowed_origins
        )
