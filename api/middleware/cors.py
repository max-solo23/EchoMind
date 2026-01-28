from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_config


def setup_cors(app: FastAPI) -> None:
    config = get_config()

    if not config.allowed_origins:
        print("Warning: No ALLOWED_ORIGINS configured. CORS will block all requests.")
        allowed_origins = []
    else:
        allowed_origins = config.allowed_origins
        print(f"CORS enabled for origins: {allowed_origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
