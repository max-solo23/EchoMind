import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.middleware.cors import setup_cors
from api.routes import admin, chat, health
from repositories.connection import close_database


# uvicorn configures only its own loggers; without this, app INFO lines (LLM usage, cache hits)
# never reach the Fly logs.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per OpenAI request is noise


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await chat.drain_background_tasks()
    await close_database()


app = FastAPI(
    title="EchoMind API",
    description="Personal AI chatbot API",
    version="1.0.0",
    redoc_url="/redoc",
    lifespan=lifespan,
)

setup_cors(app)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/", tags=["root"])
async def root():
    return JSONResponse(
        content={
            "name": "EchoMind API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "chat": "POST /api/v1/chat",
                "admin_health": "GET /api/v1/admin/health",
                "cache_stats": "GET /api/v1/admin/cache/stats",
                "clear_cache": "DELETE /api/v1/admin/cache",
                "session_history": "GET /api/v1/admin/sessions/{session_id}",
            },
        }
    )


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app, host="0.0.0.0", port=port, timeout_keep_alive=300, log_level="info", access_log=True
    )
