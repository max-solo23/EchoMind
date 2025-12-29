from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from api.middleware.cors import setup_cors
from api.middleware.rate_limit import limiter, rate_limit_exceeded_handler
from api.routes import health, chat, admin
from database import close_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    yield
    # Shutdown
    await close_database()


app = FastAPI(
    title="EchoMind API",
    description="Personal AI chatbot API",
    version="1.0.0",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

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
                "session_history": "GET /api/v1/admin/sessions/{session_id}"
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, 
                host="0.0.0.0", 
                port=8000,
                timeout_keep_alive=300,
                log_level="info",
                access_log=True
            )
