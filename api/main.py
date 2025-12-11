from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.middleware.cors import setup_cors
from api.routes import health, chat


app = FastAPI(
    title="EchoMind API",
    description="Personal AI chatbot API",
    version="1.0.0",
    redoc_url="/redoc"
)

setup_cors(app)

app.include_router(health.router)
app.include_router(chat.router)


@app.get("/", tags=["root"])
async def root():
    return JSONResponse(
        content={
            "name": "EchoMind API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "chat": "POST /api/v1/chat"
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
