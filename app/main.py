import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import auth, chat, quiz, progress, learning, gamification
from app.api.v1.websocket import router as ws_router
from app.api.v1 import curriculum, modules, notifications
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

LANGSMITH_KEYS = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
for key in LANGSMITH_KEYS:
    val = os.getenv(key)
    if val:
        os.environ[key] = val
from redis import asyncio as aioredis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = aioredis.from_url(settings.REDIS_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="pla-cache")
    # Warm shared Redis pool used by metrics/quiz/log helpers
    from app.utils.redis_client import get_redis, close_redis
    await get_redis()
    yield
    await close_redis()

app = FastAPI(
    title="Personal Learning Agent API",
    description="Backend API for Synapsa System — Multi-Agent RAG Learning Platform",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration — read allowed origins from settings (env-driven).
# In dev, defaults to localhost:3000/5173/5174. Set CORS_ORIGINS in .env for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth endpoints
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])

# Tutor Chat endpoints (RAG)
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

# Quiz endpoints
app.include_router(quiz.router, prefix="/api/v1/quiz", tags=["quiz"])

# Progress & Feedback endpoints
app.include_router(progress.router, prefix="/api/v1/progress", tags=["progress"])

# Learning Session endpoints
app.include_router(learning.router, prefix="/api/v1/learning", tags=["learning"])

# Curriculum endpoint
app.include_router(curriculum.router, prefix="/api/v1", tags=["curriculum"])

# Modules endpoint
app.include_router(modules.router, prefix="/api/v1", tags=["modules"])

# Gamification endpoints (Streak Heatmap + XP / Leveling)
app.include_router(
    gamification.router,
    prefix="/api/v1/gamification",
    tags=["gamification"],
)

# Notifications endpoint
app.include_router(notifications.router, prefix="/api/v1/notifications", tags=["notifications"])


# WebSocket endpoint for real-time agent log streaming
app.include_router(ws_router, tags=["websocket"])



@app.get("/")
async def root():
    return {"message": "Welcome to Synapsa API — Multi-Agent RAG Learning Platform"}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
