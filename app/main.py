from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import auth, chat, quiz, progress, learning, metrics
from app.api.v1.websocket import router as ws_router
from app.api.v1 import curriculum, modules

app = FastAPI(
    title="Personal Learning Agent API",
    description="Backend API for PLA System — Multi-Agent RAG Learning Platform",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend domain
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

# Metrics endpoints (RAG evaluation + UX survey)
app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])

# WebSocket endpoint for real-time agent log streaming
app.include_router(ws_router, tags=["websocket"])

# WebSocket endpoint for real-time agent log streaming
app.include_router(ws_router, tags=["websocket"])

@app.get("/")
async def root():
    return {"message": "Welcome to PLA API — Multi-Agent RAG Learning Platform"}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
