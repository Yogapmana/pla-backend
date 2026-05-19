from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import auth, chat, quiz

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

@app.get("/")
async def root():
    return {"message": "Welcome to PLA API — Fase 2 (RAG Pipeline) Active"}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
