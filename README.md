# Personal Learning Agent (PLA) System

A multi-agent RAG-based adaptive learning platform that operates as a "private tutoring team." Given a topic and duration, the system autonomously designs a personalized curriculum, gathers materials from the internet, synthesizes them into structured learning modules, and adapts the learning path in real-time based on multi-signal performance feedback.

Built for undergraduate thesis (Skripsi S1).

---

## Architecture

The system uses **LangGraph** to orchestrate five specialized agents in a pipeline:

```
[START] → [PLANNER] → [RESEARCHER] → [COMPOSER] → [TUTOR + FEEDBACK] → [END]
                        ↑
                    ┌──[FEEDBACK ENGINE]──┘
```

### Agents

| Agent | Role |
|-------|------|
| **Planner** | Decomposes topic into multi-week curriculum with search queries |
| **Researcher** | Layered multi-source scraping: Tavily + Jina + Wikipedia + ArXiv + Semantic Scholar + Course discovery |
| **Composer** | Synthesizes raw content into structured Markdown learning modules |
| **Tutor** | Interactive RAG chat (HyDE + FlashRank reranking) + adaptive MCQ quiz generation |
| **Feedback Engine** | Computes mastery from 5 signals (quiz 40%, reading time 20%, question frequency 20%, self-assessment 15%, material rating 5%) and triggers curriculum revision |

### RAG Pipeline

Advanced retrieval using **HyDE (Hypothetical Document Embedding)** + **FlashRank cross-encoder reranking**:

1. Generate hypothetical answer to the query (HyDE)
2. Dual embedding: 60% hypothetical + 40% query vector
3. Qdrant initial search (top-8)
4. FlashRank reranking → top-3 chunks
5. LLM generation with context

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Web Framework | FastAPI + Uvicorn |
| Database | PostgreSQL 16 (async, SQLAlchemy 2.0) |
| Vector DB | Qdrant |
| Cache / Broker | Redis 7 + Celery |
| AI / Agents | LangGraph, LangChain, Groq, Ollama |
| Embedding | `nomic-embed-text` via Ollama |
| RAG Evaluation | RAGAS |
| Auth | PyJWT + passlib[bcrypt] |

---

## API Endpoints

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Login, returns JWT |
| GET | `/api/v1/auth/me` | Current user profile |

### Learning Session
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/learning/start` | Start session → triggers Celery pipeline |
| GET | `/api/v1/learning/sessions` | List user's sessions |
| GET | `/api/v1/learning/{session_id}` | Session details |
| GET | `/api/v1/learning/{session_id}/curriculum` | Curriculum JSON |
| GET | `/api/v1/learning/{session_id}/topics` | Topic schedule |
| GET | `/api/v1/learning/{session_id}/modules/{topic_id}` | Learning module markdown |
| PATCH | `/api/v1/learning/{session_id}/topics/{topic_id}/complete` | Mark topic completed |

### Chat (RAG Tutor)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat/message` | Send message, returns RAG response + sources |
| GET | `/api/v1/chat/history/{topic_id}` | Chat history per topic |
| DELETE | `/api/v1/chat/history/{topic_id}` | Clear history |

### Quiz
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/v1/quiz/{topic_id}` | Generate MCQ quiz |
| POST | `/api/v1/quiz/submit` | Submit answers, returns score + feedback |
| GET  | `/api/v1/quiz/history/{session_id}` | Quiz attempt history |

### Progress & Feedback
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/progress/signal` | Submit progress signals |
| POST | `/api/v1/progress/evaluate` | Trigger feedback engine → recalculate mastery |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `WS /ws/agent-log/{session_id}?token={jwt}` | Real-time agent log streaming |

---

## Quick Start

### 1. Infrastructure

```bash
docker compose up -d postgres qdrant redis
```

### 2. Environment

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+asyncpg://pla_user:pla_password@localhost:5433/pla_db
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6380/0
OLLAMA_BASE_URL=http://localhost:11434
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
JINA_API_KEY=your_jina_api_key
SECRET_KEY=your_secret_key_here
```

### 3. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run

```bash
# API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Celery worker (separate terminal)
celery -A app.tasks.celery_app worker --loglevel=info
```

### 5. Full Docker Stack

```bash
docker compose up --build
```

---

## Project Structure

```
pla-backend/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Settings from environment
│   ├── dependencies.py            # JWT auth dependency
│   ├── api/v1/
│   │   ├── auth.py                # Auth endpoints
│   │   ├── learning.py            # Learning session endpoints
│   │   ├── chat.py                # RAG chat endpoints
│   │   ├── quiz.py                # Quiz endpoints
│   │   ├── progress.py            # Progress signals + feedback
│   │   └── websocket.py           # Real-time agent log streaming
│   ├── agents/
│   │   ├── orchestrator.py       # LangGraph StateGraph builder
│   │   ├── planner.py             # Curriculum generation
│   │   ├── researcher.py          # Multi-source scraping
│   │   ├── composer.py            # Module synthesis + RAG indexing
│   │   ├── tutor.py               # RAG chat + quiz generation
│   │   ├── feedback_engine.py     # Mastery calculation + revision logic
│   │   └── state.py               # PLAState TypedDict + models
│   ├── rag/
│   │   ├── indexer.py             # Chunk → embed → Qdrant pipeline
│   │   ├── retriever.py           # HyDE + dual embedding + FlashRank
│   │   ├── embedder.py            # Ollama nomic-embed-text wrapper
│   │   ├── vector_store.py        # Qdrant client
│   │   └── chunker.py             # Recursive text splitter
│   ├── tools/
│   │   ├── tavily_search.py       # Web search
│   │   ├── jina_reader.py         # Full-text URL extraction
│   │   ├── wikipedia_search.py    # Wikipedia search
│   │   ├── arxiv_search.py        # ArXiv paper search
│   │   ├── semantic_scholar.py    # Academic paper search
│   │   ├── youtube_transcript.py  # YouTube transcript
│   │   └── course_discovery.py    # Course platform search
│   ├── services/
│   │   └── learning_service.py    # DB operations service layer
│   ├── models/
│   │   ├── user.py
│   │   ├── learning.py
│   │   └── agent.py
│   └── db/
│       ├── database.py            # Async SQLAlchemy engine
│       └── migrations/            # Alembic migrations
├── tasks/
│   ├── celery_app.py              # Celery instance
│   └── run_orchestrator.py        # Background pipeline task
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── PRD_Personal_Learning_Agent.md
```

---

## Models

PLA uses two databases:

**PostgreSQL** — Sessions, curricula, topics, modules, chat history, quiz results, progress signals, mastery scores, agent logs, UX surveys.

**Qdrant** — Per-user vector collections (`pla_user_{user_id}`). Chunks are indexed with metadata: `topic_id`, `week`, `day`, `source_type`, `module_id`.

---

## Adaptive Feedback Loop

The feedback engine calculates mastery from five signals:

```
mastery = quiz_score × 0.40
        + reading_time_ratio × 0.20
        + question_frequency × 0.20
        + self_assessment × 0.15
        + material_rating × 0.05
```

Actions based on mastery score:

| Score | Action | Planner Revision |
|-------|--------|-----------------|
| < 0.60 | `repeat` | Repeat topic + find simpler materials |
| 0.60–0.75 | `review` | Add review sessions before next topic |
| 0.75–0.90 | `continue` | Proceed as scheduled |
| > 0.90 | `accelerate` | Speed up, skip intro if mastered |

---

## Tests

```bash
source venv/bin/activate
PYTHONPATH=. python -m pytest tests/test_agents/ -v
```

All tests pass: Feedback engine (5 tests), Progress API (2 tests).