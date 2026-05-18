# Product Requirements Document (PRD)
## Personal Learning Agent (PLA) System
**Versi:** 1.0.0  
**Tanggal:** Mei 2026  
**Status:** Draft — untuk keperluan skripsi S1  
**Author:** [Nama Mahasiswa]  
**Pembimbing:** [Nama Dosen Pembimbing]

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Latar Belakang & Masalah](#2-latar-belakang--masalah)
3. [Tujuan & Ruang Lingkup](#3-tujuan--ruang-lingkup)
4. [Arsitektur Sistem](#4-arsitektur-sistem)
5. [Spesifikasi Agent](#5-spesifikasi-agent)
6. [Tech Stack](#6-tech-stack)
7. [Struktur Repository](#7-struktur-repository)
8. [Spesifikasi API](#8-spesifikasi-api)
9. [Skema Database](#9-skema-database)
10. [RAG Pipeline](#10-rag-pipeline)
11. [Fitur & Requirement Fungsional](#11-fitur--requirement-fungsional)
12. [Requirement Non-Fungsional](#12-requirement-non-fungsional)
13. [Alur Pengguna (User Flow)](#13-alur-pengguna-user-flow)
14. [Metrik Evaluasi & KPI Skripsi](#14-metrik-evaluasi--kpi-skripsi)
15. [Docker Compose & Deployment](#15-docker-compose--deployment)
16. [Milestone & Timeline](#16-milestone--timeline)
17. [Risiko & Mitigasi](#17-risiko--mitigasi)
18. [Glosarium](#18-glosarium)

---

## 1. Ringkasan Eksekutif

Personal Learning Agent (PLA) adalah sistem multi-agent berbasis kecerdasan buatan yang beroperasi layaknya tim pendidik privat yang berkolaborasi secara otonom. Hanya dari satu instruksi awal pengguna berupa topik dan durasi belajar, sistem secara otomatis merancang kurikulum, mengumpulkan dan menyintesis materi dari berbagai sumber internet, menyajikan konten dalam bentuk modul belajar terstruktur, serta melakukan evaluasi adaptif yang terus menyesuaikan kecepatan dan strategi pembelajaran berdasarkan performa pengguna.

**Pernyataan masalah inti:** Tidak adanya sistem pembelajaran digital yang mampu secara otonom merancang kurikulum personal, mengumpulkan materi terkini dari internet, menyintesisnya menjadi modul belajar yang terkurasi, dan mengadaptasi jalur belajar secara real-time berdasarkan multi-sinyal kemajuan pengguna.

**Kontribusi penelitian:**
- Arsitektur multi-agent heterogeneous dengan pembagian peran Orchestrator, Planner, Researcher, Composer, dan Tutor
- Advanced RAG pipeline dengan HyDE + re-ranking berbasis materi yang dikurasi secara dinamis
- Adaptive feedback engine berbasis multi-sinyal (bukan hanya skor kuis)
- Evaluasi komparatif heterogeneous vs homogeneous model assignment pada sistem multi-agent edukasi

---

## 2. Latar Belakang & Masalah

### 2.1 Konteks

Era pembelajaran digital telah menghasilkan platform seperti Coursera, Udemy, dan Khan Academy, namun kesemuanya menyajikan kurikulum yang bersifat statis dan seragam. Pengguna dengan latar belakang berbeda dipaksa mengikuti jalur yang sama, tanpa mempertimbangkan kecepatan belajar, gaya kognitif, maupun topik spesifik yang relevan bagi mereka.

Perkembangan Large Language Models (LLM) dan paradigma multi-agent AI membuka peluang baru: sistem yang dapat bertindak sebagai "tim pengajar privat" yang menyesuaikan diri secara real-time terhadap kebutuhan individu.

### 2.2 Permasalahan yang Diidentifikasi

| No | Masalah | Dampak |
|----|---------|--------|
| P1 | Kurikulum platform existing bersifat statis, tidak menyesuaikan level pengguna | Pengguna bosan atau kewalahan |
| P2 | Materi tidak diperbarui secara dinamis sesuai perkembangan topik | Konten cepat usang |
| P3 | Tidak ada mekanisme adaptasi berbasis performa nyata pengguna | Tidak ada personalisasi sejati |
| P4 | Pengguna tidak dapat belajar dari sumber yang dikurasi khusus untuk mereka | Overload informasi |
| P5 | Tidak ada asisten yang memahami konteks materi yang sedang dipelajari | Tanya-jawab tidak relevan |

### 2.3 Solusi yang Diusulkan

Sistem PLA menggunakan arsitektur 5 agent yang bekerja secara kolaboratif dan otonom:
- **Orchestrator** sebagai koordinator pusat
- **Planner** untuk merancang dan merevisi kurikulum adaptif
- **Researcher** untuk mengumpulkan materi multi-sumber secara real-time
- **Composer** untuk menyintesis bahan mentah menjadi modul belajar terstruktur
- **Tutor** untuk interaksi langsung dengan pengguna melalui kuis dan chat RAG

---

## 3. Tujuan & Ruang Lingkup

### 3.1 Tujuan Penelitian

1. Merancang dan mengimplementasikan arsitektur sistem multi-agent untuk personal learning yang adaptif
2. Mengembangkan Advanced RAG pipeline (HyDE + re-ranking) berbasis materi yang dikurasi secara dinamis per pengguna
3. Mengimplementasikan adaptive feedback engine berbasis multi-sinyal kemajuan belajar
4. Mengevaluasi kualitas sistem menggunakan metrik RAG (RAGAS), efisiensi agent, kepuasan pengguna (UX), dan latensi

### 3.2 Ruang Lingkup (In-Scope)

- Sistem autentikasi pengguna (register, login, profil)
- Input learning goal dan konfigurasi awal (topik, durasi, level, jam/hari)
- Pembuatan kurikulum dinamis oleh Planner Agent
- Pencarian dan pengumpulan materi dari: web umum, YouTube (transcript), arXiv, Wikipedia, upload PDF/dokumen pengguna, artikel, dan buku
- Sintesis materi menjadi modul belajar per sub-bab oleh Composer Agent
- Kuis adaptif interaktif per topik
- Chat berbasis Advanced RAG (HyDE + FlashRank re-ranking) per sesi belajar
- Adaptive feedback loop: revisi jadwal berdasarkan multi-sinyal
- Dashboard progress pengguna
- Agent activity log untuk transparansi sistem
- Dashboard metrik evaluasi (RAGAS, latency, UX)

### 3.3 Batasan (Out-of-Scope)

- Aplikasi mobile native (iOS/Android)
- Sistem pembayaran / monetisasi
- Multi-bahasa selain Bahasa Indonesia dan Inggris
- Video conference / live tutoring
- Kolaborasi antar pengguna / fitur sosial
- Fine-tuning model LLM

---

## 4. Arsitektur Sistem

### 4.1 Gambaran Umum

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)               │
│   Dashboard · Kurikulum · Chat · Kuis · Metrik · Log    │
└────────────────────────┬────────────────────────────────┘
                         │ REST API + WebSocket
┌────────────────────────▼────────────────────────────────┐
│                  BACKEND (FastAPI)                       │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │            Orchestrator Agent (LangGraph)        │    │
│  │                                                  │    │
│  │  ┌──────────┐  ┌────────────┐  ┌─────────────┐  │    │
│  │  │ Planner  │  │ Researcher │  │   Composer  │  │    │
│  │  │  Agent   │  │   Agent    │  │    Agent    │  │    │
│  │  └──────────┘  └────────────┘  └─────────────┘  │    │
│  │                     ┌──────────┐                 │    │
│  │                     │  Tutor   │                 │    │
│  │                     │  Agent   │                 │    │
│  │                     └──────────┘                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────┐   │
│  │  PostgreSQL  │  │    Qdrant   │  │ Ollama / Groq │   │
│  │  (app data)  │  │ (vector DB) │  │ (LLM & embed) │   │
│  └──────────────┘  └─────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Pola Komunikasi Agent

Sistem menggunakan pola **Orchestrator-Subagent** via LangGraph `StateGraph`. State dibagikan antar agent melalui shared memory object, bukan dikirim manual antar fungsi. Orchestrator memutuskan agent mana yang dipanggil, kapan, dan dengan konteks apa berdasarkan kondisi state saat itu.

**State object utama:**
```python
class PLAState(TypedDict):
    user_id: str
    learning_config: LearningConfig     # topik, durasi, level, jam/hari
    curriculum: Curriculum              # output Planner
    research_results: list[RawContent]  # output Researcher
    modules: list[LearningModule]       # output Composer
    chat_history: list[Message]         # riwayat chat Tutor
    quiz_results: list[QuizResult]      # skor kuis per topik
    mastery_scores: dict[str, float]    # mastery per topik
    progress_signals: ProgressSignals   # semua sinyal adaptif
    feedback_actions: list[FeedbackAction] # aksi revisi Planner
    agent_logs: list[AgentLog]          # log aktivitas semua agent
```

### 4.3 Pola Alur Data

```
User Input
    │
    ▼
Orchestrator ──► Planner (kurikulum + query list)
                     │
                     ▼
              Researcher (raw content per topik)
                     │
                     ▼
              Composer ──► Qdrant (embed chunks)
                     │         │
                     ▼         ▼
              Modul UI     Vector Store
                               │
                               ▼
                          Tutor Agent
                          ├── Chat RAG (HyDE → retrieve → rerank → generate)
                          └── Kuis (generate → evaluate → score)
                               │
                               ▼
                     Adaptive Feedback Engine
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
               Revisi jadwal      Cari materi tambahan
               (Planner)          (Researcher)
```

---

## 5. Spesifikasi Agent

### 5.1 Orchestrator Agent

**Model:** Qwen3-32B (via Ollama lokal atau Groq fallback)  
**Framework:** LangGraph StateGraph  
**Tanggung jawab:**
- Menerima `UserLearningConfig` sebagai input awal
- Mendistribusikan task ke subagent yang tepat berdasarkan state
- Mengelola urutan eksekusi: Planner → Researcher → Composer (paralel jika memungkinkan)
- Menerima sinyal dari Adaptive Feedback Engine dan memicu Planner untuk revisi
- Mencatat semua aktivitas ke `agent_logs`

**Node LangGraph:**
```
START → validate_input → call_planner → call_researcher → 
call_composer → notify_ready → END
         ↑                                    │
         └──── feedback_loop ←── evaluate ───┘
```

### 5.2 Planner Agent

**Model:** Qwen3-14B (via Ollama)  
**Input:** `LearningConfig` (topik, durasi, level, jam/hari)  
**Output:** `Curriculum` (JSON terstruktur)  

**Tugas utama:**
1. Dekomposisi topik menjadi sub-topik terurut berdasarkan prerequisite
2. Kalkulasi jadwal harian: `total_hours = days × hours_per_day`, dibagi ke topik dengan bobot kompleksitas
3. Menghasilkan daftar search query per topik untuk Researcher
4. Menerima `FeedbackAction` dari Feedback Engine dan merevisi jadwal

**Output schema:**
```json
{
  "curriculum_id": "uuid",
  "topic": "Machine Learning",
  "total_weeks": 4,
  "weeks": [
    {
      "week": 1,
      "title": "Pengenalan ML",
      "days": [
        {
          "day": 1,
          "topic_id": "intro_ml",
          "title": "Apa itu Machine Learning?",
          "duration_minutes": 60,
          "status": "pending",
          "search_queries": ["machine learning introduction", "ML for beginners"]
        }
      ]
    }
  ]
}
```

**Feedback revision rules:**

| Mastery Score | Aksi Revisi |
|---------------|-------------|
| < 60% | Topik diulang + Researcher cari materi alternatif yang lebih sederhana |
| 60–75% | Tambah sesi review sebelum lanjut ke topik berikutnya |
| 75–90% | Lanjut normal sesuai jadwal |
| > 90% | Percepat jadwal, skip materi intro jika ada |

### 5.3 Researcher Agent

**Model:** Qwen3-8B (via Ollama) — ringan karena task utama adalah tool calling  
**Input:** Daftar search query dari Planner  
**Output:** `RawContent[]` — teks mentah beserta metadata sumber  

**Tool set:**
```python
tools = [
    TavilySearchTool(),          # web umum
    YoutubeTranscriptTool(),     # transcript YouTube
    ArxivSearchTool(),           # paper akademik
    WikipediaSearchTool(),       # Wikipedia
    PDFLoaderTool(),             # dokumen upload pengguna
    WebScraperTool(),            # artikel & buku online
]
```

**Proses per topik:**
1. Jalankan semua tools secara paralel menggunakan `asyncio.gather()`
2. Filter konten: buang hasil dengan relevansi < threshold
3. Deduplikasi konten yang mirip
4. Kirim `RawContent[]` ke Composer

**Metadata yang disimpan per sumber:**
```python
class RawContent(BaseModel):
    source_type: str       # web | youtube | arxiv | wikipedia | pdf | article
    source_url: str
    source_title: str
    raw_text: str
    topic_id: str
    relevance_score: float
    fetched_at: datetime
```

### 5.4 Composer Agent

**Model:** Qwen3-32B (via Ollama / Groq fallback — agent paling demanding)  
**Input:** `RawContent[]` per topik dari Researcher  
**Output (paralel):**
- `LearningModule` — modul belajar terstruktur untuk ditampilkan di UI
- Chunks yang telah di-embed → disimpan ke Qdrant

**Struktur modul output (wajib):**
```markdown
# [Judul Topik]
## 🎯 Learning Objectives
- Poin 1
- Poin 2
- Poin 3

## 📖 Penjelasan Konsep
[Narasi utama dengan analogi yang disesuaikan level pengguna]

## 💡 Contoh Konkret
[Studi kasus / contoh kode jika relevan]

## 🔁 Ringkasan
[Tabel atau poin-poin kunci]

## 🧪 Latihan Mandiri
[1-2 soal refleksi sebelum kuis formal]

## 📚 Referensi Sumber
- [Sumber 1](url)
- [Sumber 2](url)
```

**Proses chunking untuk RAG:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_text(module.full_text)

# Metadata per chunk
metadata = {
    "user_id": user_id,
    "topic_id": topic_id,
    "week": week_number,
    "day": day_number,
    "source_types": [s.source_type for s in raw_sources],
    "module_id": module_id
}
```

**Draft kuis:** Composer juga menghasilkan 5–10 draft soal pilihan ganda per topik yang dikirim ke Tutor Agent.

### 5.5 Tutor Agent

**Model:** Qwen3-14B (via Ollama) — prioritas latensi rendah  
**Mode operasi:**

**Mode 1 — Chat RAG:**
```
User query
    │
    ▼
Query Expansion (HyDE)
    │  LLM generate hypothetical answer
    ▼
Vector Retrieval (Qdrant)
    │  top-8 chunks, filter by user_id + topic_id
    ▼
Re-ranking (FlashRank)
    │  cross-encoder → top-3 chunks
    ▼
Augmented Generation (Qwen3-14B)
    │  prompt = system + context + history + query
    ▼
Response + RAGAS Auto-eval
```

**Mode 2 — Kuis Adaptif:**
- Generate soal dari draft Composer atau buat baru dari chunks Qdrant
- Evaluasi jawaban user → simpan `QuizResult` ke PostgreSQL
- Hitung `mastery_score` per topik → kirim ke Feedback Engine

### 5.6 Adaptive Feedback Engine

Bukan agent LLM — ini adalah **komponen logika** Python yang berjalan setiap akhir sesi belajar.

**Input:** semua sinyal progress dari `ProgressSignals`  
**Output:** `FeedbackAction[]` yang dikirim ke Orchestrator → Planner

**Multi-signal mastery score:**
```python
def calculate_mastery(signals: ProgressSignals) -> float:
    score = (
        signals.quiz_score           * 0.40 +  # skor kuis
        signals.reading_time_ratio   * 0.20 +  # waktu baca vs estimasi
        signals.question_frequency   * 0.20 +  # frekuensi tanya (inverse)
        signals.self_assessment      * 0.15 +  # self-confidence rating
        signals.material_rating      * 0.05    # feedback tombol
    )
    return score
```

**Sinyal progress yang dikumpulkan secara otomatis:**

| Sinyal | Cara Pengumpulan | Bobot |
|--------|-----------------|-------|
| Skor kuis | Sistem catat otomatis | 40% |
| Waktu baca materi | Frontend tracking time-on-page | 20% |
| Frekuensi pertanyaan ke Tutor | Counter per topik | 20% |
| Self-assessment (1–5) | Slider setelah topik selesai | 15% |
| Rating materi (paham/bingung/tidak) | Tombol satu klik | 5% |

---

## 6. Tech Stack

### 6.1 Backend Repository (`pla-backend`)

| Komponen | Teknologi | Versi | Keterangan |
|----------|-----------|-------|------------|
| Framework | FastAPI | ≥0.111 | REST API + WebSocket |
| Agent framework | LangGraph | ≥0.2 | Multi-agent orchestration |
| LLM wrapper | LangChain | ≥0.2 | Tool calling, RAG chains |
| Database ORM | SQLAlchemy + Alembic | ≥2.0 | PostgreSQL migrations |
| Vector DB client | qdrant-client | ≥1.9 | Koneksi ke Qdrant |
| LLM inference | Ollama (lokal) + Groq SDK | latest | Hybrid inference |
| Embedding model | nomic-embed-text v1.5 | via Ollama | 768 dimensi |
| LLM backbone | Qwen3-32B / 14B / 8B | via Ollama | Heterogeneous assignment |
| Re-ranking | FlashRank | ≥0.2 | Cross-encoder re-ranking |
| Web search | Tavily Python SDK | latest | Researcher tool |
| YouTube | youtube-transcript-api | latest | Transcript extraction |
| PDF processing | PyMuPDF (fitz) | latest | PDF text extraction |
| RAG evaluation | RAGAS | ≥0.1 | faithfulness, relevancy, recall |
| Auth | PyJWT + bcrypt | latest | JWT authentication |
| Task queue | Celery + Redis | latest | Background agent tasks |
| Validation | Pydantic v2 | ≥2.0 | Schema validation |
| Testing | pytest + pytest-asyncio | latest | Unit & integration test |

### 6.2 Frontend Repository (`pla-frontend`)

| Komponen | Teknologi | Versi | Keterangan |
|----------|-----------|-------|------------|
| Framework | React | ≥18 | UI library |
| Build tool | Vite | ≥5 | Fast dev server |
| Routing | React Router v6 | ≥6 | SPA routing |
| State management | Zustand | ≥4 | Global state |
| Server state | TanStack Query | ≥5 | API caching & sync |
| Styling | Tailwind CSS | ≥3 | Utility-first CSS |
| UI components | shadcn/ui | latest | Accessible components |
| Charts | Recharts | ≥2 | Dashboard visualisasi |
| Real-time | native WebSocket | — | Agent log streaming |
| HTTP client | Axios | ≥1 | API calls |
| Forms | React Hook Form + Zod | latest | Validasi form |
| Icons | Lucide React | latest | Icon set |
| Code highlight | Prism.js | latest | Syntax highlight di modul |
| Markdown | react-markdown | latest | Render modul materi |

### 6.3 Infrastructure

| Komponen | Teknologi | Keterangan |
|----------|-----------|------------|
| Container | Docker + Docker Compose | Semua service |
| Database | PostgreSQL 16 | Data utama |
| Vector DB | Qdrant | Lokal (Docker) atau cloud |
| Cache / Queue | Redis 7 | Celery broker + cache |
| LLM runtime | Ollama | Model lokal |
| Reverse proxy | Nginx | Production only |

---

## 7. Struktur Repository

### 7.1 `pla-backend`

```
pla-backend/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings (env vars)
│   ├── dependencies.py            # Dependency injection
│   │
│   ├── api/                       # REST API routes
│   │   ├── v1/
│   │   │   ├── auth.py            # POST /auth/register, /auth/login
│   │   │   ├── learning.py        # POST /learning/start, GET /learning/{id}
│   │   │   ├── curriculum.py      # GET /curriculum/{id}
│   │   │   ├── modules.py         # GET /modules/{topic_id}
│   │   │   ├── chat.py            # POST /chat/message (RAG)
│   │   │   ├── quiz.py            # GET /quiz/{topic_id}, POST /quiz/submit
│   │   │   ├── progress.py        # GET /progress/{user_id}
│   │   │   ├── signals.py         # POST /signals (reading time, rating, dll)
│   │   │   └── metrics.py         # GET /metrics/rag, /metrics/agent
│   │   └── ws/
│   │       └── agent_log.py       # WebSocket /ws/agent-log
│   │
│   ├── agents/                    # LangGraph agent definitions
│   │   ├── orchestrator.py        # StateGraph + routing logic
│   │   ├── planner.py             # Curriculum generation
│   │   ├── researcher.py          # Multi-source search
│   │   ├── composer.py            # Module synthesis + embedding
│   │   ├── tutor.py               # RAG chat + quiz
│   │   ├── feedback_engine.py     # Mastery calc + revision logic
│   │   └── state.py               # PLAState TypedDict definition
│   │
│   ├── tools/                     # LangChain tools untuk Researcher
│   │   ├── tavily_search.py
│   │   ├── youtube_transcript.py
│   │   ├── arxiv_search.py
│   │   ├── wikipedia_search.py
│   │   ├── pdf_loader.py
│   │   └── web_scraper.py
│   │
│   ├── rag/                       # RAG pipeline components
│   │   ├── hyde.py                # Hypothetical Document Embedding
│   │   ├── retriever.py           # Qdrant retrieval
│   │   ├── reranker.py            # FlashRank re-ranking
│   │   ├── embedder.py            # nomic-embed-text
│   │   └── evaluator.py           # RAGAS evaluation
│   │
│   ├── models/                    # SQLAlchemy ORM models
│   │   ├── user.py
│   │   ├── learning_session.py
│   │   ├── curriculum.py
│   │   ├── module.py
│   │   ├── quiz_result.py
│   │   ├── progress_signal.py
│   │   ├── chat_message.py
│   │   └── agent_log.py
│   │
│   ├── schemas/                   # Pydantic v2 schemas
│   │   ├── auth.py
│   │   ├── learning.py
│   │   ├── curriculum.py
│   │   ├── module.py
│   │   ├── chat.py
│   │   ├── quiz.py
│   │   ├── progress.py
│   │   └── metrics.py
│   │
│   ├── services/                  # Business logic
│   │   ├── auth_service.py
│   │   ├── learning_service.py
│   │   ├── rag_service.py
│   │   └── metrics_service.py
│   │
│   ├── tasks/                     # Celery background tasks
│   │   ├── celery_app.py
│   │   ├── run_planner.py
│   │   ├── run_researcher.py
│   │   └── run_composer.py
│   │
│   └── db/
│       ├── database.py            # SQLAlchemy engine + session
│       └── migrations/            # Alembic migrations
│
├── tests/
│   ├── test_agents/
│   ├── test_api/
│   ├── test_rag/
│   └── conftest.py
│
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

### 7.2 `pla-frontend`

```
pla-frontend/
├── src/
│   ├── main.jsx                   # Entry point
│   ├── App.jsx                    # Router setup
│   │
│   ├── pages/
│   │   ├── Landing.jsx            # Halaman awal / onboarding
│   │   ├── Auth/
│   │   │   ├── Login.jsx
│   │   │   └── Register.jsx
│   │   ├── Dashboard.jsx          # Statistik & progress overview
│   │   ├── Curriculum.jsx         # Jadwal & daftar topik
│   │   ├── Module.jsx             # Tampilan modul belajar per topik
│   │   ├── Chat.jsx               # Tutor chat (RAG)
│   │   ├── Quiz.jsx               # Kuis interaktif
│   │   ├── AgentLog.jsx           # Real-time agent activity (WebSocket)
│   │   └── Metrics.jsx            # Dashboard metrik evaluasi
│   │
│   ├── components/
│   │   ├── ui/                    # shadcn/ui base components
│   │   ├── layout/
│   │   │   ├── Sidebar.jsx
│   │   │   ├── Topbar.jsx
│   │   │   └── AppLayout.jsx
│   │   ├── dashboard/
│   │   │   ├── StatCard.jsx
│   │   │   ├── ProgressChart.jsx
│   │   │   └── FeedbackBanner.jsx
│   │   ├── curriculum/
│   │   │   ├── WeekView.jsx
│   │   │   ├── TopicRow.jsx
│   │   │   └── AdaptiveBadge.jsx
│   │   ├── module/
│   │   │   ├── ModuleContent.jsx  # Render markdown modul
│   │   │   ├── MaterialRating.jsx # Tombol paham/bingung/tidak
│   │   │   ├── SelfAssessment.jsx # Slider confidence 1-5
│   │   │   └── ReadingTracker.jsx # Track waktu baca (background)
│   │   ├── chat/
│   │   │   ├── ChatBubble.jsx
│   │   │   ├── SourceCitation.jsx
│   │   │   └── ThinkingIndicator.jsx
│   │   ├── quiz/
│   │   │   ├── QuizCard.jsx
│   │   │   ├── OptionButton.jsx
│   │   │   └── QuizFeedback.jsx
│   │   ├── agent-log/
│   │   │   ├── LogLine.jsx
│   │   │   └── AgentBadge.jsx
│   │   └── metrics/
│   │       ├── RAGScoreBar.jsx
│   │       └── UXSurveyChart.jsx
│   │
│   ├── store/                     # Zustand stores
│   │   ├── authStore.js
│   │   ├── learningStore.js
│   │   └── agentLogStore.js
│   │
│   ├── hooks/                     # Custom React hooks
│   │   ├── useAgentLog.js         # WebSocket connection
│   │   ├── useReadingTracker.js   # Auto-track time on page
│   │   └── useProgress.js
│   │
│   ├── api/                       # Axios API calls
│   │   ├── client.js              # Axios instance + interceptors
│   │   ├── auth.js
│   │   ├── learning.js
│   │   ├── chat.js
│   │   ├── quiz.js
│   │   ├── progress.js
│   │   └── metrics.js
│   │
│   └── utils/
│       ├── formatters.js
│       └── constants.js
│
├── public/
├── index.html
├── vite.config.js
├── tailwind.config.js
├── .env.example
├── Dockerfile
└── README.md
```

---

## 8. Spesifikasi API

### 8.1 Authentication

| Method | Endpoint | Deskripsi | Auth |
|--------|----------|-----------|------|
| POST | `/api/v1/auth/register` | Registrasi pengguna baru | — |
| POST | `/api/v1/auth/login` | Login, return JWT token | — |
| GET | `/api/v1/auth/me` | Info pengguna saat ini | JWT |
| POST | `/api/v1/auth/refresh` | Refresh JWT token | JWT |

### 8.2 Learning Session

| Method | Endpoint | Deskripsi | Auth |
|--------|----------|-----------|------|
| POST | `/api/v1/learning/start` | Mulai sesi baru, trigger Orchestrator | JWT |
| GET | `/api/v1/learning/{session_id}` | Detail sesi belajar | JWT |
| GET | `/api/v1/learning/sessions` | Daftar semua sesi user | JWT |
| DELETE | `/api/v1/learning/{session_id}` | Hapus sesi | JWT |

**Request body `POST /learning/start`:**
```json
{
  "topic": "Machine Learning dan Deep Learning",
  "duration_weeks": 4,
  "level": "beginner",
  "hours_per_day": 1,
  "language": "id"
}
```

### 8.3 Curriculum & Modules

| Method | Endpoint | Deskripsi | Auth |
|--------|----------|-----------|------|
| GET | `/api/v1/curriculum/{session_id}` | Kurikulum lengkap + jadwal | JWT |
| GET | `/api/v1/modules/{topic_id}` | Modul belajar per topik | JWT |
| GET | `/api/v1/modules/{topic_id}/status` | Status & progress topik | JWT |
| PATCH | `/api/v1/modules/{topic_id}/complete` | Tandai topik selesai | JWT |

### 8.4 Chat (RAG)

| Method | Endpoint | Deskripsi | Auth |
|--------|----------|-----------|------|
| POST | `/api/v1/chat/message` | Kirim pesan ke Tutor Agent | JWT |
| GET | `/api/v1/chat/history/{topic_id}` | Riwayat chat per topik | JWT |
| DELETE | `/api/v1/chat/history/{topic_id}` | Hapus riwayat | JWT |

**Request body `POST /chat/message`:**
```json
{
  "session_id": "uuid",
  "topic_id": "intro_ml",
  "message": "Apa bedanya overfitting dan variance tinggi?",
  "include_sources": true
}
```

**Response:**
```json
{
  "message_id": "uuid",
  "response": "...",
  "sources": [
    {"title": "Pattern Recognition Ch.3", "type": "pdf", "relevance": 0.92},
    {"title": "StatQuest YouTube", "type": "youtube", "relevance": 0.88}
  ],
  "rag_metrics": {
    "faithfulness": 0.91,
    "answer_relevancy": 0.87,
    "latency_ms": 2100
  }
}
```

### 8.5 Quiz

| Method | Endpoint | Deskripsi | Auth |
|--------|----------|-----------|------|
| GET | `/api/v1/quiz/{topic_id}` | Ambil soal kuis (5–10 soal) | JWT |
| POST | `/api/v1/quiz/submit` | Submit jawaban & dapatkan skor | JWT |
| GET | `/api/v1/quiz/history/{session_id}` | Riwayat skor kuis | JWT |

### 8.6 Progress Signals

| Method | Endpoint | Deskripsi | Auth |
|--------|----------|-----------|------|
| POST | `/api/v1/signals/reading-time` | Kirim durasi baca materi | JWT |
| POST | `/api/v1/signals/material-rating` | Rating materi (paham/bingung) | JWT |
| POST | `/api/v1/signals/self-assessment` | Self-confidence score 1–5 | JWT |
| GET | `/api/v1/progress/{session_id}` | Progress dashboard lengkap | JWT |

### 8.7 Metrics

| Method | Endpoint | Deskripsi | Auth |
|--------|----------|-----------|------|
| GET | `/api/v1/metrics/rag` | RAGAS scores keseluruhan | JWT |
| GET | `/api/v1/metrics/agent` | Efisiensi agent (latency, calls) | JWT |
| GET | `/api/v1/metrics/ux` | Data UX survey | JWT |
| POST | `/api/v1/metrics/ux` | Submit UX survey | JWT |

### 8.8 WebSocket

| Endpoint | Deskripsi |
|----------|-----------|
| `WS /ws/agent-log/{session_id}` | Stream real-time agent activity log |

**Event format:**
```json
{
  "timestamp": "2026-05-18T08:01:04Z",
  "agent": "planner",
  "level": "info",
  "message": "Generating curriculum for 'Machine Learning'...",
  "metadata": {}
}
```

---

## 9. Skema Database

### 9.1 PostgreSQL Tables

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Learning Sessions
CREATE TABLE learning_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    topic VARCHAR(500) NOT NULL,
    level VARCHAR(50) NOT NULL,          -- beginner | intermediate | advanced
    duration_weeks INT NOT NULL,
    hours_per_day FLOAT NOT NULL,
    language VARCHAR(10) DEFAULT 'id',
    status VARCHAR(50) DEFAULT 'active', -- active | paused | completed
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Curriculum
CREATE TABLE curricula (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES learning_sessions(id) ON DELETE CASCADE,
    version INT DEFAULT 1,               -- increment setiap revisi Planner
    curriculum_json JSONB NOT NULL,      -- full curriculum object
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Topics (sub-bab dalam kurikulum)
CREATE TABLE topics (
    id VARCHAR(100) PRIMARY KEY,         -- slug: "intro_ml", "linear_regression"
    session_id UUID REFERENCES learning_sessions(id) ON DELETE CASCADE,
    curriculum_id UUID REFERENCES curricula(id),
    title VARCHAR(500) NOT NULL,
    week_number INT NOT NULL,
    day_number INT NOT NULL,
    duration_minutes INT NOT NULL,
    status VARCHAR(50) DEFAULT 'locked', -- locked | available | in_progress | completed | review
    scheduled_date DATE,
    completed_at TIMESTAMPTZ,
    search_queries JSONB                 -- queries untuk Researcher
);

-- Learning Modules (output Composer)
CREATE TABLE learning_modules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id VARCHAR(100) REFERENCES topics(id),
    session_id UUID REFERENCES learning_sessions(id),
    title VARCHAR(500) NOT NULL,
    content_markdown TEXT NOT NULL,      -- konten modul dalam Markdown
    content_version INT DEFAULT 1,
    sources JSONB,                       -- daftar sumber yang digunakan
    word_count INT,
    estimated_read_minutes INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat Messages
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES learning_sessions(id) ON DELETE CASCADE,
    topic_id VARCHAR(100) REFERENCES topics(id),
    role VARCHAR(20) NOT NULL,           -- user | assistant
    content TEXT NOT NULL,
    sources JSONB,                       -- RAG sources yang digunakan
    rag_faithfulness FLOAT,
    rag_answer_relevancy FLOAT,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Quiz Results
CREATE TABLE quiz_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES learning_sessions(id) ON DELETE CASCADE,
    topic_id VARCHAR(100) REFERENCES topics(id),
    attempt_number INT DEFAULT 1,
    score FLOAT NOT NULL,                -- 0.0 - 1.0
    total_questions INT NOT NULL,
    correct_answers INT NOT NULL,
    answers_detail JSONB,                -- detail per soal
    time_spent_seconds INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Progress Signals (semua sinyal adaptif)
CREATE TABLE progress_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES learning_sessions(id) ON DELETE CASCADE,
    topic_id VARCHAR(100) REFERENCES topics(id),
    signal_type VARCHAR(50) NOT NULL,    -- reading_time | material_rating | self_assessment | question_count
    value FLOAT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Mastery Scores (dihitung Feedback Engine setiap akhir sesi)
CREATE TABLE mastery_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES learning_sessions(id) ON DELETE CASCADE,
    topic_id VARCHAR(100) REFERENCES topics(id),
    mastery_score FLOAT NOT NULL,        -- 0.0 - 1.0 (gabungan semua sinyal)
    quiz_score FLOAT,
    reading_time_ratio FLOAT,
    question_frequency_score FLOAT,
    self_assessment_score FLOAT,
    material_rating_score FLOAT,
    feedback_action VARCHAR(100),        -- aksi yang dipicu: repeat | review | continue | accelerate
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent Logs
CREATE TABLE agent_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES learning_sessions(id) ON DELETE CASCADE,
    agent VARCHAR(50) NOT NULL,          -- orchestrator | planner | researcher | composer | tutor
    level VARCHAR(20) DEFAULT 'info',    -- info | warning | error
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RAG Evaluation Metrics
CREATE TABLE rag_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES chat_messages(id),
    session_id UUID REFERENCES learning_sessions(id),
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_recall FLOAT,
    context_precision FLOAT,
    answer_correctness FLOAT,
    latency_ms INT,
    chunks_retrieved INT,
    chunks_after_rerank INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- UX Survey
CREATE TABLE ux_surveys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES learning_sessions(id),
    user_id UUID REFERENCES users(id),
    ease_of_use FLOAT,                   -- 1-5
    material_relevance FLOAT,
    quiz_quality FLOAT,
    adaptivity_satisfaction FLOAT,
    overall_satisfaction FLOAT,
    open_feedback TEXT,
    submitted_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 9.2 Qdrant Collection

```python
# Collection per user (isolasi data antar pengguna)
collection_name = f"pla_user_{user_id}"

# Vector config
vectors_config = VectorParams(
    size=768,           # nomic-embed-text dimensi
    distance=Distance.COSINE
)

# Payload (metadata) per point
payload = {
    "user_id": str,
    "session_id": str,
    "topic_id": str,
    "week": int,
    "day": int,
    "module_id": str,
    "chunk_index": int,
    "source_type": str,   # web | youtube | arxiv | pdf | wikipedia
    "source_title": str,
    "source_url": str,
    "text": str,          # teks chunk asli
    "created_at": str
}
```

---

## 10. RAG Pipeline

### 10.1 Indexing Pipeline (saat Composer selesai)

```
Raw content dari Researcher
    │
    ▼
Composer synthesize → Learning Module (Markdown)
    │
    ├──► Render ke UI (disimpan ke PostgreSQL: learning_modules)
    │
    └──► Chunking (RecursiveCharacterTextSplitter, 512 token, overlap 50)
              │
              ▼
         Embedding (nomic-embed-text via Ollama)
              │
              ▼
         Qdrant upsert (dengan metadata topik + sumber)
```

### 10.2 Query Pipeline (saat user chat dengan Tutor)

```
User query: "Apa bedanya overfitting dan variance tinggi?"
    │
    ▼
[1] HyDE — Query Expansion
    │  Prompt: "Tulis jawaban singkat yang ideal untuk: {query}"
    │  Model: Qwen3-14B
    │  Output: hypothetical_answer (string)
    │
    ▼
[2] Dual Embedding
    │  embed(query) + embed(hypothetical_answer)
    │  Model: nomic-embed-text
    │
    ▼
[3] Qdrant Retrieval
    │  Filter: user_id = X, topic_id IN [current_topic, related_topics]
    │  top_k = 8
    │
    ▼
[4] FlashRank Re-ranking
    │  Cross-encoder: score setiap chunk terhadap query asli
    │  Output: top_3 chunks
    │
    ▼
[5] Prompt Assembly
    │  system_prompt + context (top_3) + chat_history (last 5) + query
    │
    ▼
[6] Generation
    │  Model: Qwen3-14B
    │  Output: response + source attribution
    │
    ▼
[7] RAGAS Auto-evaluation (async, tidak blok response)
    │  faithfulness, answer_relevancy, context_recall
    └──► Simpan ke rag_metrics table
```

### 10.3 RAGAS Evaluation Setup

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)

metrics = [
    faithfulness,         # apakah jawaban sesuai context?
    answer_relevancy,     # apakah jawaban relevan dengan pertanyaan?
    context_recall,       # apakah context berisi info yang dibutuhkan?
    context_precision,    # apakah context tidak mengandung noise?
    answer_correctness,   # apakah jawaban secara faktual benar?
]
```

---

## 11. Fitur & Requirement Fungsional

### FR-01: Autentikasi
- **FR-01.1** Pengguna dapat mendaftar dengan email, username, dan password
- **FR-01.2** Pengguna dapat login dan menerima JWT token
- **FR-01.3** Token diperbarui otomatis sebelum expired

### FR-02: Inisiasi Belajar
- **FR-02.1** Pengguna dapat memasukkan topik belajar, durasi (minggu), level, dan jam/hari
- **FR-02.2** Sistem memvalidasi input sebelum memicu Orchestrator
- **FR-02.3** Pengguna dapat mengupload dokumen PDF/artikel sebagai sumber tambahan
- **FR-02.4** Sistem menampilkan indikator loading dengan agent activity log saat proses berlangsung

### FR-03: Kurikulum
- **FR-03.1** Planner Agent menghasilkan kurikulum terstruktur per minggu dan hari
- **FR-03.2** Kurikulum menampilkan estimasi durasi per topik
- **FR-03.3** Topik terkunci hingga topik prerequisite diselesaikan
- **FR-03.4** Kurikulum diperbarui otomatis setelah feedback loop berjalan
- **FR-03.5** Revisi kurikulum menyimpan versi sebelumnya

### FR-04: Modul Belajar
- **FR-04.1** Setiap topik memiliki modul belajar yang disintesis Composer dari multi-sumber
- **FR-04.2** Modul menampilkan learning objectives, penjelasan, contoh, ringkasan, latihan, dan referensi
- **FR-04.3** Pengguna dapat memberi rating materi (paham / agak bingung / tidak paham)
- **FR-04.4** Pengguna mengisi self-assessment confidence (slider 1–5) setelah selesai membaca
- **FR-04.5** Sistem mencatat waktu baca otomatis di latar belakang

### FR-05: Chat Tutor (RAG)
- **FR-05.1** Pengguna dapat bertanya kepada Tutor Agent dalam konteks topik yang sedang dipelajari
- **FR-05.2** Sistem menampilkan sumber yang digunakan untuk menjawab
- **FR-05.3** Riwayat chat disimpan per topik dan dapat dilihat kembali
- **FR-05.4** Tutor menampilkan indikator "sedang berpikir" saat RAG pipeline berjalan

### FR-06: Kuis Adaptif
- **FR-06.1** Setiap topik memiliki 5–10 soal pilihan ganda
- **FR-06.2** Pengguna mendapat feedback langsung setelah menjawab
- **FR-06.3** Skor dan waktu per soal dicatat otomatis
- **FR-06.4** Pengguna dapat mengulang kuis (attempt dicatat terpisah)

### FR-07: Adaptive Feedback Loop
- **FR-07.1** Feedback Engine berjalan otomatis setiap user menekan tombol "Selesai Belajar" di akhir sesi belajar
- **FR-07.2** Mastery score dihitung dari 5 sinyal dengan bobot masing-masing
- **FR-07.3** Planner merevisi jadwal berdasarkan mastery score (4 level aksi)
- **FR-07.4** Pengguna menerima notifikasi banner tentang perubahan jadwal

### FR-08: Dashboard Progress
- **FR-08.1** Dashboard menampilkan: topik selesai, rata-rata skor kuis, materi dikurasi, streak belajar
- **FR-08.2** Grafik progress per minggu divisualisasikan
- **FR-08.3** Notifikasi feedback adaptif terakhir ditampilkan

### FR-09: Agent Log
- **FR-09.1** Pengguna dapat melihat aktivitas real-time semua agent via WebSocket
- **FR-09.2** Log menampilkan: timestamp, nama agent, pesan, dan metadata

### FR-10: Metrik Evaluasi
- **FR-10.1** Dashboard metrik menampilkan RAGAS scores (faithfulness, relevancy, recall, precision)
- **FR-10.2** Latensi rata-rata per RAG response ditampilkan
- **FR-10.3** Hasil UX survey dapat diisi dan divisualisasikan

---

## 12. Requirement Non-Fungsional

| ID | Kategori | Requirement | Target |
|----|----------|-------------|--------|
| NFR-01 | Performa | Latency RAG response (P95) | < 10 detik |
| NFR-02 | Performa | Waktu generate kurikulum awal | < 60 detik |
| NFR-03 | Performa | Waktu Composer per topik | < 60 detik |
| NFR-04 | Ketersediaan | Uptime sistem saat demo/evaluasi | > 99% |
| NFR-05 | Keamanan | Semua endpoint dilindungi JWT | Wajib |
| NFR-06 | Keamanan | Isolasi data antar pengguna di Qdrant | Wajib (per-user collection) |
| NFR-07 | Skalabilitas | Mendukung minimal 10 concurrent user | Untuk kebutuhan evaluasi |
| NFR-08 | Kualitas RAG | RAGAS faithfulness score | ≥ 0.80 |
| NFR-09 | Kualitas RAG | RAGAS answer relevancy score | ≥ 0.75 |
| NFR-10 | UX | Skor kepuasan pengguna (1–5) | ≥ 4.0 |
| NFR-11 | Maintainability | Coverage unit test | ≥ 70% |
| NFR-12 | Portabilitas | Semua service dapat dijalankan via Docker Compose | Wajib |

---

## 13. Alur Pengguna (User Flow)

### 13.1 Onboarding (Pertama Kali)

```
Landing Page
    │ Klik "Mulai Belajar"
    ▼
Register / Login
    │
    ▼
Form Learning Goal
    │ - Topik belajar
    │ - Durasi (minggu)
    │ - Level (pemula/menengah/mahir)
    │ - Jam belajar per hari
    │ - Upload dokumen (opsional)
    │
    ▼
Loading Screen (Agent Log streaming)
    │ "Planner sedang merancang kurikulum..."
    │ "Researcher sedang mencari materi..."
    │ "Composer sedang menyusun modul..."
    │
    ▼
Dashboard (kurikulum & modul siap)
```

### 13.2 Sesi Belajar Harian

```
Dashboard → Klik topik hari ini
    │
    ▼
Baca Modul Belajar
    │ (sistem catat waktu baca di background)
    │
    ▼
Rating Materi (paham / agak bingung / tidak paham)
    │
    ▼
Self-assessment confidence (slider 1–5)
    │
    ▼
Tandai "Selesai Baca"
    │
    ├──► [Opsional] Chat dengan Tutor (pertanyaan terkait materi)
    │
    ▼
Kuis (5–10 soal)
    │
    ▼
Lihat skor & feedback per soal
    │
    ▼
Feedback Engine berjalan (otomatis)
    │
    ▼
Dashboard diperbarui + notifikasi revisi jadwal (jika ada)
```

---

## 14. Metrik Evaluasi & KPI Skripsi

### 14.1 Metrik Utama (Bab 5 Skripsi)

#### A. Kualitas RAG (RAGAS Framework)
Dievaluasi otomatis setiap RAG response, disimpan ke tabel `rag_metrics`.

| Metrik | Deskripsi | Target |
|--------|-----------|--------|
| Faithfulness | Apakah jawaban sesuai dengan context yang diambil? | ≥ 0.80 |
| Answer Relevancy | Apakah jawaban relevan dengan pertanyaan? | ≥ 0.75 |
| Context Recall | Apakah context berisi semua info yang dibutuhkan? | ≥ 0.70 |
| Context Precision | Apakah context bebas dari noise yang tidak relevan? | ≥ 0.75 |

#### B. Efisiensi Agent
Diukur dari `agent_logs` dan `rag_metrics`.

| Metrik | Deskripsi | Target |
|--------|-----------|--------|
| RAG latency (P50) | Median waktu respons RAG | < 3 detik |
| RAG latency (P95) | 95th percentile waktu respons | < 5 detik |
| Agent orchestration time | Waktu total Planner + Researcher + Composer | < 120 detik |
| LLM calls per session | Total panggilan LLM per sesi harian | Dicatat dan dianalisis |

#### C. Kepuasan Pengguna (UX Survey)
Kuesioner 5 dimensi (skala Likert 1–5), diberikan setelah minimal 3 sesi belajar.

| Dimensi | Pertanyaan |
|---------|------------|
| Kemudahan penggunaan | Seberapa mudah menggunakan aplikasi ini? |
| Relevansi materi | Seberapa relevan materi yang disajikan? |
| Kualitas kuis | Seberapa baik kuis membantu pemahaman? |
| Adaptivitas | Seberapa tepat sistem menyesuaikan kecepatan belajar? |
| Kepuasan keseluruhan | Seberapa puas secara keseluruhan? |

#### D. Skenario Perbandingan Model (Eksperimen Skripsi)

Jalankan sistem dengan 3 konfigurasi berbeda untuk dibandingkan:

| Setup | Konfigurasi | Tujuan |
|-------|-------------|--------|
| Setup A | Semua agent: Qwen3-8B | Baseline (murah, cepat) |
| Setup B | Heterogeneous (32B/14B/8B) | Rekomendasi utama |
| Setup C | Semua agent: Qwen3-32B | Upper bound (terbaik) |

Bandingkan: kualitas modul Composer, skor RAGAS Tutor, latensi total sistem, dan biaya komputasi.

### 14.2 Dataset Evaluasi

- Minimal **30 sesi chat** untuk RAGAS evaluation
- Minimal **15 responden** untuk UX survey (mahasiswa atau teman sebaya)
- Minimal **3 topik berbeda** untuk variasi eksperimen

---

## 15. Docker Compose & Deployment

### 15.1 `docker-compose.yml` (Backend)

```yaml
version: "3.9"

services:
  # FastAPI Backend
  backend:
    build: ./pla-backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://pla:pla@postgres:5432/pla_db
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379/0
      - OLLAMA_BASE_URL=http://ollama:11434
      - GROQ_API_KEY=${GROQ_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
    depends_on:
      - postgres
      - qdrant
      - redis
      - ollama
    volumes:
      - ./pla-backend:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  # Celery Worker (background agent tasks)
  celery_worker:
    build: ./pla-backend
    command: celery -A app.tasks.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql+asyncpg://pla:pla@postgres:5432/pla_db
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379/0
      - OLLAMA_BASE_URL=http://ollama:11434
      - GROQ_API_KEY=${GROQ_API_KEY}
    depends_on:
      - backend
      - redis

  # PostgreSQL
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=pla
      - POSTGRES_PASSWORD=pla
      - POSTGRES_DB=pla_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  # Redis (Celery broker + cache)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Ollama (LLM runtime lokal)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  postgres_data:
  qdrant_data:
  redis_data:
  ollama_data:
```

### 15.2 `docker-compose.yml` (Frontend)

```yaml
version: "3.9"

services:
  frontend:
    build: ./pla-frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_BASE_URL=http://localhost:8000/api/v1
      - VITE_WS_URL=ws://localhost:8000/ws
    volumes:
      - ./pla-frontend:/app
      - /app/node_modules
    command: npm run dev -- --host
```

### 15.3 Perintah Setup Awal

```bash
# Pull model Ollama yang dibutuhkan
docker exec -it pla-backend-ollama-1 ollama pull qwen3:32b
docker exec -it pla-backend-ollama-1 ollama pull qwen3:14b
docker exec -it pla-backend-ollama-1 ollama pull qwen3:8b
docker exec -it pla-backend-ollama-1 ollama pull nomic-embed-text

# Jalankan migrasi database
docker exec -it pla-backend-backend-1 alembic upgrade head

# Jalankan semua service
docker-compose up -d
```

### 15.4 Environment Variables

**`pla-backend/.env`:**
```env
# Database
DATABASE_URL=postgresql+asyncpg://pla:pla@postgres:5432/pla_db

# Vector DB
QDRANT_URL=http://qdrant:6333

# Cache & Queue
REDIS_URL=redis://redis:6379/0

# LLM Inference
OLLAMA_BASE_URL=http://ollama:11434
GROQ_API_KEY=your_groq_api_key_here

# Auth
SECRET_KEY=your_very_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# External APIs
TAVILY_API_KEY=your_tavily_api_key_here

# Model Assignment
ORCHESTRATOR_MODEL=qwen3:32b
PLANNER_MODEL=qwen3:14b
RESEARCHER_MODEL=qwen3:8b
COMPOSER_MODEL=qwen3:32b
TUTOR_MODEL=qwen3:14b
EMBEDDING_MODEL=nomic-embed-text

# Fallback to Groq if Ollama slow
USE_GROQ_FALLBACK=true
GROQ_FALLBACK_MODEL=llama-3.3-70b-versatile
```

**`pla-frontend/.env`:**
```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
```

---

## 16. Milestone & Timeline

| Fase | Durasi | Deliverable |
|------|--------|-------------|
| **Fase 0** — Setup & fondasi | Minggu 1–2 | Repo setup, Docker Compose jalan, DB schema, auth API |
| **Fase 1** — Agent core | Minggu 3–5 | LangGraph StateGraph, Planner + Researcher + Composer Agent |
| **Fase 2** — RAG pipeline | Minggu 6–7 | Indexing pipeline, HyDE, retrieval, FlashRank, Tutor Agent |
| **Fase 3** — Feedback & adaptasi | Minggu 8–9 | Adaptive Feedback Engine, progress signals, revisi jadwal |
| **Fase 4** — Frontend | Minggu 10–12 | Semua halaman React, WebSocket agent log, dashboard |
| **Fase 5** — Evaluasi | Minggu 13–14 | RAGAS benchmark, UX survey (n≥15), 3 setup eksperimen |
| **Fase 6** — Penulisan skripsi | Minggu 15–16 | Bab 4 (implementasi) & Bab 5 (evaluasi) |

---

## 17. Risiko & Mitigasi

| Risiko | Probabilitas | Dampak | Mitigasi |
|--------|-------------|--------|----------|
| Ollama terlalu lambat di hardware terbatas | Tinggi | Tinggi | Gunakan Groq sebagai fallback otomatis |
| Kualitas output Composer tidak konsisten | Sedang | Tinggi | Prompt engineering ketat + schema validation Pydantic |
| RAGAS score rendah karena retrieval buruk | Sedang | Tinggi | Tuning chunk size, overlap, dan top-k retrieval |
| Researcher gagal akses sumber tertentu | Sedang | Sedang | Graceful degradation: lanjut dengan sumber yang tersedia |
| Token API Groq habis | Sedang | Sedang | Rate limiting + monitor usage + switch ke Ollama |
| Waktu eksekusi Composer terlalu lama | Sedang | Sedang | Jalankan sebagai Celery background task |
| Data pengguna tercampur di Qdrant | Rendah | Tinggi | Isolasi per-user collection + filter wajib di setiap query |

---

## 18. Glosarium

| Istilah | Definisi |
|---------|----------|
| **PLA** | Personal Learning Agent — nama sistem |
| **Agent** | Komponen AI otonom yang dapat mengambil keputusan dan menggunakan tools |
| **Orchestrator** | Agent koordinator yang mengatur alur kerja semua subagent |
| **LangGraph** | Framework Python untuk membangun multi-agent stateful workflow |
| **RAG** | Retrieval-Augmented Generation — teknik menggabungkan retrieval dokumen dengan generasi LLM |
| **HyDE** | Hypothetical Document Embedding — teknik query expansion untuk meningkatkan kualitas retrieval |
| **Re-ranking** | Proses menilai ulang hasil retrieval menggunakan cross-encoder model |
| **FlashRank** | Library Python untuk re-ranking dokumen yang efisien |
| **RAGAS** | Framework evaluasi kualitas RAG system |
| **Qdrant** | Vector database untuk menyimpan dan mencari embedding |
| **ChromaDB** | Alternatif vector database yang lebih sederhana |
| **Mastery Score** | Skor gabungan dari semua sinyal progress yang menggambarkan penguasaan topik |
| **Feedback Loop** | Mekanisme umpan balik dari hasil evaluasi ke revisi kurikulum |
| **Celery** | Framework task queue Python untuk menjalankan tugas di background |
| **Chunk** | Potongan teks kecil hasil splitting dokumen untuk di-embed ke vector store |
| **Embedding** | Representasi vektor numerik dari teks, digunakan untuk pencarian semantik |

---

*Dokumen ini merupakan living document yang akan diperbarui seiring perkembangan implementasi.*

**Versi:** 1.0.0 | **Terakhir diperbarui:** Mei 2026
