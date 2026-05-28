import asyncio
from datetime import datetime

from app.agents.state import AgentLog, LearningModule, PLAState
from app.config import settings
from app.utils.llm_factory import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

COMPOSER_PROMPT = """
Anda adalah seorang Educator ahli di sistem Personal Learning Agent.
Tugas Anda adalah menulis sebuah Modul Belajar Markdown tentang topik: {topic_title}.

Modul ini HARUS ditulis dengan gaya penulisan artikel edukatif yang rapi, mengalir, dan langsung pada intinya (seperti artikel blog teknologi). Setiap penjelasan konsep wajib disertai dengan ilustrasi visual agar pembaca mudah memahami materi.

Gunakan sumber materi berikut sebagai basis informasi Anda:
{sources_text}

Rekomendasi Kursus Terkait:
{courses_text}

ATURAN PENULISAN DAN STRUKTUR MODUL:

1. Pendahuluan
   Tulis 2-3 paragraf pengantar yang menjelaskan gambaran umum topik ini dan mengapa hal ini penting dalam penerapannya sehari-hari atau di industri.

2. Pembahasan Konsep Utama (Alur: Teks -> Gambar -> Teks)
   Pecah materi utama menjadi beberapa sub-topik (gunakan ## atau ###).
   Untuk SETIAP sub-topik materi, WAJIB ikuti struktur berurutan ini:
   - Paragraf 1: Penjelasan dasar/definisi dari sub-topik tersebut.
   - GAMBAR ILUSTRASI: Sisipkan 1 gambar menggunakan format Markdown berikut:
     `![Deskripsi Gambar](https://gen.pollinations.ai/image/{{keyword}}?model={image_model}&width={image_width}&height={image_height}&key={api_key})`
     ATURAN KETAT GAMBAR:
     - Ganti `{{keyword}}` dengan kata kunci BAHASA INGGRIS yang SPESIFIK dan RELEVAN dengan sub-topik.
     - DILARANG KERAS menggunakan spasi - GANTI DENGAN GARIS BAWAH (_).
     - Contoh benar: `product_roadmap_strategy`, `ai_machine_learning_diagram`
     - Contoh salah: `product roadmap`, `ai machine learning`
   - Paragraf 2 dst: Lanjutkan dengan penjelasan lebih dalam, seperti jenis algoritma/cara kerja/contoh penerapannya.

3. Kesimpulan
   Tutup dengan paragraf kesimpulan ringkas yang menyimpulkan seluruh materi di atas.

4. Referensi & Pembelajaran Lanjutan
   - Tulis "## Referensi" dan buat daftar bullet point dari sumber materi di atas (Format: `* [Judul](URL)`).
   - Tulis "## Pelajari Lebih Dalam" dan buat daftar kursus dari data di atas (Format: `* **[Nama Kursus](URL)** - Platform`). Jika tidak ada, tulis "(Tidak ada kursus eksternal yang direkomendasikan)".

PENTING: Keluarkan HANYA teks Markdown murni. Dilarang keras menambahkan kalimat basa-basi di awal (seperti "Berikut adalah modulnya...") atau di akhir. Sesuaikan kedalaman materi untuk level: {level}.
"""


def composer_node(state: PLAState) -> PLAState:
    curriculum = state.get("curriculum")
    research_results = state.get("research_results", [])
    config = state.get("learning_config")

    if not curriculum or not research_results:
        return state

    if "agent_logs" not in state or state["agent_logs"] is None:
        state["agent_logs"] = []

    # Identify the topic we just researched. We assume it's the one from the research results.
    # Group results by topic_id. For now we just take the first topic found in results.
    topic_ids = list(set([r.topic_id for r in research_results]))
    if not topic_ids:
        return state

    target_topic_id = topic_ids[0]

    # Get topic title from curriculum
    topic_title = target_topic_id
    for week in curriculum.weeks:
        for day in week.days:
            if day.topic_id == target_topic_id:
                topic_title = day.title
                break

    log = AgentLog(
        timestamp=datetime.utcnow(),
        agent="composer",
        level="info",
        message=f"Composing learning module for '{topic_title}' using {len(research_results)} sources...",
    )
    state["agent_logs"].append(log)

    # Format sources — separate embeddable content from course links
    sources_text = ""
    courses_text = ""
    for idx, r in enumerate(research_results):
        if getattr(r, "embed_mode", True) and r.raw_text:
            content_snippet = (
                r.raw_text[:2000] + "..." if len(r.raw_text) > 2000 else r.raw_text
            )
            sources_text += f"Sumber {idx + 1} ({r.source_type}): {r.source_title}\nURL: {r.source_url}\nKonten: {content_snippet}\n\n"
        elif not getattr(r, "embed_mode", True) and getattr(r, "course_metadata", None):
            cm = r.course_metadata
            courses_text += f"Kursus: {cm.title}\nPlatform: {cm.platform}\nURL: {cm.url}\nHarga: {cm.price_type}\nDeskripsi: {cm.description}\n\n"

    if not courses_text:
        courses_text = "(Tidak ada rekomendasi kursus yang ditemukan untuk topik ini.)"

    llm = get_llm(settings.COMPOSER_MODEL, temperature=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COMPOSER_PROMPT),
            ("human", "Tolong buatkan modul markdown berdasarkan sumber di atas."),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        module_markdown = chain.invoke(
            {
                "topic_title": topic_title,
                "level": config.level if config else "umum",
                "sources_text": sources_text,
                "courses_text": courses_text,
                "image_model": settings.IMAGE_MODEL,
                "image_width": settings.IMAGE_WIDTH,
                "image_height": settings.IMAGE_HEIGHT,
                "api_key": settings.POLLINATIONS_API_KEY,
            }
        )

        module = LearningModule(
            topic_id=target_topic_id,
            title=topic_title,
            content_markdown=module_markdown,
            sources=[
                {"title": r.source_title, "url": r.source_url} for r in research_results
            ],
        )

        if "modules" not in state or state["modules"] is None:
            state["modules"] = []

        state["modules"].append(module)

        success_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="composer",
            level="info",
            message=f"Successfully composed learning module for '{topic_title}'.",
        )
        state["agent_logs"].append(success_log)

        from app.rag.indexer import index_module

        user_id = state.get("user_id", "anonymous")
        session_id = str(state.get("session_id", ""))

        # Find week/day for metadata
        week_num, day_num = 1, 1
        for w_idx, week in enumerate(curriculum.weeks, 1):
            for d_idx, day in enumerate(week.days, 1):
                if day.topic_id == target_topic_id:
                    week_num, day_num = w_idx, d_idx
                    break

        try:
            count = index_module(
                user_id=user_id,
                session_id=session_id,
                topic_id=target_topic_id,
                module_id=target_topic_id,
                title=topic_title,
                content_markdown=module_markdown,
                week=week_num,
                day=day_num,
                sources=module.sources,
            )
            state["agent_logs"].append(
                AgentLog(
                    timestamp=datetime.utcnow(),
                    agent="composer",
                    level="info",
                    message=f"Indexed {count} chunks into Qdrant for topic '{topic_title}'.",
                )
            )
        except Exception as e:
            state["agent_logs"].append(
                AgentLog(
                    timestamp=datetime.utcnow(),
                    agent="composer",
                    level="warning",
                    message=f"RAG indexing skipped (Qdrant/Ollama unavailable): {str(e)}",
                )
            )

        # Mark topic as completed or in_progress in curriculum (optional state mutation)
        for week in curriculum.weeks:
            for day in week.days:
                if day.topic_id == target_topic_id:
                    day.status = "active"  # Module is ready, user can start learning

    except Exception as e:
        error_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="composer",
            level="error",
            message=f"Error composing module: {str(e)}",
        )
        state["agent_logs"].append(error_log)

    return state
