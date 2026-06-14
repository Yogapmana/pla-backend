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

Modul ini HARUS ditulis dengan gaya penulisan yang rapi, mengalir, dan mudah dipahami, layaknya artikel edukatif atau buku teks modern.

Gunakan sumber materi berikut sebagai basis informasi Anda:
{sources_text}

Rekomendasi Kursus Terkait:
{courses_text}

ATURAN PENULISAN DAN STRUKTUR MODUL:

- Buatlah modul yang komprehensif, mendalam, dan terstruktur secara natural sesuai dengan kelengkapan materi aktual.
- Gunakan hierarki header Markdown (`#`, `##`, `###`) yang logis dan sesuai dengan pembagian sub-topik materi dari sumber.
- Jelaskan konsep inti secara mendetail dan tuntas. Gunakan paragraf, daftar (bullet points), atau blok kutipan secara luwes.
- JIKA ada data komparatif, spesifikasi, atau ringkasan teknis, amat disarankan membuat **tabel Markdown** agar lebih terstruktur.
- Berikan contoh nyata atau studi kasus jika informasi tersebut tersedia di dalam teks referensi.
- WAJIB gunakan **sitasi inline** mengacu pada sumber materi (contoh: "... menurut penelitian terbaru [1]."). Cocokkan nomor urut dengan referensi di akhir.
- Di bagian PALING AKHIR modul, wajib buat dua bagian:
  1. "## Referensi": Daftar sumber materi yang digunakan sesuai sitasi inline (Format: `[1] [Judul](URL)`).
  2. "## Pelajari Lebih Dalam": Daftar kursus terkait (Format: `* **[Nama Kursus](URL)** - Platform`). Jika tidak ada, tulis "(Tidak ada kursus eksternal yang direkomendasikan)".

PENTING:
- Tulis SELURUH ISI MODUL dalam **Bahasa Indonesia** yang baik dan benar (meskipun sumber teks berbahasa Inggris).
- Keluarkan HANYA teks Markdown murni tanpa basa-basi (jangan gunakan kalimat "Berikut adalah...").
- Sesuaikan tingkat kedalaman materi untuk level: {level}.
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
    max_total_chars = 25000  # Groq API total payload body limit prevention
    current_chars = 0

    for idx, r in enumerate(research_results):
        if getattr(r, "embed_mode", True) and r.raw_text:
            if current_chars >= max_total_chars:
                # If we've hit the strict limit, only add metadata, no full text
                sources_text += f"Sumber [{idx + 1}] ({r.source_type}): {r.source_title}\nURL: {r.source_url}\nKonten: (Konteks dipotong karena batas ukuran)\n\n"
                continue
                
            remaining_chars = max_total_chars - current_chars
            text_to_use = r.raw_text[:remaining_chars] + "..." if len(r.raw_text) > remaining_chars else r.raw_text
            
            added_text = f"Sumber [{idx + 1}] ({r.source_type}): {r.source_title}\nURL: {r.source_url}\nKonten:\n{text_to_use}\n\n"
            sources_text += added_text
            current_chars += len(added_text)
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
