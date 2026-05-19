import asyncio
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.agents.state import PLAState, LearningModule, AgentLog
from app.config import settings
from app.utils.llm_factory import get_llm

COMPOSER_PROMPT = """
Anda adalah Composer Agent dalam sistem Personal Learning Agent.
Tugas Anda adalah menyintesis berbagai bahan mentah dari internet menjadi Modul Belajar Markdown yang terstruktur dan mudah dipahami, khusus untuk topik: {topic_title}.

Berikut adalah sumber materi yang dikumpulkan oleh Researcher Agent:
{sources_text}

Aturan Penulisan Modul Markdown:
1. Mulai dengan Judul Topik (# [Judul Topik]).
2. Buat bagian "## 🎯 Learning Objectives" (poin-poin apa yang akan dicapai).
3. Buat bagian "## 📖 Penjelasan Konsep" (narasi utama, gunakan analogi jika sesuai untuk level {level}).
4. Buat bagian "## 💡 Contoh Konkret" (contoh kasus atau kode jika relevan).
5. Buat bagian "## 🔁 Ringkasan" (tabel atau poin-poin kunci).
6. Buat bagian "## 🧪 Latihan Mandiri" (1-2 soal refleksi singkat).
7. Buat bagian "## 📚 Referensi Sumber" (cantumkan URL sumber yang digunakan dari data di atas).

Tulis hanya konten Markdown tanpa tambahan kata-kata pembuka atau penutup dari Anda.
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
        message=f"Composing learning module for '{topic_title}' using {len(research_results)} sources..."
    )
    state["agent_logs"].append(log)

    # Format sources
    sources_text = ""
    for idx, r in enumerate(research_results):
        # Limit text length to avoid token limit issues
        content_snippet = r.raw_text[:2000] + "..." if len(r.raw_text) > 2000 else r.raw_text
        sources_text += f"Sumber {idx+1} ({r.source_type}): {r.source_title}\nURL: {r.source_url}\nKonten: {content_snippet}\n\n"

    llm = get_llm(settings.COMPOSER_MODEL, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", COMPOSER_PROMPT),
        ("human", "Tolong buatkan modul markdown berdasarkan sumber di atas.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        module_markdown = chain.invoke({
            "topic_title": topic_title,
            "level": config.level if config else "umum",
            "sources_text": sources_text
        })
        
        module = LearningModule(
            topic_id=target_topic_id,
            title=topic_title,
            content_markdown=module_markdown,
            sources=[{"title": r.source_title, "url": r.source_url} for r in research_results]
        )
        
        if "modules" not in state or state["modules"] is None:
            state["modules"] = []
            
        state["modules"].append(module)
        
        success_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="composer",
            level="info",
            message=f"Successfully composed learning module for '{topic_title}'."
        )
        state["agent_logs"].append(success_log)

        # --- Trigger RAG Indexing (fire-and-forget) ---
        # Import here to avoid circular imports at module load time
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

        async def _run_indexing():
            try:
                count = await index_module(
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
                state["agent_logs"].append(AgentLog(
                    timestamp=datetime.utcnow(),
                    agent="composer",
                    level="info",
                    message=f"Indexed {count} chunks into Qdrant for topic '{topic_title}'."
                ))
            except Exception as e:
                state["agent_logs"].append(AgentLog(
                    timestamp=datetime.utcnow(),
                    agent="composer",
                    level="warning",
                    message=f"RAG indexing skipped (Qdrant/Ollama unavailable): {str(e)}"
                ))

        # Schedule indexing as a background task; don't block graph execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_run_indexing())
            else:
                loop.run_until_complete(_run_indexing())
        except RuntimeError:
            pass  # No event loop — skip indexing in sync context
        
        # Mark topic as completed or in_progress in curriculum (optional state mutation)
        for week in curriculum.weeks:
            for day in week.days:
                if day.topic_id == target_topic_id:
                    day.status = "in_progress" # Module is ready, user can start learning
                    
    except Exception as e:
        error_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="composer",
            level="error",
            message=f"Error composing module: {str(e)}"
        )
        state["agent_logs"].append(error_log)
        
    return state
