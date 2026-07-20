from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import SynapsaState, Curriculum, AgentLog
from app.config import settings
from app.utils.llm_factory import get_llm
import uuid
import logging

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """
Anda adalah AI Planner tingkat lanjut dalam sistem Personal Learning Agent.
Tugas Anda adalah memecah topik pembelajaran secara adaptif menjadi sebuah kurikulum terstruktur.

Informasi Pembelajaran Pengguna:
- Topik: {topic}
- Durasi: {duration_weeks} minggu
- Level: {level}
- Jam per hari: {hours_per_day} jam
- Bahasa: {language}

{context_text_instruction}

Aturan:
1. Bagi topik ini secara bertahap dari konsep dasar hingga konsep lanjutan (sesuai level).
2. Tentukan jadwal belajar untuk setiap minggu dan hari (misal jika ada 5 hari belajar efektif seminggu, bagi materinya).
3. Buatlah list `search_queries` yang relevan untuk setiap sub-topik agar Researcher Agent dapat mengumpulkan materi. Query sebaiknya dalam bahasa Inggris jika topik umum di IT/Science agar materi lebih kaya, tapi sesuaikan dengan konteks.
4. Estimasi durasi dalam menit per topik harian (total durasi harian tidak boleh melebihi {hours_per_day} jam, atau 60 * {hours_per_day} menit).
5. KEMBALIKAN OUTPUT DALAM FORMAT JSON SESUAI DENGAN SKEMA YANG DIMINTA.
6. PENTING: KUNCI JSON (JSON KEYS) HARUS SAMA PERSIS DENGAN SKEMA. JANGAN TERJEMAHKAN KUNCI JSON.
7. KEWAJIBAN MUTLAK: Anda HARUS meng-generate array `weeks` TEPAT SEBANYAK {duration_weeks} minggu. Jika durasi adalah 12 minggu, maka harus ada 12 elemen di dalam `weeks`. Jangan pernah mengurangi atau meringkas jumlah minggu!
8. SEMUA TEKS KONTEN (judul, deskripsi, dll kecuali search_queries dan keys JSON) WAJIB DITULIS DALAM BAHASA: {language}.
Anda WAJIB mengikuti format JSON persis seperti contoh ini (jangan kurangi atau ubah key-nya):
```json
{{
  "curriculum_id": "string acak atau format ID",
  "topic": "{topic}",
  "total_weeks": angka,
  "weeks": [
    {{
      "week": 1,
      "title": "Judul Minggu",
      "days": [
        {{
          "day": 1,
          "topic_id": "ID Topik",
          "title": "Judul Topik Harian",
          "description": "Deskripsi topik",
          "search_queries": ["query pencarian inggris"],
          "duration_minutes": angka,
          "status": "pending"
        }}
      ]
    }}
  ]
}}
```
"""

def planner_node(state: SynapsaState) -> SynapsaState:
    config = state["learning_config"]
    
    # Log the activity
    log = AgentLog(
        timestamp=datetime.utcnow(),
        agent="planner",
        level="info",
        message=f"Generating curriculum for '{config.topic}' (Level: {config.level})..."
    )
    
    if "agent_logs" not in state or state["agent_logs"] is None:
        state["agent_logs"] = []
    state["agent_logs"].append(log)

    # Initialize LLM with structured output and large token limit for long curriculums
    llm = get_llm(settings.PLANNER_MODEL, temperature=0.2, max_tokens=4000)
    from langchain_groq import ChatGroq
    if isinstance(llm, ChatGroq):
        structured_llm = llm.with_structured_output(Curriculum, method="json_mode")
    else:
        structured_llm = llm.with_structured_output(Curriculum)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_PROMPT),
        ("human", "Tolong buatkan kurikulum untuk saya berdasarkan spesifikasi di atas.")
    ])
    
    chain = prompt | structured_llm
    
    context_text_instruction = ""
    if config.context_text:
        context_text_instruction = f"\nDokumen Referensi / Roadmap dari User:\n{config.context_text}\n\nPENTING: Gunakan dokumen referensi/roadmap di atas sebagai acuan utama dalam menyusun urutan, struktur, dan cakupan topik kurikulum.\n"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            curriculum: Curriculum = chain.invoke({
                "topic": config.topic,
                "duration_weeks": config.duration_weeks,
                "level": config.level,
                "hours_per_day": config.hours_per_day,
                "language": config.language,
                "context_text_instruction": context_text_instruction
            })
            
            # Override curriculum_id just in case
            if not curriculum.curriculum_id or curriculum.curriculum_id == "uuid":
                curriculum.curriculum_id = str(uuid.uuid4())
                
            # Make topic_ids globally unique and immune to LLM hallucination collisions
            short_session = str(state["session_id"]).split("-")[0]
            for week in curriculum.weeks:
                for day in week.days:
                    day.topic_id = f"{short_session}-W{week.week}D{day.day}"
                
            state["curriculum"] = curriculum
            
            success_log = AgentLog(
                timestamp=datetime.utcnow(),
                agent="planner",
                level="info",
                message=f"Successfully generated curriculum with {curriculum.total_weeks} weeks."
            )
            state["agent_logs"].append(success_log)
            break  # Success, exit retry loop
            
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                error_msg += f" - Response: {e.response.text}"
            elif hasattr(e, 'body'):
                error_msg += f" - Body: {e.body}"
            
            logger.warning(f"[PLANNER] Attempt {attempt + 1} failed: {error_msg}")
            
            if attempt == max_retries - 1:
                error_log = AgentLog(
                    timestamp=datetime.utcnow(),
                    agent="planner",
                    level="error",
                    message=f"Error generating curriculum after {max_retries} attempts: {error_msg}"
                )
                state["agent_logs"].append(error_log)
                import traceback
                traceback.print_exc()

    return state


REPLAN_PROMPT = """
Anda adalah AI Planner tingkat lanjut dalam sistem Personal Learning Agent.
Tugas Anda adalah membuat SATU TOPIK BARU (modul pembelajaran tunggal) untuk disisipkan ke dalam kurikulum pembelajaran pengguna berdasarkan evaluasi Mastery Score.

Topik Utama Kurikulum: {topic}
Konteks Kesalahan Kuis: {context}

Topik Terakhir yang diselesaikan (Target ID: {target_topic_id}):
{current_topic_json}

Tindakan yang harus dilakukan (Action): {action}
Aturan:
1. Jika action "remedial": Buatlah topik "Micro-Remedial" berdurasi 15-20 menit yang HANYA berfokus untuk membahas dan memperbaiki konsep yang salah berdasarkan 'Konteks Kesalahan Kuis'.
2. Jika action "enrichment": Buatlah topik "Deep Dive / Pengayaan" berdurasi 20-30 menit berupa studi kasus teknis atau materi tingkat lanjut lanjutan dari topik terakhir.
3. KEMBALIKAN OUTPUT DALAM FORMAT JSON SESUAI DENGAN SKEMA.
4. SEMUA TEKS KONTEN (judul, deskripsi) WAJIB DALAM BAHASA: {language}.

Anda WAJIB mengikuti format JSON persis seperti contoh ini:
```json
{{
  "day": 0,
  "topic_id": "ID_Topik_Baru_Unik",
  "title": "Judul Topik Harian",
  "description": "Deskripsi mendetail mengenai topik ini",
  "search_queries": ["query pencarian spesifik berbahasa inggris"],
  "duration_minutes": 20,
  "status": "pending"
}}
```
"""

def replan_node(state: SynapsaState) -> SynapsaState:
    """Node untuk merevisi kurikulum berdasarkan feedback action."""
    config = state.get("learning_config")
    current_curriculum = state.get("curriculum")
    feedback_actions = state.get("feedback_actions", [])
    
    if not current_curriculum or not feedback_actions:
        return state
        
    # Ambil feedback action terbaru
    latest_feedback = feedback_actions[-1]
    
    log = AgentLog(
        timestamp=datetime.utcnow(),
        agent="planner",
        level="info",
        message=f"Generating new {latest_feedback.action} module after topic '{latest_feedback.topic_id}'..."
    )
    
    if "agent_logs" not in state or state["agent_logs"] is None:
        state["agent_logs"] = []
    state["agent_logs"].append(log)

    if latest_feedback.action not in ("remedial", "enrichment"):
        return state

    # Cari topik saat ini
    current_topic_json = ""
    week_idx = -1
    day_idx = -1
    for w_i, week in enumerate(current_curriculum.weeks):
        for d_i, day in enumerate(week.days):
            if day.topic_id == latest_feedback.topic_id:
                current_topic_json = day.model_dump_json(indent=2)
                week_idx = w_i
                day_idx = d_i
                break
        if week_idx != -1:
            break
            
    if week_idx == -1:
        return state

    llm = get_llm(settings.PLANNER_MODEL, temperature=0.3)
    from langchain_groq import ChatGroq
    from app.agents.state import DaySchedule
    if isinstance(llm, ChatGroq):
        structured_llm = llm.with_structured_output(DaySchedule, method="json_mode")
    else:
        structured_llm = llm.with_structured_output(DaySchedule)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", REPLAN_PROMPT),
        ("human", "Tolong buat satu topik baru sesuai instruksi.")
    ])
    
    chain = prompt | structured_llm
    
    try:
        new_day: DaySchedule = chain.invoke({
            "topic": config.topic if config else current_curriculum.topic,
            "level": config.level if config else "umum",
            "language": config.language if config else "id",
            "current_topic_json": current_topic_json,
            "target_topic_id": latest_feedback.topic_id,
            "action": latest_feedback.action,
            "context": getattr(latest_feedback, "context", None) or "Tidak ada konteks spesifik."
        })
        
        # Make topic_ids globally unique to avoid Postgres PK collisions
        short_session = str(state["session_id"]).split("-")[0]
        if short_session not in new_day.topic_id:
            new_day.topic_id = f"{short_session}-{new_day.topic_id}"
            
        # Sisipkan topik baru tepat setelah topik saat ini
        current_curriculum.weeks[week_idx].days.insert(day_idx + 1, new_day)
        
        # Perbarui urutan 'day' secara sekuensial agar tidak membingungkan frontend
        day_counter = 1
        for d in current_curriculum.weeks[week_idx].days:
            d.day = day_counter
            day_counter += 1
                    
        state["curriculum"] = current_curriculum
        
        success_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="planner",
            level="info",
            message=f"Successfully inserted {latest_feedback.action} module."
        )
        state["agent_logs"].append(success_log)
        
    except Exception as e:
        error_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="planner",
            level="error",
            message=f"Error generating module: {str(e)}"
        )
        state["agent_logs"].append(error_log)
        
    return state
