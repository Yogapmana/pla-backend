from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import PLAState, Curriculum, AgentLog
from app.config import settings
from app.utils.llm_factory import get_llm
import uuid

PLANNER_PROMPT = """
Anda adalah AI Planner tingkat lanjut dalam sistem Personal Learning Agent.
Tugas Anda adalah memecah topik pembelajaran secara adaptif menjadi sebuah kurikulum terstruktur.

Informasi Pembelajaran Pengguna:
- Topik: {topic}
- Durasi: {duration_weeks} minggu
- Level: {level}
- Jam per hari: {hours_per_day} jam
- Bahasa: {language}

Aturan:
1. Bagi topik ini secara bertahap dari konsep dasar hingga konsep lanjutan (sesuai level).
2. Tentukan jadwal belajar untuk setiap minggu dan hari (misal jika ada 5 hari belajar efektif seminggu, bagi materinya).
3. Buatlah list `search_queries` yang relevan untuk setiap sub-topik agar Researcher Agent dapat mengumpulkan materi. Query sebaiknya dalam bahasa Inggris jika topik umum di IT/Science agar materi lebih kaya, tapi sesuaikan dengan konteks.
4. Estimasi durasi dalam menit per topik harian (total durasi harian tidak boleh melebihi {hours_per_day} jam, atau 60 * {hours_per_day} menit).
5. KEMBALIKAN OUTPUT DALAM FORMAT JSON SESUAI DENGAN SKEMA YANG DIMINTA.
6. PENTING: KUNCI JSON (JSON KEYS) HARUS SAMA PERSIS DENGAN SKEMA. JANGAN TERJEMAHKAN KUNCI JSON KE DALAM BAHASA INDONESIA.
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

def planner_node(state: PLAState) -> PLAState:
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

    # Initialize LLM with structured output
    llm = get_llm(settings.PLANNER_MODEL, temperature=0.2)
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
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            curriculum: Curriculum = chain.invoke({
                "topic": config.topic,
                "duration_weeks": config.duration_weeks,
                "level": config.level,
                "hours_per_day": config.hours_per_day,
                "language": config.language
            })
            
            # Override curriculum_id just in case
            if not curriculum.curriculum_id or curriculum.curriculum_id == "uuid":
                curriculum.curriculum_id = str(uuid.uuid4())
                
            # Make topic_ids globally unique to avoid Postgres PK collisions
            short_session = str(state["session_id"]).split("-")[0]
            for week in curriculum.weeks:
                for day in week.days:
                    if short_session not in day.topic_id:
                        day.topic_id = f"{short_session}-{day.topic_id}"
                
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
Tugas Anda adalah MEREVISI kurikulum pembelajaran yang sudah ada berdasarkan umpan balik (feedback) dari performa pengguna.

Informasi Pembelajaran Pengguna:
- Topik Utama: {topic}
- Level: {level}

Kurikulum Saat Ini (JSON):
{current_curriculum}

Feedback Action yang diterima untuk Topik ID [{target_topic_id}]: {action}

Aturan Revisi berdasarkan Action:
1. Jika "repeat": Pengguna kesulitan. Tambahkan hari/topik baru setelah topik ini untuk mengulang materi dengan pendekatan yang lebih mendasar/mudah. Buat search_queries yang menargetkan penjelasan konsep dasar (misal "explain like I'm 5", "basic introduction").
2. Jika "review": Pengguna cukup paham tapi perlu penguatan. Tambahkan satu sesi singkat untuk mereview topik ini sebelum lanjut ke topik berikutnya.
3. Jika "continue": Pengguna paham. Tidak perlu revisi besar, mungkin hanya sedikit penyesuaian jika diperlukan.
4. Jika "accelerate": Pengguna sangat paham. Anda boleh menghapus atau menggabungkan sub-topik pengantar di hari-hari berikutnya agar pengguna bisa belajar lebih cepat.
5. KEMBALIKAN OUTPUT DALAM FORMAT JSON SESUAI DENGAN SKEMA YANG DIMINTA.
6. PENTING: KUNCI JSON (JSON KEYS) HARUS SAMA PERSIS DENGAN SKEMA. JANGAN TERJEMAHKAN KUNCI JSON KE DALAM BAHASA INDONESIA.
Anda WAJIB mengikuti format JSON persis seperti contoh ini:
```json
{{
  "curriculum_id": "string",
  "topic": "string",
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

def replan_node(state: PLAState) -> PLAState:
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
        message=f"Revising curriculum based on feedback action '{latest_feedback.action}' for topic '{latest_feedback.topic_id}'..."
    )
    
    if "agent_logs" not in state or state["agent_logs"] is None:
        state["agent_logs"] = []
    state["agent_logs"].append(log)

    llm = get_llm(settings.PLANNER_MODEL, temperature=0.3)
    from langchain_groq import ChatGroq
    if isinstance(llm, ChatGroq):
        structured_llm = llm.with_structured_output(Curriculum, method="json_mode")
    else:
        structured_llm = llm.with_structured_output(Curriculum)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", REPLAN_PROMPT),
        ("human", "Tolong revisi kurikulum ini sesuai feedback action yang diberikan.")
    ])
    
    chain = prompt | structured_llm
    
    try:
        revised_curriculum: Curriculum = chain.invoke({
            "topic": config.topic if config else current_curriculum.topic,
            "level": config.level if config else "umum",
            "current_curriculum": current_curriculum.model_dump_json(indent=2),
            "target_topic_id": latest_feedback.topic_id,
            "action": latest_feedback.action
        })
        
        # Pertahankan ID kurikulum asli
        revised_curriculum.curriculum_id = current_curriculum.curriculum_id
        
        # Make topic_ids globally unique to avoid Postgres PK collisions
        short_session = str(state["session_id"]).split("-")[0]
        for week in revised_curriculum.weeks:
            for day in week.days:
                if short_session not in day.topic_id:
                    day.topic_id = f"{short_session}-{day.topic_id}"
                    
        state["curriculum"] = revised_curriculum
        
        success_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="planner",
            level="info",
            message=f"Successfully revised curriculum due to '{latest_feedback.action}' feedback."
        )
        state["agent_logs"].append(success_log)
        
    except Exception as e:
        error_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="planner",
            level="error",
            message=f"Error revising curriculum: {str(e)}"
        )
        state["agent_logs"].append(error_log)
        
    return state
