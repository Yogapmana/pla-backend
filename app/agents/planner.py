from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import SynapsaState, Curriculum, AgentLog
from app.config import settings
from app.utils.llm_factory import get_llm
import uuid
import logging

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """
You are an advanced AI Planner in a Personal Learning Agent system.
Your task is to adaptively break down a learning topic into a structured curriculum.

User's Learning Information:
- Topic: {topic}
- Duration: {duration_weeks} weeks
- Level: {level}
- Hours per day: {hours_per_day} hours
- Language: {language}

{context_text_instruction}

Rules:
1. Break down this topic progressively from basic concepts to advanced concepts (according to the level).
2. Determine a study schedule for each week and day (for example, if there are 5 effective study days a week, divide the material).
3. Create a list of relevant `search_queries` for each sub-topic so the Researcher Agent can gather materials. CRITICAL: The search queries MUST be written in the target language: {language}. Do NOT write English queries if {language} is not English, otherwise the user will get sources in the wrong language.
4. Estimate the duration in minutes per daily topic (the total daily duration must not exceed {hours_per_day} hours, or 60 * {hours_per_day} minutes).
5. RETURN THE OUTPUT IN JSON FORMAT ACCORDING TO THE REQUESTED SCHEMA.
6. IMPORTANT: THE JSON KEYS MUST EXACTLY MATCH THE SCHEMA. DO NOT TRANSLATE THE JSON KEYS.
7. ABSOLUTE OBLIGATION: You MUST generate an array of `weeks` EXACTLY MATCHING {duration_weeks} weeks. If the duration is 12 weeks, there must be 12 elements inside `weeks`. Never reduce or summarize the number of weeks!
8. ALL CONTENT TEXT (titles, descriptions, etc. except search_queries and JSON keys) MUST BE WRITTEN IN THE LANGUAGE: {language}.

You MUST follow the JSON format exactly like this example (do not remove or change the keys):
```json
{{
  "curriculum_id": "random string or ID format",
  "topic": "{topic}",
  "total_weeks": number,
  "weeks": [
    {{
      "week": 1,
      "title": "Week Title",
      "days": [
        {{
          "day": 1,
          "topic_id": "Topic ID",
          "title": "Daily Topic Title",
          "description": "Topic description",
          "search_queries": ["english search query"],
          "duration_minutes": number,
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
        context_text_instruction = f"\nUser Reference Document / Roadmap:\n{config.context_text}\n\nIMPORTANT: Use the reference document/roadmap above as the primary guide in structuring the sequence, outline, and scope of the curriculum topics.\n"

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
You are an advanced AI Planner in a Personal Learning Agent system.
Your task is to create ONE NEW TOPIC (a single learning module) to be inserted into the user's learning curriculum based on their Mastery Score evaluation.

Main Curriculum Topic: {topic}
Quiz Error Context: {context}

Last completed topic (Target ID: {target_topic_id}):
{current_topic_json}

Action to be taken: {action}
Rules:
1. If the action is "remedial": Create a "Micro-Remedial" topic with a duration of 15-20 minutes that ONLY focuses on discussing and correcting the erroneous concepts based on the 'Quiz Error Context'.
2. If the action is "enrichment": Create a "Deep Dive / Enrichment" topic with a duration of 20-30 minutes containing technical case studies or advanced materials following up on the last topic.
3. RETURN THE OUTPUT IN JSON FORMAT ACCORDING TO THE SCHEMA.
4. ALL CONTENT TEXT (titles, descriptions) MUST BE IN THE LANGUAGE: {language}.

You MUST follow the JSON format exactly like this example:
```json
{{
  "day": 0,
  "topic_id": "Unique_New_Topic_ID",
  "title": "Daily Topic Title",
  "description": "Detailed description of this topic",
  "search_queries": ["specific english search query"],
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
