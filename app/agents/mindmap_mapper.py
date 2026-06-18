import logging
import json
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.utils.llm_factory import get_llm
from app.agents.state import PLAState
from app.services.learning_service import LearningService
from app.db.database import SessionLocal

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pydantic schema for the LLM structured output
# --------------------------------------------------------------------------- #

class SubTopic(BaseModel):
    """A key learning point or sub-topic."""
    label: str = Field(
        description="Short sub-topic name, 1-4 words. E.g. 'useState Hook', 'Foreign Keys'."
    )
    description: str | None = Field(
        default=None,
        description="One sentence describing this sub-topic."
    )

class TopicEnrichment(BaseModel):
    """Enrichment data for a specific topic."""
    topic_id: str
    sub_topics: list[SubTopic] = Field(
        description="2-3 key sub-topics or learning points that break down this topic.",
    )

class ExtractedConcept(BaseModel):
    """A single concept the LLM extracted from a week's topics."""
    label: str = Field(
        description="Short concept name, 1-4 words. E.g. 'Recursion', 'SQL Joins'."
    )
    description: str | None = Field(
        default=None,
        description="One sentence describing what this concept covers.",
    )
    topic_ids: list[str] = Field(
        default_factory=list,
        description="Which topic ids this concept covers.",
    )

class CurriculumConceptsResult(BaseModel):
    """Result of mapping concepts and subtopics for the ENTIRE curriculum."""
    concepts: list[ExtractedConcept] = Field(
        description="List of main concepts spanning the entire curriculum.",
    )
    enriched_topics: list[TopicEnrichment] = Field(
        description="Sub-topics for EVERY topic in the curriculum.",
    )

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

CONCEPT_EXTRACTION_PROMPT = """Anda adalah asisten AI untuk membuat PETA KONSEP dari sebuah kurikulum pembelajaran.

Tugas: Dari daftar seluruh topik di bawah ini untuk kursus "{course_title}",
1. Ekstrak 5-15 **Konsep Utama** yang mewakili ide-ide kunci dari keseluruhan kursus ini.
2. Bedah / pecah setiap topik menjadi 2-3 **Sub-Topik** (Poin Kunci Pembelajaran) agar lebih mudah dipahami.

# Aturan KETAT
1. Output HARUS berupa JSON valid persis seperti format di bawah ini.
2. Setiap topik WAJIB muncul di `topic_ids` dari minimal 1 konsep.
3. Setiap topik WAJIB memiliki entri di `enriched_topics` dengan 2-3 `sub_topics`.
4. Nama konsep dan sub-topik harus singkat: 1-4 kata, bahasa {language}. Capitalize each word.
5. JANGAN duplikasi konsep.

Anda WAJIB mengembalikan JSON dengan struktur KUNCI (KEYS) berikut:
```json
{{
  "concepts": [
    {{
      "label": "Nama Konsep",
      "description": "Deskripsi singkat konsep",
      "topic_ids": ["ID_TOPIK_1", "ID_TOPIK_2"]
    }}
  ],
  "enriched_topics": [
    {{
      "topic_id": "ID_TOPIK_1",
      "sub_topics": [
        {{
          "label": "Nama Sub Topik",
          "description": "Deskripsi singkat sub topik"
        }}
      ]
    }}
  ]
}}
```

# Informasi Kursus
- Kursus: {course_title}
- Bahasa output: {language}

# Daftar Topik Seluruh Kurikulum (JSON)
{topics_json}
"""

async def mindmap_mapper_node(state: PLAState) -> dict:
    """
    Background LangGraph node to generate the concept graph structure (concepts + subtopics).
    It extracts week concepts and saves them to Curriculum.concept_graph_json.
    """
    logger.info("[MINDMAP_MAPPER] Starting mindmap generation...")
    
    curriculum_model = state.get("curriculum")
    if not curriculum_model:
        logger.warning("[MINDMAP_MAPPER] No curriculum found in state.")
        return {}

    curriculum_id = curriculum_model.curriculum_id
    course_title = curriculum_model.topic
    learning_config = state.get("learning_config")
    if hasattr(learning_config, "language"):
        language = learning_config.language
    elif isinstance(learning_config, dict):
        language = learning_config.get("language", "id")
    else:
        language = "id"
    
    # We need to fetch topics to get their IDs. 
    # The state curriculum contains weeks -> days -> topic_id, title, search_queries
    
    # Group topics by week directly from the state curriculum
    by_week = {}
    for week in curriculum_model.weeks:
        by_week[week.week] = {
            "title": week.title,
            "topics": []
        }
        for day in week.days:
            by_week[week.week]["topics"].append({
                "topic_id": day.topic_id,
                "title": day.title,
                "day_number": day.day,
                "duration_minutes": day.duration_minutes,
                "search_queries": day.search_queries or []
            })
            
    if not by_week:
        logger.warning("[MINDMAP_MAPPER] Curriculum has no weeks.")
        return {}

    topic_id_to_week = {}
    all_topics = []

    for week_no in sorted(by_week.keys()):
        week_data = by_week[week_no]
        for topic in week_data["topics"]:
            topic_id_to_week[topic["topic_id"]] = week_no
            all_topics.append({
                "week_number": week_no,
                "topic_id": topic["topic_id"],
                "title": topic["title"],
                "search_queries": topic["search_queries"]
            })

    model_used = settings.MINDMAP_MODEL
    llm = get_llm(model_used, temperature=0.2, max_tokens=3000)
    
    from langchain_groq import ChatGroq
    if isinstance(llm, ChatGroq):
        structured_llm = llm.with_structured_output(CurriculumConceptsResult, method="json_mode")
    else:
        structured_llm = llm.with_structured_output(CurriculumConceptsResult)

    all_week_concepts = {}
    all_enriched_topics = {}
    
    if not all_topics:
        logger.warning("[MINDMAP_MAPPER] No topics to process.")
        return {}

    topics_json = json.dumps(all_topics, ensure_ascii=False, indent=2)
    prompt = CONCEPT_EXTRACTION_PROMPT.format(
        course_title=course_title,
        language=language,
        topics_json=topics_json,
    )

    try:
        logger.info(f"[MINDMAP_MAPPER] Extracting concepts for the entire curriculum using one-shot approach...")
        raw = await structured_llm.ainvoke(
            [
                SystemMessage(content="You are a curriculum concept-extraction assistant."),
                HumanMessage(content=prompt),
            ]
        )
        
        result: CurriculumConceptsResult = raw
        
        # Distribute concepts into week_concepts based on their first valid topic_id
        for c in result.concepts:
            week_no = 1
            for tid in c.topic_ids:
                if tid in topic_id_to_week:
                    week_no = topic_id_to_week[tid]
                    break
            
            all_week_concepts.setdefault(week_no, []).append({
                "label": c.label.strip(),
                "description": (c.description or "").strip() or None,
                "topic_ids": c.topic_ids
            })
            
        # Store enriched topics (sub-topics)
        for et in result.enriched_topics:
            sub_topics_out = [
                {
                    "label": st.label.strip(),
                    "description": (st.description or "").strip() or None
                }
                for st in et.sub_topics
            ]
            all_enriched_topics[et.topic_id] = sub_topics_out

    except Exception as exc:
        logger.error(f"[MINDMAP_MAPPER] One-shot LLM failed: {exc}")

    # Final JSON structure to save
    concept_graph_payload = {
        "version": 1,
        "model": model_used,
        "week_concepts": all_week_concepts,
        "enriched_topics": all_enriched_topics
    }

    logger.info("[MINDMAP_MAPPER] Successfully extracted concept graph.")
    return {"concept_graph": concept_graph_payload}
