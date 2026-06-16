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

class WeekConceptsResult(BaseModel):
    """LLM output for a single week's concept extraction and topic enrichment."""
    concepts: list[ExtractedConcept] = Field(
        description="3-8 concepts that capture the key ideas of this week.",
    )
    enriched_topics: list[TopicEnrichment] = Field(
        description="Sub-topics for EACH topic provided in the input.",
    )

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

CONCEPT_EXTRACTION_PROMPT = """Anda adalah asisten AI untuk membuat PETA KONSEP dari sebuah kurikulum pembelajaran.

Tugas: Dari daftar topik di bawah ini (satu minggu kursus "{course_title}"),
1. Ekstrak 3-8 **Konsep Utama** yang mewakili ide-ide kunci dari minggu ini.
2. Bedah / pecah setiap topik menjadi 2-3 **Sub-Topik** (Poin Kunci Pembelajaran) agar lebih mudah dipahami.

# Aturan KETAT
1. Output HARUS JSON valid sesuai schema yang diminta (field `concepts` dan `enriched_topics`).
2. Setiap topik WAJIB muncul di `topic_ids` dari minimal 1 konsep.
3. Setiap topik WAJIB memiliki entri di `enriched_topics` dengan 2-3 `sub_topics`.
4. Nama konsep dan sub-topik harus singkat: 1-4 kata, bahasa {language}. Capitalize each word.
5. JANGAN duplikasi konsep.
6. Jangan buat konsep/sub-topik untuk topik yang ada di minggu lain.

# Informasi Minggu
- Kursus: {course_title}
- Minggu ke-{week_number}: {week_title}
- Bahasa output: {language}

# Daftar Topik (JSON)
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

    model_used = settings.PLANNER_MODEL
    llm = get_llm(model_used, temperature=0.2, max_tokens=3000)
    structured_llm = llm.with_structured_output(WeekConceptsResult)

    all_week_concepts = {}
    all_enriched_topics = {}
    
    for week_no in sorted(by_week.keys()):
        week_data = by_week[week_no]
        week_topics = week_data["topics"]
        
        if not week_topics:
            continue
            
        topics_json = json.dumps(week_topics, ensure_ascii=False, indent=2)
        prompt = CONCEPT_EXTRACTION_PROMPT.format(
            course_title=course_title,
            week_number=week_no,
            week_title=week_data["title"],
            language=language,
            topics_json=topics_json,
        )

        try:
            logger.info(f"[MINDMAP_MAPPER] Extracting concepts for week {week_no}...")
            raw = await structured_llm.ainvoke(
                [
                    SystemMessage(content="You are a curriculum concept-extraction assistant."),
                    HumanMessage(content=prompt),
                ]
            )
            
            if isinstance(raw, WeekConceptsResult):
                result = raw
            elif isinstance(raw, dict):
                result = WeekConceptsResult(**raw)
            else:
                raise ValueError(f"Unexpected LLM result type: {type(raw).__name__}")
                
            # Store concepts
            concepts_out = [
                {
                    "label": c.label.strip(),
                    "description": (c.description or "").strip() or None,
                    "topic_ids": list(c.topic_ids or []),
                }
                for c in result.concepts
            ]
            all_week_concepts[week_no] = concepts_out
            
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
            logger.error(f"[MINDMAP_MAPPER] LLM failed for week {week_no}: {exc}")
            # Fallback to empty concepts and subtopics for this week
            all_week_concepts[week_no] = []

    # Final JSON structure to save
    concept_graph_payload = {
        "version": 1,
        "model": model_used,
        "week_concepts": all_week_concepts,
        "enriched_topics": all_enriched_topics
    }

    logger.info("[MINDMAP_MAPPER] Successfully extracted concept graph.")
    return {"concept_graph": concept_graph_payload}
