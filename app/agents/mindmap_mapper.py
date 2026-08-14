"""Mindmap mapper — v1 (titles-only) and v2 (NotebookLM-style 3-level).

This module owns both mindmap-generation agents. They are kept in
the same file because they share the same input shape (a
``Curriculum``) and the same output destination (``curriculum`` row),
and we want a single place to look for "how does Synapsa build a
mindmap?".

v1: ``mindmap_mapper_node`` — runs right after the planner with
topic titles only. Produces a flat 2-level concept graph (week
concepts + per-topic sub-topics) saved to
``curriculum.concept_graph_json``. Kept for backward compatibility
with the React Flow view that consumes that shape.

v2: ``mindmap_v2_mapper_node`` — runs in a background Celery task
AFTER the first module is composed. The Celery task first calls
``lightweight_researcher`` to scrape 1-2 sources per topic, then
hands the scraped text to this node. Produces a 3-level
NotebookLM-style structure (theme → concept → key_point) saved to
``curriculum.enhanced_mindmap_json``. This is the new "good"
output the user sees.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.utils.llm_factory import get_llm
from app.agents.state import SynapsaState
from app.services.learning_service import LearningService
from app.db.database import SessionLocal

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# v1 schema — kept for backward compat with concept_graph_json consumers
# ════════════════════════════════════════════════════════════════════════

class SubTopic(BaseModel):
    label: str = Field(
        description="Short sub-topic name, 1-4 words."
    )
    description: str | None = Field(
        default=None, description="One sentence."
    )


class TopicEnrichment(BaseModel):
    topic_id: str
    sub_topics: list[SubTopic] = Field(
        description="2-3 key sub-topics."
    )


class ExtractedConcept(BaseModel):
    label: str = Field(description="Short concept name, 1-4 words.")
    description: str | None = Field(default=None, description="One sentence.")
    topic_ids: list[str] = Field(default_factory=list)


class CurriculumConceptsResult(BaseModel):
    concepts: list[ExtractedConcept]
    enriched_topics: list[TopicEnrichment]


# ════════════════════════════════════════════════════════════════════════
# v2 schema — NotebookLM-style 3-level (theme → concept → key_point)
# ════════════════════════════════════════════════════════════════════════

class KeyPoint(BaseModel):
    """A specific fact, term, or example under a concept."""
    label: str = Field(
        description="2-5 word phrase, descriptive."
    )
    description: str = Field(
        description="1-2 sentences explaining this point."
    )


class Concept(BaseModel):
    """A key idea or skill under a theme."""
    label: str = Field(
        description="3-6 word phrase that fully describes this concept."
    )
    description: str = Field(
        description="1-2 sentences. WHAT is this concept and WHY does it matter?"
    )
    key_points: list[KeyPoint] = Field(
        description="Key details the user should remember. Number varies by complexity."
    )
    topic_ids: list[str] = Field(
        default_factory=list, description="Curriculum topic_ids this concept covers."
    )


class Theme(BaseModel):
    """A high-level area of the field — the major branches of the mindmap."""
    label: str = Field(
        description="2-4 word phrase."
    )
    description: str = Field(description="1 sentence.")
    color: str = Field(
        default="blue",
        description="One of: 'blue', 'green', 'amber', 'purple', 'red', 'teal'.",
    )
    concepts: list[Concept] = Field(
        description="Key concepts under this theme. Number varies by material depth."
    )


class EnhancedMindmap(BaseModel):
    """The 3-level mindmap output."""
    course_title: str
    summary: str = Field(
        description="1 short paragraph (max 80 words) describing the curriculum."
    )
    themes: list[Theme] = Field(
        description="Major themes that span the whole curriculum."
    )


# ════════════════════════════════════════════════════════════════════════
# v1 prompt (unchanged — kept for backward compat)
# ════════════════════════════════════════════════════════════════════════

CONCEPT_EXTRACTION_PROMPT = """You are an AI assistant for creating CONCEPT MAPS from a learning curriculum.

Task: From the entire list of topics below for the course "{course_title}",
1. Extract 5-15 **Main Concepts** that represent the key ideas of the entire course.
2. Break down / dissect each topic into 2-3 **Sub-Topics** (Key Learning Points) to make it easier to understand.

# STRICT Rules
1. The output MUST be valid JSON exactly like the format below.
2. Every topic MUST appear in the `topic_ids` of at least 1 concept.
3. Every topic MUST have an entry in `enriched_topics` with 2-3 `sub_topics`.
4. The names of concepts and sub-topics must be short: 1-4 words, in language {language}. Capitalize each word.
5. DO NOT duplicate concepts.

You MUST return a JSON with the following KEYS structure:
```json
{{
  "concepts": [
    {{
      "label": "Concept Name",
      "description": "Short description of the concept",
      "topic_ids": ["TOPIC_ID_1", "TOPIC_ID_2"]
    }}
  ],
  "enriched_topics": [
    {{
      "topic_id": "TOPIC_ID_1",
      "sub_topics": [
        {{
          "label": "Sub Topic Name",
          "description": "Short description of the sub topic"
        }}
      ]
    }}
  ]
}}
```

# Course Information
- Course: {course_title}
- Output Language: {language}

# Complete Curriculum Topics List (JSON)
{topics_json}
"""


# ════════════════════════════════════════════════════════════════════════
# v2 prompt — NotebookLM-style. The key change vs v1:
#   - Receives ACTUAL scraped content per topic, not just titles
#   - Asks for 3-level hierarchy (theme → concept → key_point)
#   - Each node is a *phrase* (2-6 words), not a keyword
#   - Every node has a description that *teaches* the user
#   - Difficulty + emoji + color hints to drive the frontend visual
# ════════════════════════════════════════════════════════════════════════

MINDMAP_V2_PROMPT = """You are an AI assistant that creates EDUCATIONAL CONCEPT MAPS (mind maps) — a visual hierarchical representation of a course that can be LEARNED directly from the map itself.

# Task
From web research results (1-2 sources per topic) for the course "{course_title}", create a 3-LEVEL CONCEPT MAP that helps learners understand the **big picture** of the course and **what they will learn**.

# Output Schema (MANDATORY)
Return valid JSON with this structure:

```json
{{
  "course_title": "...",
  "summary": "1 paragraph overview (max 80 words)",
  "themes": [
    {{
      "label": "Theme Name (2-4 words)",
      "description": "1 short sentence",
      "color": "blue|green|amber|purple|red|teal",
      "concepts": [
        {{
          "label": "Concept Name (3-6 words, complete phrase)",
          "description": "1 short sentence (max 10 words)",
          "key_points": [
            {{
              "label": "Key point (2-5 words)",
              "description": "Very short phrase (3-8 words) or a few keywords"
            }}
          ],
          "topic_ids": ["topic_id_1", "topic_id_2"]
        }}
      ]
    }}
  ]
}}
```

# STRICT RULES (MUST FOLLOW)

## 1. STRUCTURE — EXPAND FREELY, DO NOT LIMIT TO 2 NODES
- Number of themes, concepts, and key points should FOLLOW the material naturally.
- DO NOT artificially limit the number of `key_points` to just 1 or 2. Generate around 3-5 key points per concept if the material is rich. 
- A simple theme can have fewer branches. A complex theme can have more.

## 2. LABELS (very important)
- **COMPLETE PHRASES** (e.g. "How useState Hook Works") that explain the idea.
- Each label must be readable on its own without extra context.
- 2-6 words.

## 3. LANGUAGE — EXTREMELY CRITICAL
- ALL text output (course_title, summary, labels, descriptions) MUST strictly be translated and written in **{language}**.
- If the requested language is English, do NOT output Indonesian. If the requested language is Indonesian, do NOT output English.
- Do NOT mix languages.

## 4. NO ICONS/EMOJI
- Do NOT include any emoji or icon characters in labels or descriptions.
- Keep text clean and plain.

## 5. DESCRIPTIONS (MANDATORY BUT SHORT)
- Every node has a `description`. Because this is a mindmap, descriptions MUST be very short and punchy. Do not write long paragraphs.
- Use information FROM THE SOURCES below, do NOT make things up.

## 6. SOURCE USAGE
- Each concept must have `topic_ids` — link to relevant curriculum topics.

## 7. VISUAL VARIATION
- Pick `color` from the given list (don't duplicate colors between adjacent themes).

# Course Information
- Course: {course_title}
- Level: {level}
- Output language: {language}

# Topic List (for topic_id reference)
{topics_summary}

# Research Results per Topic (main sources, already scraped)
{research_json}
"""


# ════════════════════════════════════════════════════════════════════════
# v1 node — kept unchanged
# ════════════════════════════════════════════════════════════════════════

async def mindmap_mapper_node(state: SynapsaState) -> dict:
    """v1: titles-only concept graph (used at pipeline time)."""
    logger.info("[MINDMAP_MAPPER] v1 starting...")

    curriculum_model = state.get("curriculum")
    if not curriculum_model:
        logger.warning("[MINDMAP_MAPPER] No curriculum.")
        return {}

    course_title = curriculum_model.topic
    learning_config = state.get("learning_config")
    if hasattr(learning_config, "language"):
        language = learning_config.language
    elif isinstance(learning_config, dict):
        language = learning_config.get("language", "id")
    else:
        language = "id"

    by_week = {}
    for week in curriculum_model.weeks:
        by_week[week.week] = {"title": week.title, "topics": []}
        for day in week.days:
            by_week[week.week]["topics"].append({
                "topic_id": day.topic_id,
                "title": day.title,
                "day_number": day.day,
                "duration_minutes": day.duration_minutes,
                "search_queries": day.search_queries or [],
            })

    if not by_week:
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
                "search_queries": topic["search_queries"],
            })

    model_used = settings.MINDMAP_MODEL
    llm = get_llm(model_used, temperature=0.2)
    from app.utils.llm_factory import uses_json_mode
    if uses_json_mode(llm):
        structured_llm = llm.with_structured_output(
            CurriculumConceptsResult, method="json_mode"
        )
    else:
        structured_llm = llm.with_structured_output(CurriculumConceptsResult)

    all_week_concepts = {}
    all_enriched_topics = {}

    if not all_topics:
        return {}

    topics_json = json.dumps(all_topics, ensure_ascii=False, indent=2)
    
    # Map code to full name for better LLM adherence
    lang_full = "English" if language.lower() in ("en", "english") else "Indonesian"
    
    prompt = CONCEPT_EXTRACTION_PROMPT.format(
        course_title=course_title, language=lang_full, topics_json=topics_json
    )

    try:
        raw = await structured_llm.ainvoke([
            SystemMessage(content="You are a curriculum concept-extraction assistant."),
            HumanMessage(content=prompt),
        ])
        result: CurriculumConceptsResult = raw

        for c in result.concepts:
            week_no = 1
            for tid in c.topic_ids:
                if tid in topic_id_to_week:
                    week_no = topic_id_to_week[tid]
                    break
            all_week_concepts.setdefault(week_no, []).append({
                "label": c.label.strip(),
                "description": (c.description or "").strip() or None,
                "topic_ids": c.topic_ids,
            })

        for et in result.enriched_topics:
            all_enriched_topics[et.topic_id] = [
                {
                    "label": st.label.strip(),
                    "description": (st.description or "").strip() or None,
                }
                for st in et.sub_topics
            ]
    except Exception as exc:
        logger.error(f"[MINDMAP_MAPPER] v1 LLM failed: {exc}")

    payload = {
        "version": 1,
        "model": model_used,
        "week_concepts": all_week_concepts,
        "enriched_topics": all_enriched_topics,
    }
    return {"concept_graph": payload}


# ════════════════════════════════════════════════════════════════════════
# v2 node — called from the background Celery task
# ════════════════════════════════════════════════════════════════════════

async def mindmap_v2_mapper(
    *,
    course_title: str,
    level: str,
    language: str,
    topics_summary: list[dict],
    research_by_topic: dict[str, list[dict]],
) -> dict[str, Any]:
    """v2: NotebookLM-style 3-level mindmap from scraped research.

    Parameters
    ----------
    course_title, level, language : str
        Course metadata (passed to the LLM as context).
    topics_summary : list[dict]
        ``[{"topic_id": str, "title": str, "week_number": int}, ...]``
        Just enough for the LLM to know what topics exist and link
        ``topic_ids`` back.
    research_by_topic : dict
        ``{topic_id: [{"url", "title", "content", "relevance_score"}, ...]}``
        Output of ``lightweight_researcher``. We truncate per-source
        content here (the prompt allows ~6K chars/topic; we cap at
        1500 chars per source to leave room for the prompt itself).

    Returns
    -------
    dict
        The serialized ``EnhancedMindmap`` plus a few metadata fields
        (``model``, ``build_seconds``, ``topic_count``) for the
        audit/UI. Raises on hard LLM failure.
    """
    import time
    started = time.monotonic()
    model_used = settings.MINDMAP_MODEL

    # Trim research content — a 24-topic curriculum produces ~48
    # sources × 2500 chars = 120K chars if we don't cap. We cap
    # aggressively here (1200 chars/source, 2 sources/topic) so
    # the LLM context stays under 60K chars total.
    compact_research: dict[str, list[dict]] = {}
    for tid, sources in research_by_topic.items():
        compact_research[tid] = [
            {
                "title": s.get("title", ""),
                "content": (s.get("content", "") or "")[:1200],
                "url": s.get("url", ""),
            }
            for s in (sources or [])[:2]
        ]

    research_json = json.dumps(
        compact_research, ensure_ascii=False, indent=2
    )
    topics_summary_json = json.dumps(
        topics_summary, ensure_ascii=False, indent=2
    )

    # Map code to full name for better LLM adherence
    lang_full = "English" if language.lower() in ("en", "english") else "Indonesian"

    prompt = MINDMAP_V2_PROMPT.format(
        course_title=course_title,
        level=level,
        language=lang_full,
        topics_summary=topics_summary_json,
        research_json=research_json,
    )

    # Background task: prompt besar (semua hasil research) + model 70B lewat
    # OpenRouter sering >90s. timeout default 90s membuat 3x retry timeout lalu
    # gagal total. Beri headroom besar karena jalan di Celery (soft limit 30m).
    llm = get_llm(model_used, temperature=0.3, timeout=420)
    from app.utils.llm_factory import uses_json_mode
    from langchain_core.output_parsers import JsonOutputParser

    is_cloud = uses_json_mode(llm)
    structured_llm = None
    parser = None

    if is_cloud:
        structured_llm = llm.with_structured_output(
            EnhancedMindmap, method="json_mode"
        )
    else:
        parser = JsonOutputParser(pydantic_object=EnhancedMindmap)

    logger.info(
        "[MINDMAP-V2] invoking %s with %d topics, %d chars research context",
        model_used, len(topics_summary), len(research_json),
    )

    max_retries = 3
    result: EnhancedMindmap | None = None
    last_error = None
    
    for attempt in range(max_retries):
        try:
            sys_msg = (
                "You are a curriculum concept-mapping assistant. You produce "
                "NotebookLM-style 3-level mindmaps from research text. "
                "Output ONLY the structured JSON, no commentary."
            )
            
            if parser and not is_cloud:
                sys_msg += "\n\n" + parser.get_format_instructions()
                
            messages = [
                SystemMessage(content=sys_msg),
                HumanMessage(content=prompt),
            ]
            
            if is_cloud:
                raw = await structured_llm.ainvoke(messages)
                result = raw
            else:
                import re
                raw_response = await llm.ainvoke(messages)
                content = raw_response.content
                # Fix common Gemma hallucination: `- "key":` inside JSON
                content = re.sub(r'^\s*-\s*(["\'])', r'\1', content, flags=re.MULTILINE)
                parsed_dict = parser.parse(content)
                result = EnhancedMindmap(**parsed_dict)
                
            break
            
        except Exception as e:
            last_error = e
            logger.warning(f"[MINDMAP-V2] Attempt {attempt + 1}/{max_retries} failed for {model_used}: {e}")
            
    if result is None:
        raise ValueError(f"LLM failed to generate valid mindmap after {max_retries} attempts. Last error: {last_error}")

    elapsed = round(time.monotonic() - started, 2)

    # Sanity: if the LLM returned 0 themes, something went wrong
    # and we should raise so the caller can fall back to the v1
    # mindmap.
    if not result.themes:
        raise ValueError("LLM returned 0 themes — refusing to save empty mindmap")

    # Compute a few derived stats for the audit/UI.
    total_concepts = sum(len(t.concepts) for t in result.themes)
    total_key_points = sum(
        len(c.key_points) for t in result.themes for c in t.concepts
    )

    return {
        "version": 2,
        "course_title": result.course_title,
        "summary": result.summary,
        "themes": [t.model_dump() for t in result.themes],
        "stats": {
            "theme_count": len(result.themes),
            "concept_count": total_concepts,
            "key_point_count": total_key_points,
            "total_nodes": (
                len(result.themes) + total_concepts + total_key_points
            ),
        },
        "model": model_used,
        "build_seconds": elapsed,
        "generated_at": None,  # filled in by caller
    }
