"""Mindmap mapper — v1 (titles-only) and v2 (NotebookLM-style 3-level).

This module owns both mindmap-generation agents. They are kept in
the same file because they share the same input shape (a
``Curriculum``) and the same output destination (``curriculum`` row),
and we want a single place to look for "how does PLA build a
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
from app.agents.state import PLAState
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
    """A specific fact, term, or example under a concept.

    - ``label``      : 2-5 word phrase, e.g. "Virtual DOM", "LCP metric"
    - ``description``: 1-2 sentences that *teach* the fact in context
    - ``kind``       : 'fact' | 'term' | 'example' | 'principle'
                       (used by the frontend to pick an icon)
    """
    label: str = Field(
        description="2-5 word phrase, descriptive, e.g. 'Virtual DOM diffing'."
    )
    description: str = Field(
        description=(
            "1-2 sentences explaining this fact. Should be self-contained "
            "so the user can learn from the mindmap alone."
        )
    )
    kind: str = Field(
        default="fact",
        description="One of: 'fact', 'term', 'example', 'principle'.",
    )


class Concept(BaseModel):
    """A key idea or skill under a theme.

    - ``label``      : 3-6 word phrase, e.g. "React Component Lifecycle"
    - ``description``: 1-2 sentences — what is this AND why does it matter
    - ``emoji``      : a single emoji to identify the concept visually
    - ``difficulty`` : 1-5, where 1 = easy and 5 = advanced. Used by
                       the frontend for color/intensity.
    - ``key_points`` : 2-4 details the user should remember
    - ``topic_ids``  : which curriculum topics this concept covers
    """
    label: str = Field(
        description="3-6 word phrase that fully describes this concept."
    )
    description: str = Field(
        description="1-2 sentences. WHAT is this concept and WHY does it matter?"
    )
    emoji: str = Field(
        default="💡", description="One emoji to visually identify the concept."
    )
    difficulty: int = Field(
        default=2, ge=1, le=5, description="1=easy, 5=advanced."
    )
    key_points: list[KeyPoint] = Field(
        description="2-4 specific details — facts, terms, examples, principles."
    )
    topic_ids: list[str] = Field(
        default_factory=list, description="Curriculum topic_ids this concept covers."
    )


class Theme(BaseModel):
    """A high-level area of the field — the major branches of the mindmap.

    4-7 themes per curriculum. Each theme gets a color so the
    frontend can visually separate them in the layout.
    """
    label: str = Field(
        description="2-4 word phrase, e.g. 'Frontend Foundations'."
    )
    description: str = Field(description="1 sentence.")
    emoji: str = Field(default="📚")
    color: str = Field(
        default="blue",
        description="One of: 'blue', 'green', 'amber', 'purple', 'red', 'teal'.",
    )
    concepts: list[Concept] = Field(
        description="3-5 concepts under this theme."
    )


class EnhancedMindmap(BaseModel):
    """The NotebookLM-style 3-level mindmap output."""
    course_title: str
    summary: str = Field(
        description="1 short paragraph (max 80 words) describing the curriculum."
    )
    themes: list[Theme] = Field(
        description="4-7 major themes that span the whole curriculum."
    )


# ════════════════════════════════════════════════════════════════════════
# v1 prompt (unchanged — kept for backward compat)
# ════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════
# v2 prompt — NotebookLM-style. The key change vs v1:
#   - Receives ACTUAL scraped content per topic, not just titles
#   - Asks for 3-level hierarchy (theme → concept → key_point)
#   - Each node is a *phrase* (2-6 words), not a keyword
#   - Every node has a description that *teaches* the user
#   - Difficulty + emoji + color hints to drive the frontend visual
# ════════════════════════════════════════════════════════════════════════

MINDMAP_V2_PROMPT = """Anda adalah AI assistant yang membuat PETA KONSEP (mind map) EDUKATIF seperti NotebookLM — sebuah representasi visual hierarkis dari sebuah kursus yang bisa DIPELAJARI langsung dari peta-nya saja.

# Tugas
Dari hasil riset web (1-2 sumber per topik) untuk kursus "{course_title}", buatlah PETA KONSEP 3-LEVEL yang membantu pelajar memahami **gambaran besar** kursus dan **apa yang akan mereka pelajari**.

# Skema Output (WAJIB)
Kembalikan JSON valid dengan struktur berikut:

```json
{{
  "course_title": "...",
  "summary": "1 paragraf overview (maks 80 kata)",
  "themes": [
    {{
      "label": "Nama Tema (2-4 kata)",
      "description": "1 kalimat",
      "emoji": "📱",
      "color": "blue|green|amber|purple|red|teal",
      "concepts": [
        {{
          "label": "Nama Konsep (3-6 kata, frasa lengkap)",
          "description": "1-2 kalimat — APA dan MENGAPA penting",
          "emoji": "🧩",
          "difficulty": 1-5,
          "key_points": [
            {{
              "label": "Poin kunci (2-5 kata)",
              "description": "1-2 kalimat edukatif",
              "kind": "fact|term|example|principle"
            }}
          ],
          "topic_ids": ["id_topik_1", "id_topik_2"]
        }}
      ]
    }}
  ]
}}
```

# Aturan KETAT (WAJIB DIIKUTI)

## 1. STRUKTUR
- **4-7 tema** total. Tema = cabang utama (cabang-cabang besar dari topik sentral).
- **3-5 konsep** per tema. Konsep = ide/keterampilan utama di bawah tema.
- **2-4 key_points** per konsep. Key point = fakta spesifik, terminologi, contoh, atau prinsip.
- **TOTAL node**: 4×3×2 = 24 minimum, 7×5×4 = 140 maximum. Target: 50-80 node.

## 2. LABEL (sangat penting)
- **BUKAN kata kunci** (mis. "useState") — itu JELAS, tapi tidak mendidik.
- **FRASA LENGKAP** (mis. "Cara Kerja useState Hook") yang menjelaskan ide.
- Setiap label harus bisa dibaca sendiri tanpa konteks tambahan.
- 2-6 kata. Bahasa: {language}.

## 3. DESKRIPSI (WAJIB ada & edukatif)
- Setiap node punya `description`. Bukan cuma ringkasan — harus MENGAJAR.
- Gunakan informasi DARI SUMBER di bawah ini, BUKAN mengarang.
- Jika sumber tidak menyebutkan, jangan tambahkan fakta.

## 4. PENGGUNAAN SUMBER
- Setiap konsep harus punya `topic_ids` — link ke topik kurikulum yang relevan.
- Setiap key point harus bisa ditelusuri ke sumber spesifik.

## 5. VARIASI VISUAL
- Pilih emoji yang relevan untuk setiap tema/konsep.
- Pilih `difficulty` 1-5 (1=dasar, 5=lanjut).
- Pilih `color` dari daftar yang diberikan (jangan duplikat warna antar tema yang berdekatan).

# Informasi Kursus
- Kursus: {course_title}
- Level: {level}
- Bahasa output: {language}

# Daftar Topik (untuk referensi topic_id)
{topics_summary}

# Hasil Riset per Topik (sumber utama, sudah di-scrape)
{research_json}
"""


# ════════════════════════════════════════════════════════════════════════
# v1 node — kept unchanged
# ════════════════════════════════════════════════════════════════════════

async def mindmap_mapper_node(state: PLAState) -> dict:
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
    llm = get_llm(model_used, temperature=0.2, max_tokens=3000)
    from langchain_groq import ChatGroq
    if isinstance(llm, ChatGroq):
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
    prompt = CONCEPT_EXTRACTION_PROMPT.format(
        course_title=course_title, language=language, topics_json=topics_json
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

    prompt = MINDMAP_V2_PROMPT.format(
        course_title=course_title,
        level=level,
        language=language,
        topics_summary=topics_summary_json,
        research_json=research_json,
    )

    llm = get_llm(model_used, temperature=0.3, max_tokens=6000)
    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import JsonOutputParser

    is_groq = isinstance(llm, ChatGroq)
    structured_llm = None
    parser = None
    
    if is_groq:
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
            
            if parser and not is_groq:
                sys_msg += "\n\n" + parser.get_format_instructions()
                
            messages = [
                SystemMessage(content=sys_msg),
                HumanMessage(content=prompt),
            ]
            
            if is_groq:
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
