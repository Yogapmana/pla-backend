"""
MindmapService — generates an AI-powered mind map of a curriculum.

The mind map is rendered on the frontend using Mermaid.js ``mindmap`` syntax.
This service:

1. Reads the curriculum (weeks + topics + status) from the database.
2. Calls the LLM (Groq/Ollama) to produce a Mermaid ``mindmap`` diagram.
3. Caches the result in ``curricula.mindmap_json`` so subsequent calls are free.

The cache shape is::

    {
        "syntax":       "<mermaid mindmap source>",   # the Mermaid diagram
        "summary":      "<one-paragraph summary>",     # LLM's verbal overview
        "generated_at": "<ISO8601>",                   # when generated
        "model":        "<model name>",                # which LLM was used
        "node_count":   <int>,                         # number of topic nodes
    }

Design notes
------------
- The LLM is given a **strict** prompt that requires it to emit ONLY a
  ``mindmap`` code block. Post-processing extracts that block and validates
  the syntax with a tiny Mermaid parser guard (start with ``mindmap``).
- Topics are not modified — the LLM can only group/label them.
- If the LLM call fails (timeout, invalid syntax, network error), a deterministic
  fallback mind map is built directly from the curriculum so the user always
  gets *something* useful.
- The cache is invalidated when the curriculum version changes (re-plan) or
  when the user explicitly hits the ``/regenerate`` endpoint.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.learning import Curriculum
from app.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pydantic schema for the LLM structured output
# --------------------------------------------------------------------------- #


class MindmapLLMResult(BaseModel):
    """Schema the LLM must conform to. ``with_structured_output`` enforces it."""

    mermaid_syntax: str = Field(
        description=(
            "A complete Mermaid v11 `mindmap` diagram source code. "
            "Must start with the literal line `mindmap` and use indentation "
            "to express hierarchy. Do not wrap in ``` fences."
        )
    )
    summary: str = Field(
        description=(
            "One short paragraph (max 60 words) describing the curriculum's "
            "thematic structure in plain language, written in the user's language."
        )
    )


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


MINDMAP_PROMPT_TEMSynapsaTE = """Anda adalah AI yang membantu membuat peta konsep (mind map) dari sebuah kurikulum pembelajaran.

Tugas: Mengubah daftar topik kurikulum di bawah ini menjadi sebuah diagram Mermaid ``mindmap`` yang rapi.

# Informasi Kurikulum
- Topik utama kursus: {course_title}
- Level: {level}
- Bahasa output: {language}

# Aturan KETAT (WAJIB DIIKUTI — output akan divalidasi)
1. Output HARUS berupa kode Mermaid ``mindmap`` saja. TIDAK BOLEH ada teks lain di luar blok kode.
2. Baris pertama literal: ``mindmap``
3. Gunakan indentasi 2 spasi untuk setiap level kedalaman.
4. Akar (root) adalah nama kursus: ``root(({course_title}))`` (menggunakan ikon kotak-bulat).
5. Level 1 adalah nama-nama minggu. Format: ``Minggu 1: Judul``.
6. Level 2 adalah topik di minggu tersebut. Format: ``Topik X.Y`` (tidak perlu status).
7. JANGAN tambahkan topik yang tidak ada di daftar.
8. JANGAN ubah nama topik.
9. Maksimal 50 node total (jika lebih, ringkas level 1).
10. Tulis dalam bahasa {language}.

# Daftar Topik (JSON)
{topics_json}

# Format Output (JSON)
Kembalikan HANYA JSON valid dengan struktur:
{{
  "mermaid_syntax": "<syntax Mermaid di atas, multi-line string>",
  "summary": "<1 paragraf singkat max 60 kata, bahasa {language}>"
}}
"""


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class MindmapService:
    """Generates and caches AI-powered mind maps for a learning session."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def get_or_generate_mindmap(
        self,
        session_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> dict[str, Any] | None:
        """
        Return the cached mind map for ``session_id`` or generate a new one.

        Returns ``None`` if the session has no curriculum yet.
        """
        from app.services.learning_service import LearningService

        learning_svc = LearningService(self.db)
        curriculum = await learning_svc.get_curriculum(session_id)
        if not curriculum:
            return None

        # Cache hit (and not forced to regenerate)
        if not force_regenerate and curriculum.mindmap_json:
            cached = curriculum.mindmap_json
            # Sanity check — must have required keys
            if isinstance(cached, dict) and cached.get("syntax"):
                logger.info(
                    "[MINDMAP] Cache hit for session %s (node_count=%s)",
                    session_id, cached.get("node_count"),
                )
                return cached

        # Generate fresh
        return await self._generate_and_persist(curriculum, learning_svc, session_id)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    async def _generate_and_persist(
        self,
        curriculum: Curriculum,
        learning_svc,
        session_id: UUID,
    ) -> dict[str, Any] | None:
        from sqlalchemy import update

        topics = await learning_svc.get_topics(session_id)
        if not topics:
            logger.warning("[MINDMAP] No topics found for session %s", session_id)
            return None

        # Read course title & level from curriculum_json
        cjson = curriculum.curriculum_json or {}
        course_title = cjson.get("title") or cjson.get("topic") or "Kurikulum"
        level = cjson.get("level", "beginner")

        # Build the topics payload (compact)
        topics_payload = [
            {
                "topic_id": t.id,
                "title": t.title,
                "week_number": t.week_number,
                "day_number": t.day_number,
                "status": t.status,
            }
            for t in topics
        ]
        topics_json = json.dumps(topics_payload, ensure_ascii=False, indent=2)

        # Try LLM first
        llm_result: dict | None = None
        llm_was_called = False
        model_name = settings.PLANNER_MODEL
        try:
            llm_result = await self._call_llm(
                course_title=course_title,
                level=level,
                language="id",  # default; could be sourced from session.language
                topics_json=topics_json,
                model_name=model_name,
            )
            llm_was_called = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MINDMAP] LLM generation failed: %s — using fallback", exc)

        # Fallback to deterministic mind map if LLM failed or returned invalid syntax
        used_fallback = (
            not llm_result
            or not self._is_valid_mermaid_mindmap(llm_result["syntax"])
        )
        if used_fallback:
            logger.info("[MINDMAP] Using deterministic fallback mind map")
            llm_result = MindmapService._build_fallback_mindmap(course_title, topics)

        # Persist to DB
        payload = {
            "syntax": llm_result["syntax"],
            "summary": llm_result.get("summary", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "fallback" if used_fallback else model_name,
            "node_count": llm_result["syntax"].count("\n") + 1,
        }

        await self.db.execute(
            update(Curriculum)
            .where(Curriculum.id == curriculum.id)
            .values(mindmap_json=payload)
        )
        await self.db.commit()

        logger.info(
            "[MINDMAP] Persisted mind map for session %s (node_count=%s)",
            session_id, payload["node_count"],
        )
        return payload

    async def _call_llm(
        self,
        *,
        course_title: str,
        level: str,
        language: str,
        topics_json: str,
        model_name: str,
    ) -> dict:
        """Call the LLM and parse the structured output."""
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm(model_name, temperature=0.2)
        structured_llm = llm.with_structured_output(MindmapLLMResult)

        prompt = MINDMAP_PROMPT_TEMSynapsaTE.format(
            course_title=course_title,
            level=level,
            language=language,
            topics_json=topics_json,
        )

        raw_result = await structured_llm.ainvoke(
            [
                SystemMessage(content="You are a curriculum visualization assistant."),
                HumanMessage(content=prompt),
            ]
        )
        # with_structured_output may return a Pydantic instance or a dict
        # depending on the provider wrapper. Normalize to MindmapLLMResult.
        if isinstance(raw_result, MindmapLLMResult):
            result = raw_result
        elif isinstance(raw_result, dict):
            result = MindmapLLMResult(**raw_result)
        else:
            raise ValueError(f"Unexpected LLM result type: {type(raw_result).__name__}")
        return {
            "syntax": result.mermaid_syntax.strip(),
            "summary": result.summary.strip(),
        }

    @staticmethod
    def _is_valid_mermaid_mindmap(syntax: str) -> bool:
        """
        Cheap structural validation — must start with ``mindmap`` and have
        at least 3 indented lines. Real Mermaid parsing is done on the client.
        """
        if not syntax or not isinstance(syntax, str):
            return False
        lines = [l for l in syntax.splitlines() if l.strip()]
        if not lines:
            return False
        if not lines[0].strip().lower().startswith("mindmap"):
            return False
        # At least root + 2 children
        return len(lines) >= 3

    @staticmethod
    def _escape_mermaid_label(text: str) -> str:
        """
        Mermaid mindmap node labels forbid certain characters.
        Strategy: strip parentheses inside, replace quotes, collapse whitespace.
        """
        # Mermaid v11 mindmap labels: keep alphanumerics, spaces, basic punctuation
        cleaned = re.sub(r"[()\[\]{}]", "", text)
        cleaned = cleaned.replace('"', "'")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:80]  # cap to 80 chars per node

    @staticmethod
    def _build_fallback_mindmap(course_title: str, topics: list) -> dict:
        """
        Deterministic fallback — no LLM, just groups topics by week.

        Produces a valid Mermaid v11 ``mindmap`` syntax.
        """
        root_label = MindmapService._escape_mermaid_label(course_title)
        lines = ["mindmap", f"  root(({root_label}))"]

        # Group by week, preserve order
        by_week: dict[int, list] = {}
        for t in topics:
            by_week.setdefault(t.week_number, []).append(t)

        for week_no in sorted(by_week.keys()):
            week_title = f"Minggu {week_no}"
            lines.append(f"    {week_title}")
            for t in by_week[week_no]:
                topic_label = MindmapService._escape_mermaid_label(t.title)
                lines.append(f"      {topic_label}")

        syntax = "\n".join(lines)
        summary = (
            f"Peta konsep untuk kursus '{course_title}' mencakup "
            f"{len(by_week)} minggu dengan total {len(topics)} topik. "
            "Mind map ini dihasilkan secara otomatis dari struktur kurikulum."
        )
        return {"syntax": syntax, "summary": summary}
