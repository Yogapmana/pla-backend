import time
import logging
import asyncio
from datetime import datetime
from app.agents.state import PLAState
from app.rag.retriever import retrieve_and_rerank
from app.utils.llm_factory import get_llm
from app.config import settings

logger = logging.getLogger(__name__)

TUTOR_SYSTEM_PROMPT = """Kamu adalah Tutor AI yang membantu pengguna memahami materi pembelajaran.
Gunakan konteks yang disediakan untuk menjawab pertanyaan secara akurat.
Jawab dalam Bahasa Indonesia yang jelas dan mudah dipahami.
Jika konteks tidak cukup, katakan dengan jujur bahwa kamu tidak memiliki informasi yang cukup.
Selalu sertakan referensi sumber jika relevan."""

def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    if not chunks:
        return "Tidak ada konteks yang tersedia."
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_title", "Unknown Source")
        text = chunk.get("text", "")
        blocks.append(f"[Sumber {i}: {source}]\n{text}")
    return "\n\n---\n\n".join(blocks)

def _format_chat_history(history: list[dict], max_turns: int = 5) -> str:
    """Format last N chat turns into a string for the prompt."""
    if not history:
        return ""
    recent = history[-max_turns * 2:]  # user + assistant pairs
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Tutor"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)

async def tutor_chat(
    user_id: str,
    session_id: str,
    topic_id: str,
    query: str,
    chat_history: list[dict] | None = None,
    include_sources: bool = True,
) -> dict:
    """
    RAG Chat Mode:
    1. Retrieve relevant chunks from Qdrant using HyDE + FlashRank
    2. Build augmented prompt
    3. Generate response with LLM
    4. Return response + sources + latency
    """
    start_time = time.time()
    chat_history = chat_history or []

    logger.info(f"[TUTOR] Processing query for topic '{topic_id}': {query[:80]}...")

    # Step 1: Retrieve chunks
    chunks = retrieve_and_rerank(
        user_id=user_id,
        query=query,
        topic_id=topic_id,
        top_k_retrieve=8,
        top_k_rerank=3,
        use_hyde=True,
    )

    # Step 2: Build prompt
    context_block = _build_context_block(chunks)
    history_str = _format_chat_history(chat_history)

    prompt_parts = [
        TUTOR_SYSTEM_PROMPT,
        f"\n\n## Konteks Materi:\n{context_block}",
    ]
    if history_str:
        prompt_parts.append(f"\n\n## Riwayat Percakapan:\n{history_str}")
    prompt_parts.append(f"\n\n## Pertanyaan:\n{query}")
    prompt_parts.append("\n\n## Jawaban:")
    full_prompt = "".join(prompt_parts)

    # Step 3: Generate response
    llm = get_llm(settings.TUTOR_MODEL)
    response = llm.invoke(full_prompt)
    response_text = response.content.strip()

    latency_ms = int((time.time() - start_time) * 1000)

    # Step 4: Build sources list
    sources = []
    if include_sources and chunks:
        seen = set()
        for c in chunks:
            title = c.get("source_title", "")
            if title and title not in seen:
                sources.append({
                    "title": title,
                    "type": c.get("source_type", "module"),
                    "relevance": float(round(c.get("rerank_score", 0.0), 3)),
                })
                seen.add(title)

    logger.info(f"[TUTOR] Response generated in {latency_ms}ms with {len(chunks)} context chunks.")

    return {
        "response": response_text,
        "sources": sources,
        "latency_ms": latency_ms,
        "chunks_used": len(chunks),
        "timestamp": datetime.utcnow().isoformat(),
    }


async def tutor_generate_quiz(
    user_id: str,
    topic_id: str,
    topic_title: str,
    num_questions: int = 5,
) -> list[dict]:
    """
    Quiz Generation Mode:
    Retrieves topic context from Qdrant and generates MCQ questions.
    """
    logger.info(f"[TUTOR] Generating {num_questions} quiz questions for '{topic_title}'...")

    # Get context from Qdrant
    chunks = retrieve_and_rerank(
        user_id=user_id,
        query=f"materi pembelajaran {topic_title}",
        topic_id=topic_id,
        top_k_retrieve=5,
        top_k_rerank=3,
        use_hyde=False,
    )
    context = _build_context_block(chunks)

    prompt = f"""Berdasarkan materi berikut, buat {num_questions} soal pilihan ganda (MCQ) yang menguji pemahaman konsep.

MATERI:
{context}

FORMAT OUTPUT (JSON array):
[
  {{
    "question": "Pertanyaan?",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct_answer": "A. ...",
    "explanation": "Penjelasan singkat mengapa jawaban ini benar."
  }}
]

PENTING: correct_answer harus berupa TEKS LENGKAP dari opsi yang benar (bukan hanya hurufnya). Misalnya jika opsi A adalah "A. Multimedia adalah...", maka correct_answer harus "A. Multimedia adalah...".

Hanya output JSON array, tanpa teks tambahan."""

    llm = get_llm(settings.TUTOR_MODEL)
    response = llm.invoke(prompt)

    import json
    import re
    # Extract JSON from response
    raw = response.content.strip()
    # Try to find JSON array in the response
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("[TUTOR] Could not parse quiz JSON, returning empty quiz.")
    return []
