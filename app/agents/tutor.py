import asyncio
import logging
import time
from datetime import datetime

from app.agents.state import SynapsaState
from app.config import settings
from app.rag.retriever import retrieve_and_rerank
from app.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

TUTOR_SYSTEM_PROMPT = """You are an interactive and supportive AI Tutor assigned to help users understand their learning materials.
Use the provided context (module) as your primary basis or starting point, but you are HIGHLY ENCOURAGED to:
1. Provide deeper explanations, real-world examples, and easy-to-understand analogies.
2. Expand the answer with your general knowledge if it helps the user understand the concept better.
3. Act as a guide, not just reading the text. Explain the 'why' and 'how' behind a concept.
4. Answer in the language: {language}.

IMPORTANT: IF THE USER ASKS ABOUT A TOPIC THAT IS COMPLETELY UNRELATED TO THE MODULE MATERIAL, POLITELY REFUSE AND REDIRECT THEM BACK TO THE LEARNING TOPIC (Example: "Sorry, that question is outside the scope of this module. Let's get back to focusing on [Module Topic]").

If you use specific information from the context, you can include the references. However, make sure your tone remains natural and flows like a professional teacher."""


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
    recent = history[-max_turns * 2 :]  # user + assistant pairs
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
    language: str = "id",
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
        session_id=session_id,
        topic_id=topic_id,
        top_k_retrieve=8,
        top_k_rerank=3,
        use_hyde=True,
    )

    # Step 2: Build prompt
    context_block = _build_context_block(chunks)
    history_str = _format_chat_history(chat_history)

    prompt_parts = [
        TUTOR_SYSTEM_PROMPT.format(language=language),
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
                sources.append(
                    {
                        "title": title,
                        "type": c.get("source_type", "module"),
                        "relevance": float(round(c.get("rerank_score", 0.0), 3)),
                    }
                )
                seen.add(title)

    logger.info(
        f"[TUTOR] Response generated in {latency_ms}ms with {len(chunks)} context chunks."
    )

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
    language: str = "id",
    num_questions: int = 10,
    difficulty: str = "menengah",
) -> list[dict]:
    """
    Quiz Generation Mode:
    Retrieves topic context from Qdrant and generates MCQ questions.

    On JSON parse failure (LLM returns malformed output), we retry
    once with a stricter prompt. Only after the retry fails do we
    return an empty list — and only as a last resort. The downstream
    `get_quiz` endpoint then 404s the request so the client gets a
    clear "could not generate" error instead of silently submitting
    a 0-question quiz that would record as 0/0 benar on the user's
    history.
    """
    logger.info(
        f"[TUTOR] Generating {num_questions} quiz questions (Difficulty: {difficulty}) for '{topic_title}'..."
    )

    # Get context from Qdrant, fetching more chunks and sampling randomly to increase variety
    import random
    
    chunks = retrieve_and_rerank(
        user_id=user_id,
        query=f"materi pembelajaran {topic_title}",
        topic_id=topic_id,
        top_k_retrieve=10,
        top_k_rerank=5,
        use_hyde=False,
    )
    if len(chunks) > 3:
        chunks = random.sample(chunks, 3)
        
    context = _build_context_block(chunks)

    prompt = f"""Based on the following material, generate {num_questions} Multiple Choice Questions (MCQ) RANDOMLY.
WRITE ALL QUESTIONS AND ANSWERS IN THE LANGUAGE: {language}.

QUESTION DIFFICULTY LEVEL: {difficulty}
(Difficulty Guidelines:
- Easy: Questions focus on concept recognition and simple applicable case studies. Do not just ask for pure definitions.
- Medium: Requires users to analyze relationships between concepts and solve intermediate-level scenarios.
- Hard: Focus on complex problem-solving, in-depth analysis, or logic traps (trick questions) based on the material.)

IMPORTANT FOR QUESTION VARIETY AND CREATIVITY:
- Randomize the concepts you pick from the material extremely (do not just focus on the beginning of the material).
- Create questions from different perspectives, such as application case studies, error scenarios, or concept comparisons.
- Ensure the distractor options are highly plausible, not too easy to guess, and provoke deep understanding.
- DO NOT repeat the same question patterns. Make each question feel unique and test different aspects of the material.

MATERIAL:
{context}

QUESTION RATIO: 100% MCQ
- MCQ: 4 options (A, B, C, D)

OUTPUT FORMAT (JSON array):
[
  {{
    "question": "Question text?",
    "question_type": "mcq",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct_answer": "A. ...",
    "explanation": "Brief explanation of why this answer is correct."
  }}
]

IMPORTANT:
- correct_answer for MCQ must be the FULL TEXT of the correct option
- question_type must be: "mcq"

Output only the JSON array, no additional text."""

    import json
    import re

    def _try_parse(raw_text: str) -> list[dict]:
        """Pull a JSON array out of the LLM response and validate it.
        Returns [] if the regex didn't match OR if json.loads failed
        OR if the parsed result isn't a non-empty list of dicts each
        with a `question` key.
        """
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list) or len(parsed) == 0:
            return []
        if not all(isinstance(item, dict) and item.get("question") for item in parsed):
            return []
        return parsed

    # Set temperature to 0.9 for much higher creativity and extreme question variety
    llm = get_llm(settings.TUTOR_MODEL, temperature=0.9)

    # First attempt — the normal prompt.
    response = llm.invoke(prompt)
    # `response.content` is typed as `list[Union[str, dict]]` in newer
    # LangChain versions but at runtime it's always a str for invoke().
    # The `# type: ignore` is needed because Pyright can't narrow it
    # without the full LangChain stubs.
    response_text = response.content.strip()  # type: ignore[attr-defined]
    questions = _try_parse(response_text)
    if questions:
        return questions

    # Retry once with a stricter, "JSON only" prompt. This handles
    # the common LLM failure mode where it adds a preamble like
    # "Berikut soalnya:" or wraps the array in code fences.
    logger.warning(
        "[TUTOR] First quiz JSON parse failed for '%s', retrying with stricter prompt.",
        topic_title,
    )
    strict_prompt = (
        f"Create exactly {num_questions} multiple choice questions in JSON array format.\n"
        "DO NOT add any text outside the array. The output MUST start with '[' and end with ']'.\n"
        "EACH question MUST have these keys exactly: 'question', 'question_type' (must be 'mcq'), "
        "'options' (array of 4 strings), 'correct_answer' (must match one of the options exactly), and 'explanation'.\n\n"
        f"MATERIAL:\n{context}"
    )
    response = llm.invoke(strict_prompt)
    response_text = response.content.strip()  # type: ignore[attr-defined]
    questions = _try_parse(response_text)
    if questions:
        logger.info(
            "[TUTOR] Retry succeeded for '%s' with %d questions.",
            topic_title,
            len(questions),
        )
        return questions

    # Both attempts failed — log loudly and return empty list.
    # The get_quiz endpoint will 404 the request so the client
    # never records a 0/0 attempt.
    logger.error(
        "[TUTOR] Could not parse quiz JSON for '%s' after retry. "
        "Returning empty quiz (client will see 404).",
        topic_title,
    )
    return []
