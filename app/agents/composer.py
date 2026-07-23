import asyncio
from datetime import datetime

from app.agents.state import AgentLog, LearningModule, SynapsaState
from app.config import settings
from app.utils.llm_factory import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

COMPOSER_PROMPT = """
You are an expert Educator in a Personal Learning Agent system.
Your task is to write a Markdown Learning Module about the topic: {topic_title}.

This module MUST be written in a clean, flowing, and easy-to-understand writing style, much like an educational article or a modern textbook.

Use the following source materials as your primary basis of information:
{sources_text}

Related Course Recommendations:
{courses_text}

WRITING RULES AND MODULE STRUCTURE:

- Create a comprehensive, in-depth, and naturally structured module corresponding to the actual completeness of the material.
- Use a logical Markdown header hierarchy (`#`, `##`, `###`) that matches the sub-topic divisions from the sources.
- Explain core concepts thoroughly and deeply. Use paragraphs, bullet points, or blockquotes fluidly.
- IF there are comparative data, specifications, or technical summaries, it is highly recommended to create **Markdown tables** to make it more structured.
- Provide real-world examples or case studies if such information is available in the reference text.

IMPORTANT:
- Output ONLY pure Markdown text without any conversational filler (do not use phrases like "Here is the module...").
- Adjust the depth of the material for the level: {level}.
{word_limit_instruction}
- WRITE THE ENTIRE CONTENT OF THE MODULE (including title, subtitles, and body text) IN THE LANGUAGE: {language}.
"""

REMEDIAL_PROMPT = """
You are an empathetic, patient, and highly skilled Personal Tutor.
Your task is to write a REMEDIAL Learning Module for a student who struggled with the topic: {topic_title}.

CONTEXT FROM THE STUDENT'S RECENT QUIZ MISTAKES:
{custom_context}

Use the following source materials as your basis of information:
{sources_text}

Related Course Recommendations:
{courses_text}

WRITING RULES FOR REMEDIAL:
- DO NOT use rigid textbook structures or heavy bullet points. Write in a conversational, narrative, and highly supportive tone.
- Address the student's specific mistakes directly if provided in the context, breaking down the misunderstanding gently.
- Explain concepts step-by-step using highly relatable real-world analogies.
- Focus on building a strong foundational understanding rather than overwhelming the student with complex jargon.
- Use a logical Markdown header hierarchy (`#`, `##`, `###`).
- IF there are comparative data, create Markdown tables to make it clearer.

IMPORTANT:
- Output ONLY pure Markdown text without conversational filler at the very beginning (no "Here is your remedial module...").
- Adjust the depth of the material for the level: {level}.
{word_limit_instruction}
- WRITE THE ENTIRE CONTENT OF THE MODULE IN THE LANGUAGE: {language}.
"""

DEEP_DIVE_PROMPT = """
You are an advanced Subject Matter Expert and Mentor in a Personal Learning Agent system.
Your task is to write an ENRICHMENT / DEEP DIVE Module for an advanced student who has mastered the basics of: {topic_title}.

Use the following source materials as your basis of information:
{sources_text}

Related Course Recommendations:
{courses_text}

WRITING RULES FOR DEEP DIVE:
- SKIP basic definitions. Assume the student already knows the fundamentals.
- DO NOT use standard rigid bullet-point summaries. Write in an advanced, essay-like, analytical, or article-style format.
- Focus on advanced applications, theoretical implications, edge cases, historical context, or complex real-world case studies.
- Provoke critical thinking by exploring the "why" and the systemic impact of the topic.
- Use a logical Markdown header hierarchy (`#`, `##`, `###`).
- Use Markdown tables for technical specifications or comparative analyses if applicable.

IMPORTANT:
- Output ONLY pure Markdown text without any conversational filler.
- Adjust the depth of the material for the level: {level}.
{word_limit_instruction}
- WRITE THE ENTIRE CONTENT OF THE MODULE IN THE LANGUAGE: {language}.
"""


def composer_node(state: SynapsaState) -> SynapsaState:
    curriculum = state.get("curriculum")
    research_results = state.get("research_results", [])
    config = state.get("learning_config")

    if not curriculum or not research_results:
        return {}

    if "agent_logs" not in state or state["agent_logs"] is None:
        state["agent_logs"] = []

    # Identify the topic we just researched. We assume it's the one from the research results.
    # Group results by topic_id. For now we just take the first topic found in results.
    topic_ids = list(set([r.topic_id for r in research_results]))
    if not topic_ids:
        return {}

    target_topic_id = topic_ids[0]

    # Get topic title from curriculum
    topic_title = target_topic_id
    for week in curriculum.weeks:
        for day in week.days:
            if day.topic_id == target_topic_id:
                topic_title = day.title
                break

    feedback_actions = state.get("feedback_actions", [])
    module_type = "standard"
    custom_context = ""
    if feedback_actions:
        module_type = feedback_actions[0].get("action", "standard")
        custom_context = feedback_actions[0].get("context", "")

    log = AgentLog(
        timestamp=datetime.utcnow(),
        agent="composer",
        level="info",
        message=f"Composing learning module for '{topic_title}' using {len(research_results)} sources...",
    )
    state["agent_logs"].append(log)

    # Format sources — separate embeddable content from course links
    sources_text = ""
    courses_text = ""
    max_total_chars = 40000  # Long limit since we are using local Ollama (gemma4:12b)
    current_chars = 0
    courses_count = 0

    for idx, r in enumerate(research_results):
        if getattr(r, "embed_mode", True) and r.raw_text:
            if current_chars >= max_total_chars:
                # Stop adding sources entirely to save tokens
                continue
                
            remaining_chars = max_total_chars - current_chars
            text_to_use = r.raw_text[:remaining_chars] + "..." if len(r.raw_text) > remaining_chars else r.raw_text
            
            added_text = f"Sumber [{idx + 1}] ({r.source_type}): {r.source_title}\nURL: {r.source_url}\nKonten:\n{text_to_use}\n\n"
            sources_text += added_text
            current_chars += len(added_text)
        elif not getattr(r, "embed_mode", True) and getattr(r, "course_metadata", None):
            if courses_count >= 10: # Increased limit for courses
                continue
            cm = r.course_metadata
            courses_text += f"Kursus: {cm.title}\nPlatform: {cm.platform}\nURL: {cm.url}\nHarga: {cm.price_type}\nDeskripsi: {cm.description[:500]}\n\n"
            courses_count += 1

    if not courses_text:
        courses_text = "(Tidak ada rekomendasi kursus yang ditemukan untuk topik ini.)"

    if module_type == "remedial":
        selected_prompt = REMEDIAL_PROMPT
    elif module_type == "deep_dive" or module_type == "enrichment":
        selected_prompt = DEEP_DIVE_PROMPT
    else:
        selected_prompt = COMPOSER_PROMPT

    llm = get_llm(settings.COMPOSER_MODEL, temperature=0.3, max_tokens=2500)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", selected_prompt),
            ("human", "Please create the markdown module based on the sources above."),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        hours_per_day = float(getattr(config, "hours_per_day", 1.0))
        if hours_per_day <= 0.5:
            word_limit_instruction = "- The output MUST NOT exceed 600 words. Keep it highly concise."
        elif hours_per_day <= 1.0:
            word_limit_instruction = "- The output MUST be at least 1200 words."
        elif hours_per_day <= 2.0:
            word_limit_instruction = "- The output MUST be at least 1500 words. Provide detailed explanations."
        else:
            word_limit_instruction = "- The output MUST be at least 2000 words. Provide highly comprehensive and exhaustive explanations."

        module_markdown = chain.invoke(
            {
                "topic_title": topic_title,
                "level": config.level if config else "umum",
                "language": config.language if config else "id",
                "sources_text": sources_text,
                "courses_text": courses_text,
                "image_model": settings.IMAGE_MODEL,
                "image_width": settings.IMAGE_WIDTH,
                "image_height": settings.IMAGE_HEIGHT,
                "api_key": settings.POLLINATIONS_API_KEY,
                "word_limit_instruction": word_limit_instruction,
                "custom_context": custom_context,
            }
        )

        module = LearningModule(
            topic_id=target_topic_id,
            title=topic_title,
            content_markdown=module_markdown,
            sources=[
                {"title": r.source_title, "url": r.source_url} for r in research_results
            ],
        )

        if "modules" not in state or state["modules"] is None:
            state["modules"] = []

        state["modules"].append(module)

        success_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="composer",
            level="info",
            message=f"Successfully composed learning module for '{topic_title}'.",
        )
        state["agent_logs"].append(success_log)


        # Mark topic as completed or in_progress in curriculum (optional state mutation)
        for week in curriculum.weeks:
            for day in week.days:
                if day.topic_id == target_topic_id:
                    day.status = "active"  # Module is ready, user can start learning

    except Exception as e:
        error_log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="composer",
            level="error",
            message=f"Error composing module: {str(e)}",
        )
        state["agent_logs"].append(error_log)

    return {
        "modules": state["modules"],
        "agent_logs": state["agent_logs"],
        "curriculum": curriculum
    }
