import asyncio
import logging
from datetime import datetime
from typing import List
from app.agents.state import PLAState, RawContent, CourseLink, AgentLog
from app.tools.tavily_search import tavily_search_tool
from app.tools.wikipedia_search import wikipedia_search_tool
from app.tools.arxiv_search import arxiv_search_tool
from app.tools.youtube_transcript import youtube_transcript_tool
from app.tools.jina_reader import jina_read_urls
from app.tools.semantic_scholar import search_semantic_scholar
from app.tools.course_discovery import discover_courses

logger = logging.getLogger(__name__)


async def run_tools_for_query(query: str, topic_id: str) -> List[RawContent]:
    """
    Layered scraping strategy per PRD:
    1. Tavily search → discover URLs + snippets
    2. Jina Reader → full text from Tavily URLs (parallel)
    3. Dedicated tools (Wikipedia, Arxiv, Semantic Scholar) → parallel
    4. Course discovery (Tavily site-scoped) → embed_mode=False
    """
    raw_contents: List[RawContent] = []

    # --- Layer 1: Tavily discovery + Wikipedia + Arxiv + Semantic Scholar (parallel) ---
    tasks = [
        tavily_search_tool.ainvoke({"query": query}),
        wikipedia_search_tool.ainvoke({"query": query}),
        arxiv_search_tool.ainvoke({"query": query}),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    tavily_results = []
    for res in results:
        if isinstance(res, Exception):
            logger.warning(f"[RESEARCHER] Tool error: {res}")
            continue
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict) and "raw_text" in item:
                    tavily_results.append(item) if item.get("source_type") == "web" else None
                    raw_contents.append(RawContent(
                        source_type=item["source_type"],
                        source_title=item["source_title"],
                        source_url=item["source_url"],
                        raw_text=item["raw_text"],
                        topic_id=topic_id,
                        relevance_score=item.get("relevance_score", 0.5),
                        fetched_at=datetime.utcnow(),
                        display_url=item["source_url"],
                    ))

    # --- Layer 1b: Semantic Scholar (sync, wrapped) ---
    try:
        papers = await asyncio.to_thread(search_semantic_scholar, query, 3)
        for paper in papers:
            if paper.get("abstract"):
                raw_contents.append(RawContent(
                    source_type="semantic_scholar",
                    source_title=paper["title"],
                    source_url=paper["url"],
                    raw_text=f"Title: {paper['title']}\nAuthors: {paper['authors']}\nYear: {paper.get('year')}\nCitations: {paper.get('citation_count', 0)}\n\nAbstract:\n{paper['abstract']}",
                    topic_id=topic_id,
                    relevance_score=0.7,
                    fetched_at=datetime.utcnow(),
                    display_url=paper["url"],
                ))
    except Exception as e:
        logger.warning(f"[RESEARCHER] Semantic Scholar error: {e}")

    # --- Layer 2: Jina Reader — full text from Tavily-discovered URLs ---
    tavily_urls = [item["source_url"] for item in tavily_results if item.get("source_url")]
    if tavily_urls:
        try:
            jina_results = await jina_read_urls(tavily_urls[:5], timeout=20)
            for jina_res in jina_results:
                if jina_res.get("success") and len(jina_res.get("text", "")) > 200:
                    # Replace the Tavily snippet with full Jina text for matching URL
                    for rc in raw_contents:
                        if rc.source_url == jina_res["url"] and rc.source_type == "web":
                            rc.raw_text = jina_res["text"]
                            logger.info(f"[RESEARCHER] Jina enriched: {jina_res['url'][:60]}... ({jina_res['char_count']} chars)")
                            break
        except Exception as e:
            logger.warning(f"[RESEARCHER] Jina Reader error (graceful skip): {e}")

    # --- Layer 3: Course discovery (embed_mode=False) ---
    try:
        courses = await asyncio.to_thread(discover_courses, query, 3)
        for course in courses:
            raw_contents.append(RawContent(
                source_type="course",
                source_title=course["title"],
                source_url=course["url"],
                raw_text="",  # no text for courses — link only
                topic_id=topic_id,
                relevance_score=0.6,
                fetched_at=datetime.utcnow(),
                embed_mode=False,
                display_url=course["url"],
                course_metadata=CourseLink(
                    title=course["title"],
                    platform=course.get("platform", "other"),
                    url=course["url"],
                    price_type=course.get("price_type", "free"),
                    description=course.get("description", ""),
                ),
            ))
    except Exception as e:
        logger.warning(f"[RESEARCHER] Course discovery error: {e}")

    return raw_contents


async def researcher_node(state: PLAState) -> PLAState:
    curriculum = state.get("curriculum")
    if not curriculum:
        return state

    if "agent_logs" not in state or state["agent_logs"] is None:
        state["agent_logs"] = []

    # Find the first pending topic to research
    target_topic = None
    for week in curriculum.weeks:
        for day in week.days:
            if day.status == "pending":
                target_topic = day
                break
        if target_topic:
            break

    if not target_topic:
        log = AgentLog(
            timestamp=datetime.utcnow(),
            agent="researcher",
            level="info",
            message="No pending topics to research."
        )
        state["agent_logs"].append(log)
        return state

    queries = target_topic.search_queries
    log = AgentLog(
        timestamp=datetime.utcnow(),
        agent="researcher",
        level="info",
        message=f"Researching topic '{target_topic.title}' with {len(queries)} queries (Tavily+Jina+Scholar+Courses)..."
    )
    state["agent_logs"].append(log)

    all_raw_content = []

    # Run all queries in parallel
    query_tasks = [run_tools_for_query(q, target_topic.topic_id) for q in queries]
    query_results = await asyncio.gather(*query_tasks)

    for contents in query_results:
        all_raw_content.extend(contents)

    # Deduplicate based on URL
    seen_urls = set()
    deduped_content = []
    for content in all_raw_content:
        if content.source_url not in seen_urls:
            seen_urls.add(content.source_url)
            deduped_content.append(content)

    if "research_results" not in state or state["research_results"] is None:
        state["research_results"] = []

    state["research_results"].extend(deduped_content)

    # Count by type
    embed_count = sum(1 for c in deduped_content if c.embed_mode)
    course_count = sum(1 for c in deduped_content if not c.embed_mode)

    success_log = AgentLog(
        timestamp=datetime.utcnow(),
        agent="researcher",
        level="info",
        message=f"Gathered {len(deduped_content)} sources ({embed_count} embeddable, {course_count} course links) for '{target_topic.topic_id}'."
    )
    state["agent_logs"].append(success_log)

    return state
