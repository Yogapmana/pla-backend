import asyncio
from datetime import datetime
from typing import List
from app.agents.state import PLAState, RawContent, AgentLog
from app.tools.tavily_search import tavily_search_tool
from app.tools.wikipedia_search import wikipedia_search_tool
from app.tools.arxiv_search import arxiv_search_tool
from app.tools.youtube_transcript import youtube_transcript_tool

async def run_tools_for_query(query: str, topic_id: str) -> List[RawContent]:
    """Run search tools in parallel for a single query."""
    
    # We can run tavily, wikipedia, and arxiv concurrently for the query.
    # Youtube tool requires a URL, so we skip it for general queries unless we build a YT search tool.
    # For now, we'll use Tavily, Wikipedia, and Arxiv.
    
    tasks = [
        tavily_search_tool.ainvoke({"query": query}),
        wikipedia_search_tool.ainvoke({"query": query}),
        arxiv_search_tool.ainvoke({"query": query})
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    raw_contents = []
    for res in results:
        if isinstance(res, Exception):
            print(f"Tool error: {res}")
            continue
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict) and "raw_text" in item:
                    raw_contents.append(RawContent(
                        source_type=item["source_type"],
                        source_title=item["source_title"],
                        source_url=item["source_url"],
                        raw_text=item["raw_text"],
                        topic_id=topic_id,
                        relevance_score=item.get("relevance_score", 0.5),
                        fetched_at=datetime.utcnow()
                    ))
    return raw_contents

async def researcher_node(state: PLAState) -> PLAState:
    curriculum = state.get("curriculum")
    if not curriculum:
        return state

    if "agent_logs" not in state or state["agent_logs"] is None:
        state["agent_logs"] = []

    # Find the first pending topic to research
    # In a real system, the orchestrator might specify which topic to process.
    # For now, we process the first one that is "pending".
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
        message=f"Researching topic '{target_topic.title}' with {len(queries)} queries..."
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
    
    success_log = AgentLog(
        timestamp=datetime.utcnow(),
        agent="researcher",
        level="info",
        message=f"Gathered {len(deduped_content)} unique sources for topic '{target_topic.topic_id}'."
    )
    state["agent_logs"].append(success_log)

    return state

# Wrapper if we need a sync node for LangGraph, though LangGraph supports async nodes.
# We will use researcher_node directly if async is supported in the graph.
