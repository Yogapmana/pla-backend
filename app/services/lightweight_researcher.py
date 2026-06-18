"""Lightweight researcher — fast 1-2-source-per-topic scrape for the mindmap.

Design intent
-------------
The full ``Researcher`` agent (see ``app/agents/researcher.py``) hits 7+
tools (Tavily, Jina, Wikipedia, ArXiv, Semantic Scholar, YouTube, course
discovery) and fetches a long list of URLs per topic so the Composer has
enough material to write a multi-page learning module. That level of
depth is overkill for the mindmap.

The mindmap only needs *just enough* signal to identify the main themes,
concepts, and key facts in the curriculum. Two high-quality sources per
topic — scraped end-to-end with the same Tavily + Jina stack the full
researcher uses — gives the LLM mindmap generator enough context to
produce NotebookLM-quality output while keeping the wall-clock cost
manageable (a 24-topic course completes in ~15-30 seconds).

Why this is a separate service and not just a flag on the full
researcher:
  * Different scale. 2 sources/topic vs 10+ for the full research.
  * Different output shape. Mindmap generator wants the actual scraped
    text per topic keyed by topic_id, not the heterogeneous
    ``RawContent`` rows the Composer consumes.
  * Triggered later. This runs in the background after the first module
    is composed (user has just entered the dashboard). The full
    researcher runs up front, in parallel with the planner.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import settings
from app.tools.tavily_search import tavily_client
from app.tools.jina_reader import fetch_jina_with_retry

logger = logging.getLogger(__name__)


# How many sources to keep per topic. Two is the sweet spot — enough
# for the LLM to see two angles on each concept, not so many that the
# context window fills up when the mindmap generator batches 24 topics.
SOURCES_PER_TOPIC = 2

# Hard cap on the scraped text we keep per source. Mindmap LLM context
# is precious — 2500 chars (~500 tokens) is enough for the model to
# extract a few key facts without drowning it.
MAX_SCRAPED_CHARS = 2500

# Concurrency for parallel Jina scrapes. 8 in flight keeps us under
# Jina's per-IP rate limit while still finishing 24 topics × 2 sources
# in seconds rather than minutes.
JINA_CONCURRENCY = 8


async def _scrape_one(url: str) -> str:
    """Scrape a single URL via Jina reader, return the truncated text.

    Returns empty string on failure (logged but not raised) so a single
    bad URL doesn't break the whole mindmap generation.
    """
    try:
        text = await fetch_jina_with_retry(url, timeout=20)
        return (text or "")[:MAX_SCRAPED_CHARS]
    except Exception as exc:
        logger.warning(
            "[LIGHTWEIGHT-RESEARCHER] Jina failed for %s: %s", url, exc
        )
        return ""


async def _scrape_parallel(urls: list[str]) -> list[str]:
    """Scrape multiple URLs with bounded concurrency."""
    sem = asyncio.Semaphore(JINA_CONCURRENCY)

    async def _bounded(url: str) -> str:
        async with sem:
            return await _scrape_one(url)

    return await asyncio.gather(*[_bounded(u) for u in urls])


def _tavily_search_sync(query: str, max_results: int) -> list[dict]:
    """Tavily is sync. We wrap in ``asyncio.to_thread`` at the call site."""
    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )
        return response.get("results", [])
    except Exception as exc:
        logger.warning(
            "[LIGHTWEIGHT-RESEARCHER] Tavily failed for %r: %s", query, exc
        )
        return []


async def lightweight_research(
    topics: list[dict],
    *,
    progress_cb=None,
) -> dict[str, list[dict]]:
    """Run 1-2 source Tavily + Jina scrape for every topic.

    Parameters
    ----------
    topics : list[dict]
        Each item is shaped like
        ``{"topic_id": str, "title": str, "search_queries": list[str] |
        None}``. We only need ``topic_id`` and ``title``/``search_queries``
        here; the rest of the topic metadata is irrelevant.
    progress_cb : callable, optional
        Optional callback ``progress_cb(done: int, total: int)`` invoked
        after each topic completes. Used by the Celery task to push
        WebSocket progress events.

    Returns
    -------
    dict[str, list[dict]]
        ``{topic_id: [{"url", "title", "content", "relevance_score"}, ...]}``
        with at most ``SOURCES_PER_TOPIC`` (2) entries per topic. An empty
        list means scraping failed for every source — the caller can
        decide whether to fall back or skip.
    """
    if not topics:
        return {}

    result: dict[str, list[dict]] = {}
    total = len(topics)
    logger.info(
        "[LIGHTWEIGHT-RESEARCHER] starting scrape for %d topics (%d src/topic)",
        total,
        SOURCES_PER_TOPIC,
    )

    # We process topics in batches of 4 — enough to keep Tavily + Jina
    # busy without running into API rate limits for users on the
    # free tier of either service. Sequential between batches is fine
    # because each batch is short-lived.
    BATCH = 4
    for batch_start in range(0, total, BATCH):
        batch = topics[batch_start : batch_start + BATCH]
        batch_tasks = [_process_topic(t) for t in batch]
        batch_results = await asyncio.gather(
            *batch_tasks, return_exceptions=True
        )
        for topic, res in zip(batch, batch_results):
            if isinstance(res, Exception):
                logger.warning(
                    "[LIGHTWEIGHT-RESEARCHER] topic %s failed: %s",
                    topic.get("topic_id"),
                    res,
                )
                result[topic["topic_id"]] = []
            else:
                result[topic["topic_id"]] = res
                logger.info(
                    "[LIGHTWEIGHT-RESEARCHER] %s: scraped %d sources",
                    topic.get("topic_id"),
                    len(res),
                )
            if progress_cb is not None:
                done = batch_start + batch.index(topic) + 1
                try:
                    progress_cb(done, total)
                except Exception:
                    pass

    logger.info(
        "[LIGHTWEIGHT-RESEARCHER] done — %d topics processed", total
    )
    return result


async def _process_topic(topic: dict) -> list[dict]:
    """Tavily search → top URLs → Jina scrape in parallel."""
    topic_id = topic["topic_id"]
    queries = topic.get("search_queries") or [topic.get("title", "")]
    if not queries:
        return []

    # Use the first search_query (the most specific one — the planner
    # orders them). Fall back to title if there are no search queries.
    query = queries[0]
    if not query:
        query = topic.get("title", topic_id)

    # 1. Tavily search — returns top URLs + snippets
    tavily_results = await asyncio.to_thread(
        _tavily_search_sync, query, SOURCES_PER_TOPIC
    )
    if not tavily_results:
        return []

    # 2. Jina scrape the URLs in parallel
    urls = [r.get("url", "") for r in tavily_results if r.get("url")]
    if not urls:
        return []

    scraped_texts = await _scrape_parallel(urls)

    # 3. Merge Tavily metadata with scraped text
    out: list[dict] = []
    for meta, text in zip(tavily_results, scraped_texts):
        url = meta.get("url", "")
        if not url:
            continue
        out.append(
            {
                "url": url,
                "title": meta.get("title", ""),
                "content": text or meta.get("content", "")[:MAX_SCRAPED_CHARS],
                "relevance_score": float(meta.get("score", 0.0)),
            }
        )
    return out[:SOURCES_PER_TOPIC]
