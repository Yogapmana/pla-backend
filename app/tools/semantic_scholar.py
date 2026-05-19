import httpx
import logging

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_semantic_scholar(query: str, limit: int = 5) -> list[dict]:
    """
    Search academic papers via Semantic Scholar API.
    Free, no API key required. Returns title, abstract, authors, year, citations, URL.
    """
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,citationCount,url,externalIds",
    }
    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(SEMANTIC_SCHOLAR_API, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        for paper in data.get("data", []):
            abstract = paper.get("abstract") or ""
            authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])[:3]])
            # Build a proper URL
            paper_url = paper.get("url", "")
            if not paper_url:
                paper_id = paper.get("paperId", "")
                paper_url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""

            results.append({
                "title": paper.get("title", ""),
                "abstract": abstract[:3000],
                "authors": authors,
                "year": paper.get("year"),
                "citation_count": paper.get("citationCount", 0),
                "url": paper_url,
                "source_type": "semantic_scholar",
            })
        logger.info(f"[SEMANTIC_SCHOLAR] Found {len(results)} papers for '{query}'")
        return results

    except httpx.TimeoutException:
        logger.warning(f"[SEMANTIC_SCHOLAR] Timeout for query '{query}'")
        return []
    except Exception as e:
        logger.warning(f"[SEMANTIC_SCHOLAR] Error: {e}")
        return []
