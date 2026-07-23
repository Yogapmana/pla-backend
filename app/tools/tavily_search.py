from langchain_core.tools import tool
from tavily import TavilyClient
from app.config import settings

tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

@tool
def tavily_search_tool(query: str, max_results: int = 5, language: str = "id") -> list[dict]:
    """Search the web for information using Tavily API. 
    Useful for finding up-to-date information, articles, and general knowledge.
    Returns a list of search results with 'title', 'url', and 'content'."""
    try:
        response = tavily_client.search(query=query, max_results=max_results, search_depth="advanced")
        results = []
        for res in response.get("results", []):
            results.append({
                "source_type": "web",
                "source_title": res.get("title", ""),
                "source_url": res.get("url", ""),
                "raw_text": res.get("content", ""),
                "relevance_score": res.get("score", 0.0)
            })
        return results
    except Exception as e:
        print(f"Tavily search error: {e}")
        return []
