from langchain_core.tools import tool
import wikipedia

@tool
def wikipedia_search_tool(query: str) -> list[dict]:
    """Search Wikipedia for a given topic and return the summary.
    Useful for factual information, definitions, and broad topic overviews."""
    try:
        wikipedia.set_lang("id") # Try Indonesian first
        search_results = wikipedia.search(query)
        
        if not search_results:
            wikipedia.set_lang("en") # Fallback to English
            search_results = wikipedia.search(query)
            
        if not search_results:
            return []
            
        page_title = search_results[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        
        return [{
            "source_type": "wikipedia",
            "source_title": page.title,
            "source_url": page.url,
            "raw_text": page.summary,
            "relevance_score": 0.9 # Wikipedia usually highly relevant for factual queries
        }]
    except Exception as e:
        print(f"Wikipedia search error for {query}: {e}")
        return []
