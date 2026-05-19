from langchain_core.tools import tool
import arxiv

@tool
def arxiv_search_tool(query: str, max_results: int = 3) -> list[dict]:
    """Search Arxiv for academic papers and return their abstracts.
    Useful for deep academic or scientific research."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in client.results(search):
            results.append({
                "source_type": "arxiv",
                "source_title": result.title,
                "source_url": result.pdf_url,
                "raw_text": result.summary,
                "relevance_score": 0.85
            })
        return results
    except Exception as e:
        print(f"Arxiv search error for {query}: {e}")
        return []
