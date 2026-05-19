import logging
from tavily import TavilyClient
from app.config import settings

logger = logging.getLogger(__name__)

COURSE_PLATFORMS = [
    "site:udemy.com",
    "site:coursera.org",
    "site:edx.org",
    "site:fast.ai",
    "site:freecodecamp.org",
]

def discover_courses(query: str, max_results: int = 5) -> list[dict]:
    """
    Discover relevant online courses using Tavily site-scoped search.
    Returns CourseLink-compatible metadata (not full text — embed_mode=False).
    """
    site_filter = " OR ".join(COURSE_PLATFORMS)
    search_query = f"{query} course tutorial ({site_filter})"

    try:
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        response = client.search(query=search_query, max_results=max_results, search_depth="basic")
        courses = []
        for r in response.get("results", []):
            url = r.get("url", "")
            platform = _detect_platform(url)
            courses.append({
                "title": r.get("title", ""),
                "platform": platform,
                "url": url,
                "description": r.get("content", "")[:500],
                "price_type": _guess_price_type(platform),
                "source_type": "course",
            })
        logger.info(f"[COURSE_DISCOVERY] Found {len(courses)} courses for '{query}'")
        return courses
    except Exception as e:
        logger.warning(f"[COURSE_DISCOVERY] Error: {e}")
        return []


def _detect_platform(url: str) -> str:
    """Detect the platform from URL."""
    url_lower = url.lower()
    if "udemy.com" in url_lower:
        return "udemy"
    elif "coursera.org" in url_lower:
        return "coursera"
    elif "edx.org" in url_lower:
        return "edx"
    elif "fast.ai" in url_lower:
        return "fastai"
    elif "freecodecamp.org" in url_lower:
        return "freecodecamp"
    elif "youtube.com" in url_lower:
        return "youtube"
    return "other"


def _guess_price_type(platform: str) -> str:
    """Guess price type based on platform."""
    free_platforms = {"freecodecamp", "fastai"}
    audit_platforms = {"coursera", "edx"}
    if platform in free_platforms:
        return "free"
    elif platform in audit_platforms:
        return "audit"
    elif platform == "udemy":
        return "paid"
    return "free"
