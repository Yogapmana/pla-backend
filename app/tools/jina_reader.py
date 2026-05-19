import httpx
import logging

from app.config import settings

logger = logging.getLogger(__name__)

JINA_READER_BASE = "https://r.jina.ai"

async def jina_read_url(url: str, timeout: int = 30) -> dict:
    """
    Extract full text content from a URL using Jina Reader API.
    Jina Reader converts any URL to clean, LLM-friendly text.
    Uses JINA_API_KEY if configured, otherwise falls back to anonymous access.
    """
    reader_url = f"{JINA_READER_BASE}/{url}"
    headers = {
        "Accept": "text/plain",
        "X-Return-Format": "text",
    }
    if settings.JINA_API_KEY:
        headers["Authorization"] = f"Bearer {settings.JINA_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(reader_url, headers=headers)
            response.raise_for_status()
            text = response.text.strip()
            if len(text) < 100:
                logger.warning(f"[JINA] Very short content from {url} ({len(text)} chars)")
            return {
                "url": url,
                "text": text[:15000],
                "success": True,
                "char_count": len(text),
            }
    except httpx.TimeoutException:
        logger.warning(f"[JINA] Timeout reading {url}")
        return {"url": url, "text": "", "success": False, "error": "timeout"}
    except httpx.HTTPStatusError as e:
        logger.warning(f"[JINA] HTTP {e.response.status_code} for {url}")
        return {"url": url, "text": "", "success": False, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        logger.warning(f"[JINA] Error reading {url}: {e}")
        return {"url": url, "text": "", "success": False, "error": str(e)}


async def jina_read_urls(urls: list[str], timeout: int = 30) -> list[dict]:
    """Read multiple URLs in parallel using Jina Reader."""
    import asyncio
    tasks = [jina_read_url(url, timeout) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for r in results:
        if isinstance(r, Exception):
            output.append({"url": "", "text": "", "success": False, "error": str(r)})
        else:
            output.append(r)
    return output
