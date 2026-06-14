import httpx
import logging
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger(__name__)

JINA_READER_BASE = "https://r.jina.ai"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True
)
async def fetch_jina_with_retry(url: str, timeout: int) -> str:
    reader_url = f"{JINA_READER_BASE}/{url}"
    headers = {
        "Accept": "text/plain",
        "X-Return-Format": "text",
    }
    if settings.JINA_API_KEY:
        headers["Authorization"] = f"Bearer {settings.JINA_API_KEY}"

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(reader_url, headers=headers)
        response.raise_for_status()
        return response.text.strip()

async def fallback_scrape(url: str, timeout: int) -> str:
    """Fallback scraper using BeautifulSoup if Jina fails."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; PLA/1.0)"}
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "nav", "footer"]):
                script.extract()
            text = soup.get_text(separator=" ", strip=True)
            return text
    except Exception as e:
        logger.warning(f"[FALLBACK] BeautifulSoup failed for {url}: {e}")
        return ""

async def jina_read_url(url: str, timeout: int = 30) -> dict:
    """
    Extract full text content from a URL using Jina Reader API with retries and BS4 fallback.
    """
    try:
        text = await fetch_jina_with_retry(url, timeout)
        if len(text) < 100:
            logger.warning(f"[JINA] Very short content from {url} ({len(text)} chars). Trying fallback...")
            fallback_text = await fallback_scrape(url, timeout)
            if len(fallback_text) > len(text):
                text = fallback_text
                
        return {
            "url": url,
            "text": text[:15000],
            "success": True,
            "char_count": len(text),
        }
    except Exception as e:
        logger.warning(f"[JINA] Error reading {url} after retries: {e}. Attempting fallback...")
        fallback_text = await fallback_scrape(url, timeout)
        if fallback_text:
            return {
                "url": url,
                "text": fallback_text[:15000],
                "success": True,
                "char_count": len(fallback_text),
            }
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
