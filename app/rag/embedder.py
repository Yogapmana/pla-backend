from langchain_ollama import OllamaEmbeddings
from app.config import settings

_embedder: OllamaEmbeddings | None = None

def get_embedder() -> OllamaEmbeddings:
    """Get or create singleton embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    return _embedder

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using nomic-embed-text via Ollama."""
    embedder = get_embedder()
    return embedder.embed_documents(texts)

def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    embedder = get_embedder()
    return embedder.embed_query(text)
