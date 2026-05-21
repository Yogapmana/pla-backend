import logging
from typing import Any
from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

_ranker: Ranker | None = None


def get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker()
    return _ranker


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    FlashRank cross-encoder re-ranking.
    Takes a query and list of chunk dicts (each with 'text' key),
    returns top_k chunks sorted by cross-encoder relevance score.
    """
    if not chunks:
        return []

    ranker = get_ranker()
    passages = [{"id": i, "text": r.get("text", "")} for i, r in enumerate(chunks)]

    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)

    top_results = []
    for item in reranked[:top_k]:
        orig = chunks[item.get("id", 0)]
        orig["rerank_score"] = float(item.get("score", 0.0))
        top_results.append(orig)

    logger.info(f"[Reranker] Re-ranked {len(chunks)} chunks down to {len(top_results)}.")
    return top_results