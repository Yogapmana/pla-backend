import logging
from app.rag.hyde import generate_hypothetical_answer
from app.rag.embedder import embed_query
from app.rag.vector_store import get_qdrant_client, search_chunks
from app.rag.reranker import rerank_chunks

logger = logging.getLogger(__name__)


def retrieve_and_rerank(
    user_id: str,
    query: str,
    session_id: str | None = None,
    topic_id: str | None = None,
    top_k_retrieve: int = 8,
    top_k_rerank: int = 3,
    use_hyde: bool = True,
) -> list[dict]:
    """
    Full RAG retrieval pipeline (backwards-compatible):
    1. HyDE — generate hypothetical answer
    2. Dual embedding (query + hypothetical)
    3. Qdrant search with filter
    4. FlashRank cross-encoder re-ranking
    Returns top_k_rerank chunks with score and metadata.
    """
    hypothetical = generate_hypothetical_answer(query) if use_hyde else query

    query_vec = embed_query(query)
    hyp_vec = embed_query(hypothetical)
    combined_vec = [
        0.6 * h + 0.4 * q for h, q in zip(hyp_vec, query_vec)
    ]

    client = get_qdrant_client()
    results = search_chunks(
        client=client,
        user_id=user_id,
        query_vector=combined_vec,
        session_id=session_id,
        topic_id=topic_id,
        top_k=top_k_retrieve,
    )

    if not results:
        logger.warning(f"[RETRIEVER] No chunks found for user {user_id}, topic {topic_id}.")
        return []

    reranked = rerank_chunks(query, results, top_k=top_k_rerank)

    logger.info(f"[RETRIEVER] Retrieved {len(reranked)} chunks after re-ranking.")
    return reranked