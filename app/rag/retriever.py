import logging
from flashrank import Ranker, RerankRequest
from app.rag.embedder import embed_query, embed_texts
from app.rag.vector_store import get_qdrant_client, search_chunks
from app.utils.llm_factory import get_llm
from app.config import settings

logger = logging.getLogger(__name__)

# Singleton FlashRank ranker (lightweight cross-encoder)
_ranker: Ranker | None = None

def get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker()  # Uses ms-marco-MiniLM-L-12-v2 by default
    return _ranker

def generate_hyde_answer(query: str) -> str:
    """
    HyDE: Generate a hypothetical ideal answer using a small LLM.
    This synthetic document is then embedded alongside the real query
    to improve retrieval recall.
    """
    try:
        llm = get_llm(settings.TUTOR_MODEL)
        prompt = (
            f"Tulis jawaban singkat dan padat (2-3 kalimat) untuk pertanyaan berikut "
            f"seolah-olah kamu adalah ahlinya:\n\nPertanyaan: {query}\n\nJawaban:"
        )
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"[RETRIEVER] HyDE generation failed: {e}. Using raw query.")
        return query

def retrieve_and_rerank(
    user_id: str,
    query: str,
    topic_id: str | None = None,
    top_k_retrieve: int = 8,
    top_k_rerank: int = 3,
    use_hyde: bool = True,
) -> list[dict]:
    """
    Full RAG retrieval pipeline:
    1. HyDE — generate hypothetical answer
    2. Dual embedding (query + hypothetical)
    3. Qdrant search with filter
    4. FlashRank cross-encoder re-ranking
    Returns top_k_rerank chunks with score and metadata.
    """
    # Step 1: HyDE
    hypothetical = generate_hyde_answer(query) if use_hyde else query

    # Step 2: Dual embedding — average query and hypothetical vectors
    query_vec = embed_query(query)
    hyp_vec = embed_query(hypothetical)
    # Weighted average: 60% hypothetical, 40% original query
    combined_vec = [
        0.6 * h + 0.4 * q for h, q in zip(hyp_vec, query_vec)
    ]

    # Step 3: Vector retrieval from Qdrant
    client = get_qdrant_client()
    results = search_chunks(
        client=client,
        user_id=user_id,
        query_vector=combined_vec,
        topic_id=topic_id,
        top_k=top_k_retrieve,
    )

    if not results:
        logger.warning(f"[RETRIEVER] No chunks found for user {user_id}, topic {topic_id}.")
        return []

    # Step 4: FlashRank re-ranking
    ranker = get_ranker()
    passages = [{"id": i, "text": r["text"]} for i, r in enumerate(results)]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)

    # Map back to original result dicts, sorted by rerank score
    top_results = []
    for item in reranked[:top_k_rerank]:
        orig = results[item.get("id", 0)]
        orig["rerank_score"] = item.get("score", 0.0)
        top_results.append(orig)

    logger.info(f"[RETRIEVER] Retrieved {len(top_results)} chunks after re-ranking.")
    return top_results
