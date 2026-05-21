import logging
from typing import Any
from app.rag.hyde import generate_hypothetical_answer
from app.rag.embedder import embed_query
from app.rag.vector_store import get_qdrant_client, search_chunks
from app.rag.reranker import rerank_chunks
from app.utils.llm_factory import get_llm
from app.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service layer for RAG operations: retrieval, generation, and evaluation."""

    def __init__(self):
        self.llm = get_llm(settings.TUTOR_MODEL)

    async def retrieve_context(
        self,
        user_id: str,
        query: str,
        topic_id: str | None = None,
        top_k_retrieve: int = 8,
        top_k_rerank: int = 3,
        use_hyde: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Full RAG retrieval pipeline:
        1. HyDE — generate hypothetical answer
        2. Dual embedding (query + hypothetical)
        3. Qdrant search with filter
        4. FlashRank cross-encoder re-ranking
        Returns top_k_rerank chunks with score and metadata.
        """
        try:
            hypothetical = await generate_hypothetical_answer(query) if use_hyde else query

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
                topic_id=topic_id,
                top_k=top_k_retrieve,
            )

            if not results:
                logger.warning(f"[RAG] No chunks found for user {user_id}, topic {topic_id}.")
                return []

            reranked = rerank_chunks(query, results, top_k=top_k_rerank)

            logger.info(f"[RAG] Retrieved {len(reranked)} chunks after re-ranking.")
            return reranked

        except Exception as e:
            logger.error(f"[RAG] Retrieval error: {e}")
            return []

    def build_prompt(
        self,
        system_prompt: str,
        context_chunks: list[dict[str, Any]],
        chat_history: list[dict[str, str]] | None = None,
        user_query: str | None = None,
    ) -> str:
        """
        Build a full prompt from context chunks, optional chat history, and user query.
        """
        context_text = "\n\n".join(
            f"[Source: {c.get('source_title', 'Unknown')}]({c.get('source_url', '')})\\n{c.get('text', '')}"
            for c in context_chunks
        )

        history_text = ""
        if chat_history:
            history_text = "\\n".join(
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in chat_history[-5:]
            )

        prompt_parts = [
            system_prompt,
            "\\n--- Konteks ---",
            context_text,
        ]

        if history_text:
            prompt_parts.extend(["\\n--- Riwayat Chat ---", history_text])

        if user_query:
            prompt_parts.extend(["\\n--- Pertanyaan ---", user_query])

        return "\\n".join(prompt_parts)

    async def generate_response(
        self,
        user_id: str,
        user_query: str,
        topic_id: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Full RAG query pipeline:
        1. Retrieve context
        2. Build prompt
        3. Generate response
        Returns response with sources and latency.
        """
        import time
        start = time.time()

        if system_prompt is None:
            system_prompt = (
                "Anda adalah Tutor AI yang helpful dan knowledgeable. "
                "Jawab berdasarkan konteks yang diberikan. "
                "Jika informasi tidak cukup, katakan dengan jujur."
            )

        chunks = await self.retrieve_context(
            user_id=user_id,
            query=user_query,
            topic_id=topic_id,
        )

        if not chunks:
            return {
                "response": "Maaf, saya tidak menemukan informasi yang cukup untuk menjawab pertanyaan Anda. Silakan coba topik lain atau ajukan pertanyaan yang berbeda.",
                "sources": [],
                "latency_ms": int((time.time() - start) * 1000),
            }

        prompt = self.build_prompt(
            system_prompt=system_prompt,
            context_chunks=chunks,
            chat_history=chat_history,
            user_query=user_query,
        )

        try:
            llm_response = self.llm.invoke(prompt)
            response_text = llm_response.content.strip()
        except Exception as e:
            logger.error(f"[RAG] Generation failed: {e}")
            return {
                "response": "Maaf, terjadi kesalahan saat generating respons.",
                "sources": [],
                "latency_ms": int((time.time() - start) * 1000),
            }

        sources = [
            {
                "title": c.get("source_title", "Unknown"),
                "type": c.get("source_type", "web"),
                "url": c.get("source_url", ""),
                "relevance": float(c.get("rerank_score", 0.0)),
            }
            for c in chunks
        ]

        return {
            "response": response_text,
            "sources": sources,
            "latency_ms": int((time.time() - start) * 1000),
        }


def get_rag_service() -> RAGService:
    return RAGService()