import logging
from app.utils.llm_factory import get_llm
from app.config import settings

logger = logging.getLogger(__name__)


def generate_hypothetical_answer(query: str) -> str:
    """
    HyDE: Generate a hypothetical ideal answer using a small LLM.
    This synthetic document is then embedded alongside the real query
    to improve retrieval recall.
    """
    try:
        llm = get_llm(settings.TUTOR_MODEL)
        prompt = (
            f"Write a concise and direct answer (2-3 sentences) to the following question "
            f"as if you are an expert:\n\nQuestion: {query}\n\nAnswer:"
        )
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"[HyDE] Generation failed: {e}. Using raw query.")
        return query