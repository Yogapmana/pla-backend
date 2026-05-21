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
            f"Tulis jawaban singkat dan padat (2-3 kalimat) untuk pertanyaan berikut "
            f"seolah-olah kamu adalah ahlinya:\n\nPertanyaan: {query}\n\nJawaban:"
        )
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"[HyDE] Generation failed: {e}. Using raw query.")
        return query