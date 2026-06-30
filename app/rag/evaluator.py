"""
RAGAS-based evaluation for RAG responses.

Computes two LLM-as-judge metrics per chat message:
  1. Faithfulness       — does the answer stay within the retrieved context?
  2. Answer Relevancy   — does the answer address the user's question?

Status: RAGAS path is DISABLED by default because the RAGAS library
has known incompatibility with Groq (RAGAS calls the LLM with
`n > 1` for some metrics, but Groq only allows `n=1`). We rely on
the lightweight LLM-as-judge fallback, which is also faster and
cheaper. Set env ENABLE_RAGAS=1 to re-enable the RAGAS path.

Both metrics are written to `chat_messages.rag_faithfulness` and
`chat_messages.rag_answer_relevancy` for later analysis.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


def _try_ragas():
    """Lazy import of RAGAS. Returns (RAGEvaluator, available).

    Disabled by default (RAGAS_Groq incompatibility). Set env
    ENABLE_RAGAS=1 to opt in.
    """
    if not os.getenv("ENABLE_RAGAS", "").lower() in ("1", "true", "yes"):
        return {"available": False, "reason": "ENABLE_RAGAS not set"}
    try:
        from ragas import evaluate  # noqa: F401
        from ragas.metrics import faithfulness, answer_relevancy  # noqa: F401
        return {
            "available": True,
            "evaluate": evaluate,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        }
    except ImportError:
        return {"available": False, "reason": "ragas package not installed"}


_RAGAS = _try_ragas()


class RAGEvaluator:
    """Service that scores a RAG response with RAGAS metrics."""

    def __init__(self, llm, embeddings=None):
        self.llm = llm
        self.embeddings = embeddings

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, Any]:
        """
        Score a RAG response. Returns dict with at least
        `{ "rag_faithfulness": float, "rag_answer_relevancy": float, "method": str }`.
        Never raises — failures are caught and logged.
        """
        if not answer or not question:
            return {"rag_faithfulness": None, "rag_answer_relevancy": None, "method": "skipped"}

        # Try RAGAS first; fall back to lightweight LLM judge.
        if _RAGAS.get("available") and self.embeddings is not None:
            try:
                return await asyncio.to_thread(self._run_ragas, question, answer, contexts)
            except Exception as e:
                logger.warning(f"[RAGAS] RAGAS eval failed, falling back: {e}")

        # Lightweight LLM-as-judge fallback
        try:
            return await self._lightweight_eval(question, answer, contexts)
        except Exception as e:
            logger.error(f"[RAGAS] Even lightweight eval failed: {e}")
            return {"rag_faithfulness": None, "rag_answer_relevancy": None, "method": "failed"}

    def _run_ragas(self, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
        """Run real RAGAS evaluation (sync; called via to_thread)."""
        from datasets import Dataset  # ragas depends on it

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts] if contexts else [[""]],
        }
        ds = Dataset.from_dict(data)
        result = _RAGAS["evaluate"](
            ds,
            metrics=[_RAGAS["faithfulness"], _RAGAS["answer_relevancy"]],
            llm=self.llm,
            embeddings=self.embeddings,
        )
        return {
            "rag_faithfulness": _extract_score(result, "faithfulness"),
            "rag_answer_relevancy": _extract_score(result, "answer_relevancy"),
            "method": "ragas",
        }

    def _extract_scalar(value) -> float | None:
        """RAGAS can return a scalar, a list of one element, or a string.

        Normalize to float or None.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        if isinstance(value, list):
            # RAGAS sometimes returns a list of per-row scores
            if not value:
                return None
            first = value[0]
            return _extract_scalar(first)
        return None

    async def _lightweight_eval(
        self, question: str, answer: str, contexts: list[str]
    ) -> dict[str, Any]:
        """
        Lightweight LLM-as-judge fallback. Asks the LLM to rate two
        dimensions on 0-1 scale and parses the numeric answer.

        Forces n=1 in the LLM call (Groq compatibility).
        Prompt is bilingual (Indonesian) to match Synapsa's domain.
        """
        ctx_str = "\n\n---\n\n".join(
            (c or "")[:1200] for c in (contexts or [])[:5]
        )
        prompt = (
            "Anda adalah penilai kualitas jawaban AI. Skor 0.0-1.0 untuk dua metrik.\n\n"
            f"PERTANYAAN USER:\n{question[:500]}\n\n"
            f"JAWABAN AI:\n{answer[:1000]}\n\n"
            f"KONTEKS YANG DIRETRIEVE:\n{ctx_str[:3000]}\n\n"
            "TUGAS:\n"
            "1. FAITHFULNESS (0.0-1.0): Apakah semua klaim faktual di JAWABAN "
            "didukung oleh KONTEKS? (1.0 = semua klaim ada di konteks; 0.0 = banyak halusinasi)\n"
            "2. ANSWER_RELEVANCY (0.0-1.0): Apakah JAWABAN relevan dengan "
            "PERTANYAAN USER? (1.0 = langsung menjawab; 0.0 = melenceng)\n\n"
            "Format jawaban HANYA sebagai JSON valid, tanpa teks lain:\n"
            '{"faithfulness": 0.85, "answer_relevancy": 0.92}'
        )
        # Call LLM with explicit n=1 (Groq only allows n=1). Some
        # langchain providers support `n` via .bind(); we set the
        # attribute on the request if the underlying client supports it.
        try:
            raw = await asyncio.to_thread(self._invoke_with_n1, prompt)
        except Exception as e:
            logger.error(f"[RAGAS] LLM invoke failed: {e}")
            return {
                "rag_faithfulness": None,
                "rag_answer_relevancy": None,
                "method": "invoke_failed",
            }
        text = raw.content.strip() if hasattr(raw, "content") else str(raw)

        # Try several parsing strategies
        faith = self._parse_score_from_text(text, "faithfulness")
        relev = self._parse_score_from_text(text, "answer_relevancy")

        return {
            "rag_faithfulness": faith,
            "rag_answer_relevancy": relev,
            "method": "llm_judge",
        }

    def _invoke_with_n1(self, prompt: str):
        """
        Invoke the underlying LLM with n=1 explicitly.

        Different providers expose this differently:
        - ChatGroq / ChatOpenAI: pass `n=1` as a kwarg
        - ChatOllama: doesn't support `n`
        We try multiple signatures for robustness.
        """
        # Strategy 1: pass n=1 as a kwarg
        try:
            return self.llm.invoke(prompt, n=1)
        except TypeError:
            pass
        # Strategy 2: use .bind() to set n
        try:
            bound = self.llm.bind(n=1)
            return bound.invoke(prompt)
        except Exception:
            pass
        # Strategy 3: invoke without n (works for Ollama)
        return self.llm.invoke(prompt)

    def _parse_score_from_text(self, text: str, key: str) -> float | None:
        """
        Extract a numeric score for a given key from a JSON-ish text.
        Handles scalars, lists, and the messy text that LLMs sometimes
        return (e.g. '0.85' or '0.85/1.0' or '~0.85').
        """
        # 1. Try to find a JSON object containing the key
        # Look for { ... "key": value ... } patterns (greedy non-greedy with quotes)
        # Use a forgiving pattern that handles nested braces minimally.
        pattern = r'"' + re.escape(key) + r'"\s*:\s*([-+]?\d*\.?\d+)'
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
            except (ValueError, TypeError):
                pass

        # 2. Try key: value (no quotes)
        pattern2 = r'(?:^|[\s,])' + re.escape(key) + r'\s*:\s*([-+]?\d*\.?\d+)'
        m = re.search(pattern2, text)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
            except (ValueError, TypeError):
                pass

        # 3. Try JSON parse with bracket matching
        try:
            # Find the first '{' and match balanced braces
            start = text.find('{')
            if start >= 0:
                depth = 0
                end = -1
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end > 0:
                    json_str = text[start:end]
                    data = json.loads(json_str)
                    raw = data.get(key)
                    if raw is None:
                        return None
                    if isinstance(raw, list):
                        raw = raw[0] if raw else None
                    if raw is None:
                        return None
                    return max(0.0, min(1.0, float(raw)))
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            pass

        return None


def _extract_score(result: Any, key: str) -> float | None:
    """Extract a single score from a RAGAS EvaluationResult."""
    try:
        # RAGAS may return a dict, an EvaluationResult, or a list
        if isinstance(result, dict):
            v = result.get(key)
        else:
            v = getattr(result, key, None)
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list):
            return float(v[0]) if v else None
        if isinstance(v, str):
            return float(v)
        return None
    except (TypeError, ValueError, IndexError):
        return None


def get_rag_evaluator() -> RAGEvaluator | None:
    """
    Factory for a global RAG evaluator singleton.
    Returns None if LLM cannot be constructed.
    """
    try:
        from app.utils.llm_factory import get_llm
        from app.config import settings
        from app.rag.embedder import get_embedder

        # Build the LLM with generous token limit (RAGAS prompts are
        # often long; default 1024 is too low and causes LLMDidNotFinish).
        # We explicitly use a smaller model for evaluation to save API tokens and avoid Rate Limits!
        try:
            llm = get_llm(settings.RAGAS_MODEL, max_tokens=2048)
        except TypeError:
            # llm_factory doesn't accept max_tokens; fall back to default
            llm = get_llm(settings.RAGAS_MODEL)

        embeddings = None
        try:
            embeddings = get_embedder()
        except Exception as e:
            logger.warning(f"[RAGAS] Embeddings unavailable: {e}")
        return RAGEvaluator(llm=llm, embeddings=embeddings)
    except Exception as e:
        logger.error(f"[RAGAS] Could not build evaluator: {e}")
        return None
