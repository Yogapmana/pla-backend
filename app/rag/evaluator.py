import logging
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.agent import RAGMetric
from app.schemas.metrics import RAGMetricsCreate

logger = logging.getLogger(__name__)

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("[RAGAS] ragas not installed. Evaluation disabled.")


class RAGEvaluator:
    """RAGAS-based evaluation for RAG responses."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def evaluate_response(
        self,
        message_id: str,
        user_id: str,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, float] | None:
        """
        Evaluate a RAG response using RAGAS metrics.
        Returns dict of metric names to scores, or None if RAGAS unavailable.
        """
        if not RAGAS_AVAILABLE:
            logger.debug("[RAGAS] Skipping eval — ragas not installed.")
            return None

        try:
            from datasets import Dataset

            data = {
                "user_input": [query],
                "response": [answer],
                "retrieved_contexts": [contexts],
                "reference": [""],
            }

            dataset = Dataset.from_dict(data)

            metrics = [
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
                answer_correctness,
            ]

            results = evaluate(dataset, metrics=metrics)
            scores = results.scores[0]

            rag_metrics = RAGMetricsCreate(
                message_id=message_id,
                session_id=user_id,
                faithfulness=scores.get("faithfulness", 0.0),
                answer_relevancy=scores.get("answer_relevancy", 0.0),
                context_recall=scores.get("context_recall", 0.0),
                context_precision=scores.get("context_precision", 0.0),
                answer_correctness=scores.get("answer_correctness", 0.0),
                latency_ms=0,
                chunks_retrieved=len(contexts),
                chunks_after_rerank=len(contexts),
            )

            db_metric = RAGMetric(**rag_metrics.model_dump())
            self.db.add(db_metric)
            await self.db.commit()

            return {
                "faithfulness": scores.get("faithfulness", 0.0),
                "answer_relevancy": scores.get("answer_relevancy", 0.0),
                "context_recall": scores.get("context_recall", 0.0),
                "context_precision": scores.get("context_precision", 0.0),
                "answer_correctness": scores.get("answer_correctness", 0.0),
            }

        except Exception as e:
            logger.error(f"[RAGAS] Evaluation error: {e}")
            return None