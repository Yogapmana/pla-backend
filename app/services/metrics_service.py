from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select, func
from app.models.agent import RAGMetric, UXSurvey, QuizResult, AgentLog, MasteryScore
from app.models.learning import LearningSession


class MetricsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_rag_metrics(
        self, session_id: UUID, days: int = 7
    ) -> dict:
        """Get aggregated RAG metrics for a session over the last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.db.execute(
            select(RAGMetric).where(
                RAGMetric.session_id == session_id,
                RAGMetric.created_at >= cutoff,
            )
        )
        metrics = result.scalars().all()

        if not metrics:
            return {
                "avg_faithfulness": 0.0,
                "avg_answer_relevancy": 0.0,
                "avg_context_recall": 0.0,
                "avg_context_precision": 0.0,
                "avg_latency_ms": 0,
                "total_evaluations": 0,
            }

        total = len(metrics)
        return {
            "avg_faithfulness": sum(m.faithfulness or 0 for m in metrics) / total,
            "avg_answer_relevancy": sum(m.answer_relevancy or 0 for m in metrics) / total,
            "avg_context_recall": sum(m.context_recall or 0 for m in metrics) / total,
            "avg_context_precision": sum(m.context_precision or 0 for m in metrics) / total,
            "avg_latency_ms": int(sum(m.latency_ms or 0 for m in metrics) / total),
            "total_evaluations": total,
        }

    async def get_agent_metrics(
        self, session_id: UUID, days: int = 7
    ) -> dict:
        """Get agent efficiency metrics: call counts, avg latency per agent."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.db.execute(
            select(AgentLog).where(
                AgentLog.session_id == session_id,
                AgentLog.created_at >= cutoff,
            )
        )
        logs = result.scalars().all()

        agent_counts: dict[str, int] = {}
        for log in logs:
            agent_counts[log.agent] = agent_counts.get(log.agent, 0) + 1

        return {
            "total_logs": len(logs),
            "agent_call_counts": agent_counts,
        }

    async def get_quiz_metrics(
        self, session_id: UUID
    ) -> dict:
        """Get quiz performance summary for a session."""
        result = await self.db.execute(
            select(QuizResult).where(QuizResult.session_id == session_id)
        )
        quizzes = result.scalars().all()

        if not quizzes:
            return {
                "avg_score": 0.0,
                "total_attempts": 0,
                "total_correct": 0,
            }

        scores = [q.score for q in quizzes]
        return {
            "avg_score": sum(scores) / len(scores),
            "total_attempts": len(quizzes),
            "total_correct": sum(q.correct_answers for q in quizzes),
            "max_score": max(scores),
            "min_score": min(scores),
        }

    async def get_ux_surveys(
        self, session_id: UUID
    ) -> list[UXSurvey]:
        """Get all UX survey responses for a session."""
        result = await self.db.execute(
            select(UXSurvey).where(UXSurvey.session_id == session_id)
        )
        return list(result.scalars().all())

    async def submit_ux_survey(
        self,
        session_id: UUID,
        user_id: UUID,
        ease_of_use: float,
        material_relevance: float,
        quiz_quality: float,
        adaptivity_satisfaction: float,
        overall_satisfaction: float,
        open_feedback: str | None = None,
    ) -> UXSurvey:
        """Submit a UX survey for a session."""
        survey = UXSurvey(
            session_id=session_id,
            user_id=user_id,
            ease_of_use=ease_of_use,
            material_relevance=material_relevance,
            quiz_quality=quiz_quality,
            adaptivity_satisfaction=adaptivity_satisfaction,
            overall_satisfaction=overall_satisfaction,
            open_feedback=open_feedback,
        )
        self.db.add(survey)
        await self.db.commit()
        await self.db.refresh(survey)
        return survey

    async def get_session_summary(
        self, session_id: UUID
    ) -> dict:
        """Aggregate all metrics for a session dashboard."""
        rag = await self.get_rag_metrics(session_id)
        agent = await self.get_agent_metrics(session_id)
        quiz = await self.get_quiz_metrics(session_id)
        surveys = await self.get_ux_surveys(session_id)

        avg_satisfaction = (
            sum(s.overall_satisfaction or 0 for s in surveys) / len(surveys)
            if surveys else 0.0
        )

        return {
            "rag": rag,
            "agent": agent,
            "quiz": quiz,
            "ux": {
                "submission_count": len(surveys),
                "avg_overall_satisfaction": avg_satisfaction,
            },
        }


def get_metrics_service(db: AsyncSession) -> MetricsService:
    return MetricsService(db)