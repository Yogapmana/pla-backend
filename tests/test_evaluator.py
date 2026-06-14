"""
Test the RAGAS lightweight LLM-as-judge fallback.

This test deliberately does NOT depend on the RAGAS package being
installed. We mock the LLM to return a valid JSON response and
verify the evaluator parses it correctly.
"""

import asyncio
import sys
from unittest.mock import MagicMock

sys.path.insert(0, '.')


def test_lightweight_eval_parses_valid_json():
    """When the LLM returns valid JSON, scores are extracted correctly."""
    from app.rag.evaluator import RAGEvaluator

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"faithfulness": 0.85, "answer_relevancy": 0.92}'
    mock_llm.invoke.return_value = mock_response

    evaluator = RAGEvaluator(llm=mock_llm, embeddings=None)
    result = asyncio.run(evaluator._lightweight_eval(
        question="Apa itu IoT?",
        answer="IoT adalah jaringan perangkat pintar.",
        contexts=["IoT adalah singkatan dari Internet of Things."],
    ))

    assert result["method"] == "llm_judge"
    assert result["rag_faithfulness"] == 0.85
    assert result["rag_answer_relevancy"] == 0.92
    print("✓ test_lightweight_eval_parses_valid_json passed")


def test_lightweight_eval_handles_invalid_json():
    """When the LLM returns garbage, scores are None and method=llm_judge."""
    from app.rag.evaluator import RAGEvaluator

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "I cannot score this — sorry!"
    mock_llm.invoke.return_value = mock_response

    evaluator = RAGEvaluator(llm=mock_llm, embeddings=None)
    result = asyncio.run(evaluator._lightweight_eval(
        question="q",
        answer="a",
        contexts=["c"],
    ))

    # Method is now always 'llm_judge' (the method is how we ATTEMPTED),
    # but the scores are None when parsing fails
    assert result["method"] == "llm_judge"
    assert result["rag_faithfulness"] is None
    assert result["rag_answer_relevancy"] is None
    print("✓ test_lightweight_eval_handles_invalid_json passed")


def test_lightweight_eval_clamps_scores():
    """Scores outside [0, 1] should be clamped to that range."""
    from app.rag.evaluator import RAGEvaluator

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"faithfulness": 1.5, "answer_relevancy": -0.3}'
    mock_llm.invoke.return_value = mock_response

    evaluator = RAGEvaluator(llm=mock_llm, embeddings=None)
    result = asyncio.run(evaluator._lightweight_eval(
        question="q", answer="a", contexts=["c"]
    ))

    assert result["rag_faithfulness"] == 1.0
    assert result["rag_answer_relevancy"] == 0.0
    print("✓ test_lightweight_eval_clamps_scores passed")


def test_evaluate_skipped_when_empty():
    """Empty question/answer should return skipped status, not crash."""
    from app.rag.evaluator import RAGEvaluator

    evaluator = RAGEvaluator(llm=MagicMock(), embeddings=None)

    result1 = asyncio.run(evaluator.evaluate("", "answer", ["ctx"]))
    assert result1["method"] == "skipped"
    print("✓ test_evaluate_skipped_when_empty (empty question)")

    result2 = asyncio.run(evaluator.evaluate("question", "", ["ctx"]))
    assert result2["method"] == "skipped"
    print("✓ test_evaluate_skipped_when_empty (empty answer)")


def test_parse_score_handles_list_value():
    """RAGAS may return scores as a list of one element. Parser must handle."""
    from app.rag.evaluator import RAGEvaluator

    mock_llm = MagicMock()
    evaluator = RAGEvaluator(llm=mock_llm, embeddings=None)

    # RAGAS-style: scores wrapped in lists
    text = '{"faithfulness": [0.85], "answer_relevancy": [0.92]}'
    faith = evaluator._parse_score_from_text(text, "faithfulness")
    relev = evaluator._parse_score_from_text(text, "answer_relevancy")
    assert faith == 0.85, f"expected 0.85, got {faith}"
    assert relev == 0.92, f"expected 0.92, got {relev}"
    print("✓ test_parse_score_handles_list_value passed")


def test_parse_score_clamps_out_of_range():
    """Scores outside [0, 1] should be clamped."""
    from app.rag.evaluator import RAGEvaluator

    mock_llm = MagicMock()
    evaluator = RAGEvaluator(llm=mock_llm, embeddings=None)

    text = '{"faithfulness": 1.5, "answer_relevancy": -0.5}'
    faith = evaluator._parse_score_from_text(text, "faithfulness")
    relev = evaluator._parse_score_from_text(text, "answer_relevancy")
    assert faith == 1.0
    assert relev == 0.0
    print("✓ test_parse_score_clamps_out_of_range passed")


if __name__ == "__main__":
    test_lightweight_eval_parses_valid_json()
    test_lightweight_eval_handles_invalid_json()
    test_lightweight_eval_clamps_scores()
    test_evaluate_skipped_when_empty()
    test_parse_score_handles_list_value()
    test_parse_score_clamps_out_of_range()
    print("\nAll 6 RAGAS tests passed.")
