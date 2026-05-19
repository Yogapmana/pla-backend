import pytest
from app.agents.state import ProgressSignals
from app.agents.feedback_engine import calculate_mastery, evaluate_mastery, run_feedback_loop

def test_calculate_mastery_all_high():
    signals = ProgressSignals(
        quiz_score=1.0,
        reading_time_ratio=1.0,
        question_frequency=1.0, # (dianggap 1.0 berarti bagus/jarang nanya, dlm PRD "inverse")
        self_assessment=1.0,
        material_rating=1.0
    )
    score = calculate_mastery(signals)
    assert score == 1.0

def test_calculate_mastery_all_low():
    signals = ProgressSignals(
        quiz_score=0.0,
        reading_time_ratio=0.0,
        question_frequency=0.0,
        self_assessment=0.0,
        material_rating=0.0
    )
    score = calculate_mastery(signals)
    assert score == 0.0
    
def test_calculate_mastery_mixed():
    signals = ProgressSignals(
        quiz_score=0.5,           # 0.5 * 0.40 = 0.20
        reading_time_ratio=0.8,   # 0.8 * 0.20 = 0.16
        question_frequency=0.6,   # 0.6 * 0.20 = 0.12
        self_assessment=0.7,      # 0.7 * 0.15 = 0.105
        material_rating=0.9       # 0.9 * 0.05 = 0.045
    )                             # Total = 0.63
    score = calculate_mastery(signals)
    assert abs(score - 0.63) < 0.001
    
def test_evaluate_mastery_actions():
    assert evaluate_mastery(0.5, "t1").action == "repeat"
    assert evaluate_mastery(0.59, "t1").action == "repeat"
    
    assert evaluate_mastery(0.60, "t2").action == "review"
    assert evaluate_mastery(0.74, "t2").action == "review"
    
    assert evaluate_mastery(0.75, "t3").action == "continue"
    assert evaluate_mastery(0.90, "t3").action == "continue"
    
    assert evaluate_mastery(0.91, "t4").action == "accelerate"
    assert evaluate_mastery(1.0, "t4").action == "accelerate"
    
def test_run_feedback_loop():
    signals = ProgressSignals(
        quiz_score=0.5,
        reading_time_ratio=0.5,
        question_frequency=0.5,
        self_assessment=0.5,
        material_rating=0.5
    ) # 0.5 total
    score, action = run_feedback_loop(signals, "t1")
    assert score == 0.5
    assert action.action == "repeat"
    assert action.topic_id == "t1"
