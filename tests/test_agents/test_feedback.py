import pytest
from app.agents.state import ProgressSignals
from app.agents.feedback_engine import calculate_mastery, evaluate_mastery, run_feedback_loop

def test_calculate_mastery_all_high():
    signals = ProgressSignals(
        quiz_score=1.0,
        reading_time_ratio=1.0,
        question_frequency=1.0,
        self_assessment=1.0,
        material_rating=1.0,
    )
    score = calculate_mastery(signals)
    assert score == 1.0

def test_calculate_mastery_all_low():
    signals = ProgressSignals(
        quiz_score=0.0,
        reading_time_ratio=0.0,
        question_frequency=0.0,
        self_assessment=0.0,
        material_rating=0.0,
    )
    score = calculate_mastery(signals)
    assert score == 0.0

def test_calculate_mastery_mixed():
    # Bobot produksi: quiz 0.60, reading 0.15, question_freq 0.10,
    # self_assess 0.10, material_rating 0.05
    signals = ProgressSignals(
        quiz_score=0.5,           # 0.5 * 0.60 = 0.30
        reading_time_ratio=0.8,   # 0.8 * 0.15 = 0.12
        question_frequency=0.6,   # 0.6 * 0.10 = 0.06
        self_assessment=0.7,      # 0.7 * 0.10 = 0.07
        material_rating=0.9,      # 0.9 * 0.05 = 0.045
    )                             # Total = 0.595
    score = calculate_mastery(signals)
    assert abs(score - 0.595) < 0.001

def test_evaluate_mastery_actions():
    assert evaluate_mastery(0.5, "t1").action == "remedial"
    assert evaluate_mastery(0.59, "t1").action == "remedial"

    assert evaluate_mastery(0.60, "t2").action == "continue"
    assert evaluate_mastery(0.85, "t2").action == "continue"

    assert evaluate_mastery(0.86, "t3").action == "enrichment"
    assert evaluate_mastery(1.0, "t3").action == "enrichment"

def test_run_feedback_loop():
    signals = ProgressSignals(
        quiz_score=0.5,
        reading_time_ratio=0.5,
        question_frequency=0.5,
        self_assessment=0.5,
        material_rating=0.5,
    )  # total 0.5 → remedial
    score, action = run_feedback_loop(signals, "t1")
    assert score == 0.5
    assert action.action == "remedial"
    assert action.topic_id == "t1"
