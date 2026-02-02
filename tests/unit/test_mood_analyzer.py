"""Unit tests for MoodAnalyzer."""

import pytest
from src.models.mood_analyzer import MoodAnalyzer


def test_mood_analyzer_initialization(mood_analyzer):
    """Test MoodAnalyzer initialization."""
    assert mood_analyzer is not None
    assert hasattr(mood_analyzer, 'analyze')


def test_analyze_positive_text(mood_analyzer):
    """Test analysis of positive text."""
    text = "I'm so excited and motivated! Today I achieved all my goals."
    result = mood_analyzer.analyze(text)
    
    assert result["mood"] in ["positive", "very_positive"]
    assert result["motivation_category"] == "high"
    assert result["motivation_level"] > 60
    assert result["sentiment"]["polarity"] > 0


def test_analyze_negative_text(mood_analyzer):
    """Test analysis of negative text."""
    text = "I feel so lazy and unmotivated. Everything is difficult and I want to give up."
    result = mood_analyzer.analyze(text)
    
    assert result["mood"] in ["negative", "very_negative"]
    assert result["motivation_category"] == "low"
    assert result["motivation_level"] < 40
    assert result["sentiment"]["polarity"] < 0


def test_analyze_neutral_text(mood_analyzer):
    """Test analysis of neutral text."""
    text = "Today was just another ordinary day. Nothing special happened."
    result = mood_analyzer.analyze(text)
    
    assert result["mood"] == "neutral"
    assert result["sentiment"]["polarity"] >= -0.2
    assert result["sentiment"]["polarity"] <= 0.2


def test_analyze_empty_text(mood_analyzer):
    """Test analysis of empty text."""
    result = mood_analyzer.analyze("")
    
    assert "error" in result
    assert result["mood"] == "neutral"
    assert result["motivation_level"] == 50.0


def test_motivation_score_range(mood_analyzer):
    """Test that motivation scores are in valid range."""
    texts = [
        "I'm extremely motivated and excited!",
        "Feeling quite lazy today.",
        "Just normal day."
    ]
    
    for text in texts:
        result = mood_analyzer.analyze(text)
        assert 0 <= result["motivation_level"] <= 100


def test_sentiment_analysis(mood_analyzer):
    """Test sentiment analysis component."""
    polarity, subjectivity = mood_analyzer.analyze_sentiment("I love this!")
    
    assert -1 <= polarity <= 1
    assert 0 <= subjectivity <= 1
    assert polarity > 0  # "love" should be positive


def test_mood_categories(mood_analyzer):
    """Test mood category classification."""
    test_cases = [
        (-0.8, "very_negative"),
        (-0.4, "negative"),
        (0.0, "neutral"),
        (0.4, "positive"),
        (0.8, "very_positive")
    ]
    
    for polarity, expected_mood in test_cases:
        mood = mood_analyzer.get_mood_category(polarity)
        assert mood == expected_mood
