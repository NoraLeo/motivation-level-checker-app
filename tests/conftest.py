"""Test configuration and fixtures."""

import pytest
from motivation_checker.models.mood_analyzer import MoodAnalyzer
from motivation_checker.data.preprocessing import TextPreprocessor


@pytest.fixture
def mood_analyzer():
    """Fixture for MoodAnalyzer."""
    return MoodAnalyzer()


@pytest.fixture
def text_preprocessor():
    """Fixture for TextPreprocessor."""
    return TextPreprocessor()


@pytest.fixture
def sample_journal_entries():
    """Sample journal entries for testing."""
    return [
        {
            "text": "Today was amazing! I accomplished so much and feel really motivated to continue.",
            "expected_mood": "very_positive",
            "expected_motivation": "high"
        },
        {
            "text": "I'm feeling lazy and unmotivated. Can't seem to get anything done today.",
            "expected_mood": "negative",
            "expected_motivation": "low"
        },
        {
            "text": "Just another day. Nothing special happened.",
            "expected_mood": "neutral",
            "expected_motivation": "moderate"
        }
    ]
