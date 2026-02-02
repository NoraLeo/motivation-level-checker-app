"""
Mood and motivation analyzer using sentiment analysis and NLP.
"""

import logging
from typing import Dict, List, Tuple
from textblob import TextBlob
import numpy as np

logger = logging.getLogger(__name__)


class MoodAnalyzer:
    """Analyzes mood and motivation levels from text."""
    
    # Mood categories based on sentiment scores
    MOOD_CATEGORIES = {
        "very_negative": (-1.0, -0.6),
        "negative": (-0.6, -0.2),
        "neutral": (-0.2, 0.2),
        "positive": (0.2, 0.6),
        "very_positive": (0.6, 1.0)
    }
    
    # Keywords indicating high motivation
    HIGH_MOTIVATION_KEYWORDS = [
        'accomplish', 'achieve', 'goal', 'productive', 'motivated',
        'excited', 'determined', 'progress', 'success', 'complete',
        'finish', 'energy', 'focused', 'driven', 'inspired'
    ]
    
    # Keywords indicating low motivation
    LOW_MOTIVATION_KEYWORDS = [
        'lazy', 'tired', 'unmotivated', 'procrastinate', 'stuck',
        'overwhelmed', 'exhausted', 'burnout', 'pointless', 'give up',
        'quit', 'difficult', 'struggle', 'frustrated', 'hopeless'
    ]
    
    def __init__(self):
        pass
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.
        
        Returns:
            Tuple of (polarity, subjectivity)
            - polarity: -1 (negative) to 1 (positive)
            - subjectivity: 0 (objective) to 1 (subjective)
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    def get_mood_category(self, polarity: float) -> str:
        """Get mood category from polarity score."""
        for category, (lower, upper) in self.MOOD_CATEGORIES.items():
            if lower <= polarity < upper:
                return category
        return "very_positive"  # Catch-all for polarity >= 0.6
    
    def calculate_motivation_score(self, text: str, polarity: float) -> float:
        """
        Calculate motivation score (0-100).
        
        Combines sentiment polarity with keyword analysis.
        """
        text_lower = text.lower()
        
        # Count motivation keywords
        high_motivation_count = sum(1 for keyword in self.HIGH_MOTIVATION_KEYWORDS if keyword in text_lower)
        low_motivation_count = sum(1 for keyword in self.LOW_MOTIVATION_KEYWORDS if keyword in text_lower)
        
        # Keyword-based score (-1 to 1)
        keyword_score = 0
        if high_motivation_count + low_motivation_count > 0:
            keyword_score = (high_motivation_count - low_motivation_count) / (high_motivation_count + low_motivation_count)
        
        # Combine polarity and keyword score (weighted average)
        # 60% sentiment, 40% keywords
        combined_score = 0.6 * polarity + 0.4 * keyword_score
        
        # Convert to 0-100 scale
        motivation_score = (combined_score + 1) * 50
        
        return np.clip(motivation_score, 0, 100)
    
    def analyze(self, text: str) -> Dict:
        """
        Perform complete mood and motivation analysis.
        
        Args:
            text: Journal entry text
        
        Returns:
            Dictionary containing analysis results
        """
        if not text or not text.strip():
            return {
                "error": "Empty text provided",
                "mood": "neutral",
                "motivation_level": 50.0,
                "sentiment": {
                    "polarity": 0.0,
                    "subjectivity": 0.0
                }
            }
        
        try:
            # Get sentiment
            polarity, subjectivity = self.analyze_sentiment(text)
            
            # Get mood category
            mood = self.get_mood_category(polarity)
            
            # Calculate motivation
            motivation_level = self.calculate_motivation_score(text, polarity)
            
            # Determine motivation category
            if motivation_level >= 70:
                motivation_category = "high"
            elif motivation_level >= 40:
                motivation_category = "moderate"
            else:
                motivation_category = "low"
            
            return {
                "mood": mood,
                "motivation_level": round(motivation_level, 2),
                "motivation_category": motivation_category,
                "sentiment": {
                    "polarity": round(polarity, 3),
                    "subjectivity": round(subjectivity, 3)
                },
                "text_length": len(text),
                "word_count": len(text.split())
            }
        
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                "error": str(e),
                "mood": "neutral",
                "motivation_level": 50.0,
                "sentiment": {
                    "polarity": 0.0,
                    "subjectivity": 0.0
                }
            }
