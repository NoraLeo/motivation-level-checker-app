"""
Mood and motivation analyzer using sentiment analysis and NLP.
"""

import logging
from typing import Dict, List, Tuple
from textblob import TextBlob
import numpy as np
from dotenv import load_dotenv
import os

#importing a HuggingFace Transformer model for sentiment analysis
from transformers import pipeline, AutoTokenizer, AutoModel

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

    def __init__(self):
        """Initialize the MoodAnalyzer with a HuggingFace pipeline."""
        # Load environment variables from the specified .env file
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

        # Retrieve the token from the .env file
        token = os.getenv("HF_TOKEN")

        # Log the token loading for debugging
        logger.info(f"Loaded HuggingFace token: {token}")

        # Initialize the HuggingFace pipeline with the correct model identifier
        self.transformer_pipeline = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True,
            use_auth_token=token
        )

        #for finding similarity between predifined mood categories and the transformer labels
        self.tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        self.model = AutoModel.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a given text using the transformer model."""
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        # Use the mean of the last hidden state as the embedding
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embedding.flatten()
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
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
        
        Combines sentiment polarity with the highest score from the HuggingFace transformer.
        """
        # Get transformer scores 
        transformer_scores= self.transformer_pipeline(text)

        # Combine polarity and transformer score (weighted average)
        # 50% sentiment, 50% transformer score
        #Calculate the transformer score based on the similarity to motivation-related labels (
        # how similar is the label value to any of the mood categories defined above)
        tf_score = transformer_scores[0].get('score')
        tf_label = transformer_scores[0].get('label')
        highest_score = tf_score  # Default to the raw score
        label_embedding = self.get_embedding(tf_label)
        for mood in self.MOOD_CATEGORIES.keys():
            mood_embedding = self.get_embedding(mood)
            sim = self.similarity(label_embedding, mood_embedding)
            if sim > highest_score:
                highest_score = sim
        combined_score = 0.5 * polarity + (0.5 * highest_score * tf_score)

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
            # TODO: Refine thresholds based on empirical data
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
            logger.error("Error analyzing text", exc_info=True)
            return {
                "error": str(e),
                "mood": "neutral",
                "motivation_level": 50.0,
                "sentiment": {
                    "polarity": 0.0,
                    "subjectivity": 0.0
                }
            }
