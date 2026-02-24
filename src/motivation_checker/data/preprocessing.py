"""
Text preprocessing utilities for journal entries.
"""

import re
import logging
from typing import List
import nltk

logger = logging.getLogger(__name__)

# Download required NLTK data (should be done during setup)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Preprocesses text data for model consumption."""
    
    def __init__(self, remove_stopwords: bool = False, lowercase: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        
        if remove_stopwords:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess(self, text: str) -> str:
        """Apply full preprocessing pipeline."""
        # Clean text
        text = self.clean_text(text)
        
        # Lowercase if needed
        if self.lowercase:
            text = text.lower()
        
        # Remove stopwords if needed
        if self.remove_stopwords:
            tokens = text.split()
            tokens = [word for word in tokens if word not in self.stopwords]
            text = ' '.join(tokens)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        return nltk.sent_tokenize(text)
