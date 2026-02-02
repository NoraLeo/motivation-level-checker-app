"""Unit tests for data preprocessing."""

import pytest
from src.data.preprocessing import TextPreprocessor


def test_text_preprocessor_initialization(text_preprocessor):
    """Test TextPreprocessor initialization."""
    assert text_preprocessor is not None


def test_clean_text_removes_urls(text_preprocessor):
    """Test URL removal."""
    text = "Check this out https://example.com and www.test.com"
    cleaned = text_preprocessor.clean_text(text)
    
    assert "https://" not in cleaned
    assert "www." not in cleaned


def test_clean_text_removes_emails(text_preprocessor):
    """Test email removal."""
    text = "Contact me at test@example.com for more info"
    cleaned = text_preprocessor.clean_text(text)
    
    assert "@" not in cleaned
    assert "test@example.com" not in cleaned


def test_clean_text_removes_extra_whitespace(text_preprocessor):
    """Test whitespace normalization."""
    text = "Too   many    spaces     here"
    cleaned = text_preprocessor.clean_text(text)
    
    assert "  " not in cleaned
    assert cleaned == "Too many spaces here"


def test_preprocess_lowercase(text_preprocessor):
    """Test lowercase conversion."""
    text = "HELLO World"
    processed = text_preprocessor.preprocess(text)
    
    assert processed == "hello world"


def test_preprocess_no_lowercase():
    """Test preprocessing without lowercase."""
    preprocessor = TextPreprocessor(lowercase=False)
    text = "HELLO World"
    processed = preprocessor.preprocess(text)
    
    assert "HELLO" in processed


def test_tokenize_sentences(text_preprocessor):
    """Test sentence tokenization."""
    text = "First sentence. Second sentence! Third sentence?"
    sentences = text_preprocessor.tokenize(text)
    
    assert len(sentences) == 3
    assert "First sentence." in sentences[0]
