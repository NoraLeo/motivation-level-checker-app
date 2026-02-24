"""
FastAPI application for mood and motivation analysis.
Provides REST API endpoints for analyzing journal entries.
"""

import logging
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from motivation_checker.models.mood_analyzer import MoodAnalyzer
from motivation_checker.data.preprocessing import TextPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Motivation Level Checker API",
    description="Analyze mood and motivation levels from journal entries",
    version="0.1.0"
)


# Initialize components
mood_analyzer = MoodAnalyzer()
text_preprocessor = TextPreprocessor(lowercase=False, remove_stopwords=False)


# Request/Response models
class JournalEntryRequest(BaseModel):
    """Request model for journal entry analysis."""
    text: str = Field(..., description="Journal entry text", min_length=1)
    preprocess: bool = Field(default=False, description="Whether to preprocess the text")
    date: Optional[str] = Field(default=None, description="Entry date (ISO format)")


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    mood: str
    motivation_level: float
    motivation_category: str
    sentiment: dict
    text_length: int
    word_count: int
    timestamp: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


# API endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Motivation Level Checker API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="0.1.0"
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_journal_entry(entry: JournalEntryRequest):
    """
    Analyze a journal entry for mood and motivation levels.
    
    Args:
        entry: Journal entry request with text and optional parameters
    
    Returns:
        Analysis results including mood, motivation level, and sentiment
    """
    try:
        # Preprocess text if requested
        text = entry.text
        if entry.preprocess:
            text = text_preprocessor.preprocess(text)
        
        # Perform analysis
        result = mood_analyzer.analyze(text)
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return AnalysisResponse(**result)
    
    except Exception as e:
        logger.error(f"Error analyzing entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=dict)
async def get_stats():
    """Get API statistics."""
    return {
        "mood_categories": list(MoodAnalyzer.MOOD_CATEGORIES.keys()),
        "motivation_levels": ["low", "moderate", "high"],
        "available_features": [
            "sentiment_analysis",
            "mood_classification",
            "motivation_scoring",
            "text_preprocessing"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
