# Motivation Level Checker App

An MLOps application for analyzing mood and motivation levels from daily journal entries and reflections. This project demonstrates practical MLOps concepts including data pipelines, model training, API serving, monitoring, and CI/CD.

## 🎯 Project Overview

This application uses Natural Language Processing (NLP) to analyze journal entries and provide insights on:
- **Mood Detection**: Classifies mood into 5 categories (very negative, negative, neutral, positive, very positive)
- **Motivation Level**: Scores motivation on a 0-100 scale
- **Sentiment Analysis**: Provides polarity and subjectivity metrics

## 🏗️ Project Structure

```
motivation-level-checker-app/
├── .github/workflows/       # CI/CD pipelines
│   ├── ci-cd.yml           # Main CI/CD workflow
│   └── train-model.yml     # Model training workflow
├── config/                  # Configuration files
│   └── config.yaml         # Application configuration
├── data/                    # Data storage
│   ├── raw/                # Raw journal entries
│   ├── processed/          # Processed data
│   └── models/             # Trained models
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/                 # Utility scripts
│   ├── example_usage.py    # Usage examples
│   ├── run_api.py          # API server launcher
│   └── train_model.py      # Model training script
├── src/                     # Source code
│   ├── api/                # FastAPI application
│   │   └── main.py         # API endpoints
│   ├── data/               # Data processing
│   │   ├── ingestion.py    # Data loading
│   │   └── preprocessing.py # Text preprocessing
│   ├── models/             # ML models
│   │   ├── mood_analyzer.py # Mood analysis
│   │   └── train.py        # Model training
│   └── monitoring/         # Monitoring & metrics
│       └── metrics.py      # Prometheus metrics
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── .gitignore
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container setup
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/NoraLeo/motivation-level-checker-app.git
cd motivation-level-checker-app
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -e .
```

4. **Download NLTK data**:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Quick Start

#### Option 1: Run Example Script

```bash
python scripts/example_usage.py
```

#### Option 2: Start the API Server

```bash
python scripts/run_api.py
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

#### Option 3: Use Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# API at http://localhost:8000
# MLflow UI at http://localhost:5000
```

## 📊 Usage Examples

### Using the API

**Analyze a journal entry**:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Today was amazing! I accomplished so much and feel really motivated to continue."
  }'
```

**Response**:
```json
{
  "mood": "very_positive",
  "motivation_level": 87.5,
  "motivation_category": "high",
  "sentiment": {
    "polarity": 0.625,
    "subjectivity": 0.75
  },
  "text_length": 82,
  "word_count": 14,
  "timestamp": "2024-01-01T12:00:00"
}
```

### Using the Python API

```python
from motivation_checker.models.mood_analyzer import MoodAnalyzer

analyzer = MoodAnalyzer()
result = analyzer.analyze("I'm feeling great and motivated today!")

print(f"Mood: {result['mood']}")
print(f"Motivation Level: {result['motivation_level']}/100")
print(f"Category: {result['motivation_category']}")
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_mood_analyzer.py -v
```

## 🔧 MLOps Features

### 1. Data Management
- Structured data ingestion pipeline
- Support for JSON and CSV formats
- Data validation and preprocessing

### 2. Model Training & Experimentation
- MLflow integration for experiment tracking
- Scikit-learn based models
- Reproducible training pipeline

### 3. Model Serving
- FastAPI-based REST API
- Pydantic models for request/response validation
- OpenAPI documentation (Swagger UI)

### 4. Monitoring
- Prometheus metrics integration
- API request tracking
- Model performance monitoring

### 5. CI/CD
- Automated testing with GitHub Actions
- Docker containerization
- Linting and code quality checks

### 6. Containerization
- Dockerfile for single container
- Docker Compose for multi-service setup
- MLflow server for experiment tracking

## 📈 MLflow Tracking

View experiment results:

```bash
mlflow ui --backend-store-uri mlruns
```

Access the UI at `http://localhost:5000`

## 🛠️ Configuration

Edit `config/config.yaml` to customize:
- API settings (host, port)
- Model parameters
- Data paths
- Logging configuration
- Monitoring settings

## 📝 API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `POST /analyze` - Analyze journal entry
- `GET /stats` - API statistics and capabilities
- `GET /docs` - Interactive API documentation

## 🔍 Model Details

The mood analyzer uses:
- **TextBlob** for sentiment analysis
- **Keyword matching** for motivation detection
- **Weighted scoring** combining sentiment and keywords
- Classification into 5 mood categories and 3 motivation levels

## 🤝 Contributing

This is a learning project for mastering MLOps concepts. Feel free to:
- Add new features
- Improve existing components
- Add more sophisticated ML models
- Enhance monitoring capabilities

## 📚 Learning Resources

This project covers:
- Data pipeline design
- ML model training and evaluation
- API development with FastAPI
- Containerization with Docker
- CI/CD with GitHub Actions
- Experiment tracking with MLflow
- Model monitoring with Prometheus

## 🎓 Next Steps

To continue your MLOps journey:
1. Add more sophisticated NLP models (BERT, transformers)
2. Implement A/B testing
3. Add data versioning with DVC
4. Set up model registry
5. Implement automated retraining
6. Add more comprehensive monitoring
7. Deploy to cloud platforms (AWS, GCP, Azure)

## 📄 License

MIT License - feel free to use this project for learning and development.

## 🙋 Support

For issues or questions, please open an issue on GitHub.

---

**Happy Learning! 🚀**
