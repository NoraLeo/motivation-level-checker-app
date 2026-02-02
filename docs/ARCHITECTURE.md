# MLOps Motivation Level Checker - Documentation

## Architecture Overview

This application follows MLOps best practices with a modular architecture:

### Components

1. **Data Layer** (`src/data/`)
   - **Ingestion**: Load journal entries from various sources (JSON, CSV)
   - **Preprocessing**: Clean and prepare text data for analysis
   
2. **Model Layer** (`src/models/`)
   - **Mood Analyzer**: Real-time sentiment and motivation analysis
   - **Training Pipeline**: Batch model training with MLflow tracking
   
3. **API Layer** (`src/api/`)
   - **FastAPI Server**: REST endpoints for predictions
   - **Request Validation**: Pydantic models for type safety
   
4. **Monitoring Layer** (`src/monitoring/`)
   - **Metrics Collection**: Prometheus metrics for observability
   - **Performance Tracking**: API and model performance monitoring

### Data Flow

```
Journal Entry → Preprocessing → Mood Analysis → API Response
                                      ↓
                              MLflow Tracking
                                      ↓
                              Prometheus Metrics
```

## API Documentation

### Endpoints

#### POST /analyze
Analyze a journal entry for mood and motivation.

**Request Body**:
```json
{
  "text": "Your journal entry here",
  "preprocess": false,
  "date": "2024-01-01T12:00:00"
}
```

**Response**:
```json
{
  "mood": "positive",
  "motivation_level": 75.5,
  "motivation_category": "high",
  "sentiment": {
    "polarity": 0.45,
    "subjectivity": 0.60
  },
  "text_length": 150,
  "word_count": 28,
  "timestamp": "2024-01-01T12:00:00"
}
```

## Development Guide

### Adding New Features

1. **New Data Sources**: Extend `src/data/ingestion.py`
2. **New Models**: Add to `src/models/`
3. **New Endpoints**: Update `src/api/main.py`
4. **New Metrics**: Extend `src/monitoring/metrics.py`

### Testing

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Run with: `pytest tests/ -v --cov=src`

### Model Training

```bash
python scripts/train_model.py --data-path data/raw/entries.csv
```

## Deployment

### Local Development
```bash
python scripts/run_api.py
```

### Docker
```bash
docker-compose up --build
```

### Production Considerations
- Use environment variables for configuration
- Set up proper logging
- Enable authentication/authorization
- Use HTTPS
- Set up load balancing
- Configure auto-scaling

## Monitoring

### Metrics Available
- `api_requests_total`: Total API requests
- `api_request_duration_seconds`: Request latency
- `analysis_mood_distribution`: Mood category distribution
- `analysis_motivation_level`: Latest motivation score
- `prediction_errors_total`: Error count

### Access Metrics
```bash
curl http://localhost:9090/metrics
```

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

2. **Port Already in Use**
   - Change port in `config/config.yaml`
   - Or kill process: `lsof -ti:8000 | xargs kill`

3. **Import Errors**
   - Ensure package is installed: `pip install -e .`
   - Check PYTHONPATH includes project root
