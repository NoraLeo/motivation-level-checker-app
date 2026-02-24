FROM python:3.10-slim as builder
WORKDIR /app
# Install Poetry
RUN pip install poetry
# Copy the dependencies in the TOML file and lock it
COPY pyproject.toml poetry.lock ./
# Export locked dependencies to a standard format
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
# Install your local package in editable mode if needed, or just set PYTHONPATH
ENV PYTHONPATH=/app/src
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "motivation_checker.api.main:app", "--host", "0.0.0.0"]