FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching)
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy application source code
COPY models.py ./models.py
COPY data_generator.py ./data_generator.py
COPY support_env.py ./support_env.py
COPY grader.py ./grader.py
COPY inference.py ./inference.py
COPY server/ ./server/
COPY openenv.yaml ./openenv.yaml
COPY README.md ./README.md

# Create output directories
RUN mkdir -p outputs/logs outputs/evals

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Environment variables (overridden at runtime via HF Space secrets)
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
