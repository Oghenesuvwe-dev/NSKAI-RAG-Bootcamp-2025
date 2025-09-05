FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p logs data chroma_db

EXPOSE 8010 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "uvicorn", "optimized_rag:app", "--host", "0.0.0.0", "--port", "8010"]