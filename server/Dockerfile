FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for PostgreSQL client and building some requirements
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip cache purge || true

# Copy application
COPY . /app

EXPOSE 8000

# Use gunicorn to serve Flask app (app:app)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
