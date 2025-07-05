# AGI Forex Trading System - Production Dockerfile
# ================================================

# Use Python 3.11 slim image as base
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    gcc \
    g++ \
    make \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib (Technical Analysis Library)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create application user
RUN groupadd -r agi && useradd -r -g agi agi

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data backups config/secrets && \
    chown -R agi:agi /app

# Switch to non-root user
USER agi

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py"]

# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black flake8 mypy jupyter

# Install debugging tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

USER agi

# Override command for development
CMD ["python", "main.py", "--debug"]

# Production stage
FROM base as production

# Copy only necessary files for production
COPY --from=base /app /app

# Set production environment
ENV ENVIRONMENT=production

# Use production command
CMD ["python", "main.py"]

# Testing stage
FROM development as testing

# Copy test files
COPY tests/ tests/

# Run tests
RUN python -m pytest tests/ -v --cov=. --cov-report=html

# N8N Integration stage
FROM n8nio/n8n:latest as n8n

# Copy N8N workflows
COPY n8n_workflows/ /home/node/.n8n/workflows/

# Set N8N environment variables
ENV N8N_BASIC_AUTH_ACTIVE=true \
    N8N_BASIC_AUTH_USER=admin \
    N8N_BASIC_AUTH_PASSWORD=admin123 \
    WEBHOOK_URL=http://localhost:5678 \
    N8N_PORT=5678

EXPOSE 5678

# Multi-service stage (for docker-compose)
FROM base as multi-service

# Install supervisor for process management
USER root
RUN apt-get update && apt-get install -y supervisor && \
    rm -rf /var/lib/apt/lists/*

# Copy supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create log directories for supervisor
RUN mkdir -p /var/log/supervisor && \
    chown -R agi:agi /var/log/supervisor

USER agi

# Expose multiple ports
EXPOSE 8000 9090 5678

# Use supervisor to manage multiple processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]