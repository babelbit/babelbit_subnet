# Stage 1: Base image with build dependencies
FROM python:3.12-slim-bullseye AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel hatchling


# Stage 2: Test stage with test dependencies and test execution
FROM base AS test

# Install Docker for testcontainers (PostgreSQL tests)
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Copy source code and test files
COPY . /app

# Install package with test dependencies
RUN pip install ".[dev]"

# Run tests
RUN pytest tests/ -v --tb=short


# Stage 3: Production image (minimal, no test dependencies)
FROM base AS production

# Copy only necessary application files
COPY . /app

# Install package without test dependencies
RUN pip install .

ENV PATH="/root/.local/bin:$PATH"

# Default command (can be overridden)
CMD ["python", "-m", "babelbit"]
