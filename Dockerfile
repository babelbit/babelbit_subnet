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


# Stage 2: Production dependencies only
FROM base AS prod-dependencies

# Copy only dependency files first for better caching
COPY pyproject.toml README.md ./

# Install production dependencies
RUN pip install --no-cache-dir .


# Stage 3: Test dependencies - extends production dependencies
FROM prod-dependencies AS test-dependencies

# Install dev dependencies for testing
RUN pip install --no-cache-dir pytest pytest-asyncio python-dotenv requests httpx 'testcontainers[postgresql]'


# Stage 4: Test stage - copy code and run tests
FROM test-dependencies AS test

# Install Docker for testcontainers (PostgreSQL tests)
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Copy source code and test files
COPY . /app

# Install package in editable mode (dependencies already installed)
RUN pip install --no-cache-dir --no-deps -e .

# Run tests and write a success marker if they pass
RUN python -m pytest tests/ -v --tb=short && touch /tests-passed


# Stage 5: Production - copy pre-installed dependencies from prod-dependencies stage
FROM base AS production

# Copy installed packages from prod-dependencies stage (no test deps)
COPY --from=prod-dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=prod-dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Install package (non-editable, dependencies already present)
RUN pip install --no-cache-dir --no-deps .

ENV PATH="/root/.local/bin:$PATH"

CMD ["python", "-m", "babelbit"]


# Stage 6: Production with tests - production image that only builds if tests pass
FROM test AS production-tested

# Final production image - uses production stage but only builds if tests passed
FROM production AS production-with-tests

# This stage ensures tests must pass before the production image is created
# Force-build the test stage by copying its success marker; build fails if tests fail
COPY --from=test /tests-passed /tests-passed
