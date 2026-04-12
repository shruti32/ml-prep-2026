# ── Stage 1: Builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a venv and install everything into it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the entire venv from builder — executables and packages all in one place
COPY --from=builder /opt/venv /opt/venv

# Put the venv on PATH so uvicorn, python etc. are found
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ ./src/

EXPOSE 8000