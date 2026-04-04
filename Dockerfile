FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    numpy \
    pandas>=2.0 \
    scikit-learn \
    scipy \
    fastapi \
    uvicorn \
    jupyter \
    typer \
    click \
    jinja2

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

CMD ["python"]