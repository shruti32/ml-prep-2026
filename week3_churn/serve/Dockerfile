FROM python:3.11-slim

WORKDIR /app

# Install dependencies — pinned for reproducibility
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    psycopg2-binary==2.9.9 \
    pandas==2.2.2 \
    scikit-learn==1.5.1 \
    xgboost==2.1.1 \
    lightgbm==4.5.0 \
    pydantic==2.8.2

# Copy the pipeline source (needed so pickle can reconstruct custom transformers)
COPY week3_churn/src/pipeline.py /app/week3_churn/serve/pipeline.py

# Copy the serving layer
COPY week3_churn/serve/ /app/week3_churn/serve/

# Copy the trained model artifact
COPY week3_churn/artifacts/ /app/week3_churn/artifacts/

WORKDIR /app/week3_churn/serve

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
