FROM python:3.11-slim
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY serve serve

COPY artifacts/model.joblib artifacts/model.joblib
COPY data/processed/item_lookup.parquet  data/processed/item_lookup.parquet
COPY data/processed/store_lookup.parquet data/processed/store_lookup.parquet

EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
