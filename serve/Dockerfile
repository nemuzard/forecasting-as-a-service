FROM python:3.11-slim
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 复制运行资产
COPY artifacts/model.joblib artifacts/model.joblib
COPY data/processed/item_lookup.parquet data/processed/item_lookup.parquet
COPY data/processed/store_lookup.parquet data/processed/store_lookup.parquet
COPY data/processed/history_minimal.parquet data/processed/history_minimal.parquet
COPY ml ml
COPY serve serve

WORKDIR /app/serve
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
