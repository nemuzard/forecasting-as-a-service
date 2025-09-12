# serve/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from inference import build_features, pipe

app = FastAPI(title="Favorita Forecast API", version="1.1.0")

# Prometheus 指标
REQ_CNT = Counter("requests_total", "Total predict requests")
REQ_ERR = Counter("requests_errors_total", "Total predict errors")
LATENCY = Histogram("predict_latency_ms", "Predict latency (ms)")

class PredictIn(BaseModel):
    store_nbr: int = Field(..., ge=1)
    item_nbr: int = Field(..., ge=1)
    date: str
    dcoilwtico: float | None = None
    is_holiday: int | None = None
    onpromotion: int | None = None
    transactions: float | None = None

@app.get("/healthz")
def healthz(): return {"status":"ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(inp: PredictIn):
    REQ_CNT.inc()
    t0 = time.time()
    try:
        X = build_features(inp.model_dump(exclude_none=True))
        yhat = float(pipe.predict(X)[0])
        lat = int((time.time()-t0)*1000)
        LATENCY.observe(lat)
        return {"yhat": round(max(yhat,0.0),2), "latency_ms": lat, "model":"v1"}
    except Exception as e:
        REQ_ERR.inc()
        raise HTTPException(status_code=400, detail=str(e))
