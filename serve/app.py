"""
Routes:
- GET  /healthz : liveness probe
- POST /predict : single-point forecast
- GET  /metrics : Prometheus metrics
"""
from fastapi import FastAPI, HTTPException, Response
from .inference import PredictIn, predict_one, metrics

app = FastAPI(title="Favorita Forecast API", version="1.1.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        return predict_one(inp)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def prom():
    ctype, body = metrics()
    return Response(content=body, media_type=ctype)
