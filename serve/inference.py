from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ROOT / "data" / "processed"

# ---------- Load artifacts (fail fast if missing) ----------
MODEL_PATH = ART / "model.joblib"
ITEM_LK_PATH = DATA / "item_lookup.parquet"
STORE_LK_PATH = DATA / "store_lookup.parquet"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not ITEM_LK_PATH.exists() or not STORE_LK_PATH.exists():
    raise FileNotFoundError("Lookup parquet missing. Expect "
                            f"{ITEM_LK_PATH} and {STORE_LK_PATH}")

pipe = joblib.load(MODEL_PATH)
ITEM_LK = pd.read_parquet(ITEM_LK_PATH)
STORE_LK = pd.read_parquet(STORE_LK_PATH)

# ---------- Prometheus metrics ----------
PREDICT_CT = Counter("predict_requests_total", "Total /predict requests")
PREDICT_ERR = Counter("predict_errors_total", "Total /predict errors")
PREDICT_LAT = Histogram(
    "predict_latency_ms",
    "Predict latency (ms)",
    buckets=(5, 10, 20, 50, 100, 200, 500, 1000, 2000),
)

def metrics():
    """Return (content_type, body) for /metrics endpoint."""
    return CONTENT_TYPE_LATEST, generate_latest()

# ---------- Input schema (for internal use) ----------
class PredictIn(BaseModel):
    store_nbr: int
    item_nbr: int
    date: str  # YYYY-MM-DD
    dcoilwtico: float | None = None
    is_holiday: int | None = None
    onpromotion: int | None = None
    transactions: float | None = None

    @field_validator("is_holiday", "onpromotion", mode="before")
    @classmethod
    def _to_int(cls, v):
        if v is None: return 0
        return int(v)

    @field_validator("transactions", mode="before")
    @classmethod
    def _to_float_nonneg(cls, v):
        if v is None: return 0.0
        return float(v)

# ---------- Feature building ----------
def _calendar_from_date(date_str: str) -> Dict[str, Any]:
    d = pd.to_datetime(date_str)
    return {
        "year": d.year,
        "month": d.month,
        "day": d.day,
        "dow": d.dayofweek,
        "weekofyear": int(d.isocalendar().week),
    }

_ZERO_FEATS = [
    # lags
    "lag_sales_1","lag_sales_7","lag_sales_14","lag_sales_28",
    # rolling stats
    "roll_unit_sales_7_mean","roll_unit_sales_7_std",
    "roll_unit_sales_28_mean","roll_unit_sales_28_std",
    "roll_onpromotion_7_mean","roll_onpromotion_28_mean",
    "roll_transactions_7_mean","roll_transactions_28_mean",
    # historical stats / relatives
    "hist_mean","hist_median","rel_to_mean","rel_to_median",
]

def build_features(inp: PredictIn) -> pd.DataFrame:
    """Build a single-row feature frame that matches the training schema."""
    # base numerics
    base = {
        "dcoilwtico": float(inp.dcoilwtico or 0.0),
        "is_holiday": int(inp.is_holiday or 0),
        "onpromotion": int(inp.onpromotion or 0),
        "transactions": float(inp.transactions or 0.0),
        **_calendar_from_date(inp.date),
    }
    # zeros for online-only unavailable features
    zeros = {k: 0.0 for k in _ZERO_FEATS}

    # categorical lookups
    item_row = ITEM_LK[ITEM_LK["item_nbr"] == inp.item_nbr].head(1)
    store_row = STORE_LK[STORE_LK["store_nbr"] == inp.store_nbr].head(1)
    if item_row.empty or store_row.empty:
        raise ValueError("Unknown store_nbr or item_nbr (not found in lookups).")

    cats = {
        "family": item_row["family"].iloc[0],
        "class": item_row["class"].iloc[0],
        "perishable": item_row["perishable"].iloc[0],
        "city": store_row["city"].iloc[0],
        "state": store_row["state"].iloc[0],
        "type": store_row["type"].iloc[0],
        "cluster": store_row["cluster"].iloc[0],
    }

    row = {**base, **zeros, **cats}
    X = pd.DataFrame([row])
    return X

# ---------- Prediction ----------
def predict_one(inp: PredictIn) -> dict:
    """Run a single prediction with latency/metrics."""
    PREDICT_CT.inc()
    t0 = time.time()
    try:
        X = build_features(inp)
        yhat = float(pipe.predict(X)[0]) 
        yhat = float(np.expm1(yhat))
        latency_ms = int((time.time() - t0) * 1000)
        PREDICT_LAT.observe(latency_ms)
        return {"yhat": round(max(yhat, 0.0), 2), "latency_ms": latency_ms, "model": "v1"}
    except Exception:
        PREDICT_ERR.inc()
        raise
