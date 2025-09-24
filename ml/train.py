import pandas as pd, numpy as np
from pathlib import Path
import mlflow, mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from mlflow.models.signature import infer_signature

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ART  = ROOT / "artifacts"; ART.mkdir(parents=True, exist_ok=True)

TARGET = "unit_sales"
LOG_TARGET = True

CAT = ["family","class","perishable","city","state","type","cluster"]
NUM = [
    "dcoilwtico","is_holiday","year","month","day","dow","weekofyear",
    "onpromotion","transactions",
    "lag_sales_1","lag_sales_7","lag_sales_14","lag_sales_28",
    "roll_unit_sales_7_mean","roll_unit_sales_7_std",
    "roll_unit_sales_28_mean","roll_unit_sales_28_std",
    "roll_onpromotion_7_mean","roll_onpromotion_28_mean",
    "roll_transactions_7_mean","roll_transactions_28_mean",
    "hist_mean","hist_median","rel_to_mean","rel_to_median"
]

def load_split():
    tr = pd.read_parquet(DATA/"train.parquet")
    va = pd.read_parquet(DATA/"val.parquet")
    te = pd.read_parquet(DATA/"test.parquet")
    return tr, va, te

def build_pipeline():
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])
    prep = ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])
    reg = HistGradientBoostingRegressor(
        learning_rate=0.06, max_depth=8, max_iter=400,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=42
    )
    return Pipeline([("prep", prep), ("reg", reg)])

def smape(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    denom = (np.abs(y) + np.abs(yhat)); denom[denom==0]=1e-9
    return 100 * np.mean(2*np.abs(yhat-y)/denom)

def to_y(y): return np.log1p(y) if LOG_TARGET else y
def from_yhat(yhat): return np.expm1(yhat) if LOG_TARGET else yhat

def main():
    mlflow.set_tracking_uri("file:" + str(ROOT / "ml" / "mlruns"))
    mlflow.set_experiment("favorita-forecast")

    tr, va, te = load_split()
    feats = CAT + NUM

    with mlflow.start_run() as run:
        pipe = build_pipeline()
        pipe.fit(tr[feats], to_y(tr[TARGET].values))

        yva_hat = from_yhat(pipe.predict(va[feats]))
        yte_hat = from_yhat(pipe.predict(te[feats]))

        metrics = {
            "val_rmse": float(np.sqrt(mean_squared_error(va[TARGET], yva_hat))),
            "val_mae": float(mean_absolute_error(va[TARGET], yva_hat)),
            "val_smape": float(smape(va[TARGET], yva_hat)),
            "test_rmse": float(np.sqrt(mean_squared_error(te[TARGET], yte_hat))),
            "test_mae": float(mean_absolute_error(te[TARGET], yte_hat)),
            "test_smape": float(smape(te[TARGET], yte_hat)),
        }
        for k,v in metrics.items(): mlflow.log_metric(k, v)

        # 带签名的 log（一遍）
        Xva = va[feats].head(10); pred_sample = pipe.predict(Xva)
        signature = infer_signature(Xva, pred_sample); input_example = va[feats].head(1)
        mlflow.sklearn.log_model(pipe, artifact_path="model", signature=signature, input_example=input_example)

        joblib.dump(pipe, ART/"model.joblib")
        print("RUN:", run.info.run_id)
        print(metrics)

if __name__ == "__main__":
    main()
