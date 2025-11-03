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

def build_pipeline(model_params: dict, categorical_features_list: list)-> Pipeline:
    regressor = HistGradientBoostingRegressor(
            **model_params,
            categorical_features=categorical_features_list
        )
    return Pipeline([
            ("regressor",regressor)
        ])

def smape(y, yhat):
    """ calculate msape"""
    y, yhat = np.array(y), np.array(yhat)
    denom = (np.abs(y) + np.abs(yhat)); denom[denom==0]=1e-9
    return 100 * np.mean(2*np.abs(yhat-y)/denom)

def to_y(y): 
    """Perform log1p transformation on the target based on the LOG_TARGET switch"""
    return np.log1p(y) if LOG_TARGET else y
def from_yhat(yhat): 
    """Perform an expm1 inverse transform on the prediction based on the LOG_TARGET switch"""

    return np.expm1(yhat) if LOG_TARGET else yhat

def main():

    # -- MLflow setting __
    mlflow.set_tracking_uri("file:" + str(ROOT / "ml" / "mlruns"))
    mlflow.set_experiment("favorita-forecast")

    # -- load Data --
    tr, va, te = load_split()

    # -- ensure dtype of loaded parquet data is 'category'
    for col in CAT:
        tr[col] = tr[col].astype('category')
        va[col] = va[col].astype('category')
        te[col] = te[col].astype('category')
    
    # -- define X and y --
    feats = CAT + NUM
    y_tr = to_y(tr[TARGET].values)
    y_va = va[TARGET].values
    y_te = te[TARGET].values


    # -- ML flow --
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting MLflow run : ",{run_id})

        #  -- define and record params --
        model_params = {
            "learning_rate":0.06,
            "max_depth":8,
            "max_iter":400,
            "l2_regularization":1.0,
            "early_stopping":True,
            "validation_fraction":0.1,
            "random_state":42
        }

        print("Logging params to MLflow ...")
        mlflow.log_params(model_params)

        # -- build and train pipeline
        pipe = build_pipeline(model_params,CAT)
        print("Start training ..")
        pipe.fit(tr[feats],y_tr)
        print("Complete.")

        # -- model eval --
        print("Evaluating model ..")

        # -- prediction, values are on a log scale
        yva_hat_log = pipe.predict(va[feats])
        yte_hat_log = pipe.predict(te[feats])

        # -- convert back to actual scale --
        yva_hat_actual = from_yhat(yva_hat_log)
        yte_hat_actual = from_yhat(yte_hat_log)

        # -- metrics based on the actual sales scale 

        metrics = {
            "val_rmse":float(np.sqrt(mean_squared_error(y_va, yva_hat_actual))),
            "val_mae":float(mean_absolute_error(y_va, yva_hat_actual)),
            "val_smape": float(smape(y_va, yva_hat_actual)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_te, yte_hat_actual))),
            "test_mae": float(mean_absolute_error(y_te, yte_hat_actual)),
            "test_smape": float(smape(y_te, yte_hat_actual)),
        }

        print("Logging metrics to MLflow..")
        mlflow.log_metrics(metrics)

        # -- Artifacts -- 
        print("Logging model artifacts to MLflow...")
        Xva_sample = va[feats].head(10)
        pred_sample = pipe.predict(Xva_sample)
        signature = infer_signature(Xva_sample,pred_sample)
        input_example = va[feats].head(1)

        mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                signature=signature,
                input_example=input_example
        )

        # -- save local copy --
        print("Saving local model artifact ...")
        joblib.dump(pipe,ART/"model.joblib")

        print("\n -- Run complete --")
        print(f"RUN ID: {run_id}")
        print("Metrics:")
        print(metrics)
        print(f"Model saved to: {ART/'model.joblib'}")



if __name__ == "__main__":
    main()
