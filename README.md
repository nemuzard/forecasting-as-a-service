# forecasting-as-a-service
### Business Background

- **Retail Demand Forecasting**: Predicts the future sales volume of a specific product in a store for inventory, replenishment, and promotional decisions.
- **Kaggle Favorita Dataset**: Contains daily store-product historical sales, promotional information, holidays, fuel prices, store and product attributes, and more. [link](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)

Tech：Python、pandas、scikit-learn、MLflow、FastAPI、Prometheus、Docker、GitHub Actions、Terraform、AWS S3

### Project Structure 

```
.
├─ data/
│  ├─ raw/favorita-grocery-sales-forecasting/ 
│  └─ processed/                               
├─ ml/
│  ├─ data_prep.py                            
│  ├─ train.py                                 
│  
├─ serve/
│  ├─ app.py                                   
│  ├─ inference.py                             
│  └─ Dockerfile
├─ tests/
│  └─ smoke_test.py                            
├─ artifacts/
│  └─ model.joblib                             
├─ infra/
│  └─ main.tf                                  
├─ .github/workflows/ci.yml                    
├─ .env.example
├─ requirements-server.txt
├─ requirements-train.txt                       
└─ README.md

```

### Data and feature engineer
- **Time Range**: default to the latest `RECENT_DAYS=365`
- **Entity Selection**: Top-N stores and products by sales volume
- **Exogenous Variables**: `dcoilwtico`, holiday, transactions
- **Time series lag and rolling**
    - lags: 1/7/14/28
    - rolling mean/std: 7/28
- **Cleaning**: negative sales filtering, null forward/zero filling

### Model and Evaluation
- Model: `HistGradientBoostingRegressor`(lightGBM/XGBoost alternative)
- Objective: Perform log1p training and expm1 inverse transformation prediction on `unit_sales`
- Metrics: rmse, mae, smape
- Versioning
    - MLflow: parameters, metrics, and artifact logging
    - Output: artifaces/model.joblib


