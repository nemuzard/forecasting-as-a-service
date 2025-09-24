import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "favorita-grocery-sales-forecasting"
OUT = ROOT / "data" / "processed"; OUT.mkdir(parents=True, exist_ok=True)

RECENT_DAYS = 365
N_STORES = 20
N_ITEMS = 200

def load_tables():
    train = pd.read_csv(RAW/"train.csv", parse_dates=["date"])
    items = pd.read_csv(RAW/"items.csv")
    stores = pd.read_csv(RAW/"stores.csv")
    oil = pd.read_csv(RAW/"oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(RAW/"holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(RAW/"transactions.csv", parse_dates=["date"])
    return train, items, stores, oil, holidays, transactions

def make_holiday_features(holidays: pd.DataFrame) -> pd.DataFrame:
    h = holidays.copy()
    h["is_holiday"] = 1
    return h[["date","is_holiday"]].drop_duplicates("date")

def recent_subset(train: pd.DataFrame) -> pd.DataFrame:
    max_date = train["date"].max()
    min_date = max_date - pd.Timedelta(days=RECENT_DAYS)
    sub = train[(train["date"]>=min_date) & (train["date"]<=max_date)].copy()
    store_top = sub.groupby("store_nbr")["unit_sales"].sum().sort_values(ascending=False).head(N_STORES).index
    item_top  = sub.groupby("item_nbr")["unit_sales"].sum().sort_values(ascending=False).head(N_ITEMS).index
    return sub[sub["store_nbr"].isin(store_top) & sub["item_nbr"].isin(item_top)]

def add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    return df

def main():
    train, items, stores, oil, holidays, transactions = load_tables()
    sub = recent_subset(train)

    df = sub.merge(items, on="item_nbr", how="left") \
            .merge(stores, on="store_nbr", how="left")

    oil = oil.sort_values("date").copy()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
    df = df.merge(oil, on="date", how="left")
    df["dcoilwtico"] = df["dcoilwtico"].ffill().fillna(0)

    h = make_holiday_features(holidays)
    df = df.merge(h, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    df = df.merge(transactions, on=["date","store_nbr"], how="left")
    df["transactions"] = df["transactions"].fillna(0).astype(int)
    df["onpromotion"] = df.get("onpromotion", 0)
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)

    df = add_calendar_feats(df)
    df = df.sort_values(["store_nbr","item_nbr","date"])
    grp = df.groupby(["store_nbr","item_nbr"], group_keys=False)

    for l in [1,7,14,28]:
        df[f"lag_sales_{l}"] = grp["unit_sales"].shift(l)

    def add_roll(col, wins):
        for w in wins:
            df[f"roll_{col}_{w}_mean"] = grp[col].shift(1).rolling(w, min_periods=3).mean()
            df[f"roll_{col}_{w}_std"]  = grp[col].shift(1).rolling(w, min_periods=3).std()
    add_roll("unit_sales",[7,28])
    add_roll("onpromotion",[7,28])
    add_roll("transactions",[7,28])

    lag_cols = [c for c in df.columns if c.startswith(("lag_","roll_"))]
    df = df.dropna(subset=lag_cols, how="any")  # 更稳，丢掉不完整窗口
    df["hist_mean"] = grp["unit_sales"].shift(1).expanding(min_periods=10).mean()
    df["hist_median"] = grp["unit_sales"].shift(1).expanding(min_periods=10).median()
    eps = 1e-6
    df["rel_to_mean"] = (df["lag_sales_1"] - df["hist_mean"]) / (df["hist_mean"] + eps)
    df["rel_to_median"] = (df["lag_sales_1"] - df["hist_median"]) / (df["hist_median"] + eps)
    df = df[df["unit_sales"] >= 0]

    df = df.sort_values("date")
    last_date = df["date"].max()
    val_start = last_date - pd.Timedelta(weeks=12)
    test_start = last_date - pd.Timedelta(weeks=8)
    train_df = df[df["date"] < val_start]
    val_df   = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test_df  = df[df["date"] >= test_start]

    train_df.to_parquet(OUT/"train.parquet", index=False)
    val_df.to_parquet(OUT/"val.parquet", index=False)
    test_df.to_parquet(OUT/"test.parquet", index=False)

    df[["item_nbr","family","class","perishable"]].drop_duplicates("item_nbr").to_parquet(OUT/"item_lookup.parquet", index=False)
    df[["store_nbr","city","state","type","cluster"]].drop_duplicates("store_nbr").to_parquet(OUT/"store_lookup.parquet", index=False)
    print("Prepared:", OUT)

if __name__ == "__main__":
    main()