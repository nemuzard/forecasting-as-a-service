from pathlib import Path
import joblib
import pandas as pd
from dateutil.parser import isoparse

# 路径稳健
ROOT = Path(__file__).resolve().parents[1]
ART  = ROOT / "artifacts"
DATA = ROOT / "data" / "processed"

# 模型与查表
pipe = joblib.load(ART / "model.joblib")
item_lkp = pd.read_parquet(DATA / "item_lookup.parquet").set_index("item_nbr")
store_lkp = pd.read_parquet(DATA / "store_lookup.parquet").set_index("store_nbr")
history   = pd.read_parquet(DATA / "history_minimal.parquet")  # 只含必要历史列

# 与训练保持一致的列
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

def _calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    return df

def _compute_time_feats(sid: int, iid: int, asof_date: pd.Timestamp,
                        onpromotion: int, transactions: float):
    """从历史表按 (store,item) 截取 asof_date 之前的数据，算 lag/roll/hist。"""
    g = history[(history.store_nbr==sid)&(history.item_nbr==iid)].copy()
    g = g[g["date"] < asof_date].sort_values("date")
    # 若历史太少，直接返回全0特征（不理想，但保证服务可用）
    if len(g) == 0:
        return {
            "lag_sales_1":0,"lag_sales_7":0,"lag_sales_14":0,"lag_sales_28":0,
            "roll_unit_sales_7_mean":0,"roll_unit_sales_7_std":0,
            "roll_unit_sales_28_mean":0,"roll_unit_sales_28_std":0,
            "roll_onpromotion_7_mean":0,"roll_onpromotion_28_mean":0,
            "roll_transactions_7_mean":0,"roll_transactions_28_mean":0,
            "hist_mean":0,"hist_median":0,"rel_to_mean":0,"rel_to_median":0
        }
    # 滞后
    g["lag1"]  = g["unit_sales"].shift(1)
    g["lag7"]  = g["unit_sales"].shift(7)
    g["lag14"] = g["unit_sales"].shift(14)
    g["lag28"] = g["unit_sales"].shift(28)

    # rolling（与训练一致：先 shift(1) 再 rolling）
    s_sales = g["unit_sales"].shift(1)
    s_promo = g["onpromotion"].shift(1)
    s_tran  = g["transactions"].shift(1)

    r7m  = s_sales.rolling(7,  min_periods=3).mean()
    r7s  = s_sales.rolling(7,  min_periods=3).std()
    r28m = s_sales.rolling(28, min_periods=3).mean()
    r28s = s_sales.rolling(28, min_periods=3).std()

    rp7  = s_promo.rolling(7,  min_periods=3).mean()
    rp28 = s_promo.rolling(28, min_periods=3).mean()
    rt7  = s_tran.rolling(7,  min_periods=3).mean()
    rt28 = s_tran.rolling(28, min_periods=3).mean()

    # 历史均值/中位数（expanding）
    hist_mean   = s_sales.expanding(min_periods=10).mean()
    hist_median = s_sales.expanding(min_periods=10).median()

    # 取最后一个可用窗口
    lag1  = g["lag1"].iloc[-1]  if not g["lag1"].dropna().empty  else 0.0
    lag7  = g["lag7"].iloc[-1]  if not g["lag7"].dropna().empty  else 0.0
    lag14 = g["lag14"].iloc[-1] if not g["lag14"].dropna().empty else 0.0
    lag28 = g["lag28"].iloc[-1] if not g["lag28"].dropna().empty else 0.0

    roll7m  = r7m.iloc[-1]  if not r7m.dropna().empty  else 0.0
    roll7s  = r7s.iloc[-1]  if not r7s.dropna().empty  else 0.0
    roll28m = r28m.iloc[-1] if not r28m.dropna().empty else 0.0
    roll28s = r28s.iloc[-1] if not r28s.dropna().empty else 0.0

    rollp7  = rp7.iloc[-1]  if not rp7.dropna().empty  else 0.0
    rollp28 = rp28.iloc[-1] if not rp28.dropna().empty else 0.0
    rollt7  = rt7.iloc[-1]  if not rt7.dropna().empty  else 0.0
    rollt28 = rt28.iloc[-1] if not rt28.dropna().empty else 0.0

    hmean   = hist_mean.iloc[-1]   if not hist_mean.dropna().empty   else 0.0
    hmedian = hist_median.iloc[-1] if not hist_median.dropna().empty else 0.0
    eps = 1e-6
    rel_mean   = (lag1 - hmean)   / (hmean+eps)   if hmean>0 else 0.0
    rel_median = (lag1 - hmedian) / (hmedian+eps) if hmedian>0 else 0.0

    return {
        "lag_sales_1":lag1, "lag_sales_7":lag7, "lag_sales_14":lag14, "lag_sales_28":lag28,
        "roll_unit_sales_7_mean":roll7m, "roll_unit_sales_7_std":roll7s,
        "roll_unit_sales_28_mean":roll28m, "roll_unit_sales_28_std":roll28s,
        "roll_onpromotion_7_mean":rollp7, "roll_onpromotion_28_mean":rollp28,
        "roll_transactions_7_mean":rollt7, "roll_transactions_28_mean":rollt28,
        "hist_mean":hmean, "hist_median":hmedian,
        "rel_to_mean":rel_mean, "rel_to_median":rel_median
    }

def build_features(payload: dict) -> pd.DataFrame:
    """
    输入至少需要：
      store_nbr, item_nbr, date(YYYY-MM-DD)
    可选（强烈建议传）：dcoilwtico, is_holiday, onpromotion, transactions
    """
    d = payload.copy()
    d["date"] = pd.to_datetime(isoparse(d["date"]).date())
    sid = int(d["store_nbr"]); iid = int(d["item_nbr"])

    # 画像补齐
    if sid in store_lkp.index:
      for k,v in store_lkp.loc[sid].to_dict().items(): d.setdefault(k, v)
    if iid in item_lkp.index:
      for k,v in item_lkp.loc[iid].to_dict().items(): d.setdefault(k, v)

    # 默认值（若前端不传）
    d.setdefault("dcoilwtico", 0.0)
    d.setdefault("is_holiday", 0)
    d.setdefault("onpromotion", 0)
    d.setdefault("transactions", 0.0)

    # 时间派生
    df = pd.DataFrame([d])
    df = _calendar_feats(df)

    # 基于历史的时序特征
    time_feats = _compute_time_feats(
        sid=sid, iid=iid, asof_date=df["date"].iloc[0],
        onpromotion=int(d["onpromotion"]), transactions=float(d["transactions"])
    )
    for k,v in time_feats.items():
        df[k] = v

    # 只保留模型需要的列
    for c in CAT + NUM:
        if c not in df.columns: df[c] = 0
    return df[CAT + NUM]
