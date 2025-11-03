import pandas as pd
from pathlib import Path
import time
import gc # c

# --- 1. SETUP: parth ---

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "favorita-grocery-sales-forecasting"


# --- 2. define dtypes ---

train_dtypes = {
    'id': 'int64',
    'store_nbr': 'int8',
    'item_nbr': 'int32',
    'onpromotion': pd.BooleanDtype(),
    'sales': 'float32'
}

items_dtypes = {
    'item_nbr': 'int32',
    'family': 'category',
    'class': 'category',
    'perishable': pd.BooleanDtype() 
}

stores_dtype = {
    'store_nbr': 'int8',
    'city': 'category',
    'state': 'category',
    'type': 'category',
    'cluster': 'category'
}

oil_dtypes = {'dcoilwtico': 'float32'}

holidays_dtypes = {
    'type': 'category',
    'locale': 'category',
    'locale_name': 'category',
    'description': 'object',
    'transferred': 'bool'
}

transactions_dtypes = {
    'store_nbr': 'int8',
    'transactions': 'int32'
}

# --- 3. measure memory usage ---
def get_memory_mb(df: pd.DataFrame) -> float:
    """use deep=True to calculate DataFrame real memory usage (MB)"""
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / 1024**2 # byte -> megabyte
    return mem_mb

# --- 4. Comparsion ---

# track total memory usage
total_default_mb = 0
total_optimized_mb = 0

print("="*40)
print("--- `items.csv` ---")
# 1. Default
start = time.time()
df_default = pd.read_csv(RAW/"items.csv")
mem = get_memory_mb(df_default)
print(f"Default:   {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_default_mb += mem
del df_default; gc.collect() # clean 

# 2. Optimized
start = time.time()
df_optimized = pd.read_csv(RAW/"items.csv", dtype=items_dtypes)
mem = get_memory_mb(df_optimized)
print(f"Optimized: {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_optimized_mb += mem
del df_optimized; gc.collect()


print("\n" + "="*40)
print("--- `stores.csv` ---")
# 1. Default
start = time.time()
df_default = pd.read_csv(RAW/"stores.csv")
mem = get_memory_mb(df_default)
print(f"Default:   {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_default_mb += mem
del df_default; gc.collect()

# 2. Optimized
start = time.time()
df_optimized = pd.read_csv(RAW/"stores.csv", dtype=stores_dtype)
mem = get_memory_mb(df_optimized)
print(f"Optimized: {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_optimized_mb += mem
del df_optimized; gc.collect()


print("\n" + "="*40)
print("---  `holidays_events.csv` ---")
# 1. Default
start = time.time()
df_default = pd.read_csv(RAW/"holidays_events.csv")
mem = get_memory_mb(df_default)
print(f"Default:   {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_default_mb += mem
del df_default; gc.collect()

# 2. Optimized
start = time.time()
df_optimized = pd.read_csv(RAW/"holidays_events.csv", parse_dates=["date"], dtype=holidays_dtypes)
mem = get_memory_mb(df_optimized)
print(f"Optimized: {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_optimized_mb += mem
del df_optimized; gc.collect()


print("\n" + "="*40)
print("--- `transactions.csv` ---")
# 1. Default
start = time.time()
df_default = pd.read_csv(RAW/"transactions.csv")
mem = get_memory_mb(df_default)
print(f"Default:   {mem:,.2f} MB (耗时: {time.time()-start:.2f}s)")
total_default_mb += mem
del df_default; gc.collect()

# 2. Optimized
start = time.time()
df_optimized = pd.read_csv(RAW/"transactions.csv", parse_dates=["date"], dtype=transactions_dtypes)
mem = get_memory_mb(df_optimized)
print(f"Optimized: {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_optimized_mb += mem
del df_optimized; gc.collect()


print("\n" + "="*40)
print("--- `oil.csv` ---")
# 1. Default
start = time.time()
df_default = pd.read_csv(RAW/"oil.csv")
mem = get_memory_mb(df_default)
print(f"Default:   {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_default_mb += mem
del df_default; gc.collect()

# 2. Optimized
start = time.time()
df_optimized = pd.read_csv(RAW/"oil.csv", parse_dates=["date"], dtype=oil_dtypes)
mem = get_memory_mb(df_optimized)
print(f"Optimized: {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_optimized_mb += mem
del df_optimized; gc.collect()


print("\n" + "="*40)
print("--- Comparing `train.csv` ( ---")
# 1. Default
start = time.time()
df_default = pd.read_csv(RAW/"train.csv")
mem = get_memory_mb(df_default)
print(f"Default:   {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_default_mb += mem
del df_default; gc.collect()

# 2. Optimized
start = time.time()
df_optimized = pd.read_csv(RAW/"train.csv", parse_dates=["date"], dtype=train_dtypes)
mem = get_memory_mb(df_optimized)
print(f"Optimized: {mem:,.2f} MB (Time required: {time.time()-start:.2f}s)")
total_optimized_mb += mem
del df_optimized; gc.collect()


# --- 5. final comparison ---
print("\n" + "="*60)
print("---  Final result (use deep=True )  ---")
print(f"Optimized: {total_optimized_mb:,.2f} MB")
print(f"Default:   {total_default_mb:,.2f} MB")
print("-" * 60)

if total_default_mb > 0:
    reduction_pct = (total_default_mb - total_optimized_mb) / total_default_mb * 100
    print(f"Total saved {total_default_mb - total_optimized_mb:,.2f} MB memory")
    print(f"Memory usage reduced by : {reduction_pct:.2f}%")
else:
    print("Cannot calculate")

print("="*60)