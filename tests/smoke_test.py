# tests/smoke_test.py
from __future__ import annotations
import os
import sys
import time
import json
import signal
import socket
import subprocess
from pathlib import Path

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP  = "serve.app:app"
HOST = os.environ.get("SMOKE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SMOKE_PORT", "8000"))
BASE = f"http://{HOST}:{PORT}"

def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) != 0

def start_server() -> subprocess.Popen:
    """Start uvicorn as a subprocess from project root."""
    if not _is_port_free(HOST, PORT):
        print(f"[smoke] ERROR: port {PORT} is already in use on {HOST}", file=sys.stderr)
        sys.exit(2)

    env = os.environ.copy()
    cmd = [sys.executable, "-m", "uvicorn", APP, "--host", HOST, "--port", str(PORT)]
    print(f">>> Starting uvicorn: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc

def wait_health(timeout_s: int = 20) -> dict:
    """Poll /healthz until healthy or timeout."""
    url = f"{BASE}/healthz"
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                return r.json()
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"/healthz not ready within {timeout_s}s; last_err={last_err}")

def stop_server(proc: subprocess.Popen, timeout_s: int = 5) -> None:
    """Gracefully terminate uvicorn subprocess."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if proc.poll() is not None:
                return
            time.sleep(0.2)
        proc.kill()
    except Exception:
        pass

def pick_store_item() -> tuple[int, int]:
    store_lk = pd.read_parquet(ROOT / "data/processed/store_lookup.parquet")
    item_lk  = pd.read_parquet(ROOT / "data/processed/item_lookup.parquet")
    s = int(store_lk["store_nbr"].iloc[0])
    i = int(item_lk["item_nbr"].iloc[0])
    return s, i

def call_predict(store: int, item: int) -> dict:
    url = f"{BASE}/predict"
    payload = {
        "store_nbr": store,
        "item_nbr": item,
        "date": "2017-07-01",
        "dcoilwtico": 47.8,
        "is_holiday": 0,
        "onpromotion": 1,
        "transactions": 800,
    }
    r = requests.post(url, json=payload, timeout=5)
    r.raise_for_status()
    return r.json()

def print_metrics_head(n: int = 20) -> None:
    url = f"{BASE}/metrics"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    lines = r.text.splitlines()[:n]
    print("Metrics (head):")
    print("\n".join(lines))

def tail_process(proc: subprocess.Popen, n: int = 60) -> None:
    """Print last lines from server stdout (best effort)."""
    try:
        if proc.stdout:
            out = proc.stdout.read()
            if out:
                tail = "\n".join(out.splitlines()[-n:])
                if tail:
                    print("---- uvicorn stdout (tail) ----")
                    print(tail)
                    print("---- end stdout ----")
    except Exception:
        pass

if __name__ == "__main__":
    os.chdir(ROOT)
    rc = 0
    proc = None
    try:
        proc = start_server()
        health = wait_health(timeout_s=25)
        print("Healthz:", health)

        s, i = pick_store_item()
        print(f"Using store={s}, item={i}")
        pred = call_predict(s, i)
        print("Predict:", json.dumps(pred, ensure_ascii=False))

        print_metrics_head(30)

    except Exception as e:
        rc = 1
        print(f"[smoke] FAILED: {e}", file=sys.stderr)
        if proc is not None:
            tail_process(proc)
    finally:
        print(">>> Stopping server ...")
        if proc is not None:
            stop_server(proc)
        sys.exit(rc)
