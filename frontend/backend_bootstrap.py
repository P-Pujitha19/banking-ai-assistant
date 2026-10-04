"""Starts the mock bank (8001) and assistant API (8000) inside the Streamlit Cloud container."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def _wait(port: int, timeout: int = 180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _port_open(port):
            return
        time.sleep(1)
    raise RuntimeError(f"Backend on port {port} did not start within {timeout}s")


@st.cache_resource(show_spinner="Starting banking backend (first load can take a few minutes)...")
def ensure_backend():
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "BANK_API_MODE": "http",
        "BANK_API_URL": "http://127.0.0.1:8001",
        "ASSISTANT_API_URL": "http://127.0.0.1:8000",
        "LOG_FILE": "none",
    }
    # Generate whatever is missing (these folders are gitignored, so they are absent on Cloud)
    if not (ROOT / "data" / "bank").exists():
        _run("scripts/generate_data.py", "--anchor", "2026-10-03")
    if not (ROOT / "models" / "intent").exists():
        _run("scripts/train_intent_model.py")
    if not (ROOT / "models" / "vector_store").exists():
        _run("scripts/build_vector_store.py")

    if not _port_open(8001):
        subprocess.Popen([sys.executable, "-m", "uvicorn", "banking_ai.api.bank_app:app",
                          "--host", "127.0.0.1", "--port", "8001"], cwd=ROOT, env=env)
    _wait(8001)
    if not _port_open(8000):
        subprocess.Popen([sys.executable, "-m", "uvicorn", "banking_ai.api.assistant_app:app",
                          "--host", "127.0.0.1", "--port", "8000"], cwd=ROOT, env=env)
    _wait(8000)
    return True
