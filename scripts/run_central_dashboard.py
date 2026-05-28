"""
Central dashboard that aggregates live numbers from every store.

Reads config/central.yaml for the list of stores + their Tailscale URLs.
Fans out parallel HTTP requests to each store's API and assembles a single
combined view.

Run this on your Mac (or any always-on machine on the Tailnet).

Usage:
    cp config/central.example.yaml config/central.yaml
    # edit to add the right Tailscale IPs for each store
    python scripts/run_central_dashboard.py
    # then open http://localhost:8080

Environment:
    CENTRAL_PORT=8080       port to bind (default 8080)
    CENTRAL_HOST=127.0.0.1  bind interface (default localhost; 0.0.0.0 to share)
    PER_STORE_TIMEOUT=4     seconds before declaring a store offline (default 4)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse

CONFIG_PATH = ROOT / "config" / "central.yaml"
EXAMPLE_PATH = ROOT / "config" / "central.example.yaml"
DASHBOARD_DIR = ROOT / "dashboard"
TIMEOUT = float(os.environ.get("PER_STORE_TIMEOUT", "4"))

app = FastAPI(title="Customer Tracking — All Stores")


def load_stores() -> list[dict]:
    if not CONFIG_PATH.exists():
        return []
    return yaml.safe_load(CONFIG_PATH.read_text()).get("stores", []) or []


async def _fetch_store(client: httpx.AsyncClient, store: dict) -> dict:
    base = store["url"].rstrip("/")
    started = time.time()
    try:
        r1, r2 = await asyncio.gather(
            client.get(f"{base}/api/entry_stats", timeout=TIMEOUT),
            client.get(f"{base}/api/stats", timeout=TIMEOUT),
            return_exceptions=True,
        )
        entry = r1.json() if isinstance(r1, httpx.Response) and r1.status_code == 200 else None
        stats_raw = r2.json() if isinstance(r2, httpx.Response) and r2.status_code == 200 else None
        stats = None
        if stats_raw:
            stats = {
                "unique_persons": stats_raw.get("unique_persons") or stats_raw.get("unique_tracks"),
                "employees_count": stats_raw.get("employees_count") or 0,
            }
        latency_ms = int((time.time() - started) * 1000)
        return {
            "id": store.get("id"),
            "name": store.get("name", store.get("id")),
            "url": store["url"],
            "online": True,
            "entry": entry if entry and "error" not in entry else None,
            "stats": stats,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {
            "id": store.get("id"),
            "name": store.get("name", store.get("id")),
            "url": store["url"],
            "online": False,
            "error": str(e)[:60],
        }


@app.get("/api/all_stores")
async def all_stores():
    stores = load_stores()
    if not stores:
        return {"stores": [], "error": f"No stores configured. Edit {CONFIG_PATH}"}
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[_fetch_store(client, s) for s in stores])
    return {"stores": results}


@app.get("/")
def root():
    return FileResponse(DASHBOARD_DIR / "central.html")


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"No {CONFIG_PATH}.")
        print(f"  Copy: cp {EXAMPLE_PATH} {CONFIG_PATH}")
        print(f"  Then edit the Tailscale URLs and re-run.")
        return 1

    host = os.environ.get("CENTRAL_HOST", "127.0.0.1")
    port = int(os.environ.get("CENTRAL_PORT", "8080"))
    print(f"Central dashboard starting on http://{host}:{port}")
    print(f"Aggregating from {len(load_stores())} stores listed in {CONFIG_PATH}")
    uvicorn.run("scripts.run_central_dashboard:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
