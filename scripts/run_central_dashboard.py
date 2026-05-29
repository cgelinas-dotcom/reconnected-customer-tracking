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
from pydantic import BaseModel

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
        r1, r2, r3 = await asyncio.gather(
            client.get(f"{base}/api/entry_stats", timeout=TIMEOUT),
            client.get(f"{base}/api/stats", timeout=TIMEOUT),
            client.get(f"{base}/api/admin/health", timeout=TIMEOUT),
            return_exceptions=True,
        )
        entry = r1.json() if isinstance(r1, httpx.Response) and r1.status_code == 200 else None
        stats_raw = r2.json() if isinstance(r2, httpx.Response) and r2.status_code == 200 else None
        health = r3.json() if isinstance(r3, httpx.Response) and r3.status_code == 200 else None
        stats = None
        if stats_raw:
            stats = {
                "unique_persons": stats_raw.get("unique_persons") or stats_raw.get("unique_tracks"),
                "employees_count": stats_raw.get("employees_count") or 0,
            }

        # Distill health into a single status label + summary
        # Detection age is the primary signal — if detections are happening,
        # everything upstream (NVR, camera, pipeline) must be working.
        status = "ok"
        status_detail = []
        import datetime
        now = datetime.datetime.now()
        is_business_hours = 9 <= now.hour < 21  # matches BUSINESS_HOURS=9-21
        if health:
            age = health.get("last_detection_age_sec")
            n_today = health.get("n_detections_today", 0)
            if age is None and n_today == 0:
                # Pipeline has never recorded a detection
                if is_business_hours:
                    status = "stale"
                    status_detail.append("No detections recorded yet")
                else:
                    status_detail.append("Outside business hours — pipeline idle (expected)")
            elif age is not None and age > 7200 and is_business_hours:
                # 2+ hours without detection during business hours = camera issue
                status = "stale"
                status_detail.append(f"No detection in {int(age/60)} min (during business hours)")
            else:
                if n_today:
                    status_detail.append(f"{n_today} detections today")
                if age is not None and age < 3600:
                    status_detail.append(f"last detection {int(age)}s ago")
            task = health.get("pipeline_task_status")
            if task and task.lower() not in ("running", "ready"):
                if status == "ok":
                    status = "warn"
                status_detail.append(f"Pipeline task: {task}")

        latency_ms = int((time.time() - started) * 1000)
        return {
            "id": store.get("id"),
            "name": store.get("name", store.get("id")),
            "url": store["url"],
            "online": True,
            "entry": entry if entry and "error" not in entry else None,
            "stats": stats,
            "health": health,
            "status": status,
            "status_detail": " · ".join(status_detail) if status_detail else "All good",
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {
            "id": store.get("id"),
            "name": store.get("name", store.get("id")),
            "url": store["url"],
            "online": False,
            "status": "offline",
            "status_detail": "Mini PC unreachable — power/internet/Tailscale issue",
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


async def _pull_one(client: httpx.AsyncClient, store: dict) -> dict:
    base = store["url"].rstrip("/")
    started = time.time()
    try:
        r = await client.post(f"{base}/api/admin/git_pull", timeout=60)
        d = r.json() if r.status_code == 200 else {"ok": False, "error": f"HTTP {r.status_code}"}
        return {
            "id": store.get("id"),
            "name": store.get("name"),
            "ok": d.get("ok", False),
            "pull": (d.get("pull") or "")[-500:],
            "restart": (d.get("restart") or "")[-300:],
            "took_sec": round(time.time() - started, 1),
        }
    except Exception as e:
        return {
            "id": store.get("id"),
            "name": store.get("name"),
            "ok": False,
            "error": str(e)[:120],
            "took_sec": round(time.time() - started, 1),
        }


@app.post("/api/all_stores/pull_all")
async def pull_all():
    """Trigger 'git pull + restart' on every store in parallel."""
    stores = load_stores()
    if not stores:
        return {"stores": [], "error": "No stores configured"}
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[_pull_one(client, s) for s in stores])
    n_ok = sum(1 for r in results if r.get("ok"))
    return {"results": results, "ok_count": n_ok, "total": len(results)}


@app.get("/api/all_stores/settings")
async def all_stores_settings():
    """Return the settings catalog from the first reachable store, plus the
    current per-store value for each setting. Used by the central tune panel
    so Cam can see drift across stores."""
    stores = load_stores()
    if not stores:
        return {"catalog": {}, "per_store": [], "error": "No stores configured"}

    async def fetch_settings(client: httpx.AsyncClient, store: dict) -> dict:
        try:
            r = await client.get(f"{store['url'].rstrip('/')}/api/settings", timeout=TIMEOUT)
            return {
                "id": store.get("id"),
                "name": store.get("name"),
                "online": r.status_code == 200,
                "settings": r.json() if r.status_code == 200 else None,
            }
        except Exception as e:
            return {
                "id": store.get("id"),
                "name": store.get("name"),
                "online": False,
                "error": str(e)[:80],
                "settings": None,
            }

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch_settings(client, s) for s in stores])

    catalog = next((r["settings"] for r in results if r.get("settings")), {})
    per_store = []
    for r in results:
        values = {k: r["settings"][k].get("value") for k in catalog} if r.get("settings") else {}
        per_store.append({
            "id": r["id"],
            "name": r["name"],
            "online": r["online"],
            "values": values,
            "error": r.get("error"),
        })
    return {"catalog": catalog, "per_store": per_store}


class SettingFanout(BaseModel):
    key: str
    value: float


@app.post("/api/all_stores/set_setting")
async def set_setting_all_stores(payload: SettingFanout):
    """Push one setting value to every store in parallel."""
    stores = load_stores()
    if not stores:
        return {"results": [], "error": "No stores configured"}

    async def push_one(client: httpx.AsyncClient, store: dict) -> dict:
        base = store["url"].rstrip("/")
        started = time.time()
        try:
            r = await client.put(
                f"{base}/api/settings/{payload.key}",
                json={"value": payload.value},
                timeout=TIMEOUT,
            )
            ok = r.status_code == 200
            body = r.json() if ok else {"detail": (r.text or "")[:200]}
            return {
                "id": store.get("id"),
                "name": store.get("name"),
                "ok": ok,
                "status": r.status_code,
                "detail": None if ok else body.get("detail"),
                "took_sec": round(time.time() - started, 2),
            }
        except Exception as e:
            return {
                "id": store.get("id"),
                "name": store.get("name"),
                "ok": False,
                "error": str(e)[:120],
                "took_sec": round(time.time() - started, 2),
            }

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[push_one(client, s) for s in stores])
    n_ok = sum(1 for r in results if r.get("ok"))
    return {"results": results, "ok_count": n_ok, "total": len(results)}


@app.post("/api/all_stores/wipe_data")
async def wipe_all_stores():
    """Destructive: hit /api/admin/wipe_data on every store in parallel.
    Clears persons / track_persons / detections / entry_events / visits /
    employees at each store, preserves settings, restarts pipeline+dashboard.
    Use this after big code changes to start fresh from a clean baseline."""
    stores = load_stores()
    if not stores:
        return {"results": [], "error": "No stores configured"}

    async def wipe_one(client: httpx.AsyncClient, store: dict) -> dict:
        base = store["url"].rstrip("/")
        started = time.time()
        try:
            r = await client.post(
                f"{base}/api/admin/wipe_data",
                json={"confirm": "yes"},
                timeout=20,
            )
            ok = r.status_code == 200
            body = r.json() if ok else {}
            return {
                "id": store.get("id"),
                "name": store.get("name"),
                "ok": ok,
                "deleted": body.get("deleted", {}),
                "restart": body.get("restart"),
                "took_sec": round(time.time() - started, 2),
            }
        except Exception as e:
            return {
                "id": store.get("id"),
                "name": store.get("name"),
                "ok": False,
                "error": str(e)[:120],
                "took_sec": round(time.time() - started, 2),
            }

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[wipe_one(client, s) for s in stores])
    n_ok = sum(1 for r in results if r.get("ok"))
    return {"results": results, "ok_count": n_ok, "total": len(results)}


@app.post("/api/all_stores/restart_pipelines")
async def restart_pipelines_all_stores():
    """Hit /api/admin/restart_pipeline on every store in parallel. Needed
    after pushing a setting that's only read at pipeline startup (like
    reid.similarity_threshold)."""
    stores = load_stores()
    if not stores:
        return {"results": [], "error": "No stores configured"}

    async def restart_one(client: httpx.AsyncClient, store: dict) -> dict:
        base = store["url"].rstrip("/")
        started = time.time()
        try:
            r = await client.post(f"{base}/api/admin/restart_pipeline", timeout=15)
            ok = r.status_code == 200
            body = r.json() if ok else {}
            return {
                "id": store.get("id"),
                "name": store.get("name"),
                "ok": ok and body.get("ok", False),
                "skipped": body.get("skipped", False),
                "took_sec": round(time.time() - started, 2),
            }
        except Exception as e:
            return {
                "id": store.get("id"),
                "name": store.get("name"),
                "ok": False,
                "error": str(e)[:120],
                "took_sec": round(time.time() - started, 2),
            }

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[restart_one(client, s) for s in stores])
    n_ok = sum(1 for r in results if r.get("ok"))
    return {"results": results, "ok_count": n_ok, "total": len(results)}


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
