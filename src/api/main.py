"""
Local dashboard API. Reads from data/events.sqlite.

Start with:
    python scripts/run_dashboard.py
Then open http://localhost:8000
"""

from pathlib import Path
import sqlite3
import time

from datetime import datetime
import csv
import io

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "events.sqlite"
DASHBOARD_DIR = ROOT / "dashboard"
CONFIG_PATH = ROOT / "config" / "stores.yaml"

app = FastAPI(title="Customer Tracking — Local Dev")


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def has_data() -> bool:
    if not DB_PATH.exists():
        return False
    with db() as conn:
        return conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='detections'"
        ).fetchone() is not None


def ensure_employees_table() -> None:
    """Create the employees table on demand — first call to any employee endpoint."""
    if not DB_PATH.exists():
        return
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                person_id    INTEGER PRIMARY KEY,
                name         TEXT,
                role         TEXT,
                enrolled_at  REAL NOT NULL
            )
            """
        )
        conn.commit()


@app.get("/api/health")
def health():
    return {"status": "ok", "db_exists": DB_PATH.exists(), "has_data": has_data()}


@app.get("/api/stats")
def stats():
    if not has_data():
        return {"error": "No events yet. Run: python scripts/run_pipeline.py data/samples/people-detection.mp4"}
    with db() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_detections,
                COUNT(DISTINCT d.store_id || '|' || d.camera_id || '|' || d.track_id) AS unique_tracks,
                COUNT(DISTINCT d.store_id) AS stores,
                COUNT(DISTINCT d.camera_id) AS cameras,
                MIN(d.ts) AS first_ts,
                MAX(d.ts) AS last_ts
            FROM detections d
            """
        ).fetchone()
        result = dict(row)

        persons_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='persons'"
        ).fetchone()
        if persons_table:
            result["unique_persons"] = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        else:
            result["unique_persons"] = None

        emp_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone()
        if emp_table and persons_table:
            result["employees_count"] = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
            result["customer_persons"] = (result["unique_persons"] or 0) - result["employees_count"]
        else:
            result["employees_count"] = 0
            result["customer_persons"] = result["unique_persons"]

        return result


@app.get("/api/by_store")
def by_store():
    if not has_data():
        return []
    with db() as conn:
        rows = conn.execute(
            """
            SELECT
                d.store_id,
                COUNT(DISTINCT d.camera_id || '|' || d.track_id) AS unique_tracks,
                COUNT(*) AS total_detections,
                COUNT(DISTINCT d.camera_id) AS cameras
            FROM detections d
            GROUP BY d.store_id
            ORDER BY d.store_id
            """
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/visit_stats")
def visit_stats():
    with db() as conn:
        has_visits = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='visits'"
        ).fetchone()
        if not has_visits:
            return {"error": "No visits computed. Run: python scripts/compute_visits.py"}

        has_emp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone()

        # Always compute the gross stats
        gross = conn.execute(
            """
            SELECT
                COUNT(*)                          AS total_visits,
                COUNT(DISTINCT person_id)         AS unique_visitors,
                SUM(is_returning)                 AS returning_visits,
                AVG(duration_sec)                 AS avg_duration_sec,
                MAX(duration_sec)                 AS max_duration_sec,
                AVG(n_cameras)                    AS avg_cameras_per_visit,
                MAX(session_timeout_sec)          AS session_timeout_sec
            FROM visits
            """
        ).fetchone()
        result = dict(gross)

        if has_emp:
            cust = conn.execute(
                """
                SELECT
                    COUNT(*) AS customer_visits,
                    COUNT(DISTINCT v.person_id) AS customer_unique_visitors,
                    SUM(v.is_returning) AS customer_returning_visits,
                    AVG(v.duration_sec) AS customer_avg_duration_sec
                FROM visits v
                LEFT JOIN employees e ON v.person_id = e.person_id
                WHERE e.person_id IS NULL
                """
            ).fetchone()
            result.update(dict(cust))
            emp = conn.execute(
                """
                SELECT
                    COUNT(*) AS employee_visits,
                    COUNT(DISTINCT v.person_id) AS employee_unique
                FROM visits v
                INNER JOIN employees e ON v.person_id = e.person_id
                """
            ).fetchone()
            result.update(dict(emp))
        else:
            # No employees enrolled yet — customer == total
            result["customer_visits"] = result["total_visits"]
            result["customer_unique_visitors"] = result["unique_visitors"]
            result["customer_returning_visits"] = result["returning_visits"]
            result["customer_avg_duration_sec"] = result["avg_duration_sec"]
            result["employee_visits"] = 0
            result["employee_unique"] = 0

        return result


@app.get("/api/visits")
def visits():
    with db() as conn:
        has_visits = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='visits'"
        ).fetchone()
        if not has_visits:
            return []
        has_emp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone()
        emp_join = (
            "LEFT JOIN employees e ON v.person_id = e.person_id"
            if has_emp else ""
        )
        emp_col = "CASE WHEN e.person_id IS NOT NULL THEN 1 ELSE 0 END" if has_emp else "0"
        rows = conn.execute(
            f"""
            SELECT v.visit_id, v.person_id, v.store_id, v.started_ts, v.ended_ts,
                   v.duration_sec, v.n_detections, v.n_cameras, v.is_returning,
                   {emp_col} AS is_employee
            FROM visits v
            {emp_join}
            ORDER BY v.started_ts DESC
            LIMIT 200
            """
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/persons")
def persons():
    """Cross-camera person registry (populated when REID=1 pipeline runs)."""
    with db() as conn:
        persons_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='persons'"
        ).fetchone()
        if not persons_table:
            return []
        has_emp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone()
        emp_join = (
            "LEFT JOIN employees e ON p.person_id = e.person_id"
            if has_emp else ""
        )
        emp_cols = (
            ", CASE WHEN e.person_id IS NOT NULL THEN 1 ELSE 0 END AS is_employee, "
            "e.name AS employee_name, e.role AS employee_role, e.tagged_via AS tagged_via"
            if has_emp else
            ", 0 AS is_employee, NULL AS employee_name, NULL AS employee_role, NULL AS tagged_via"
        )
        rows = conn.execute(
            f"""
            SELECT
                p.person_id,
                p.first_seen_ts,
                p.last_seen_ts,
                p.first_store,
                p.first_camera,
                p.n_samples,
                COUNT(DISTINCT tp.store_id || '|' || tp.camera_id) AS cameras_seen
                {emp_cols}
            FROM persons p
            LEFT JOIN track_persons tp ON p.person_id = tp.person_id
            {emp_join}
            GROUP BY p.person_id
            ORDER BY p.last_seen_ts DESC
            LIMIT 200
            """
        ).fetchall()
        return [dict(r) for r in rows]


class EmployeeEnroll(BaseModel):
    person_id: int
    name: str | None = None
    role: str | None = None


@app.get("/api/employees")
def list_employees():
    ensure_employees_table()
    if not DB_PATH.exists():
        return []
    with db() as conn:
        # Backfill column for older DBs
        try:
            conn.execute("ALTER TABLE employees ADD COLUMN tagged_via TEXT NOT NULL DEFAULT 'manual'")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        rows = conn.execute(
            """
            SELECT e.person_id, e.name, e.role, e.enrolled_at, e.tagged_via,
                   p.first_seen_ts, p.last_seen_ts
            FROM employees e
            LEFT JOIN persons p ON e.person_id = p.person_id
            ORDER BY e.enrolled_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/employees")
def enroll_employee(payload: EmployeeEnroll):
    ensure_employees_table()
    if not DB_PATH.exists():
        raise HTTPException(404, "DB doesn't exist yet")
    with db() as conn:
        person = conn.execute(
            "SELECT person_id FROM persons WHERE person_id = ?", (payload.person_id,)
        ).fetchone()
        if not person:
            raise HTTPException(404, f"person_id {payload.person_id} not found")
        conn.execute(
            "INSERT OR REPLACE INTO employees (person_id, name, role, enrolled_at) VALUES (?, ?, ?, ?)",
            (payload.person_id, payload.name, payload.role, time.time()),
        )
        conn.commit()
        return {"ok": True, "person_id": payload.person_id}


@app.delete("/api/employees/{person_id}")
def unenroll_employee(person_id: int):
    ensure_employees_table()
    if not DB_PATH.exists():
        raise HTTPException(404, "DB doesn't exist yet")
    with db() as conn:
        conn.execute("DELETE FROM employees WHERE person_id = ?", (person_id,))
        conn.commit()
        return {"ok": True, "person_id": person_id}


class SettingUpdate(BaseModel):
    value: float


@app.get("/api/settings")
def list_settings():
    """All tunable settings with current values + metadata for the dashboard."""
    from src import settings as settings_mod
    with db() as conn:
        return settings_mod.get_all(conn)


@app.put("/api/settings/{key:path}")
def update_setting(key: str, payload: SettingUpdate):
    from src import settings as settings_mod
    with db() as conn:
        try:
            settings_mod.set_value(conn, key, payload.value)
        except KeyError:
            raise HTTPException(404, f"Unknown setting: {key}")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "key": key, "value": payload.value}


@app.delete("/api/settings/{key:path}")
def reset_setting(key: str):
    """Reset a setting to its default by deleting the override row."""
    from src import settings as settings_mod
    if key not in settings_mod.DEFAULTS:
        raise HTTPException(404, f"Unknown setting: {key}")
    with db() as conn:
        settings_mod.ensure_table(conn)
        conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        conn.commit()
        return {"ok": True, "key": key, "value": settings_mod.DEFAULTS[key].default}


def _today_bounds():
    """Returns (start_of_today_ts, now_ts) in local time."""
    import datetime
    now = datetime.datetime.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp(), now.timestamp()


@app.get("/api/validation")
def validation_window(
    start_iso: str = Query(..., description="ISO datetime, e.g. 2026-05-26T10:00"),
    end_iso: str = Query(..., description="ISO datetime, e.g. 2026-05-26T12:00"),
    store_id: str = Query("", description="optional, filter to one store"),
):
    """For accuracy validation: returns what the system counted in a specific
    time window, so you can compare against a manual ground-truth count."""
    import datetime
    try:
        start_ts = datetime.datetime.fromisoformat(start_iso).timestamp()
        end_ts = datetime.datetime.fromisoformat(end_iso).timestamp()
    except Exception:
        raise HTTPException(400, "start_iso and end_iso must be ISO datetimes (e.g. 2026-05-26T10:00)")

    if end_ts <= start_ts:
        raise HTTPException(400, "end_iso must be after start_iso")

    with db() as conn:
        has_entries = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entry_events'"
        ).fetchone() is not None
        has_emp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone() is not None

        store_clause = "AND store_id = ?" if store_id else ""
        params: list = [start_ts, end_ts]
        if store_id:
            params.append(store_id)

        # Total detections in window
        det_total = conn.execute(
            f"SELECT COUNT(*) FROM detections WHERE ts BETWEEN ? AND ? {store_clause}",
            params,
        ).fetchone()[0]
        det_tracks = conn.execute(
            f"SELECT COUNT(DISTINCT store_id || '|' || camera_id || '|' || track_id) "
            f"FROM detections WHERE ts BETWEEN ? AND ? {store_clause}",
            params,
        ).fetchone()[0]

        # Entry events in window
        entries_in = entries_out = 0
        entries_in_customer = 0
        unique_persons_entered = 0
        if has_entries:
            entry_store_clause = "AND store_id = ?" if store_id else ""
            entries_in = conn.execute(
                f"SELECT COUNT(*) FROM entry_events WHERE direction='in' AND ts BETWEEN ? AND ? {entry_store_clause}",
                params,
            ).fetchone()[0]
            entries_out = conn.execute(
                f"SELECT COUNT(*) FROM entry_events WHERE direction='out' AND ts BETWEEN ? AND ? {entry_store_clause}",
                params,
            ).fetchone()[0]

            if has_emp:
                emp_filter = "AND ee.person_id NOT IN (SELECT person_id FROM employees)"
            else:
                emp_filter = ""
            entries_in_customer = conn.execute(
                f"SELECT COUNT(*) FROM entry_events ee WHERE direction='in' "
                f"AND ts BETWEEN ? AND ? {entry_store_clause.replace('store_id', 'ee.store_id')} {emp_filter}",
                params,
            ).fetchone()[0]
            unique_persons_entered = conn.execute(
                f"SELECT COUNT(DISTINCT ee.person_id) FROM entry_events ee "
                f"WHERE direction='in' AND ts BETWEEN ? AND ? "
                f"{entry_store_clause.replace('store_id', 'ee.store_id')} {emp_filter} "
                f"AND ee.person_id IS NOT NULL",
                params,
            ).fetchone()[0]

        return {
            "window_start": start_iso,
            "window_end": end_iso,
            "store_id": store_id or None,
            "system_entries_in_gross": entries_in,
            "system_entries_in_customer": entries_in_customer,
            "system_entries_out": entries_out,
            "system_unique_customer_persons": unique_persons_entered,
            "system_total_detections": det_total,
            "system_unique_tracks": det_tracks,
        }


@app.get("/api/entry_stats")
def entry_stats(scope: str = Query("today", pattern="^(today|all)$")):
    """Headline numbers driven by line-crossing entry events. This is the
    accurate count for conversion-rate purposes."""
    with db() as conn:
        has_entries = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entry_events'"
        ).fetchone()
        if not has_entries:
            return {"error": "No entry events yet. Run a camera with ENTRY_LINE set."}
        has_emp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone()

        where_clauses = ["direction = 'in'"]
        params: list = []
        if scope == "today":
            start_ts, _ = _today_bounds()
            where_clauses.append("ts >= ?")
            params.append(start_ts)
        where_sql = " AND ".join(where_clauses)

        emp_join = "LEFT JOIN employees e ON ee.person_id = e.person_id" if has_emp else ""
        emp_pred = "AND e.person_id IS NULL" if has_emp else ""

        # Door crossings IN — gross count of entrance line crosses going inward
        gross = conn.execute(
            f"SELECT COUNT(*) FROM entry_events ee {emp_join} WHERE {where_sql} {emp_pred}",
            params,
        ).fetchall()
        customer_in = gross[0][0]

        # Unique customers — distinct person_id with at least one IN crossing in scope
        unique = conn.execute(
            f"SELECT COUNT(DISTINCT ee.person_id) FROM entry_events ee {emp_join} "
            f"WHERE {where_sql} {emp_pred} AND ee.person_id IS NOT NULL",
            params,
        ).fetchall()
        unique_customers = unique[0][0]

        # Employee crossings (for reference)
        employee_in = 0
        if has_emp:
            r = conn.execute(
                f"SELECT COUNT(*) FROM entry_events ee INNER JOIN employees e "
                f"ON ee.person_id = e.person_id WHERE {where_sql}",
                params,
            ).fetchall()
            employee_in = r[0][0]

        # Total OUT crossings (everyone) for sanity check
        out_clauses = ["direction = 'out'"]
        out_params: list = []
        if scope == "today":
            start_ts, _ = _today_bounds()
            out_clauses.append("ts >= ?")
            out_params.append(start_ts)
        out_where = " AND ".join(out_clauses)
        out_row = conn.execute(
            f"SELECT COUNT(*) FROM entry_events WHERE {out_where}", out_params
        ).fetchone()

        return {
            "scope": scope,
            "customer_crossings_in": customer_in,
            "unique_customers": unique_customers,
            "employee_crossings_in": employee_in,
            "total_crossings_out": out_row[0],
        }


@app.get("/api/entries")
def entry_events_recent(limit: int = Query(100, ge=1, le=500)):
    with db() as conn:
        has_entries = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entry_events'"
        ).fetchone()
        if not has_entries:
            return []
        has_emp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone()
        emp_col = "CASE WHEN e.person_id IS NOT NULL THEN 1 ELSE 0 END" if has_emp else "0"
        emp_join = "LEFT JOIN employees e ON ee.person_id = e.person_id" if has_emp else ""
        rows = conn.execute(
            f"""
            SELECT ee.id, ee.ts, ee.store_id, ee.camera_id, ee.track_id,
                   ee.person_id, ee.direction, {emp_col} AS is_employee
            FROM entry_events ee
            {emp_join}
            ORDER BY ee.ts DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/timeseries")
def timeseries(bucket: str = Query("hour", pattern="^(minute|hour|day)$")):
    if not has_data():
        return []
    bucket_sec = {"minute": 60, "hour": 3600, "day": 86400}[bucket]
    with db() as conn:
        rows = conn.execute(
            f"""
            SELECT
                CAST(ts / {bucket_sec} AS INTEGER) * {bucket_sec} AS bucket_ts,
                COUNT(DISTINCT store_id || '|' || camera_id || '|' || track_id) AS unique_people,
                COUNT(*) AS total_detections
            FROM detections
            GROUP BY bucket_ts
            ORDER BY bucket_ts
            """
        ).fetchall()
        return [dict(r) for r in rows]


def _csv_response(rows: list[dict], filename: str) -> StreamingResponse:
    """Serve a list of dicts as a downloadable CSV."""
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    else:
        buf.write("(no data)\n")
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _iso(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).isoformat() if ts else ""


@app.get("/api/export/entries.csv")
def export_entries(store_id: str | None = None, since: str | None = None):
    """Every door crossing event. Use this to cross-reference with POS data.
    Filters: store_id (optional), since (ISO date 'YYYY-MM-DD', optional)."""
    with db() as conn:
        has_entries = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entry_events'"
        ).fetchone()
        if not has_entries:
            return _csv_response([], "entries.csv")
        has_emp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone()
        emp_col = "CASE WHEN e.person_id IS NOT NULL THEN 1 ELSE 0 END" if has_emp else "0"
        emp_join = "LEFT JOIN employees e ON ee.person_id = e.person_id" if has_emp else ""

        where, params = ["1=1"], []
        if store_id:
            where.append("ee.store_id = ?")
            params.append(store_id)
        if since:
            try:
                start_ts = datetime.fromisoformat(since).timestamp()
            except ValueError:
                raise HTTPException(400, f"Bad 'since' date: {since!r} (use YYYY-MM-DD)")
            where.append("ee.ts >= ?")
            params.append(start_ts)
        where_sql = " AND ".join(where)

        rows = conn.execute(
            f"""
            SELECT ee.ts AS ts_unix, ee.store_id, ee.camera_id, ee.track_id,
                   ee.person_id, ee.direction, {emp_col} AS is_employee
            FROM entry_events ee
            {emp_join}
            WHERE {where_sql}
            ORDER BY ee.ts
            """,
            params,
        ).fetchall()

        out = []
        for r in rows:
            d = dict(r)
            d = {"timestamp": _iso(d["ts_unix"]), **d}
            out.append(d)
        return _csv_response(out, "entries.csv")


@app.get("/api/export/visits.csv")
def export_visits(store_id: str | None = None, since: str | None = None):
    """Every visit (session-coalesced). Useful for dwell-time and visit-count
    analysis. Run scripts/compute_visits.py first if you haven't lately."""
    with db() as conn:
        has_visits = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='visits'"
        ).fetchone()
        if not has_visits:
            return _csv_response([], "visits.csv")
        has_emp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone()
        emp_col = "CASE WHEN e.person_id IS NOT NULL THEN 1 ELSE 0 END" if has_emp else "0"
        emp_join = "LEFT JOIN employees e ON v.person_id = e.person_id" if has_emp else ""

        where, params = ["1=1"], []
        if store_id:
            where.append("v.store_id = ?")
            params.append(store_id)
        if since:
            try:
                start_ts = datetime.fromisoformat(since).timestamp()
            except ValueError:
                raise HTTPException(400, f"Bad 'since' date: {since!r} (use YYYY-MM-DD)")
            where.append("v.started_ts >= ?")
            params.append(start_ts)
        where_sql = " AND ".join(where)

        rows = conn.execute(
            f"""
            SELECT v.visit_id, v.person_id, v.store_id,
                   v.started_ts, v.ended_ts, v.duration_sec,
                   v.n_detections, v.n_cameras, v.is_returning,
                   {emp_col} AS is_employee
            FROM visits v
            {emp_join}
            WHERE {where_sql}
            ORDER BY v.started_ts
            """,
            params,
        ).fetchall()

        out = []
        for r in rows:
            d = dict(r)
            d = {
                "visit_id": d["visit_id"],
                "person_id": d["person_id"],
                "store_id": d["store_id"],
                "started": _iso(d["started_ts"]),
                "ended": _iso(d["ended_ts"]),
                "started_ts_unix": d["started_ts"],
                "ended_ts_unix": d["ended_ts"],
                "duration_sec": d["duration_sec"],
                "n_detections": d["n_detections"],
                "n_cameras": d["n_cameras"],
                "is_returning": d["is_returning"],
                "is_employee": d["is_employee"],
            }
            out.append(d)
        return _csv_response(out, "visits.csv")


@app.get("/api/export/daily_summary.csv")
def export_daily_summary(store_id: str | None = None):
    """One row per (date, store_id) — the right granularity for conversion-rate
    cross-referencing with daily POS ticket totals. Customer columns exclude
    auto/manually-tagged employees."""
    with db() as conn:
        has_entries = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entry_events'"
        ).fetchone()
        if not has_entries:
            return _csv_response([], "daily_summary.csv")
        has_emp = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='employees'"
        ).fetchone()
        is_emp_expr = "CASE WHEN e.person_id IS NOT NULL THEN 1 ELSE 0 END" if has_emp else "0"
        emp_join = "LEFT JOIN employees e ON ee.person_id = e.person_id" if has_emp else ""

        where, params = ["ee.direction = 'in'"], []
        if store_id:
            where.append("ee.store_id = ?")
            params.append(store_id)
        where_sql = " AND ".join(where)

        # SQLite stores ts as unix seconds; group by local YYYY-MM-DD
        rows = conn.execute(
            f"""
            SELECT
                DATE(ee.ts, 'unixepoch', 'localtime') AS date,
                ee.store_id,
                SUM(CASE WHEN {is_emp_expr} = 0 THEN 1 ELSE 0 END) AS customer_crossings_in,
                COUNT(DISTINCT CASE WHEN {is_emp_expr} = 0 AND ee.person_id IS NOT NULL THEN ee.person_id END) AS unique_customers,
                SUM(CASE WHEN {is_emp_expr} = 1 THEN 1 ELSE 0 END) AS employee_crossings_in,
                COUNT(DISTINCT CASE WHEN {is_emp_expr} = 1 THEN ee.person_id END) AS unique_employees
            FROM entry_events ee
            {emp_join}
            WHERE {where_sql}
            GROUP BY date, ee.store_id
            ORDER BY date, ee.store_id
            """,
            params,
        ).fetchall()
        return _csv_response([dict(r) for r in rows], "daily_summary.csv")


@app.get("/")
def root():
    return FileResponse(DASHBOARD_DIR / "index.html")


# ============================================================================
# Remote admin endpoints (Phase 9)
# Lets the dashboard browser edit config, view live snapshots, restart the
# pipeline, and pull latest code — without AnyDesk-ing into the mini PC.
# ============================================================================

@app.get("/api/admin/identity")
def admin_identity():
    """Return basic info about this mini PC so the central dashboard can
    tell which store it's talking to."""
    import socket
    try:
        with open(CONFIG_PATH) as f:
            import yaml
            cfg = yaml.safe_load(f)
        stores = cfg.get("stores", [])
        store_id = stores[0]["id"] if stores else "unknown"
        store_name = stores[0].get("name", store_id) if stores else "unknown"
    except Exception:
        store_id = store_name = "unknown"
    return {
        "store_id": store_id,
        "store_name": store_name,
        "hostname": socket.gethostname(),
    }


@app.get("/api/admin/config")
def admin_get_config():
    """Read current stores.yaml as text + parsed dict."""
    if not CONFIG_PATH.exists():
        return {"raw": "", "parsed": None, "error": "stores.yaml not found"}
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        import yaml
        parsed = yaml.safe_load(raw)
    except Exception as e:
        parsed = None
        return {"raw": raw, "parsed": None, "error": str(e)}
    return {"raw": raw, "parsed": parsed, "error": None}


class ConfigUpdate(BaseModel):
    raw: str


@app.put("/api/admin/config")
def admin_put_config(payload: ConfigUpdate):
    """Save updated stores.yaml. Validates as YAML before writing."""
    import yaml
    try:
        parsed = yaml.safe_load(payload.raw)
    except Exception as e:
        raise HTTPException(400, f"Invalid YAML: {e}")
    if not isinstance(parsed, dict) or "stores" not in parsed:
        raise HTTPException(400, "Config must have a top-level 'stores' key")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(payload.raw, encoding="utf-8")
    return {"ok": True, "bytes_written": len(payload.raw)}


@app.get("/api/admin/snapshot/{camera_name}")
def admin_snapshot(camera_name: str, overlay: int = 1):
    """Pull a fresh frame from the named camera in stores.yaml. Optionally
    overlay the entry_line (yellow) and exclusion_line (red). Returns JPEG."""
    import yaml
    import cv2
    import numpy as np

    if not CONFIG_PATH.exists():
        raise HTTPException(404, "No stores.yaml")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    stores = cfg.get("stores", [])
    if not stores:
        raise HTTPException(404, "No stores in config")
    store = stores[0]
    camera = next((c for c in store.get("cameras", []) if c.get("name") == camera_name), None)
    if camera is None:
        raise HTTPException(404, f"Camera {camera_name!r} not found")

    nvr = store.get("nvr", {})
    user = nvr.get("username", "admin")
    pw = nvr.get("password", "")
    host = nvr.get("host")
    port = nvr.get("port", 554)
    channel = camera.get("channel")
    subtype = 1 if camera.get("stream", "sub") == "sub" else 0
    url = f"rtsp://{user}:{pw}@{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise HTTPException(502, "Could not read frame from camera")

    if overlay:
        def _draw_line(spec: str, color, label: str):
            try:
                parts = [int(x) for x in spec.split(",")]
                x1, y1, x2, y2, idx, idy = parts
            except Exception:
                return
            cv2.line(frame, (x1, y1), (x2, y2), color, 12)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            mag = (idx * idx + idy * idy) ** 0.5
            if mag > 0:
                ax = int(mx + 250 * idx / mag)
                ay = int(my + 250 * idy / mag)
                cv2.arrowedLine(frame, (mx, my), (ax, ay), color, 8, tipLength=0.3)
                cv2.putText(frame, label, (ax + 20, ay),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, color, 6)
        if camera.get("entry_line"):
            _draw_line(camera["entry_line"], (0, 200, 255), "INSIDE")
        if camera.get("exclusion_line"):
            _draw_line(camera["exclusion_line"], (50, 50, 220), "STAFF")

    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "JPEG encode failed")
    from fastapi.responses import Response
    return Response(content=jpg.tobytes(), media_type="image/jpeg")


@app.post("/api/admin/restart_pipeline")
def admin_restart_pipeline():
    """Stop+start the CustomerTracking_Pipeline scheduled task (Windows only).
    On Mac/Linux this is a no-op and returns a warning."""
    import sys
    import subprocess
    if sys.platform != "win32":
        return {"ok": False, "skipped": True, "reason": "not running on Windows"}
    out: list[str] = []
    for cmd in (
        ["schtasks", "/End", "/TN", "CustomerTracking_Pipeline"],
        ["schtasks", "/Run", "/TN", "CustomerTracking_Pipeline"],
    ):
        r = subprocess.run(cmd, capture_output=True, text=True)
        out.append(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return {"ok": True, "log": "\n".join(out)}


@app.get("/api/admin/health")
def admin_health():
    """Health check. Detection age is the primary signal — the NVR TCP
    test is unreliable because Dahua NVRs reject parallel connections when
    a live RTSP stream is already open."""
    out: dict = {
        "ok": True,
        "now_ts": time.time(),
        "last_detection_ts": None,
        "last_detection_age_sec": None,
        "last_entry_event_ts": None,
        "last_entry_event_age_sec": None,
        "n_detections_today": 0,
        "pipeline_task_status": None,
        "config_loaded": False,
        "store_id": None,
    }

    # Store id (so central can verify which store responded)
    if CONFIG_PATH.exists():
        try:
            import yaml
            cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            stores = cfg.get("stores", [])
            if stores:
                out["config_loaded"] = True
                out["store_id"] = stores[0].get("id")
        except Exception:
            pass

    # Last detection + entry timestamps
    if DB_PATH.exists():
        try:
            with db() as conn:
                row = conn.execute("SELECT MAX(ts) FROM detections").fetchone()
                if row and row[0]:
                    out["last_detection_ts"] = float(row[0])
                    out["last_detection_age_sec"] = time.time() - float(row[0])
                # Today's detection count
                import datetime
                start_today = datetime.datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).timestamp()
                row = conn.execute(
                    "SELECT COUNT(*) FROM detections WHERE ts >= ?",
                    (start_today,),
                ).fetchone()
                if row:
                    out["n_detections_today"] = int(row[0])
                # Last entry event
                row = conn.execute("SELECT MAX(ts) FROM entry_events").fetchone()
                if row and row[0]:
                    out["last_entry_event_ts"] = float(row[0])
                    out["last_entry_event_age_sec"] = time.time() - float(row[0])
        except Exception as e:
            out["db_error"] = str(e)[:120]

    # Pipeline scheduled task status (Windows)
    import sys
    if sys.platform == "win32":
        import subprocess
        try:
            r = subprocess.run(
                ["schtasks", "/Query", "/TN", "CustomerTracking_Pipeline", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = [p.strip().strip('"') for p in r.stdout.strip().split(",")]
                if len(parts) >= 3:
                    out["pipeline_task_status"] = parts[2]
        except Exception:
            pass
    return out


@app.post("/api/admin/git_pull")
def admin_git_pull():
    """Run `git pull` in the project root, then restart the pipeline.
    Surfaces all errors verbosely so debugging via the dashboard is possible."""
    import subprocess
    import sys
    import shutil

    # Find git — SYSTEM user's PATH may not have it even if it's installed.
    git_exe = shutil.which("git")
    if git_exe is None:
        for guess in (r"C:\Program Files\Git\bin\git.exe", r"C:\Program Files\Git\cmd\git.exe", "/usr/bin/git", "/opt/homebrew/bin/git"):
            if Path(guess).exists():
                git_exe = guess
                break
    if git_exe is None:
        return {"ok": False, "pull": "git executable not found in PATH or known locations", "restart": None}

    # Disable interactive credential prompts (otherwise git hangs forever when
    # running under SYSTEM user with no credential helper). For public repos
    # we never need auth.
    import os as _os
    env = _os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "echo"  # any failed prompt just gets "echo" and fails fast
    env["GCM_INTERACTIVE"] = "Never"

    try:
        r = subprocess.run(
            [git_exe, "-C", str(ROOT), "pull"],
            capture_output=True, text=True, timeout=30,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "pull": "git pull timed out after 30 seconds", "restart": None}
    except FileNotFoundError as e:
        return {"ok": False, "pull": f"git not found: {e}", "restart": None}
    except Exception as e:
        return {"ok": False, "pull": f"git pull crashed: {type(e).__name__}: {e}", "restart": None}

    pull_out = (r.stdout or "") + (r.stderr or "")
    if not pull_out.strip():
        pull_out = f"(git exited with code {r.returncode}, no output)"
    if r.returncode != 0:
        return {"ok": False, "pull": pull_out, "restart": None, "exit_code": r.returncode}

    restart_log = None
    if sys.platform == "win32":
        restart_lines = []
        for cmd in (
            ["schtasks", "/End", "/TN", "CustomerTracking_Pipeline"],
            ["schtasks", "/End", "/TN", "CustomerTracking_Dashboard"],
            ["schtasks", "/Run", "/TN", "CustomerTracking_Pipeline"],
            ["schtasks", "/Run", "/TN", "CustomerTracking_Dashboard"],
        ):
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                restart_lines.append(f"$ {' '.join(cmd)}\n{(res.stdout or '') + (res.stderr or '')}")
            except Exception as e:
                restart_lines.append(f"$ {' '.join(cmd)}\nERROR: {e}")
        restart_log = "\n".join(restart_lines)
    return {"ok": True, "pull": pull_out, "restart": restart_log}
