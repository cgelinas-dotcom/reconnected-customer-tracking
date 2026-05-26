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
