"""
Compute customer visits from raw detection events.

A "visit" = a continuous presence of one person at one store, where the gap
between consecutive detections never exceeds session_timeout_sec. If the
same person reappears after a longer gap, that's a new visit (and flagged
as "returning").

This is the Phase 4 logic from your 11-step plan. Re-run any time you want
to recompute (the visits table is wiped and rebuilt).

Usage:
    python scripts/compute_visits.py
    SESSION_TIMEOUT_SEC=1800 python scripts/compute_visits.py   # 30 min (default)
    SESSION_TIMEOUT_SEC=300  python scripts/compute_visits.py   # 5 min — strict
    SESSION_TIMEOUT_SEC=3600 python scripts/compute_visits.py   # 60 min — lax

If no env var is set, the value from the dashboard Settings panel is used,
falling back to the default in src/settings.py.
"""

import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / "data" / "events.sqlite"


def main() -> int:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}. Run scripts/run_pipeline.py first.")
        return 1

    conn = sqlite3.connect(DB_PATH)

    # Read session timeout: env var wins, otherwise DB setting, otherwise default.
    from src import settings as settings_mod
    env_val = os.environ.get("SESSION_TIMEOUT_SEC")
    session_timeout_sec = (
        float(env_val) if env_val
        else settings_mod.get(conn, "visits.session_timeout_sec")
    )
    print(f"Session timeout: {int(session_timeout_sec)}s ({session_timeout_sec/60:.1f} min)")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS visits (
            visit_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id           INTEGER NOT NULL,
            store_id            TEXT    NOT NULL,
            started_ts          REAL    NOT NULL,
            ended_ts            REAL    NOT NULL,
            duration_sec        REAL    NOT NULL,
            n_detections        INTEGER NOT NULL,
            n_cameras           INTEGER NOT NULL,
            is_returning        INTEGER NOT NULL DEFAULT 0,
            session_timeout_sec REAL    NOT NULL,
            computed_at         REAL    NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );
        CREATE INDEX IF NOT EXISTS idx_visits_store_started
            ON visits (store_id, started_ts);
        CREATE INDEX IF NOT EXISTS idx_visits_person
            ON visits (person_id);
        """
    )

    has_tp = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='track_persons'"
    ).fetchone()
    if not has_tp:
        print("No track_persons table — run the pipeline with REID=1 first.")
        return 2

    conn.execute("DELETE FROM visits")

    rows = conn.execute(
        """
        SELECT tp.person_id, d.store_id, d.camera_id, d.ts
        FROM detections d
        JOIN track_persons tp
          ON d.store_id  = tp.store_id
         AND d.camera_id = tp.camera_id
         AND d.track_id  = tp.track_id
        ORDER BY tp.person_id, d.store_id, d.ts
        """
    ).fetchall()

    if not rows:
        print("No detections joined to persons. Did the pipeline run with REID=1?")
        return 3

    visits: list[dict] = []
    cur = {"person_id": None, "store_id": None,
           "started_ts": None, "ended_ts": None,
           "n_detections": 0, "cameras": set()}

    def flush() -> None:
        if cur["person_id"] is None:
            return
        visits.append({
            "person_id":    cur["person_id"],
            "store_id":     cur["store_id"],
            "started_ts":   cur["started_ts"],
            "ended_ts":     cur["ended_ts"],
            "duration_sec": cur["ended_ts"] - cur["started_ts"],
            "n_detections": cur["n_detections"],
            "n_cameras":    len(cur["cameras"]),
        })

    for person_id, store_id, camera_id, ts in rows:
        new_group = (person_id != cur["person_id"] or store_id != cur["store_id"])
        gap = (ts - cur["ended_ts"]) if cur["ended_ts"] is not None else 0.0
        if new_group or gap > session_timeout_sec:
            flush()
            cur = {"person_id": person_id, "store_id": store_id,
                   "started_ts": ts, "ended_ts": ts,
                   "n_detections": 1, "cameras": {camera_id}}
        else:
            cur["ended_ts"] = ts
            cur["n_detections"] += 1
            cur["cameras"].add(camera_id)
    flush()

    # Flag returning visits: any visit that isn't the earliest one for its person.
    seen: set[int] = set()
    for v in sorted(visits, key=lambda x: x["started_ts"]):
        if v["person_id"] in seen:
            v["is_returning"] = 1
        else:
            v["is_returning"] = 0
            seen.add(v["person_id"])

    now = time.time()
    conn.executemany(
        """INSERT INTO visits
           (person_id, store_id, started_ts, ended_ts, duration_sec,
            n_detections, n_cameras, is_returning, session_timeout_sec, computed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (v["person_id"], v["store_id"], v["started_ts"], v["ended_ts"],
             v["duration_sec"], v["n_detections"], v["n_cameras"],
             v["is_returning"], session_timeout_sec, now)
            for v in visits
        ],
    )
    conn.commit()

    total = len(visits)
    returning = sum(v["is_returning"] for v in visits)
    avg_dur = sum(v["duration_sec"] for v in visits) / total if total else 0
    print(f"Built {total} visits (session timeout = {int(session_timeout_sec)}s)")
    print(f"  First-time visits:  {total - returning}")
    print(f"  Returning visits:   {returning}")
    print(f"  Avg duration:       {avg_dur:.1f}s")
    print(f"  Unique visitors:    {len(seen)}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
