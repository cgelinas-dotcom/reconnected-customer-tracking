"""
Tunable settings stored in the events DB and editable via the dashboard.

Values can be changed without code edits — the pipeline reads them at
startup, compute_visits.py reads them at runtime.

Add a new setting by appending to DEFAULTS below; the dashboard picks it up
automatically based on the metadata.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

# Each setting: (default, min, max, description)
@dataclass
class SettingSpec:
    default: float
    min: float
    max: float
    description: str
    unit: str = ""


DEFAULTS: dict[str, SettingSpec] = {
    "reid.similarity_threshold": SettingSpec(
        default=0.88, min=0.5, max=0.99,
        description="How similar two person embeddings must be to be considered the same person. "
                    "Higher = stricter (fewer false merges, more false splits). Lower = looser. "
                    "Good starting points: 0.88 for OSNet, 0.78 for ResNet18. Tune at a real "
                    "store against gut-feel counts.",
        unit="",
    ),
    "reid.auto_scan_interval_sec": SettingSpec(
        default=30, min=5, max=3600,
        description="How often the pipeline scans the entire gallery for pairs of P "
                    "numbers similar enough to be the same person, and auto-merges them. "
                    "Default 30s keeps the gallery clean continuously without much CPU "
                    "cost (scan is O(n²) on a small gallery). Drop to 10s if duplicates "
                    "are bothering you; raise to 300s to reduce CPU.",
        unit="seconds",
    ),
    "reid.auto_merge_threshold": SettingSpec(
        default=0.80, min=0.70, max=0.99,
        description="After a person's fingerprint is updated, if it's now this similar "
                    "to ANOTHER existing person in the same store, the two get auto-merged. "
                    "0.80 catches same-person drift across the day (different angles, "
                    "lighting); distinct customers usually score 0.3-0.6 pairwise so there's "
                    "a lot of margin. Crank up to 0.85+ if you see false merges of two real "
                    "customers; crank down to 0.75 if obvious duplicates still aren't merging.",
        unit="",
    ),
    "reid.recency_window_sec": SettingSpec(
        default=2592000, min=300, max=2592000,  # 5 min .. 30 days, default 30 days
        description="How far back to look when matching a new track to existing persons. "
                    "Default 30 days enables returning-customer detection across the month. "
                    "Drop to 86400 (1 day) if you only want same-day matching.",
        unit="seconds",
    ),
    "visits.session_timeout_sec": SettingSpec(
        default=1800, min=60, max=86400,  # 1 min .. 1 day
        description="Maximum gap between detections of the same person at the same store before "
                    "starting a new visit. 1800 = 30 min (typical retail). 86400 = treat all of "
                    "today as one visit.",
        unit="seconds",
    ),
    "detection.min_confidence": SettingSpec(
        default=0.5, min=0.1, max=0.95,
        description="YOLO detection confidence threshold. Higher = fewer false positives "
                    "(missed people in poor lighting), lower = more detections (more false alarms).",
        unit="",
    ),
}


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key         TEXT PRIMARY KEY,
            value       REAL NOT NULL,
            updated_at  REAL NOT NULL
        )
        """
    )


def get(conn: sqlite3.Connection, key: str) -> float:
    """Return the current value of a setting, or its default if not set."""
    ensure_table(conn)
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    if row is not None:
        return float(row[0])
    if key in DEFAULTS:
        return DEFAULTS[key].default
    raise KeyError(f"Unknown setting: {key}")


def set_value(conn: sqlite3.Connection, key: str, value: float) -> None:
    if key not in DEFAULTS:
        raise KeyError(f"Unknown setting: {key}")
    spec = DEFAULTS[key]
    if value < spec.min or value > spec.max:
        raise ValueError(f"{key} must be between {spec.min} and {spec.max} (got {value})")
    ensure_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
        (key, float(value), time.time()),
    )
    conn.commit()


def get_all(conn: sqlite3.Connection) -> dict[str, dict]:
    """Return all settings with their current value + metadata for the dashboard."""
    ensure_table(conn)
    overrides = {
        row[0]: (row[1], row[2])
        for row in conn.execute("SELECT key, value, updated_at FROM settings").fetchall()
    }
    out: dict[str, dict] = {}
    for key, spec in DEFAULTS.items():
        value, updated_at = overrides.get(key, (spec.default, None))
        out[key] = {
            "key": key,
            "value": value,
            "default": spec.default,
            "min": spec.min,
            "max": spec.max,
            "description": spec.description,
            "unit": spec.unit,
            "is_default": key not in overrides,
            "updated_at": updated_at,
        }
    return out
