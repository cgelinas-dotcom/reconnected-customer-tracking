"""
Online person registry — matches new tracks to known persons via cosine sim.

Each time a new (store_id, camera_id, track_id) shows up, we embed the crop
and ask the registry: "have we seen someone like this recently?" If yes,
reuse that person_id (same person crossing cameras or returning shortly).
If no, mint a new person_id.

Thresholds and recency window are tunable via the dashboard settings panel
(see src/settings.py). The constants here are fallback defaults.

Phase 2.1 additions:
- evaluate() returns a non-committal MatchResult so the caller can
  implement active wait (accumulate more samples before deciding)
- Color signature (HSV torso histogram) is used as a same-day tiebreaker
  in the borderline confidence zone
- commit_match / commit_new are the two ways the caller actually commits
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SIMILARITY_THRESHOLD = 0.88  # tuned for OSNet; ResNet18 wants ~0.78
DEFAULT_RECENCY_WINDOW_SEC = 2592000  # 30 days — enables cross-day returning-customer matching

# Confidence zones for the active-wait pattern. "borderline" cases let
# the caller defer the decision and gather more samples first.
HIGH_CONF_DELTA_ABOVE_THRESHOLD = 0.05  # threshold + this = confident match
LOW_CONF_DELTA_BELOW_THRESHOLD = 0.15   # threshold - this = confident new

# Color tiebreaker only kicks in when OSNet is uncertain
COLOR_SIM_BOOST_ABOVE = 0.70  # strong color match → boost OSNet sim
COLOR_SIM_REJECT_BELOW = 0.30  # strong color mismatch → penalize OSNet sim
COLOR_SIM_BOOST_AMOUNT = 0.06
COLOR_SIM_PENALTY_AMOUNT = 0.10


@dataclass
class Person:
    person_id: int
    embedding: np.ndarray
    n_samples: int
    first_seen_ts: float
    last_seen_ts: float
    first_store: str
    first_camera: str
    embedding_model: str = "unknown"
    # In-memory only (not persisted to disk yet) — same-day tiebreaker
    color_signature: list = field(default_factory=list)

    def update(self, emb: np.ndarray, ts: float, color_sig: list | None = None) -> None:
        avg = (self.embedding * self.n_samples + emb) / (self.n_samples + 1)
        norm = float(np.linalg.norm(avg))
        self.embedding = avg / norm if norm > 0 else avg
        self.n_samples += 1
        self.last_seen_ts = max(self.last_seen_ts, ts)
        if color_sig and len(color_sig) > 0:
            if self.color_signature and len(self.color_signature) == len(color_sig):
                self.color_signature = [
                    (a * (self.n_samples - 1) + b) / self.n_samples
                    for a, b in zip(self.color_signature, color_sig)
                ]
            else:
                self.color_signature = list(color_sig)


@dataclass
class MatchResult:
    """Non-committal evaluation of a candidate embedding against the registry.
    Caller chooses whether to commit_match, commit_new, or wait for more
    samples (active-wait pattern). best_pid is None when no candidate cleared
    the (adjusted) threshold."""
    best_pid: int | None
    best_sim: float
    color_sim: float | None
    adjusted_sim: float
    confidence: str  # "match" | "new" | "borderline"


def _cosine(a, b) -> float:
    """Cosine similarity between two equal-length lists/arrays of floats.
    Both are assumed L2-normalized for embeddings, or sum-normalized for
    histograms — either way this returns 0..1 for nonneg inputs."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        return 0.0
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a_arr))
    nb = float(np.linalg.norm(b_arr))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (na * nb))


class PersonRegistry:
    def __init__(self, start_id: int = 1,
                 similarity_threshold: float | None = None,
                 recency_window_sec: float | None = None,
                 active_model: str = "unknown"):
        self.persons: dict[int, Person] = {}
        self._next_id = start_id
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else DEFAULT_SIMILARITY_THRESHOLD
        )
        self.recency_window_sec = (
            recency_window_sec if recency_window_sec is not None
            else DEFAULT_RECENCY_WINDOW_SEC
        )
        self.active_model = active_model

    def evaluate(self, emb: np.ndarray, ts: float,
                 color_sig: list | None = None) -> MatchResult:
        """Find the best candidate match for this embedding WITHOUT committing.
        Caller decides whether to commit or wait for more samples."""
        best_id, best_sim = None, -1.0
        for pid, p in self.persons.items():
            if p.embedding_model != self.active_model:
                continue
            if ts - p.last_seen_ts > self.recency_window_sec:
                continue
            if p.embedding.shape != emb.shape:
                continue
            sim = float(np.dot(emb, p.embedding))
            if sim > best_sim:
                best_sim = sim
                best_id = pid

        # Color tiebreaker — only meaningfully adjusts borderline cases
        color_sim_val = None
        adjusted = best_sim
        if best_id is not None and color_sig:
            cand_color = self.persons[best_id].color_signature
            if cand_color:
                color_sim_val = _cosine(color_sig, cand_color)
                in_borderline = (
                    best_sim >= self.similarity_threshold - LOW_CONF_DELTA_BELOW_THRESHOLD
                    and best_sim < self.similarity_threshold + HIGH_CONF_DELTA_ABOVE_THRESHOLD
                )
                if in_borderline:
                    if color_sim_val >= COLOR_SIM_BOOST_ABOVE:
                        adjusted = best_sim + COLOR_SIM_BOOST_AMOUNT
                    elif color_sim_val <= COLOR_SIM_REJECT_BELOW:
                        adjusted = best_sim - COLOR_SIM_PENALTY_AMOUNT

        # Confidence label
        high_conf = self.similarity_threshold + HIGH_CONF_DELTA_ABOVE_THRESHOLD
        low_conf = self.similarity_threshold - LOW_CONF_DELTA_BELOW_THRESHOLD
        if adjusted >= high_conf and best_id is not None:
            confidence = "match"
            chosen_pid = best_id
        elif adjusted < low_conf or best_id is None:
            confidence = "new"
            chosen_pid = None
        elif adjusted >= self.similarity_threshold and best_id is not None:
            # Above threshold but not high confidence — borderline-leaning-match
            confidence = "borderline"
            chosen_pid = best_id
        else:
            # Below threshold but not clearly new — borderline-leaning-new
            confidence = "borderline"
            chosen_pid = None

        return MatchResult(
            best_pid=chosen_pid,
            best_sim=best_sim,
            color_sim=color_sim_val,
            adjusted_sim=adjusted,
            confidence=confidence,
        )

    def commit_match(self, pid: int, emb: np.ndarray, ts: float,
                     color_sig: list | None = None) -> None:
        """Commit a confident match by updating the existing person's fingerprint."""
        if pid in self.persons:
            self.persons[pid].update(emb, ts, color_sig)

    def commit_new(self, emb: np.ndarray, ts: float, store_id: str, camera_id: str,
                   color_sig: list | None = None) -> int:
        """Create a new person and return its id."""
        pid = self._next_id
        self._next_id += 1
        self.persons[pid] = Person(
            person_id=pid,
            embedding=emb.copy(),
            n_samples=1,
            first_seen_ts=ts,
            last_seen_ts=ts,
            first_store=store_id,
            first_camera=camera_id,
            embedding_model=self.active_model,
            color_signature=list(color_sig) if color_sig else [],
        )
        return pid

    def assign(self, emb: np.ndarray, ts: float,
               store_id: str, camera_id: str,
               color_sig: list | None = None) -> tuple[int, float | None]:
        """Backward-compatible wrapper that evaluates and immediately commits.
        Returns (person_id, matched_similarity). matched_similarity is None
        if a new person was minted."""
        result = self.evaluate(emb, ts, color_sig)
        if result.best_pid is not None:
            self.commit_match(result.best_pid, emb, ts, color_sig)
            return result.best_pid, result.best_sim
        pid = self.commit_new(emb, ts, store_id, camera_id, color_sig)
        return pid, None
