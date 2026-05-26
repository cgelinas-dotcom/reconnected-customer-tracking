"""
Online person registry — matches new tracks to known persons via cosine sim.

Each time a new (store_id, camera_id, track_id) shows up, we embed the crop
and ask the registry: "have we seen someone like this recently?" If yes,
reuse that person_id (same person crossing cameras or returning shortly).
If no, mint a new person_id.

Thresholds and recency window are tunable via the dashboard settings panel
(see src/settings.py). The constants here are fallback defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SIMILARITY_THRESHOLD = 0.88  # tuned for OSNet; ResNet18 wants ~0.78
DEFAULT_RECENCY_WINDOW_SEC = 43200  # 12 hours


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

    def update(self, emb: np.ndarray, ts: float) -> None:
        avg = (self.embedding * self.n_samples + emb) / (self.n_samples + 1)
        norm = float(np.linalg.norm(avg))
        self.embedding = avg / norm if norm > 0 else avg
        self.n_samples += 1
        self.last_seen_ts = max(self.last_seen_ts, ts)


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
        # The model currently producing embeddings; matching only considers
        # persons with the same model (different models = different vector spaces).
        self.active_model = active_model

    def assign(self, emb: np.ndarray, ts: float,
               store_id: str, camera_id: str) -> tuple[int, float | None]:
        """Returns (person_id, matched_similarity). matched_similarity is None
        if a new person was minted, otherwise the cosine sim of the match."""
        best_id, best_sim = None, self.similarity_threshold
        for pid, p in self.persons.items():
            if p.embedding_model != self.active_model:
                continue  # skip persons embedded by a different model
            if ts - p.last_seen_ts > self.recency_window_sec:
                continue
            if p.embedding.shape != emb.shape:
                continue  # safety: dim mismatch
            sim = float(np.dot(emb, p.embedding))
            if sim > best_sim:
                best_sim = sim
                best_id = pid

        if best_id is not None:
            self.persons[best_id].update(emb, ts)
            return best_id, best_sim

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
        )
        return pid, None
