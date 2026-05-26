"""
Line-crossing detector for entry counting.

Given a configured "entry line" (two pixel points on the camera image and
which side is "inside the store"), tracks each person's centroid frame to
frame and detects when they cross the line. Crossings are tagged 'in' or
'out' based on the configured inside-direction vector.

The pipeline calls update_track() each frame; it returns a list of
crossing events that happened on that frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EntryLine:
    """Define an entry line: from p1 to p2, with inside_dir pointing INTO the store."""
    p1: tuple[float, float]
    p2: tuple[float, float]
    inside_dir: tuple[float, float]  # (dx, dy), need not be unit length

    @classmethod
    def from_spec(cls, spec: str) -> "EntryLine":
        """Parse "x1,y1,x2,y2,inside_dx,inside_dy" comma-separated string."""
        parts = [float(x.strip()) for x in spec.split(",")]
        if len(parts) != 6:
            raise ValueError(
                f"ENTRY_LINE must be 6 comma-separated numbers (x1,y1,x2,y2,inside_dx,inside_dy); got {spec!r}"
            )
        return cls(p1=(parts[0], parts[1]), p2=(parts[2], parts[3]),
                   inside_dir=(parts[4], parts[5]))

    def signed_distance(self, x: float, y: float) -> float:
        """Signed distance from point (x,y) to the line, positive on the 'inside' side.

        Uses the line normal aligned with inside_dir so we don't have to think about
        which side is left/right of the directed line.
        """
        # Line vector
        lx = self.p2[0] - self.p1[0]
        ly = self.p2[1] - self.p1[1]
        # Perpendicular (rotate 90° ccw)
        nx, ny = -ly, lx
        # Flip if not pointing toward inside_dir
        if nx * self.inside_dir[0] + ny * self.inside_dir[1] < 0:
            nx, ny = -nx, -ny
        # Vector from p1 to point
        vx = x - self.p1[0]
        vy = y - self.p1[1]
        # Signed projection onto unit normal
        length = (nx * nx + ny * ny) ** 0.5
        if length == 0:
            return 0.0
        return (vx * nx + vy * ny) / length


@dataclass
class CrossingEvent:
    track_id: int
    direction: Literal["in", "out"]


@dataclass
class LineCrossingDetector:
    line: EntryLine
    # Per-track last signed distance (so we can tell when sign flips).
    _last_dist: dict[int, float] = field(default_factory=dict)

    def update_track(self, track_id: int, cx: float, cy: float) -> CrossingEvent | None:
        """Call once per frame per tracked person. Returns a crossing event if the
        centroid crossed the line this frame, otherwise None."""
        d = self.line.signed_distance(cx, cy)
        prev = self._last_dist.get(track_id)
        self._last_dist[track_id] = d

        if prev is None:
            return None  # first observation, can't say if crossed
        if prev * d >= 0:
            return None  # same side
        # Sign flipped — crossed
        direction: Literal["in", "out"] = "in" if d > 0 else "out"
        return CrossingEvent(track_id=track_id, direction=direction)

    def forget(self, track_id: int) -> None:
        self._last_dist.pop(track_id, None)
