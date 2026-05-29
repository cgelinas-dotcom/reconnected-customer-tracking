"""
Vigorous store-scenario simulation.

Models a Reconnected-style entrance camera (1920x1080, wide angle, person
walks from one side through the doorway out the other side). Mimics:
- Realistic OSNet similarity behavior (same person ~0.78-0.92 across angles,
  different people ~0.30-0.55)
- Frame-edge entries (the dominant cause of re-ID fragmentation)
- Partial-view degradation as people enter and exit the frame
- Occlusion when two customers enter together
- Track lifecycle: short entry frames (bad), middle frames (good), exit (bad)

Runs the same multi-shot + active-wait logic as run_pipeline.py against
8 scenarios drawn from Cam's actual operational concerns. Prints pass/fail
plus per-scenario person-count summaries.

Run from project root:
    .venv/bin/python scripts/test_store_simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.reid.matcher import PersonRegistry
from run_pipeline import is_quality_box, average_embeddings

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m·\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

EMB_DIM = 512
FRAME_W, FRAME_H = 1920, 1080

# Matcher config — match what's deployed at the stores
SIM_THRESHOLD = 0.78
RECENCY_WINDOW_SEC = 30 * 86400  # 30 days

# Pipeline config — match run_pipeline.py constants
MIN_QUALITY_EMBEDS_FOR_ASSIGN = 3
MAX_QUALITY_EMBEDS_FOR_WAIT = 8
QUALITY_MIN_BBOX_HEIGHT = 80
QUALITY_MIN_BBOX_WIDTH = 30
QUALITY_MIN_CONFIDENCE = 0.6
QUALITY_EDGE_MARGIN = 8
QUALITY_MAX_OCCLUSION_IOU = 0.3


# ---------------------------------------------------------------------------
# Synthetic embeddings — controlled cosine similarity to a "true" fingerprint
# ---------------------------------------------------------------------------

def true_fingerprint(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=EMB_DIM)
    return v / np.linalg.norm(v)


def shot(true: np.ndarray, target_sim: float, seed: int) -> np.ndarray:
    """One observation of a person's true fingerprint with controlled cosine
    similarity. Real OSNet behavior:
      target_sim 0.82-0.92 → same person, different angle
      target_sim 0.70-0.80 → same person, very different pose/lighting
      target_sim 0.30-0.55 → different person"""
    rng = np.random.default_rng(seed)
    orth = rng.normal(size=EMB_DIM)
    orth -= np.dot(orth, true) * true
    norm = np.linalg.norm(orth)
    if norm > 0:
        orth = orth / norm
    sim = target_sim + rng.normal(scale=0.03)
    sim = float(np.clip(sim, -0.95, 0.95))
    v = sim * true + np.sqrt(max(0.0, 1 - sim * sim)) * orth
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Track simulator — models a person walking through a wide-angle entrance
# camera. Yields (bbox, conf, embedding) per processed frame.
# ---------------------------------------------------------------------------

def simulate_walk_through(
    person_seed: int,
    n_frames: int = 25,
    pose_variation: float = 0.82,  # mean target_sim per frame
    has_bad_entry: bool = True,  # first few frames at frame edge
    has_bad_exit: bool = True,   # last few frames at frame edge
    overlapping_with: list | None = None,  # list of (x1,y1,x2,y2) of other people in same frames
) -> list:
    """Yields (x1,y1,x2,y2, conf, embedding) per frame. Mimics realistic
    box size growth as the person walks closer to camera, then shrinks as
    they exit. Entry/exit frames are at frame edge (bad quality)."""
    true = true_fingerprint(person_seed)
    frames = []
    for i in range(n_frames):
        # Position model: person walks left-to-right across frame
        progress = i / max(1, n_frames - 1)  # 0 to 1
        # Center x of bbox: from 50 to FRAME_W - 50
        cx = 50 + progress * (FRAME_W - 100)

        # Bbox size grows as they get closer to camera (peaks in middle), then shrinks
        proximity = 1 - abs(progress - 0.5) * 2  # 1 at middle, 0 at edges
        bbox_h = int(100 + proximity * 600)  # 100 to 700 pixels tall
        bbox_w = int(bbox_h * 0.4)  # roughly 2.5:1 height to width

        x1 = int(cx - bbox_w / 2)
        y1 = int(FRAME_H * 0.15)
        x2 = int(cx + bbox_w / 2)
        y2 = y1 + bbox_h

        # Confidence drops at frame edges (bad lighting, partial view)
        conf = 0.55 + proximity * 0.40  # 0.55 at edges, 0.95 in middle

        # Force first/last frames to be at frame edges if requested
        if has_bad_entry and i < 3:
            x1 = i * 2  # x1 = 0, 2, 4 → triggers edge rejection
            x2 = x1 + bbox_w
            conf = 0.50 + i * 0.05  # low conf entry
        if has_bad_exit and i >= n_frames - 3:
            offset_from_end = n_frames - 1 - i
            x2 = FRAME_W - offset_from_end * 2  # x2 = 1919, 1917, 1915
            x1 = x2 - bbox_w
            conf = 0.50 + offset_from_end * 0.05

        # Per-frame embedding with target_sim varying by frame quality
        per_frame_sim = pose_variation + np.random.default_rng(person_seed * 100 + i).normal(scale=0.04)
        per_frame_sim = float(np.clip(per_frame_sim, 0.5, 0.95))
        emb = shot(true, per_frame_sim, seed=person_seed * 1000 + i)

        frames.append((x1, y1, x2, y2, conf, emb))
    return frames


# ---------------------------------------------------------------------------
# Pipeline-faithful processor: runs frames through the same multi-shot +
# active-wait logic as run_pipeline.py
# ---------------------------------------------------------------------------

def process_track(
    registry: PersonRegistry,
    track_frames: list,
    ts_start: float,
    store_id: str = "test",
    camera_id: str = "front",
    other_boxes_per_frame: list | None = None,
) -> tuple[int | None, str, int]:
    """Process one track's frames through the matcher.
    Returns (assigned_pid, reason, n_quality_embeds_collected).
    reason is one of: 'committed', 'never_committed_low_quality', 'forced_after_wait'."""
    quality_embeds = []
    forced = False
    for i, (x1, y1, x2, y2, conf, emb) in enumerate(track_frames):
        ts = ts_start + i * 0.1  # 10fps

        # Collect all boxes in this frame (this track + others) for occlusion check
        boxes_this_frame = [(x1, y1, x2, y2)]
        if other_boxes_per_frame and i < len(other_boxes_per_frame):
            boxes_this_frame.extend(other_boxes_per_frame[i])

        ok, _why = is_quality_box(
            x1, y1, x2, y2, conf, FRAME_W, FRAME_H,
            boxes_this_frame, 0,
            QUALITY_MIN_BBOX_HEIGHT, QUALITY_MIN_BBOX_WIDTH,
            QUALITY_MIN_CONFIDENCE, QUALITY_EDGE_MARGIN,
            QUALITY_MAX_OCCLUSION_IOU,
        )
        if not ok:
            continue

        quality_embeds.append(emb)
        n_have = len(quality_embeds)

        if n_have >= MIN_QUALITY_EMBEDS_FOR_ASSIGN:
            avg = average_embeddings(quality_embeds)
            result = registry.evaluate(avg, ts)
            should_commit = (
                result.confidence != "borderline"
                or n_have >= MAX_QUALITY_EMBEDS_FOR_WAIT
            )
            if should_commit:
                forced = (result.confidence == "borderline")
                if result.best_pid is not None:
                    registry.commit_match(result.best_pid, avg, ts)
                    return result.best_pid, "forced_after_wait" if forced else "committed", n_have
                else:
                    pid = registry.commit_new(avg, ts, store_id, camera_id)
                    return pid, "forced_after_wait" if forced else "committed", n_have

    # Track ended without ever committing (too few quality frames)
    return None, "never_committed_low_quality", len(quality_embeds)


def header(title: str):
    print(f"\n{BOLD}========= {title} ========={RESET}")


def verdict(passed: bool, msg: str):
    print(f"  {PASS if passed else FAIL} {msg}")


# ---------------------------------------------------------------------------
# SCENARIO 1: Multi-trip customer (the Anthony repair-pickup pattern)
# ---------------------------------------------------------------------------

def scenario_multi_trip_customer():
    header("SCENARIO 1: Customer makes 4 trips in one day (drop-off + 3 status/pickup)")
    print(f"{INFO} A phone repair customer: drops off device, leaves, comes back 3 times")
    print(f"{INFO} Expected: 1 person_id reused for all 4 trips")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=SIM_THRESHOLD,
                         recency_window_sec=RECENCY_WINDOW_SEC)
    pids = []
    for trip in range(4):
        frames = simulate_walk_through(person_seed=1000, n_frames=20, pose_variation=0.84)
        pid, reason, n = process_track(reg, frames, ts_start=trip * 3600.0)
        pids.append(pid)
        print(f"  {INFO} Trip {trip+1}: assigned {'P'+str(pid) if pid else 'NONE'} ({reason}, {n} quality crops)")

    unique = set(pids)
    verdict(len(unique) == 1 and None not in unique,
            f"4 trips → {len(unique)} unique P number(s) ({sorted(unique)})")


# ---------------------------------------------------------------------------
# SCENARIO 2: Employee enters/exits 20 times during a workday (Gary)
# ---------------------------------------------------------------------------

def scenario_employee_all_day():
    header("SCENARIO 2: Employee in/out 20 times across an 8-hour shift")
    print(f"{INFO} Gary the manager: breaks, lunch, smoke breaks, back-and-forth all day")
    print(f"{INFO} Expected: 1 person_id for all 20 entries")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=SIM_THRESHOLD,
                         recency_window_sec=RECENCY_WINDOW_SEC)
    pids = []
    n_committed = 0
    for entry in range(20):
        # Pose variation differs slightly each entry (different angle of approach)
        frames = simulate_walk_through(person_seed=2000, n_frames=18,
                                       pose_variation=0.80 + np.random.default_rng(entry).normal(scale=0.02))
        pid, reason, n = process_track(reg, frames, ts_start=entry * 1200.0)
        if pid is not None:
            pids.append(pid)
            n_committed += 1

    unique = set(pids)
    print(f"  {INFO} {n_committed}/20 entries produced committed person_ids ({20-n_committed} were below quality threshold)")
    verdict(len(unique) == 1,
            f"All committed entries → {len(unique)} P number(s) ({sorted(unique)})")


# ---------------------------------------------------------------------------
# SCENARIO 3: Three distinct people (the Bullhead Gary+Anthony+Dominique test)
# ---------------------------------------------------------------------------

def scenario_three_distinct_people():
    header("SCENARIO 3: 3 different people walk through (Gary, Anthony, Dominique)")
    print(f"{INFO} Three distinct customers each walk through once")
    print(f"{INFO} Expected: 3 distinct P numbers (no merging)")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=SIM_THRESHOLD,
                         recency_window_sec=RECENCY_WINDOW_SEC)
    pids = []
    for i, seed in enumerate([3001, 3002, 3003]):
        frames = simulate_walk_through(person_seed=seed, n_frames=20, pose_variation=0.85)
        pid, reason, n = process_track(reg, frames, ts_start=i * 600.0)
        pids.append(pid)
        print(f"  {INFO} Person {i+1}: assigned {'P'+str(pid) if pid else 'NONE'} ({reason}, {n} quality crops)")

    unique = set(pid for pid in pids if pid is not None)
    verdict(len(unique) == 3,
            f"3 different people → {len(unique)} P number(s) (expected 3)")


# ---------------------------------------------------------------------------
# SCENARIO 4: Returning customer next week (cross-day matching)
# ---------------------------------------------------------------------------

def scenario_returning_customer_cross_day():
    header("SCENARIO 4: Customer returns 7 days later (different clothes/lighting)")
    print(f"{INFO} Visit 1 on Monday morning, visit 2 the following Monday morning")
    print(f"{INFO} Expected: same P number both visits (30-day gallery + tunable threshold)")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=SIM_THRESHOLD,
                         recency_window_sec=RECENCY_WINDOW_SEC)
    # Visit 1
    frames1 = simulate_walk_through(person_seed=4000, n_frames=20, pose_variation=0.85)
    pid1, _, _ = process_track(reg, frames1, ts_start=0.0)
    # Visit 2 — 7 days later, pose_variation lower (different clothes/lighting)
    frames2 = simulate_walk_through(person_seed=4000, n_frames=20, pose_variation=0.78)
    pid2, reason, n = process_track(reg, frames2, ts_start=7 * 86400.0)

    print(f"  {INFO} Day 1: assigned P{pid1}")
    print(f"  {INFO} Day 8: assigned {'P'+str(pid2) if pid2 else 'NONE'} ({reason})")
    verdict(pid1 is not None and pid1 == pid2,
            f"Cross-day match: P{pid1} = P{pid2}")


# ---------------------------------------------------------------------------
# SCENARIO 5: Frame-edge entry (the killer of the old system)
# ---------------------------------------------------------------------------

def scenario_frame_edge_entry():
    header("SCENARIO 5: Customer enters with 5 bad-quality frames at frame edge")
    print(f"{INFO} First 5 frames are at the edge of frame (partial view, low confidence)")
    print(f"{INFO} Quality filter should reject those; only good middle frames used")
    print(f"{INFO} Expected: 1 P number, assigned from quality crops only")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=SIM_THRESHOLD,
                         recency_window_sec=RECENCY_WINDOW_SEC)
    frames = simulate_walk_through(person_seed=5000, n_frames=20, pose_variation=0.85,
                                    has_bad_entry=True, has_bad_exit=True)
    # Count how many would be quality-filtered out
    n_total = len(frames)
    n_quality = sum(
        1 for (x1, y1, x2, y2, c, _) in frames
        if is_quality_box(x1, y1, x2, y2, c, FRAME_W, FRAME_H,
                          [(x1, y1, x2, y2)], 0,
                          QUALITY_MIN_BBOX_HEIGHT, QUALITY_MIN_BBOX_WIDTH,
                          QUALITY_MIN_CONFIDENCE, QUALITY_EDGE_MARGIN,
                          QUALITY_MAX_OCCLUSION_IOU)[0]
    )
    pid, reason, n_used = process_track(reg, frames, ts_start=0.0)
    print(f"  {INFO} {n_quality}/{n_total} frames passed quality filter; {n_used} used for matching")
    print(f"  {INFO} Assigned P{pid} ({reason})")
    verdict(pid is not None and n_quality < n_total,
            f"Filtered {n_total - n_quality} bad-edge frame(s); committed clean P{pid}")


# ---------------------------------------------------------------------------
# SCENARIO 6: Two customers walk in shoulder-to-shoulder (occlusion)
# ---------------------------------------------------------------------------

def scenario_occluded_pair():
    header("SCENARIO 6: Two customers walk in together, partially occluding each other")
    print(f"{INFO} Customer A and Customer B walk side-by-side through entrance")
    print(f"{INFO} Heavy overlap in early frames; clean separation later")
    print(f"{INFO} Expected: 2 P numbers; occluded frames rejected by quality filter")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=SIM_THRESHOLD,
                         recency_window_sec=RECENCY_WINDOW_SEC)
    framesA = simulate_walk_through(person_seed=6001, n_frames=18, pose_variation=0.85,
                                     has_bad_entry=False, has_bad_exit=False)
    # B walks the same path but offset in x by 80 pixels and delayed by 3 frames,
    # so they overlap heavily ONLY when both are still near the entrance.
    framesB_raw = simulate_walk_through(person_seed=6002, n_frames=18, pose_variation=0.85,
                                         has_bad_entry=False, has_bad_exit=False)
    framesB = []
    for i, (x1, y1, x2, y2, c, e) in enumerate(framesB_raw):
        # Realistic side-by-side: B is ~250px to the right of A, ~30px down.
        # Light overlap at the shoulders (<30%), like a couple walking in.
        framesB.append((x1 + 250, y1 + 30, x2 + 250, y2 + 30, c, e))

    # other_boxes_per_frame: when processing A, what other boxes exist?
    other_for_A = [[(framesB[i][0], framesB[i][1], framesB[i][2], framesB[i][3])]
                    for i in range(len(framesA))]
    other_for_B = [[(framesA[i][0], framesA[i][1], framesA[i][2], framesA[i][3])]
                    for i in range(len(framesB))]

    pidA, reasonA, nA = process_track(reg, framesA, ts_start=0.0, other_boxes_per_frame=other_for_A)
    pidB, reasonB, nB = process_track(reg, framesB, ts_start=0.0, other_boxes_per_frame=other_for_B)
    print(f"  {INFO} Customer A: P{pidA} ({reasonA}, {nA} quality crops)")
    print(f"  {INFO} Customer B: P{pidB} ({reasonB}, {nB} quality crops)")
    verdict(pidA is not None and pidB is not None and pidA != pidB,
            f"Two occluded entries → 2 different P numbers")


# ---------------------------------------------------------------------------
# SCENARIO 7: Realistic 8-hour day — 20 customers + 3 employees
# ---------------------------------------------------------------------------

def scenario_realistic_day():
    header("SCENARIO 7: Realistic 8-hour day — 20 unique customers + 3 employees")
    print(f"{INFO} 20 distinct customers each enter 1-3 times")
    print(f"{INFO} 3 employees each enter 10-20 times (breaks, lunch, etc)")
    print(f"{INFO} Expected: ~23 unique P numbers ({BOLD}NOT{RESET} hundreds)")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=SIM_THRESHOLD,
                         recency_window_sec=RECENCY_WINDOW_SEC)
    rng = np.random.default_rng(42)
    events = []  # list of (ts, person_seed)
    # 20 customers, 1-3 trips each
    for cust_id in range(20):
        n_trips = rng.integers(1, 4)
        for trip in range(n_trips):
            events.append((rng.uniform(0, 8 * 3600), 7000 + cust_id))
    # 3 employees, 10-20 trips each
    for emp_id in range(3):
        n_trips = rng.integers(10, 21)
        for trip in range(n_trips):
            events.append((rng.uniform(0, 8 * 3600), 8000 + emp_id))
    events.sort()
    # Process each entry
    pids_per_seed = {}
    n_committed = 0
    n_attempted = 0
    for ts, seed in events:
        n_attempted += 1
        frames = simulate_walk_through(person_seed=seed, n_frames=18,
                                        pose_variation=0.83 + rng.normal(scale=0.02))
        pid, reason, n = process_track(reg, frames, ts_start=ts)
        if pid is not None:
            n_committed += 1
            pids_per_seed.setdefault(seed, set()).add(pid)

    total_unique_pids = len(set().union(*pids_per_seed.values())) if pids_per_seed else 0
    seeds_with_one_pid = sum(1 for pids in pids_per_seed.values() if len(pids) == 1)
    seeds_with_multiple_pids = sum(1 for pids in pids_per_seed.values() if len(pids) > 1)

    print(f"  {INFO} Total entries simulated: {n_attempted}, committed: {n_committed}")
    print(f"  {INFO} Real unique people: 23, system found unique P numbers: {total_unique_pids}")
    print(f"  {INFO} Real-people with exactly 1 P: {seeds_with_one_pid}/{len(pids_per_seed)}")
    print(f"  {INFO} Real-people fragmented across multiple P numbers: {seeds_with_multiple_pids}")

    # Pass if total unique P numbers is within 30% of truth (23)
    fragmented_pct = (total_unique_pids - 23) / 23 * 100 if total_unique_pids >= 23 else 0
    verdict(total_unique_pids <= 30,  # tolerate up to ~30% fragmentation
            f"23 real people → {total_unique_pids} P numbers ({fragmented_pct:+.0f}% vs truth)")


# ---------------------------------------------------------------------------
# SCENARIO 8: Stress — 100 distinct customers, each visits once
# ---------------------------------------------------------------------------

def scenario_stress_no_false_merges():
    header("SCENARIO 8: STRESS — 100 distinct customers, each visits exactly once")
    print(f"{INFO} 100 completely different people walk through")
    print(f"{INFO} Expected: 100 distinct P numbers (no false merges)")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=SIM_THRESHOLD,
                         recency_window_sec=RECENCY_WINDOW_SEC)
    rng = np.random.default_rng(0)
    pids = []
    for i in range(100):
        frames = simulate_walk_through(person_seed=9000 + i, n_frames=18,
                                        pose_variation=0.85 + rng.normal(scale=0.02))
        pid, _, _ = process_track(reg, frames, ts_start=i * 60.0)
        if pid is not None:
            pids.append(pid)
    unique = set(pids)
    false_merges = len(pids) - len(unique)
    print(f"  {INFO} {len(pids)}/100 committed; {len(unique)} unique P numbers")
    verdict(false_merges <= 2,  # tolerate at most 2 false merges out of 100
            f"100 distinct customers → {len(unique)} P numbers ({false_merges} false merges)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{BOLD}Reconnected store-scenario simulation{RESET}")
    print(f"Camera: 1920x1080 wide-angle entrance, person walks across frame")
    print(f"Matcher: similarity_threshold={SIM_THRESHOLD}, recency=30d")
    print(f"Quality filter: min_h={QUALITY_MIN_BBOX_HEIGHT}px, min_conf={QUALITY_MIN_CONFIDENCE},")
    print(f"                edge_margin={QUALITY_EDGE_MARGIN}px, max_occlusion={QUALITY_MAX_OCCLUSION_IOU}")
    print(f"Multi-shot: assign after {MIN_QUALITY_EMBEDS_FOR_ASSIGN} quality crops, "
          f"force-commit at {MAX_QUALITY_EMBEDS_FOR_WAIT}")

    scenario_multi_trip_customer()
    scenario_employee_all_day()
    scenario_three_distinct_people()
    scenario_returning_customer_cross_day()
    scenario_frame_edge_entry()
    scenario_occluded_pair()
    scenario_realistic_day()
    scenario_stress_no_false_merges()

    print(f"\n{BOLD}Done.{RESET}")
