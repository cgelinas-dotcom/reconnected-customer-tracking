"""
End-to-end test of the Phase 2.1 changes using synthetic embeddings.

Runs every behavior change we shipped tonight against controlled scenarios
and prints pass/fail for each. Lets us verify the logic works as designed
without needing a real camera + walk test.

Run from project root:
    .venv/bin/python scripts/test_phase21.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.reid.matcher import (
    PersonRegistry, MatchResult,
    HIGH_CONF_DELTA_ABOVE_THRESHOLD,
    LOW_CONF_DELTA_BELOW_THRESHOLD,
    COLOR_SIM_BOOST_ABOVE,
    COLOR_SIM_REJECT_BELOW,
)

# Import the quality + color helpers from the pipeline module
sys.path.insert(0, str(Path(__file__).parent))
from run_pipeline import is_quality_box, average_embeddings, color_signature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
INFO = "\033[94m·\033[0m"

EMB_DIM = 512

def make_true_fingerprint(seed: int) -> np.ndarray:
    """Synthesize a person's 'true' L2-normalized fingerprint."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=EMB_DIM)
    return v / np.linalg.norm(v)

def noisy(true: np.ndarray, target_sim: float, seed: int) -> np.ndarray:
    """Generate a single 'shaky frame' observation with controlled cosine
    similarity to the true fingerprint. Real OSNet variation across angles/
    poses lands around target_sim=0.78–0.92 for the same person, 0.30–0.55
    for different people."""
    rng = np.random.default_rng(seed)
    # Get a direction orthogonal to true
    orth = rng.normal(size=EMB_DIM)
    orth -= np.dot(orth, true) * true
    orth_norm = np.linalg.norm(orth)
    if orth_norm > 0:
        orth = orth / orth_norm
    # Per-shot variance around target similarity (mimics angle/lighting jitter)
    sim = target_sim + rng.normal(scale=0.04)
    sim = float(np.clip(sim, -0.95, 0.95))
    # Build: sim * true + sqrt(1 - sim^2) * orth (exact cosine sim to true)
    v = sim * true + np.sqrt(max(0.0, 1 - sim * sim)) * orth
    return v / np.linalg.norm(v)

def section(title: str) -> None:
    print(f"\n\033[1m=== {title} ===\033[0m")


# ---------------------------------------------------------------------------
# TEST 1: Quality filter rejects bad crops
# ---------------------------------------------------------------------------

def test_quality_filter():
    section("TEST 1 — Quality filter rejects bad crops")

    frame_w, frame_h = 1920, 1080

    # Helper: single box at given (x1, y1, x2, y2, conf)
    def check(name, x1, y1, x2, y2, conf, others=None):
        boxes = [(x1, y1, x2, y2)] + (others or [])
        ok, why = is_quality_box(
            x1, y1, x2, y2, conf, frame_w, frame_h,
            boxes, 0,
            min_h=80, min_w=30, min_conf=0.6,
            edge_margin=8, max_occlusion_iou=0.3,
        )
        return ok, why

    cases = [
        ("Good crop, center of frame, large, high confidence",
         800, 300, 1000, 900, 0.92, None, True),
        ("Tiny crop (50px tall) — should reject",
         800, 500, 900, 550, 0.92, None, False),
        ("Crop at frame edge (x1=2) — should reject",
         2, 300, 400, 900, 0.92, None, False),
        ("Crop at right edge (x2=1919) — should reject",
         1700, 300, 1919, 900, 0.92, None, False),
        ("Low confidence (0.4) — should reject",
         800, 300, 1000, 900, 0.4, None, False),
        ("Heavily occluded by another box — should reject",
         800, 300, 1000, 900, 0.92, [(820, 350, 990, 880)], False),
    ]

    for name, x1, y1, x2, y2, conf, others, expect_pass in cases:
        ok, why = check(name, x1, y1, x2, y2, conf, others)
        passed = (ok == expect_pass)
        symbol = PASS if passed else FAIL
        verdict = "accepted" if ok else f"rejected ({why})"
        print(f"  {symbol}  {name}")
        print(f"         expected: {'accept' if expect_pass else 'reject'} · got: {verdict}")


# ---------------------------------------------------------------------------
# TEST 2: Multi-shot averaging produces a stable fingerprint
# ---------------------------------------------------------------------------

def test_multi_shot_averaging():
    section("TEST 2 — Multi-shot averaging stabilizes the fingerprint")

    true = make_true_fingerprint(seed=42)

    # Single noisy shot vs averaging 3 noisy shots
    # Each shot represents one frame of OSNet output, target_sim=0.80 mimics
    # the same person from a slightly different angle/lighting
    single = noisy(true, target_sim=0.80, seed=1)
    multi = average_embeddings([
        noisy(true, target_sim=0.80, seed=1),
        noisy(true, target_sim=0.80, seed=2),
        noisy(true, target_sim=0.80, seed=3),
    ])

    single_sim = float(np.dot(single, true))
    multi_sim = float(np.dot(multi, true))

    print(f"  {INFO}  Similarity of 1 noisy shot to true:    {single_sim:.3f}")
    print(f"  {INFO}  Similarity of 3-shot average to true:  {multi_sim:.3f}")
    improved = multi_sim > single_sim
    symbol = PASS if improved else FAIL
    print(f"  {symbol}  3-shot average is {'closer' if improved else 'NOT closer'} to true fingerprint")


# ---------------------------------------------------------------------------
# TEST 3: Same person walks in 3 separate times → 1 person ID
# ---------------------------------------------------------------------------

def test_same_person_three_visits():
    section("TEST 3 — Same person walks in 3 separate times → should be 1 P number")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=0.78)
    gary_true = make_true_fingerprint(seed=100)

    # Three visits, hours apart, each with multi-shot embeddings of Gary
    # target_sim=0.82 mimics realistic OSNet variation across angles/poses
    assigned_ids = []
    for visit_idx, base_ts in enumerate([1000.0, 10000.0, 25000.0]):
        embs = [noisy(gary_true, target_sim=0.82, seed=visit_idx * 10 + i) for i in range(4)]
        avg = average_embeddings(embs)
        pid, sim = reg.assign(avg, base_ts, "bullhead", "front")
        assigned_ids.append(pid)
        sim_str = f"{sim:.3f}" if sim is not None else "new"
        print(f"  {INFO}  Visit {visit_idx + 1}: assigned P{pid}  (sim={sim_str})")

    unique = set(assigned_ids)
    symbol = PASS if len(unique) == 1 else FAIL
    print(f"  {symbol}  Got {len(unique)} unique P number(s); expected 1")
    if len(unique) != 1:
        print(f"         (P numbers assigned: {assigned_ids})")


# ---------------------------------------------------------------------------
# TEST 4: Different people get DIFFERENT P numbers
# ---------------------------------------------------------------------------

def test_different_people_dont_merge():
    section("TEST 4 — Two different people → should be 2 P numbers")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=0.78)
    gary = make_true_fingerprint(seed=200)
    anthony = make_true_fingerprint(seed=201)

    # target_sim=0.85 for same-person multi-shot crops (each shot has high
    # similarity to that person's true fingerprint; different people are
    # naturally orthogonal because make_true_fingerprint uses unique seeds)
    avg_gary = average_embeddings([noisy(gary, target_sim=0.85, seed=i) for i in range(4)])
    avg_anthony = average_embeddings([noisy(anthony, target_sim=0.85, seed=i + 50) for i in range(4)])

    gary_pid, _ = reg.assign(avg_gary, 1000.0, "bullhead", "front")
    anthony_pid, sim = reg.assign(avg_anthony, 1001.0, "bullhead", "front")

    sim_str = f"{sim:.3f}" if sim is not None else "new"
    print(f"  {INFO}  Gary assigned P{gary_pid}")
    print(f"  {INFO}  Anthony assigned P{anthony_pid}  (Anthony's similarity to Gary: {sim_str})")

    symbol = PASS if gary_pid != anthony_pid else FAIL
    print(f"  {symbol}  Two different people got {'different' if gary_pid != anthony_pid else 'SAME'} P numbers")


# ---------------------------------------------------------------------------
# TEST 5: Borderline match triggers active-wait
# ---------------------------------------------------------------------------

def test_active_wait_on_borderline():
    section("TEST 5 — Borderline confidence triggers active wait")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=0.78)
    gary = make_true_fingerprint(seed=300)

    # Seed Gary in the registry from a clean visit (high-quality embeddings)
    avg = average_embeddings([noisy(gary, target_sim=0.88, seed=i) for i in range(5)])
    reg.commit_new(avg, 1000.0, "bullhead", "front")

    # Now simulate a new track with noisy-enough embeddings that it lands in
    # the borderline zone vs Gary
    print(f"  {INFO}  similarity_threshold = {reg.similarity_threshold}")
    print(f"  {INFO}  high-conf zone:    sim >= {reg.similarity_threshold + HIGH_CONF_DELTA_ABOVE_THRESHOLD:.2f}")
    print(f"  {INFO}  low-conf zone:     sim <  {reg.similarity_threshold - LOW_CONF_DELTA_BELOW_THRESHOLD:.2f}")
    print(f"  {INFO}  borderline zone:   in between")

    # Build a borderline embedding (target_sim landing right in the borderline
    # zone — between 0.63 low-conf and 0.83 high-conf with threshold=0.78)
    borderline = noisy(gary, target_sim=0.72, seed=999)
    res = reg.evaluate(borderline, 1500.0)
    print(f"  {INFO}  Single shaky embedding → confidence='{res.confidence}', sim={res.best_sim:.3f}")

    # Active wait: add more samples of the same person — average should
    # converge toward the true fingerprint and cross into the confident zone
    more = [noisy(gary, target_sim=0.78, seed=900 + i) for i in range(6)]
    avg_with_more = average_embeddings([borderline] + more)
    res2 = reg.evaluate(avg_with_more, 1501.0)
    print(f"  {INFO}  After 7-shot average    → confidence='{res2.confidence}', sim={res2.best_sim:.3f}")

    stable_match = (res2.confidence == "match" and res2.best_pid is not None)
    symbol = PASS if stable_match else FAIL
    print(f"  {symbol}  More samples produce {'confident match' if stable_match else 'still uncertain'}")


# ---------------------------------------------------------------------------
# TEST 6: Color tiebreaker in borderline zone
# ---------------------------------------------------------------------------

def test_color_tiebreaker():
    section("TEST 6 — Color tiebreaker adjusts borderline decisions")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=0.78)
    gary = make_true_fingerprint(seed=400)

    # Seed Gary with a known shirt color (e.g. red hue dominant)
    red_shirt = [0.0] * 16
    red_shirt[0] = 0.5  # H=0 bin = red
    red_shirt[15] = 0.5
    avg = average_embeddings([noisy(gary, target_sim=0.88, seed=i) for i in range(5)])
    reg.commit_new(avg, 1000.0, "bullhead", "front", red_shirt)

    # Borderline OSNet match — need average sim to land clearly in 0.63–0.83
    # zone (with threshold 0.78). Use lower per-shot target to land in
    # borderline after averaging.
    borderline = average_embeddings([noisy(gary, target_sim=0.65, seed=i + 200) for i in range(3)])

    # Same color → should boost
    blue_shirt = [0.0] * 16
    blue_shirt[7] = 1.0  # H=blue
    res_red = reg.evaluate(borderline, 1500.0, red_shirt)
    res_blue = reg.evaluate(borderline, 1500.0, blue_shirt)

    print(f"  {INFO}  Borderline OSNet sim:        {res_red.best_sim:.3f}")
    print(f"  {INFO}  Same color (red) → adjusted: {res_red.adjusted_sim:.3f}  conf='{res_red.confidence}'")
    print(f"  {INFO}  Diff color (blue) → adjusted:{res_blue.adjusted_sim:.3f}  conf='{res_blue.confidence}'")

    boost_works = res_red.adjusted_sim > res_red.best_sim
    penalty_works = res_blue.adjusted_sim < res_blue.best_sim
    symbol1 = PASS if boost_works else FAIL
    symbol2 = PASS if penalty_works else FAIL
    print(f"  {symbol1}  Matching colors boosted similarity")
    print(f"  {symbol2}  Mismatched colors penalized similarity")


# ---------------------------------------------------------------------------
# TEST 7: 30-day gallery — returning customer across days
# ---------------------------------------------------------------------------

def test_30_day_gallery():
    section("TEST 7 — 30-day gallery: customer returning after 5 days matches")

    reg = PersonRegistry(active_model="osnet", similarity_threshold=0.78)
    gary = make_true_fingerprint(seed=500)

    # Day 1: Gary's first visit (multi-shot averaged embeddings)
    avg1 = average_embeddings([noisy(gary, target_sim=0.85, seed=i) for i in range(4)])
    gary_pid, _ = reg.assign(avg1, 0.0, "bullhead", "front")
    print(f"  {INFO}  Day 1: Gary first seen → assigned P{gary_pid}")

    # Day 6 (5 days later — 432000 seconds): Gary returns
    five_days = 5 * 86400
    avg2 = average_embeddings([noisy(gary, target_sim=0.85, seed=i + 100) for i in range(4)])
    return_pid, sim = reg.assign(avg2, five_days, "bullhead", "front")
    sim_str = f"{sim:.3f}" if sim is not None else "new"
    print(f"  {INFO}  Day 6: Gary returns → matched to P{return_pid}  (sim={sim_str})")

    symbol = PASS if return_pid == gary_pid else FAIL
    msg = "matched" if return_pid == gary_pid else f"got NEW id P{return_pid}"
    print(f"  {symbol}  Returning customer correctly {msg}")


# ---------------------------------------------------------------------------
# TEST 8: track_persons pre-load bug is gone (regression test)
# ---------------------------------------------------------------------------

def test_no_track_persons_preload():
    section("TEST 8 — track_persons pre-load bug stays dead (regression check)")
    import re
    pipeline_path = Path(__file__).parent / "run_pipeline.py"
    src = pipeline_path.read_text()
    # The bad pattern: pre-loading track→person mappings on startup
    bad_pattern = re.compile(
        r"for\s+tid,\s*pid\s+in\s+db\.execute\(\s*\n?\s*\"SELECT\s+track_id,\s+person_id\s+FROM\s+track_persons",
        re.MULTILINE,
    )
    found = bool(bad_pattern.search(src))
    symbol = PASS if not found else FAIL
    print(f"  {symbol}  Pre-load loop is {'gone' if not found else 'STILL PRESENT — bug regressed'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\033[1mPhase 2.1 behavior verification\033[0m")
    print("Synthetic test of every change shipped tonight.\n")

    test_quality_filter()
    test_multi_shot_averaging()
    test_same_person_three_visits()
    test_different_people_dont_merge()
    test_active_wait_on_borderline()
    test_color_tiebreaker()
    test_30_day_gallery()
    test_no_track_persons_preload()

    print("\n\033[1mDone.\033[0m  Each PASS proves the corresponding code change does what we said it does.")
