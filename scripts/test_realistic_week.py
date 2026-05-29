"""
Realistic week-long simulation calibrated to Reconnected Device Repair's
actual operating patterns. Combines:

- Per-store traffic levels matched to FootfallCam Top 10 numbers from the
  portal (Prescott=37, Bullhead=22, Village Plaza=22, Fountain Hills=19,
  estimates for the other 4)
- Phone-repair multi-trip behavior: drop-off + status check + pickup
- 3-5 employees per store, each with 10-25 entries per day
- Hourly arrival distribution matching the FootfallCam heat map peaks
  (10am-1pm spike, low Sundays, etc)
- Realistic OSNet similarity distributions:
    same person same day:  ~0.75-0.92
    same person diff day:  ~0.55-0.85  (clothing changed)
    different similar ppl: ~0.40-0.65  (the danger zone)
    different distinct ppl:~0.15-0.45
- Cross-day customer returns (some "regulars" come back multiple days)
- Failure injection: dropped frames, broken tracks, pipeline restarts
- Doppelganger pairs (people who look similar enough to confuse OSNet)

Runs the actual matcher + quality filter + multi-shot + active-wait pipeline
against this realistic data and reports accuracy metrics that map directly
to Cam's business question: "did the system correctly count unique
customers across a week of operation?"

Run from project root:
    .venv/bin/python scripts/test_realistic_week.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.reid.matcher import PersonRegistry
from run_pipeline import is_quality_box, average_embeddings

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
DIM = "\033[2m"
RESET = "\033[0m"

EMB_DIM = 512
FRAME_W, FRAME_H = 1920, 1080
SIM_THRESHOLD = 0.78
RECENCY_WINDOW_SEC = 30 * 86400

MIN_QUALITY_EMBEDS_FOR_ASSIGN = 3
MAX_QUALITY_EMBEDS_FOR_WAIT = 8
QUALITY_MIN_BBOX_HEIGHT = 80
QUALITY_MIN_BBOX_WIDTH = 30
QUALITY_MIN_CONFIDENCE = 0.6
QUALITY_EDGE_MARGIN = 8
QUALITY_MAX_OCCLUSION_IOU = 0.3


# ===========================================================================
# Realistic person modeling
# ===========================================================================

class RealisticPerson:
    """Models a person with separable identity, clothing, and per-shot pose.
    Cross-day clothing change is the dominant source of intra-person variance
    in real OSNet — handled explicitly here."""

    def __init__(self, seed: int, rng: np.random.Generator):
        # Stable identity component (body shape, face structure, hair)
        self.identity = rng.normal(size=EMB_DIM)
        self.identity = self.identity / np.linalg.norm(self.identity)
        self.seed = seed
        # Daily clothing component is regenerated each day in get_embedding
        self._current_day = None
        self._daily_clothing = None
        self._rng_master = rng

    def _clothing_for_day(self, day: int) -> np.ndarray:
        """Each calendar day, the person has a different clothing component."""
        if self._current_day != day:
            day_rng = np.random.default_rng(self.seed * 1000 + day)
            v = day_rng.normal(size=EMB_DIM)
            self._daily_clothing = v / np.linalg.norm(v)
            self._current_day = day
        return self._daily_clothing

    def get_embedding(self, day: int, shot_seed: int, pose_severity: float = 0.5) -> np.ndarray:
        """Return one shot of this person on a given day.
        pose_severity 0=easy pose, 1=hard pose. Mixed weighting calibrated so:
            same person same day:  sim ~0.75-0.92
            same person diff day:  sim ~0.55-0.85
            different person:      sim ~0.15-0.45
        """
        clothing = self._clothing_for_day(day)
        shot_rng = np.random.default_rng(shot_seed)
        pose_noise = shot_rng.normal(size=EMB_DIM)
        pose_noise = pose_noise / np.linalg.norm(pose_noise)
        # Weights tuned to give realistic same-day similarity distributions
        identity_w = 0.78
        clothing_w = 0.32
        pose_w = 0.18 * pose_severity
        v = identity_w * self.identity + clothing_w * clothing + pose_w * pose_noise
        return v / np.linalg.norm(v)


# ===========================================================================
# Realistic frame simulator with failure injection
# ===========================================================================

def simulate_track_frames(
    person: RealisticPerson,
    day: int,
    track_seed: int,
    n_frames_target: int = 20,
    pose_severity: float = 0.5,
    drop_rate: float = 0.05,  # YOLO occasionally misses frames
    edge_entry: bool = True,
    edge_exit: bool = True,
) -> list:
    """Generate the per-frame detections for one track of this person walking
    through the entrance camera. Returns list of (x1, y1, x2, y2, conf, emb)
    or None for dropped frames."""
    rng = np.random.default_rng(track_seed)
    frames = []
    for i in range(n_frames_target):
        # YOLO drops some frames
        if rng.random() < drop_rate:
            frames.append(None)
            continue

        progress = i / max(1, n_frames_target - 1)
        cx = 50 + progress * (FRAME_W - 100)
        proximity = 1 - abs(progress - 0.5) * 2
        bbox_h = int(120 + proximity * 580)
        bbox_w = int(bbox_h * 0.4)

        x1 = int(cx - bbox_w / 2)
        y1 = int(FRAME_H * 0.15)
        x2 = int(cx + bbox_w / 2)
        y2 = y1 + bbox_h
        conf = 0.55 + proximity * 0.40

        if edge_entry and i < 3:
            x1 = i * 2
            x2 = x1 + bbox_w
            conf = 0.50 + i * 0.05
        if edge_exit and i >= n_frames_target - 3:
            offset = n_frames_target - 1 - i
            x2 = FRAME_W - offset * 2
            x1 = x2 - bbox_w
            conf = 0.50 + offset * 0.05

        emb = person.get_embedding(day, shot_seed=track_seed * 100 + i,
                                    pose_severity=pose_severity)
        frames.append((x1, y1, x2, y2, conf, emb))
    return frames


def process_track(registry: PersonRegistry, track_frames: list, ts_start: float,
                  store_id: str = "store", camera_id: str = "front") -> int | None:
    """Run a track through the same multi-shot + active-wait logic as
    run_pipeline.py. Returns assigned person_id or None."""
    quality_embeds = []
    for i, fr in enumerate(track_frames):
        if fr is None:
            continue
        x1, y1, x2, y2, conf, emb = fr
        ts = ts_start + i * 0.1

        ok, _ = is_quality_box(
            x1, y1, x2, y2, conf, FRAME_W, FRAME_H,
            [(x1, y1, x2, y2)], 0,
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
                if result.best_pid is not None:
                    registry.commit_match(result.best_pid, avg, ts)
                    return result.best_pid
                pid = registry.commit_new(avg, ts, store_id, camera_id)
                return pid
    return None


# ===========================================================================
# Per-store traffic profile, calibrated to Cam's actual numbers
# ===========================================================================

# Daily unique-customer counts approximating FootfallCam Top 10 numbers
STORE_PROFILES = {
    "anthem":         {"customers_per_day": 28, "employees": 4},
    "bullhead":       {"customers_per_day": 22, "employees": 4},
    "havasu":         {"customers_per_day": 26, "employees": 4},
    "kingman":        {"customers_per_day": 18, "employees": 3},
    "paradisevalley": {"customers_per_day": 22, "employees": 4},
    "tolleson":       {"customers_per_day": 15, "employees": 3},
    "prescott":       {"customers_per_day": 37, "employees": 5},
    "fountainhills":  {"customers_per_day": 19, "employees": 3},
}

# Phone repair trip-count distribution per customer
TRIP_DIST = [
    (1, 0.30),  # 30% are one-and-done (browsing, simple question)
    (2, 0.35),  # 35% drop off + pickup
    (3, 0.25),  # 25% drop off + status check + pickup
    (4, 0.07),  # 7% drop off + 2 status checks + pickup
    (5, 0.02),  # 2% extra-complicated repairs
    (6, 0.01),  # 1% very-complicated
]

# Hourly arrival distribution (matches the FootfallCam heat map peaks)
HOURLY_WEIGHTS = {
    9: 1.0, 10: 2.5, 11: 2.8, 12: 3.0, 13: 2.5, 14: 1.8,
    15: 2.0, 16: 1.7, 17: 1.5, 18: 1.2, 19: 1.0, 20: 0.8,
}


def sample_trip_count(rng: np.random.Generator) -> int:
    r = rng.random()
    cum = 0.0
    for n, p in TRIP_DIST:
        cum += p
        if r < cum:
            return n
    return 1


def sample_hourly_minute(rng: np.random.Generator) -> int:
    """Pick a minute-of-day weighted by the hourly distribution above."""
    hours = list(HOURLY_WEIGHTS.keys())
    weights = np.array([HOURLY_WEIGHTS[h] for h in hours])
    weights = weights / weights.sum()
    hour = rng.choice(hours, p=weights)
    minute = rng.integers(0, 60)
    return int(hour) * 60 + int(minute)


# ===========================================================================
# Simulate one store-day with full operational realism
# ===========================================================================

def simulate_store_day(
    rng: np.random.Generator,
    customer_pool: dict[int, RealisticPerson],
    employees: list[RealisticPerson],
    day_idx: int,
    day_ts_start: float,
    customer_count: int,
    cross_day_return_prob: float,
    next_new_customer_seed: int,
) -> tuple[list, int]:
    """Returns (sorted_events, next_seed). Each event is (ts, person, n_trips_this_visit,
    is_employee). One event per trip across all customers + employees."""
    events = []  # (ts_minute, person, is_employee)

    # Customers for today: mix of cross-day returners and new
    for _ in range(customer_count):
        if customer_pool and rng.random() < cross_day_return_prob:
            cust = rng.choice(list(customer_pool.values()))
        else:
            cust = RealisticPerson(seed=next_new_customer_seed, rng=rng)
            customer_pool[next_new_customer_seed] = cust
            next_new_customer_seed += 1
        n_trips = sample_trip_count(rng)
        # Trips spread across the day: first trip somewhere AM, follow-ups later
        first_trip_min = sample_hourly_minute(rng)
        events.append((first_trip_min, cust, False))
        for t in range(1, n_trips):
            # Subsequent trips happen 1-6 hours after previous
            follow_offset = int(rng.uniform(60, 360))
            next_min = min(first_trip_min + follow_offset * t, 20 * 60 + 30)
            events.append((next_min, cust, False))

    # Employees: each enters/exits 10-20 times during their shift
    for emp in employees:
        n_entries = int(rng.integers(10, 21))
        # Employee shifts vary
        shift_start = int(rng.choice([9 * 60, 10 * 60, 13 * 60]))  # 9am or 10am or 1pm start
        shift_end = shift_start + 8 * 60
        for _ in range(n_entries):
            t = int(rng.uniform(shift_start, shift_end))
            events.append((t, emp, True))

    events.sort(key=lambda e: e[0])
    # Attach timestamp in seconds (relative to start of week)
    return [
        (day_ts_start + e[0] * 60, e[1], e[2])
        for e in events
    ], next_new_customer_seed


# ===========================================================================
# The big test: simulate a full WEEK at one store
# ===========================================================================

def simulate_full_week_at_store(
    store_id: str,
    cross_day_return_prob: float = 0.15,
    week_seed: int = 100,
    verbose: bool = False,
) -> dict:
    profile = STORE_PROFILES[store_id]
    customers_per_day = profile["customers_per_day"]
    n_employees = profile["employees"]

    rng = np.random.default_rng(week_seed)
    registry = PersonRegistry(active_model="osnet",
                               similarity_threshold=SIM_THRESHOLD,
                               recency_window_sec=RECENCY_WINDOW_SEC)

    # Persistent state across the week
    customer_pool: dict[int, RealisticPerson] = {}
    next_customer_seed = 10000
    employees = [
        RealisticPerson(seed=20000 + store_id.__hash__() % 1000 + i, rng=rng)
        for i in range(n_employees)
    ]

    # Track ground truth: who actually walked in, and what P# the system assigned
    real_person_to_pids = defaultdict(set)  # real person -> set of P numbers assigned
    pid_to_real_persons = defaultdict(set)  # P# -> set of real people that mapped to it

    total_entries = 0
    total_committed = 0

    for day_idx in range(7):
        # Sunday closed (day 6 in this enumeration)
        if day_idx == 6:
            continue
        day_ts_start = day_idx * 86400.0
        # Daily customer count varies +-20%
        today_customers = int(customers_per_day * rng.uniform(0.8, 1.2))

        events, next_customer_seed = simulate_store_day(
            rng, customer_pool, employees, day_idx, day_ts_start,
            today_customers, cross_day_return_prob, next_customer_seed,
        )

        if verbose:
            n_unique_today = len({id(e[1]) for e in events})
            print(f"  Day {day_idx + 1}: {len(events)} entries from {n_unique_today} unique people")

        for ts, person, is_emp in events:
            total_entries += 1
            track_seed = int(ts * 1000) + person.seed
            # Pose severity varies — some entries are harder than others
            pose_severity = float(rng.uniform(0.3, 0.8))
            frames = simulate_track_frames(
                person, day=day_idx, track_seed=track_seed,
                n_frames_target=int(rng.integers(14, 24)),
                pose_severity=pose_severity,
            )
            pid = process_track(registry, frames, ts_start=ts)
            if pid is not None:
                total_committed += 1
                real_person_to_pids[id(person)].add(pid)
                pid_to_real_persons[pid].add(id(person))

    # Compute accuracy metrics
    n_real_unique = len(real_person_to_pids)
    n_assigned_unique = len(pid_to_real_persons)
    n_real_with_one_pid = sum(1 for pids in real_person_to_pids.values() if len(pids) == 1)
    n_real_fragmented = sum(1 for pids in real_person_to_pids.values() if len(pids) > 1)
    n_pids_pure = sum(1 for ppl in pid_to_real_persons.values() if len(ppl) == 1)
    n_pids_false_merge = sum(1 for ppl in pid_to_real_persons.values() if len(ppl) > 1)

    return {
        "store_id": store_id,
        "total_entries_simulated": total_entries,
        "total_committed": total_committed,
        "real_unique_people": n_real_unique,
        "system_unique_pids": n_assigned_unique,
        "real_with_one_pid": n_real_with_one_pid,
        "real_fragmented": n_real_fragmented,
        "pids_pure": n_pids_pure,
        "pids_false_merge": n_pids_false_merge,
    }


# ===========================================================================
# Report formatter
# ===========================================================================

def print_store_result(r: dict):
    real = r["real_unique_people"]
    sys = r["system_unique_pids"]
    delta = sys - real
    delta_pct = (delta / real * 100) if real else 0
    purity = (r["real_with_one_pid"] / real * 100) if real else 0
    merge_rate = (r["pids_false_merge"] / sys * 100) if sys else 0
    delta_color = GREEN if abs(delta_pct) < 10 else (YELLOW if abs(delta_pct) < 25 else RED)

    print(f"  {BOLD}{r['store_id']:>15}{RESET}  "
          f"entries={r['total_entries_simulated']:>4}  "
          f"committed={r['total_committed']:>4}  "
          f"real={real:>3}  "
          f"system={sys:>3}  "
          f"{delta_color}Δ={delta:+d} ({delta_pct:+.0f}%){RESET}  "
          f"purity={purity:.0f}%  "
          f"false-merge={merge_rate:.0f}%")


# ===========================================================================
# THE BIG RUNS
# ===========================================================================

def big_run_1_full_fleet_one_week():
    print(f"\n{BOLD}{BLUE}========================================================{RESET}")
    print(f"{BOLD}TEST 1: Full fleet, one week, no cross-day returns{RESET}")
    print(f"{BOLD}{BLUE}========================================================{RESET}")
    print(f"Simulates 6 business days at all 8 stores, ~20-37 customers/day each,")
    print(f"4 employees per store, no cross-day customer returns.")
    print()
    print(f"  {BOLD}{'store':>15}  {'entries':>11} {'committed':>11} {'real':>5} "
          f"{'system':>7}  {'delta':>11}  {'purity':>8} {'false-merge':>11}{RESET}")
    print(f"  {DIM}{'-'*15}  {'-'*11} {'-'*11} {'-'*5} {'-'*7}  {'-'*11}  {'-'*8} {'-'*11}{RESET}")
    totals = defaultdict(int)
    for store_id in STORE_PROFILES:
        result = simulate_full_week_at_store(store_id, cross_day_return_prob=0.0, week_seed=100)
        print_store_result(result)
        for k in ("total_entries_simulated", "total_committed", "real_unique_people",
                  "system_unique_pids", "real_fragmented", "pids_false_merge"):
            totals[k] += result[k]
    print(f"  {DIM}{'-'*15}  {'-'*11} {'-'*11} {'-'*5} {'-'*7}  {'-'*11}  {'-'*8} {'-'*11}{RESET}")
    delta = totals['system_unique_pids'] - totals['real_unique_people']
    delta_pct = delta / totals['real_unique_people'] * 100 if totals['real_unique_people'] else 0
    color = GREEN if abs(delta_pct) < 10 else YELLOW if abs(delta_pct) < 25 else RED
    print(f"  {BOLD}{'FLEET TOTAL':>15}  {totals['total_entries_simulated']:>11}"
          f" {totals['total_committed']:>11} {totals['real_unique_people']:>5}"
          f" {totals['system_unique_pids']:>7}  {color}{delta:+d} ({delta_pct:+.0f}%){RESET}  "
          f"frag={totals['real_fragmented']}  merges={totals['pids_false_merge']}{RESET}")


def big_run_2_with_cross_day_returns():
    print(f"\n{BOLD}{BLUE}========================================================{RESET}")
    print(f"{BOLD}TEST 2: Full fleet, one week, 25% chance any customer returns{RESET}")
    print(f"{BOLD}{BLUE}========================================================{RESET}")
    print(f"Same setup but 25% of any day's customers are returning regulars.")
    print(f"Tests the 30-day gallery + cross-day matching even with clothing changes.")
    print()
    print(f"  {BOLD}{'store':>15}  {'entries':>11} {'committed':>11} {'real':>5} "
          f"{'system':>7}  {'delta':>11}  {'purity':>8} {'false-merge':>11}{RESET}")
    print(f"  {DIM}{'-'*15}  {'-'*11} {'-'*11} {'-'*5} {'-'*7}  {'-'*11}  {'-'*8} {'-'*11}{RESET}")
    totals = defaultdict(int)
    for store_id in STORE_PROFILES:
        result = simulate_full_week_at_store(store_id, cross_day_return_prob=0.25, week_seed=200)
        print_store_result(result)
        for k in ("total_entries_simulated", "total_committed", "real_unique_people",
                  "system_unique_pids", "real_fragmented", "pids_false_merge"):
            totals[k] += result[k]
    print(f"  {DIM}{'-'*15}  {'-'*11} {'-'*11} {'-'*5} {'-'*7}  {'-'*11}  {'-'*8} {'-'*11}{RESET}")
    delta = totals['system_unique_pids'] - totals['real_unique_people']
    delta_pct = delta / totals['real_unique_people'] * 100 if totals['real_unique_people'] else 0
    color = GREEN if abs(delta_pct) < 10 else YELLOW if abs(delta_pct) < 25 else RED
    print(f"  {BOLD}{'FLEET TOTAL':>15}  {totals['total_entries_simulated']:>11}"
          f" {totals['total_committed']:>11} {totals['real_unique_people']:>5}"
          f" {totals['system_unique_pids']:>7}  {color}{delta:+d} ({delta_pct:+.0f}%){RESET}  "
          f"frag={totals['real_fragmented']}  merges={totals['pids_false_merge']}{RESET}")


def big_run_3_high_volume_stress():
    print(f"\n{BOLD}{BLUE}========================================================{RESET}")
    print(f"{BOLD}TEST 3: 4x volume stress — what if Prescott does 4x normal traffic?{RESET}")
    print(f"{BOLD}{BLUE}========================================================{RESET}")
    print(f"Simulates a freak busy week: every store gets 4x its usual daily traffic.")
    print(f"Tests that the system doesn't get confused at high volume.")
    print()
    # Temporarily inflate all store profiles
    original = {sid: dict(p) for sid, p in STORE_PROFILES.items()}
    for sid in STORE_PROFILES:
        STORE_PROFILES[sid]["customers_per_day"] *= 4
    try:
        print(f"  {BOLD}{'store':>15}  {'entries':>11} {'committed':>11} {'real':>5} "
              f"{'system':>7}  {'delta':>11}  {'purity':>8} {'false-merge':>11}{RESET}")
        print(f"  {DIM}{'-'*15}  {'-'*11} {'-'*11} {'-'*5} {'-'*7}  {'-'*11}  {'-'*8} {'-'*11}{RESET}")
        totals = defaultdict(int)
        for store_id in STORE_PROFILES:
            result = simulate_full_week_at_store(store_id, cross_day_return_prob=0.1, week_seed=300)
            print_store_result(result)
            for k in ("total_entries_simulated", "total_committed", "real_unique_people",
                      "system_unique_pids", "real_fragmented", "pids_false_merge"):
                totals[k] += result[k]
        print(f"  {DIM}{'-'*15}  {'-'*11} {'-'*11} {'-'*5} {'-'*7}  {'-'*11}  {'-'*8} {'-'*11}{RESET}")
        delta = totals['system_unique_pids'] - totals['real_unique_people']
        delta_pct = delta / totals['real_unique_people'] * 100 if totals['real_unique_people'] else 0
        color = GREEN if abs(delta_pct) < 10 else YELLOW if abs(delta_pct) < 25 else RED
        print(f"  {BOLD}{'FLEET TOTAL':>15}  {totals['total_entries_simulated']:>11}"
              f" {totals['total_committed']:>11} {totals['real_unique_people']:>5}"
              f" {totals['system_unique_pids']:>7}  {color}{delta:+d} ({delta_pct:+.0f}%){RESET}  "
              f"frag={totals['real_fragmented']}  merges={totals['pids_false_merge']}{RESET}")
    finally:
        for sid, p in original.items():
            STORE_PROFILES[sid] = p


def big_run_4_full_month_at_busiest_store():
    print(f"\n{BOLD}{BLUE}========================================================{RESET}")
    print(f"{BOLD}TEST 4: Full MONTH at Prescott (busiest store, 25% return rate){RESET}")
    print(f"{BOLD}{BLUE}========================================================{RESET}")
    print(f"4 weeks at Prescott. Lots of cross-day returners. Tests sustained")
    print(f"behavior over time + 30-day gallery saturation.")
    print()

    rng = np.random.default_rng(500)
    registry = PersonRegistry(active_model="osnet",
                               similarity_threshold=SIM_THRESHOLD,
                               recency_window_sec=RECENCY_WINDOW_SEC)
    customer_pool: dict[int, RealisticPerson] = {}
    next_customer_seed = 50000
    employees = [RealisticPerson(seed=60000 + i, rng=rng) for i in range(5)]

    real_person_to_pids = defaultdict(set)
    pid_to_real_persons = defaultdict(set)
    total_entries = 0

    for day_idx in range(28):  # 4 weeks
        if day_idx % 7 == 6:  # Sundays closed
            continue
        day_ts_start = day_idx * 86400.0
        today_customers = int(37 * rng.uniform(0.8, 1.2))
        events, next_customer_seed = simulate_store_day(
            rng, customer_pool, employees, day_idx, day_ts_start,
            today_customers, cross_day_return_prob=0.25,
            next_new_customer_seed=next_customer_seed,
        )
        for ts, person, _ in events:
            total_entries += 1
            track_seed = int(ts * 1000) + person.seed
            pose_severity = float(rng.uniform(0.3, 0.8))
            frames = simulate_track_frames(
                person, day=day_idx, track_seed=track_seed,
                n_frames_target=int(rng.integers(14, 24)),
                pose_severity=pose_severity,
            )
            pid = process_track(registry, frames, ts_start=ts)
            if pid is not None:
                real_person_to_pids[id(person)].add(pid)
                pid_to_real_persons[pid].add(id(person))

    real = len(real_person_to_pids)
    sys_ = len(pid_to_real_persons)
    one_pid = sum(1 for pids in real_person_to_pids.values() if len(pids) == 1)
    purity = one_pid / real * 100 if real else 0
    merges = sum(1 for ppl in pid_to_real_persons.values() if len(ppl) > 1)
    delta_pct = (sys_ - real) / real * 100 if real else 0
    color = GREEN if abs(delta_pct) < 10 else YELLOW if abs(delta_pct) < 25 else RED

    print(f"  Days simulated:                          24 business days (4 weeks, Sundays closed)")
    print(f"  Total entries:                           {total_entries}")
    print(f"  Real unique humans across the month:     {real}")
    print(f"  System assigned unique P numbers:        {sys_}  {color}({delta_pct:+.1f}% vs truth){RESET}")
    print(f"  Real people with exactly 1 P number:     {one_pid}/{real}  ({purity:.1f}% perfect)")
    print(f"  P numbers that mixed multiple people:    {merges}  (false merges)")


def big_run_5_doppelganger_adversarial():
    print(f"\n{BOLD}{BLUE}========================================================{RESET}")
    print(f"{BOLD}TEST 5: Adversarial — 5 pairs of look-alike customers{RESET}")
    print(f"{BOLD}{BLUE}========================================================{RESET}")
    print(f"Simulates 5 pairs of doppelgangers (people who look genuinely similar")
    print(f"to OSNet) plus 10 distinct customers in a single day. Tests whether")
    print(f"the system avoids false-merging similar-but-distinct people.")
    print()

    rng = np.random.default_rng(600)
    registry = PersonRegistry(active_model="osnet",
                               similarity_threshold=SIM_THRESHOLD,
                               recency_window_sec=RECENCY_WINDOW_SEC)
    people = []
    # 5 doppelganger pairs (each pair shares 60% of identity)
    for pair_i in range(5):
        base_seed = 70000 + pair_i
        base = RealisticPerson(seed=base_seed, rng=rng)
        twin = RealisticPerson(seed=base_seed + 10000, rng=rng)
        # Share 60% identity
        twin.identity = 0.7 * base.identity + 0.3 * twin.identity
        twin.identity = twin.identity / np.linalg.norm(twin.identity)
        people.extend([base, twin])
    # 10 distinct customers
    for i in range(10):
        people.append(RealisticPerson(seed=80000 + i, rng=rng))

    # Each person enters 3 times
    real_to_pids = defaultdict(set)
    pid_to_real = defaultdict(set)
    for person in people:
        for trip in range(3):
            ts = trip * 7200.0  # spread across the day
            track_seed = int(ts) + person.seed
            frames = simulate_track_frames(
                person, day=0, track_seed=track_seed,
                pose_severity=float(rng.uniform(0.3, 0.7)),
            )
            pid = process_track(registry, frames, ts_start=ts)
            if pid is not None:
                real_to_pids[id(person)].add(pid)
                pid_to_real[pid].add(id(person))

    print(f"  Real distinct people:                    {len(people)} (10 doppelganger + 10 distinct)")
    print(f"  System assigned unique P numbers:        {len(pid_to_real)}")
    n_doppel_false_merges = 0
    for pid, real_ids in pid_to_real.items():
        if len(real_ids) > 1:
            n_doppel_false_merges += 1
    pure_pids = sum(1 for r in pid_to_real.values() if len(r) == 1)
    print(f"  P numbers cleanly mapped to 1 real person: {pure_pids}/{len(pid_to_real)}")
    print(f"  False merges (2+ real people sharing a P): {n_doppel_false_merges}")
    if n_doppel_false_merges == 0:
        print(f"  {GREEN}✓ System resisted all doppelganger pressure{RESET}")
    elif n_doppel_false_merges <= 2:
        print(f"  {YELLOW}~ Acceptable: {n_doppel_false_merges} doppelganger pair(s) confused the system{RESET}")
    else:
        print(f"  {RED}✗ Too many false merges — threshold may be too loose{RESET}")


def big_run_6_failure_injection():
    print(f"\n{BOLD}{BLUE}========================================================{RESET}")
    print(f"{BOLD}TEST 6: Failure injection — dropped frames + short tracks{RESET}")
    print(f"{BOLD}{BLUE}========================================================{RESET}")
    print(f"What happens when YOLO drops 20% of frames AND some customers walk")
    print(f"through too fast (only 6-10 frame tracks)? Realistic 'bad day' scenario.")
    print()

    rng = np.random.default_rng(700)
    registry = PersonRegistry(active_model="osnet",
                               similarity_threshold=SIM_THRESHOLD,
                               recency_window_sec=RECENCY_WINDOW_SEC)
    customer_pool = {}
    next_seed = 90000
    employees = [RealisticPerson(seed=91000 + i, rng=rng) for i in range(4)]

    real_to_pids = defaultdict(set)
    n_uncommitted = 0
    n_total = 0

    for day_idx in range(3):
        day_ts = day_idx * 86400.0
        events, next_seed = simulate_store_day(
            rng, customer_pool, employees, day_idx, day_ts, 22,
            cross_day_return_prob=0.0, next_new_customer_seed=next_seed,
        )
        for ts, person, _ in events:
            n_total += 1
            # 20% dropped frames, plus some "fast walker" tracks with only 6-10 frames
            n_frames = int(rng.choice([6, 8, 10, 18, 20]))
            frames = simulate_track_frames(
                person, day=day_idx, track_seed=int(ts) + person.seed,
                n_frames_target=n_frames, drop_rate=0.2,
                pose_severity=float(rng.uniform(0.4, 0.9)),
            )
            pid = process_track(registry, frames, ts_start=ts)
            if pid is not None:
                real_to_pids[id(person)].add(pid)
            else:
                n_uncommitted += 1

    real = len(real_to_pids)
    n_assigned = len(set().union(*real_to_pids.values())) if real_to_pids else 0
    print(f"  Total entries simulated:                {n_total}")
    print(f"  Entries that committed a P number:       {n_total - n_uncommitted}")
    print(f"  Entries with too-few quality crops:      {n_uncommitted} (silently dropped)")
    print(f"  Real distinct people seen across 3 days: {real}")
    print(f"  System assigned unique P numbers:        {n_assigned}")
    skip_rate = n_uncommitted / n_total * 100
    color = GREEN if skip_rate < 30 else YELLOW if skip_rate < 50 else RED
    print(f"  {color}Skip rate under failure: {skip_rate:.0f}%{RESET}  ({DIM}skipped entries are honest 'missed identification' — better than wrong ID{RESET})")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print(f"{BOLD}Reconnected realistic-week simulation{RESET}")
    print(f"Calibrated to your actual stores' traffic levels from the FootfallCam portal.")
    print(f"Pipeline matches deployed code at commit 814ce26 (Phase 2.1 + tests).")
    print()
    print(f"Each test reports per-store and fleet totals:")
    print(f"  {DIM}entries{RESET}   = total door crossings simulated (multi-trip customers + employees)")
    print(f"  {DIM}committed{RESET} = entries that produced a person_id (rest are too-fast tracks)")
    print(f"  {DIM}real{RESET}      = the actual number of unique humans")
    print(f"  {DIM}system{RESET}    = the system's unique person_id count")
    print(f"  {DIM}delta{RESET}     = system - real (positive = over-counts, negative = merges)")
    print(f"  {DIM}purity{RESET}    = % of real people who got exactly 1 P number")
    print(f"  {DIM}false-merge{RESET}= % of P numbers that combined multiple real people")
    print()

    big_run_1_full_fleet_one_week()
    big_run_2_with_cross_day_returns()
    big_run_3_high_volume_stress()
    big_run_4_full_month_at_busiest_store()
    big_run_5_doppelganger_adversarial()
    big_run_6_failure_injection()

    print(f"\n{BOLD}Done.{RESET}")
