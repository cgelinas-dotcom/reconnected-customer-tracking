"""
Phase 1+3: detect + track + (optionally) re-identify people across cameras/time.

Reads from RTSP / webcam / file, runs YOLO person detection with built-in
ByteTrack tracking, draws bounding boxes + IDs on each frame, writes an
annotated mp4 to data/out_annotated.mp4, and logs every detection to
data/events.sqlite. If REID=1, also computes a person re-ID embedding for
each new track and assigns a stable person_id that persists across cameras
and across re-runs.

Usage:
    python scripts/run_pipeline.py <source>

    <source>:
      RTSP URL:    "rtsp://user:pass@ip:554/..."
      Webcam idx:  0
      File path:   data/samples/people-detection.mp4

Optional env vars:
    STORE_ID=store-01       tag events with this store (default: "dev")
    CAMERA_ID=cam-front     tag events with this camera (default: "test")
    MAX_FRAMES=900          cap frames processed (default: no cap)
    SHOW=1                  open a live preview window (default: off)
    REID=1                  enable cross-camera person re-identification
                            (downloads ~45MB ResNet18 weights on first use)
    ENTRY_LINE="x1,y1,x2,y2,inside_dx,inside_dy"
                            enable line-crossing entry counting. Numbers are pixel
                            coords on the camera frame: line from (x1,y1) to (x2,y2),
                            inside_dx/dy = direction vector pointing INTO the store.
                            Use scripts/pick_entry_line.py to get the right values.
    EXCLUSION_LINE="x1,y1,x2,y2,inside_dx,inside_dy"
                            same format. Crossing this line (either direction)
                            auto-tags the person as an employee. Use this for
                            staff-only boundaries the camera can see — e.g. the
                            line into the area behind the counter, the back of
                            the store, or an employee back door. Customers
                            shouldn't ever cross it; if they do, you can unmark
                            the false positive in the dashboard.
    FRAME_SKIP=3            process every Nth frame instead of all. Default 1.
                            FRAME_SKIP=3 cuts CPU ~66% while still tracking
                            retail-pace foot traffic well.
    BUSINESS_HOURS=9-21     only detect during these hours (24h local clock).
                            Outside the window, frames are read+discarded.
                            Empty = always on.
"""

import os
import sqlite3
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_VIDEO = ROOT / "data" / "out_annotated.mp4"
LIVE_DIR = ROOT / "data" / "live"
DB_PATH = ROOT / "data" / "events.sqlite"
MODEL_NAME = "yolov8n.pt"
PERSON_CLASS_ID = 0


def is_quality_box(x1: float, y1: float, x2: float, y2: float, conf: float,
                   frame_w: int, frame_h: int,
                   other_boxes: list, my_idx: int,
                   min_h: int, min_w: int, min_conf: float,
                   edge_margin: int, max_occlusion_iou: float) -> tuple[bool, str]:
    """Decide whether a detection box is good enough to embed for re-ID.
    Returns (is_quality, reason_if_not). Used to refuse garbage crops
    (frame-edge, tiny, occluded, low-confidence) instead of letting them
    pollute a person's fingerprint."""
    bh = y2 - y1
    bw = x2 - x1
    if bh < min_h:
        return False, f"too short ({bh:.0f}px)"
    if bw < min_w:
        return False, f"too narrow ({bw:.0f}px)"
    if conf < min_conf:
        return False, f"low conf ({conf:.2f})"
    if (x1 < edge_margin or y1 < edge_margin or
            x2 > frame_w - edge_margin or y2 > frame_h - edge_margin):
        return False, "frame edge"
    # Occlusion check against other detections in the same frame
    my_area = max(1.0, bh * bw)
    for j, (ox1, oy1, ox2, oy2) in enumerate(other_boxes):
        if j == my_idx:
            continue
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        if inter / my_area > max_occlusion_iou:
            return False, f"occluded {inter/my_area:.0%}"
    return True, ""


def color_signature(crop) -> list:
    """Cheap dominant-color signature: HSV histogram of the torso region
    (middle vertical third of the crop, which is most likely shirt/jacket).
    Returned as a 16-bin normalized list. Used as a same-day tiebreaker
    for OSNet matches."""
    import cv2
    import numpy as np
    if crop is None or crop.size == 0:
        return [0.0] * 16
    h = crop.shape[0]
    torso = crop[h // 3: 2 * h // 3]
    if torso.size == 0:
        return [0.0] * 16
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s = float(hist.sum())
    return (hist / s).tolist() if s > 0 else [0.0] * 16


def average_embeddings(embs: list):
    """L2-normalized mean of a list of unit embeddings."""
    import numpy as np
    if not embs:
        return None
    avg = np.mean(np.stack(embs, axis=0), axis=0)
    n = float(np.linalg.norm(avg))
    return (avg / n) if n > 0 else avg


def parse_source(arg: str):
    if arg.isdigit():
        return int(arg), "webcam"
    if arg.startswith(("rtsp://", "http://", "https://")):
        return arg, "stream"
    return arg, "file"


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL    NOT NULL,
            store_id    TEXT    NOT NULL,
            camera_id   TEXT    NOT NULL,
            track_id    INTEGER NOT NULL,
            x1          REAL    NOT NULL,
            y1          REAL    NOT NULL,
            x2          REAL    NOT NULL,
            y2          REAL    NOT NULL,
            confidence  REAL    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_detections_store_ts
            ON detections (store_id, ts);
        CREATE INDEX IF NOT EXISTS idx_detections_track
            ON detections (store_id, camera_id, track_id);

        CREATE TABLE IF NOT EXISTS persons (
            person_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            first_seen_ts    REAL    NOT NULL,
            last_seen_ts    REAL    NOT NULL,
            first_store      TEXT    NOT NULL,
            first_camera     TEXT    NOT NULL,
            embedding        BLOB    NOT NULL,
            n_samples        INTEGER NOT NULL DEFAULT 1,
            embedding_model  TEXT    NOT NULL DEFAULT 'unknown'
        );

        CREATE TABLE IF NOT EXISTS track_persons (
            store_id   TEXT    NOT NULL,
            camera_id  TEXT    NOT NULL,
            track_id   INTEGER NOT NULL,
            person_id  INTEGER NOT NULL,
            match_sim  REAL,
            assigned_ts REAL   NOT NULL,
            PRIMARY KEY (store_id, camera_id, track_id),
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );

        CREATE TABLE IF NOT EXISTS entry_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         REAL    NOT NULL,
            store_id   TEXT    NOT NULL,
            camera_id  TEXT    NOT NULL,
            track_id   INTEGER NOT NULL,
            person_id  INTEGER,
            direction  TEXT    NOT NULL CHECK (direction IN ('in', 'out'))
        );
        CREATE INDEX IF NOT EXISTS idx_entry_events_store_ts
            ON entry_events (store_id, ts);
        CREATE INDEX IF NOT EXISTS idx_entry_events_person
            ON entry_events (person_id);

        CREATE TABLE IF NOT EXISTS employees (
            person_id    INTEGER PRIMARY KEY,
            name         TEXT,
            role         TEXT,
            enrolled_at  REAL NOT NULL,
            tagged_via   TEXT NOT NULL DEFAULT 'manual'
        );
        """
    )
    # Best-effort schema upgrades for pre-existing DBs:
    for ddl in (
        "ALTER TABLE employees ADD COLUMN tagged_via TEXT NOT NULL DEFAULT 'manual'",
        "ALTER TABLE persons ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'unknown'",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass
    return conn


def load_registry_from_db(conn: sqlite3.Connection, active_model: str):
    """Reload existing person embeddings so re-runs continue to match."""
    import numpy as np
    from src.reid.matcher import Person, PersonRegistry

    registry = PersonRegistry(active_model=active_model)
    rows = conn.execute(
        "SELECT person_id, embedding, n_samples, first_seen_ts, last_seen_ts, "
        "first_store, first_camera, embedding_model FROM persons"
    ).fetchall()
    same_model = 0
    other_model = 0
    for pid, emb_blob, n, first_ts, last_ts, first_store, first_cam, emb_model in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        registry.persons[pid] = Person(
            person_id=pid, embedding=emb, n_samples=n,
            first_seen_ts=first_ts, last_seen_ts=last_ts,
            first_store=first_store, first_camera=first_cam,
            embedding_model=emb_model or "unknown",
        )
        if emb_model == active_model:
            same_model += 1
        else:
            other_model += 1
    if rows:
        registry._next_id = max(r[0] for r in rows) + 1
    print(f"[reid] loaded {len(rows)} existing person(s) from DB "
          f"({same_model} match active model {active_model!r}, "
          f"{other_model} from a different model — ignored for matching)")
    return registry


def persist_person(conn: sqlite3.Connection, person) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO persons
           (person_id, first_seen_ts, last_seen_ts, first_store, first_camera,
            embedding, n_samples, embedding_model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (person.person_id, person.first_seen_ts, person.last_seen_ts,
         person.first_store, person.first_camera,
         person.embedding.astype("float32").tobytes(), person.n_samples,
         person.embedding_model),
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    source, kind = parse_source(sys.argv[1])
    store_id = os.environ.get("STORE_ID", "dev")
    camera_id = os.environ.get("CAMERA_ID", "test")
    max_frames = int(os.environ.get("MAX_FRAMES", "0"))
    show_preview = os.environ.get("SHOW") == "1"
    reid_enabled = os.environ.get("REID") == "1"
    entry_line_spec = os.environ.get("ENTRY_LINE", "").strip()
    exclusion_line_spec = os.environ.get("EXCLUSION_LINE", "").strip()
    frame_skip = max(1, int(os.environ.get("FRAME_SKIP", "1")))
    business_hours_spec = os.environ.get("BUSINESS_HOURS", "").strip()
    business_hours: tuple[int, int] | None = None
    if business_hours_spec:
        try:
            start_h, end_h = [int(x) for x in business_hours_spec.split("-")]
            business_hours = (start_h, end_h)
        except Exception:
            print(f"[warn] BUSINESS_HOURS={business_hours_spec!r} unparseable; ignoring")

    print(f"Source ({kind}): {source}")
    print(f"Tagging events as store='{store_id}' camera='{camera_id}'")
    print(f"Re-ID enabled: {reid_enabled}")
    if entry_line_spec:
        print(f"Entry line: {entry_line_spec}")
    if exclusion_line_spec:
        print(f"Exclusion line (staff-only zone — auto-tags as employee): {exclusion_line_spec}")
    if frame_skip > 1:
        print(f"Frame skip: processing every {frame_skip} frames (~{30/frame_skip:.0f}fps from 30fps source)")
    if business_hours:
        print(f"Business hours: {business_hours[0]:02d}:00 - {business_hours[1]:02d}:00 local time")
    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if kind == "stream" else cv2.VideoCapture(source)
    if not cap.isOpened():
        print("FAILED to open source.")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input: {w}x{h} @ {fps:.1f}fps")

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    live_path = LIVE_DIR / f"{store_id}_{camera_id}.jpg"
    writer = cv2.VideoWriter(
        str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    db = init_db(DB_PATH)

    # Line-crossing setup (optional)
    line_detector = None
    exclusion_detector = None
    if entry_line_spec or exclusion_line_spec:
        from src.ingest.line_crossing import EntryLine, LineCrossingDetector
    if entry_line_spec:
        line_detector = LineCrossingDetector(line=EntryLine.from_spec(entry_line_spec))
    if exclusion_line_spec:
        exclusion_detector = LineCrossingDetector(line=EntryLine.from_spec(exclusion_line_spec))

    # Tunable settings read from DB (editable via dashboard)
    from src import settings as settings_mod
    min_conf = settings_mod.get(db, "detection.min_confidence")
    print(f"Detection min confidence: {min_conf}")

    embed_crop = None
    registry = None
    track_to_person: dict[int, int] = {}  # local cache for this run
    if reid_enabled:
        from src.reid.embeddings import embed_crop, MODEL_NAME as REID_MODEL_NAME
        reid_threshold = settings_mod.get(db, "reid.similarity_threshold")
        reid_window = settings_mod.get(db, "reid.recency_window_sec")
        print(f"Re-ID model: {REID_MODEL_NAME}")
        print(f"Re-ID similarity threshold: {reid_threshold}")
        print(f"Re-ID recency window: {int(reid_window)}s ({reid_window/3600:.1f}h)")
        registry = load_registry_from_db(db, active_model=REID_MODEL_NAME)
        registry.similarity_threshold = reid_threshold
        registry.recency_window_sec = reid_window
        # Do NOT pre-load track->person from DB. ByteTrack restarts its
        # track_id counter from 1 on every pipeline restart — those numbers
        # carry no meaning across runs. Pre-loading would silently map
        # today's new track_id 1 to yesterday's person_id, skipping OSNet
        # entirely and corrupting unique-customer counts. Cross-restart
        # continuity is OSNet's job (via embedding similarity), not
        # ByteTrack's integer IDs.

    unique_tracks: set[int] = set()
    unique_persons: set[int] = set()
    entries_in = 0
    entries_out = 0
    auto_tagged_employees: set[int] = set()
    frames = 0
    detections = 0
    start = time.time()
    last_live_write_ts = 0.0  # for the live-view JPEG throttle

    # ----- Phase 2.1 re-ID state -----
    # Per-track frame counter for periodic re-embedding.
    track_frames: dict[int, int] = {}
    REID_REEMBED_EVERY = 30  # ~3 sec at 10fps processed

    # Multi-shot first embedding: accumulate quality embeddings until we have
    # enough to average into a stable first fingerprint. Skips bad-crop tracks
    # entirely instead of locking in a doorway-edge half-glimpse.
    track_warmup_embeds: dict[int, list] = {}  # tid -> list of embeddings
    track_warmup_colors: dict[int, list] = {}  # tid -> list of color signatures
    MIN_QUALITY_EMBEDS_FOR_ASSIGN = 3  # average this many before deciding identity

    # Quality crop thresholds — refuse to embed garbage frames.
    QUALITY_MIN_BBOX_HEIGHT = 80         # pixels — anything smaller is too low-detail
    QUALITY_MIN_BBOX_WIDTH = 30          # pixels
    QUALITY_MIN_CONFIDENCE = 0.6         # YOLO confidence floor
    QUALITY_EDGE_MARGIN = 8              # pixels — boxes within this of frame edge are partial views
    QUALITY_MAX_OCCLUSION_IOU = 0.3      # if another detection overlaps more than this, the crop is occluded

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frames += 1

            # FRAME_SKIP: process every Nth frame, discard the rest.
            if frame_skip > 1 and (frames % frame_skip) != 0:
                continue

            # BUSINESS_HOURS: outside the window, just read+discard frames.
            if business_hours is not None:
                import datetime
                hour_now = datetime.datetime.now().hour
                start_h, end_h = business_hours
                in_hours = (start_h <= hour_now < end_h) if start_h < end_h \
                           else (hour_now >= start_h or hour_now < end_h)
                if not in_hours:
                    time.sleep(0.5)
                    continue

            results = model.track(
                frame, persist=True, classes=[PERSON_CLASS_ID],
                conf=min_conf,
                verbose=False, tracker="bytetrack.yaml",
            )
            ts = time.time()
            r = results[0]

            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()

                rows = []
                # Materialize boxes list so we can check occlusion within frame
                box_list = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in boxes]
                for box_idx, ((x1, y1, x2, y2), tid, conf) in enumerate(zip(boxes, ids, confs)):
                    tid = int(tid)
                    unique_tracks.add(tid)
                    rows.append((ts, store_id, camera_id, tid,
                                 float(x1), float(y1), float(x2), float(y2), float(conf)))

                    pid_label = ""
                    if reid_enabled:
                        track_frames[tid] = track_frames.get(tid, 0) + 1
                        is_first_sighting = tid not in track_to_person
                        is_periodic_refresh = (
                            not is_first_sighting
                            and track_frames[tid] % REID_REEMBED_EVERY == 0
                        )

                        if is_first_sighting or is_periodic_refresh:
                            # Quality gate: refuse to embed garbage crops
                            quality_ok, why = is_quality_box(
                                float(x1), float(y1), float(x2), float(y2), float(conf),
                                w, h, box_list, box_idx,
                                QUALITY_MIN_BBOX_HEIGHT, QUALITY_MIN_BBOX_WIDTH,
                                QUALITY_MIN_CONFIDENCE, QUALITY_EDGE_MARGIN,
                                QUALITY_MAX_OCCLUSION_IOU,
                            )
                            if quality_ok:
                                x1i, y1i = max(0, int(x1)), max(0, int(y1))
                                x2i, y2i = min(w, int(x2)), min(h, int(y2))
                                crop = frame[y1i:y2i, x1i:x2i]
                                if crop.size > 0:
                                    emb = embed_crop(crop)
                                    csig = color_signature(crop)

                                    if is_first_sighting:
                                        # Multi-shot first embedding: accumulate
                                        # quality samples until we have enough,
                                        # then average and decide identity.
                                        track_warmup_embeds.setdefault(tid, []).append(emb)
                                        track_warmup_colors.setdefault(tid, []).append(csig)
                                        if len(track_warmup_embeds[tid]) >= MIN_QUALITY_EMBEDS_FOR_ASSIGN:
                                            avg_emb = average_embeddings(track_warmup_embeds[tid])
                                            pid, sim = registry.assign(avg_emb, ts, store_id, camera_id)
                                            track_to_person[tid] = pid
                                            db.execute(
                                                "INSERT OR REPLACE INTO track_persons "
                                                "(store_id, camera_id, track_id, person_id, match_sim, assigned_ts) "
                                                "VALUES (?, ?, ?, ?, ?, ?)",
                                                (store_id, camera_id, tid, pid, sim, ts),
                                            )
                                            persist_person(db, registry.persons[pid])
                                            match_note = (
                                                f"matched sim={sim:.2f}"
                                                if sim is not None else "new"
                                            )
                                            n_samples = len(track_warmup_embeds[tid])
                                            print(f"  [reid] track {tid} -> person {pid} "
                                                  f"({match_note}, multi-shot from {n_samples} crops)")
                                            # Free warmup memory now that we've committed
                                            track_warmup_embeds.pop(tid, None)
                                            track_warmup_colors.pop(tid, None)
                                    else:
                                        # Periodic refresh of an existing assignment
                                        pid = track_to_person[tid]
                                        if pid in registry.persons:
                                            registry.persons[pid].update(emb, ts)
                                            persist_person(db, registry.persons[pid])
                    if reid_enabled:
                        pid = track_to_person.get(tid)
                        if pid is not None:
                            unique_persons.add(pid)
                            pid_label = f"  P{pid}"

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 220, 0), 2)
                    cv2.putText(frame, f"ID {tid}{pid_label}",
                                (int(x1), max(0, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

                    # Line crossing — uses the bbox centroid (feet are more stable
                    # in practice, but centroid works for testing)
                    cx = (float(x1) + float(x2)) / 2
                    cy = (float(y1) + float(y2)) / 2
                    if line_detector is not None:
                        event = line_detector.update_track(tid, cx, cy)
                        if event is not None:
                            pid = track_to_person.get(tid) if reid_enabled else None
                            db.execute(
                                "INSERT INTO entry_events "
                                "(ts, store_id, camera_id, track_id, person_id, direction) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (ts, store_id, camera_id, tid, pid, event.direction),
                            )
                            if event.direction == "in":
                                entries_in += 1
                            else:
                                entries_out += 1
                            print(f"  [entry] track {tid} {event.direction.upper()}"
                                  f"{f' (person {pid})' if pid else ''}")
                    if exclusion_detector is not None:
                        evt = exclusion_detector.update_track(tid, cx, cy)
                        if evt is not None:
                            pid = track_to_person.get(tid) if reid_enabled else None
                            if pid is not None and pid not in auto_tagged_employees:
                                db.execute(
                                    "INSERT OR IGNORE INTO employees "
                                    "(person_id, name, role, enrolled_at, tagged_via) "
                                    "VALUES (?, ?, ?, ?, ?)",
                                    (pid, f"auto ({camera_id})", None, ts, "staff_zone"),
                                )
                                auto_tagged_employees.add(pid)
                                print(f"  [excl] track {tid} crossed staff-zone line "
                                      f"-> person {pid} auto-tagged as employee")

                db.executemany(
                    "INSERT INTO detections (ts, store_id, camera_id, track_id, x1, y1, x2, y2, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                detections += len(rows)

            overlay = f"tracks: {len(unique_tracks)}"
            if reid_enabled:
                overlay += f"  persons: {len(unique_persons)}"
            if line_detector is not None:
                overlay += f"  IN: {entries_in}  OUT: {entries_out}"
                p1 = (int(line_detector.line.p1[0]), int(line_detector.line.p1[1]))
                p2 = (int(line_detector.line.p2[0]), int(line_detector.line.p2[1]))
                cv2.line(frame, p1, p2, (0, 200, 255), 3)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                idx, idy = line_detector.line.inside_dir
                mag = (idx * idx + idy * idy) ** 0.5
                if mag > 0:
                    arrow_end = (int(mid[0] + 40 * idx / mag), int(mid[1] + 40 * idy / mag))
                    cv2.arrowedLine(frame, mid, arrow_end, (0, 200, 255), 2, tipLength=0.4)
            if exclusion_detector is not None:
                overlay += f"  AUTO-EMP: {len(auto_tagged_employees)}"
                p1 = (int(exclusion_detector.line.p1[0]), int(exclusion_detector.line.p1[1]))
                p2 = (int(exclusion_detector.line.p2[0]), int(exclusion_detector.line.p2[1]))
                cv2.line(frame, p1, p2, (50, 50, 220), 3)
                cv2.putText(frame, "STAFF ZONE", (p1[0], p1[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 220), 2)
            cv2.putText(frame, overlay, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)

            # Live view: write the latest annotated frame to disk ~2x/sec.
            # The dashboard polls this file and shows it to Cam so he can
            # watch the YOLO boxes + track_id + P# labels in near-real-time.
            # File is overwritten in place, so disk usage stays trivial.
            now_ts = time.time()
            if now_ts - last_live_write_ts > 0.5:
                try:
                    cv2.imwrite(str(live_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    last_live_write_ts = now_ts
                except Exception as _e:
                    pass  # don't crash the pipeline if disk write hiccups

            if show_preview:
                cv2.imshow("pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames and frames >= max_frames:
                break
            if frames % 50 == 0:
                db.commit()
                msg = f"  frame {frames}  tracks={len(unique_tracks)}  det={detections}"
                if reid_enabled:
                    msg += f"  persons={len(unique_persons)}"
                print(msg)
    finally:
        db.commit()
        db.close()
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    elapsed = time.time() - start
    proc_fps = frames / elapsed if elapsed > 0 else 0

    print()
    print(f"  Frames processed:  {frames}")
    print(f"  Total detections:  {detections}")
    print(f"  Unique track IDs:  {len(unique_tracks)}")
    if reid_enabled:
        print(f"  Unique persons:    {len(unique_persons)}  (cross-camera, cross-time)")
    if line_detector is not None:
        print(f"  Entries IN:        {entries_in}")
        print(f"  Entries OUT:       {entries_out}")
    if exclusion_detector is not None:
        print(f"  Auto-tagged emps:  {len(auto_tagged_employees)}  (crossed staff-zone line)")
    print(f"  Processing speed:  {proc_fps:.1f} fps")
    print(f"  Annotated video:   {OUT_VIDEO}")
    print(f"  Event DB:          {DB_PATH}")
    print()
    print(f"  open {OUT_VIDEO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
