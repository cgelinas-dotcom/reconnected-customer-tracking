# Customer Tracking — Local Build

Computer-vision system that counts **unique customers** across 8 retail stores using existing Lorex 4K cameras.

Built locally first. Deploys to the main website (reconnected.info) once the pipeline produces reliable numbers at one store.

---

## Where we are

**Phase 0 — Dev environment + one-camera proof.** Get a single RTSP stream from one store's NVR pulling into this Mac. That proves the foundation works. Everything else is built on top.

## The roadmap (from your 11-step plan)

| Phase | What it does | Status |
|---|---|---|
| 0 | Pull one camera stream to this Mac | in progress |
| 1 | Detect + track people on that one feed | not started |
| 2 | Filter out employees | needs decision (see [DECISIONS.md](docs/DECISIONS.md)) |
| 3 | Re-identify same person across cameras + time | not started |
| 4 | Define "unique customer" rules | needs decision (see [DECISIONS.md](docs/DECISIONS.md)) |
| 5 | Local dashboard (counts, charts) | not started |
| 6 | Validate accuracy vs manual count at one store | not started |
| 7 | Networking so central box reaches all 8 stores | not started |
| 8 | Roll out + port to website | not started |

### Bonus: multi-source supervisor

`scripts/run_multi.py` reads [config/stores.yaml](config/stores.example.yaml) and launches one pipeline subprocess per enabled camera, restarting any that crash. This is how you scale beyond one camera in one terminal — it's the "actually deploy this" entry point.

```
cp config/stores.example.yaml config/stores.yaml
# edit stores.yaml, enable the cameras you want
python scripts/run_multi.py
```

Logs land in `data/logs/<store>_<camera>.log`. Tail with `tail -F data/logs/*.log`.

## What you do next — three things

1. **Install the dev tools.** Open Terminal and follow [docs/SETUP.md](docs/SETUP.md). Takes 10–15 minutes.
2. **Get one camera's RTSP URL.** Pick whichever store is easiest to access (probably the one this Mac sits at). Instructions in [docs/CAMERA_URL.md](docs/CAMERA_URL.md).
3. **Run the test script.** `python scripts/test_stream.py <your-rtsp-url>` — it'll save a snapshot and tell you the frame rate. If that works, we're past Phase 0 and I can build the rest.

## Project layout

```
customer-tracking/
├── config/         # store + camera config, settings
├── src/
│   ├── ingest/     # pull RTSP streams from NVRs
│   ├── detect/     # YOLO person detection
│   ├── track/      # frame-to-frame tracking
│   ├── reid/       # cross-camera person re-identification
│   ├── employees/  # employee filter
│   ├── events/     # SQLite event log
│   └── api/        # FastAPI backend for dashboard
├── dashboard/      # local web UI
├── data/           # SQLite db, snapshots, embeddings
├── scripts/        # runnable utilities (test_stream, enroll, etc.)
└── docs/           # SETUP, CAMERA_URL, DECISIONS
```

## Stack

- **Python 3.11+** — vision/ML libraries
- **OpenCV + ffmpeg** — pull RTSP, decode video
- **Ultralytics YOLO** — person detection + tracking (ByteTrack built in)
- **SQLite** — local event store (will become D1 / Postgres on production)
- **FastAPI** — backend
- **plain HTML/JS** — dashboard (keep it simple, port to real stack on rollout)
