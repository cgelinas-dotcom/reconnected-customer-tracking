"""
Phase 0 sanity check: can we read video frames?

Usage:
    python scripts/test_stream.py <source>

Where <source> is one of:
  - An RTSP URL:    "rtsp://user:pass@ip:554/..."
  - A webcam index: 0  (built-in Mac camera) or 1, 2, ...
  - A file path:    data/samples/people-detection.mp4

Reads ~10 seconds, saves the first frame as data/test_snapshot.jpg,
prints the measured frame rate.
"""

import sys
import time
from pathlib import Path

import cv2

DURATION_SEC = 10
SNAPSHOT_PATH = Path(__file__).parent.parent / "data" / "test_snapshot.jpg"


def parse_source(arg: str):
    if arg.isdigit():
        return int(arg), "webcam"
    if arg.startswith(("rtsp://", "http://", "https://")):
        return arg, "stream"
    return arg, "file"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    source, kind = parse_source(sys.argv[1])
    print(f"Source ({kind}): {source if kind != 'stream' else source.split('@')[-1]}")

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if kind == "stream" else cv2.VideoCapture(source)
    if not cap.isOpened():
        print("FAILED to open source.")
        if kind == "stream":
            print("  - Confirm URL plays in VLC first")
            print("  - Check NVR settings: Network -> RTSP enabled on port 554")
        elif kind == "webcam":
            print("  - macOS will prompt for camera permission the first time. Allow it and retry.")
            print("  - Try a different index (0, 1, 2)")
        else:
            print(f"  - Check the file exists: {source}")
        return 2

    print(f"Connected. Reading for ~{DURATION_SEC}s...")
    start = time.time()
    frames = 0
    first_frame_saved = False

    while time.time() - start < DURATION_SEC:
        ok, frame = cap.read()
        if not ok:
            if kind == "file":
                break
            time.sleep(0.05)
            continue

        frames += 1
        if not first_frame_saved:
            SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(SNAPSHOT_PATH), frame)
            h, w = frame.shape[:2]
            print(f"  First frame: {w}x{h}, saved -> {SNAPSHOT_PATH}")
            first_frame_saved = True

    cap.release()
    elapsed = time.time() - start
    fps = frames / elapsed if elapsed > 0 else 0

    print()
    print(f"  Frames read:  {frames}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  Measured FPS: {fps:.1f}")

    if frames == 0:
        print()
        print("FAILED: source opened but no frames received.")
        return 3

    print()
    print("SUCCESS. Phase 0 complete.")
    print(f"  Open {SNAPSHOT_PATH} to verify the image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
