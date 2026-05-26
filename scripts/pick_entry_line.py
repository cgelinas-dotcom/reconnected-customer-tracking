"""
Interactive helper to define the entry line for a camera.

Pulls one frame from the source you give it, opens a window. Click two
points to define the line across the doorway, then click a third point
on the side of the line that is INSIDE the store. The script prints the
ENTRY_LINE env var string you should paste into your run command (or into
config/stores.yaml).

Usage:
    python scripts/pick_entry_line.py <source>

    <source>: same as run_pipeline.py — RTSP URL, webcam index, or file path.
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent


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
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if kind == "stream" else cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return 2

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("Could not read a frame from source.")
        return 3

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # Scale the displayed image to fit comfortably on screen while still
    # recording click coordinates at the ORIGINAL frame resolution (so the
    # ENTRY_LINE values match what the pipeline sees).
    MAX_DISPLAY_W = 1280
    MAX_DISPLAY_H = 720
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    if scale < 1.0:
        print(f"Display scaled to {disp_w}x{disp_h} (scale={scale:.3f}); click coords reported in original {w}x{h}")

    print("Click two points to draw the entry line across the doorway.")
    print("Then click a third point ON THE INSIDE SIDE of the store.")
    print("Press 'r' to restart, 'q' to quit.")

    state = {"points": [], "frame_orig": frame.copy()}

    def redraw():
        f = state["frame_orig"].copy()
        # draw in original resolution
        pts = state["points"]
        for p in pts:
            cv2.circle(f, p, max(6, int(8 / scale)), (0, 200, 255), -1)
        if len(pts) >= 2:
            cv2.line(f, pts[0], pts[1], (0, 200, 255), max(2, int(3 / scale)))
        if len(pts) == 3:
            cv2.circle(f, pts[2], max(8, int(10 / scale)), (50, 220, 50), 2)
            cv2.putText(f, "INSIDE", (pts[2][0] + 12, pts[2][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2 / scale if scale < 1 else 0.7,
                        (50, 220, 50), max(2, int(3 / scale)))
        # show the downscaled version
        if scale < 1.0:
            f = cv2.resize(f, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        cv2.imshow("pick entry line", f)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < 3:
            # rescale clicked display coords back to original frame coords
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            state["points"].append((orig_x, orig_y))
            redraw()

    cv2.namedWindow("pick entry line")
    cv2.setMouseCallback("pick entry line", on_click)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return 0
        if key == ord("r"):
            state["points"] = []
            redraw()
            continue
        if len(state["points"]) == 3:
            break

    cv2.destroyAllWindows()

    (x1, y1), (x2, y2), (ix, iy) = state["points"]
    # inside_dir = vector from line midpoint to the "inside" click
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = ix - mx, iy - my

    spec = f"{x1},{y1},{x2},{y2},{dx:.0f},{dy:.0f}"
    print()
    print("=" * 60)
    print(f"ENTRY_LINE={spec}")
    print("=" * 60)
    print()
    print("Paste this into your run command, e.g.:")
    print(f'  ENTRY_LINE="{spec}" REID=1 STORE_ID=store-01 CAMERA_ID=entrance \\')
    print(f"    python scripts/run_pipeline.py <source>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
