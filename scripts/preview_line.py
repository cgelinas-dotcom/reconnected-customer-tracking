"""
Grab one fresh frame from a camera and draw the configured entry line on it.
Quick way to verify whether the line is positioned where customers actually
cross, without depending on the in-progress annotated mp4.

Usage:
    python scripts/preview_line.py <rtsp-url> "<entry_line_string>"

Example:
    python scripts/preview_line.py "rtsp://..." "1741,905,2210,909,-24,97"

Output: data/line_preview.jpg
"""

import sys
from pathlib import Path

import cv2


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    url, line_spec = sys.argv[1], sys.argv[2]
    parts = [int(x) for x in line_spec.split(",")]
    if len(parts) != 6:
        print("entry line must be 6 numbers: x1,y1,x2,y2,inside_dx,inside_dy")
        return 2
    x1, y1, x2, y2, idx, idy = parts

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("Could not read frame from source")
        return 3

    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 12)
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    mag = (idx * idx + idy * idy) ** 0.5
    if mag > 0:
        ax = int(mx + 250 * idx / mag)
        ay = int(my + 250 * idy / mag)
        cv2.arrowedLine(frame, (mx, my), (ax, ay), (0, 200, 255), 8, tipLength=0.3)
        cv2.putText(frame, "INSIDE", (ax + 20, ay),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 200, 255), 6)

    out = Path("data") / "line_preview.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
