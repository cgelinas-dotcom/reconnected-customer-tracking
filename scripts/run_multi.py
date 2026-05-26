"""
Multi-source pipeline supervisor.

Reads config/stores.yaml, spawns one run_pipeline.py subprocess per enabled
camera, captures each one's logs to data/logs/<store>_<camera>.log,
restarts crashed children automatically, and shuts everything down cleanly
on Ctrl+C.

This is what you run for an actual deployment — one supervisor per
processing box, scaling up to all 8 stores.

Usage:
    cp config/stores.example.yaml config/stores.yaml
    # edit stores.yaml to fill in real RTSP URLs (or use "source: sample" for the test video)
    python scripts/run_multi.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "stores.yaml"
EXAMPLE_PATH = ROOT / "config" / "stores.example.yaml"
SAMPLE_VIDEO = ROOT / "data" / "samples" / "people-detection.mp4"
LOGS_DIR = ROOT / "data" / "logs"
PIPELINE_SCRIPT = ROOT / "scripts" / "run_pipeline.py"
RESTART_DELAY_SEC = 5


def build_url(camera_cfg: dict, store_cfg: dict) -> str:
    """Build an RTSP URL from NVR + channel, or use a special source like 'sample' / webcam index / raw URL."""
    src = camera_cfg.get("source")
    if src == "sample":
        return str(SAMPLE_VIDEO)
    if isinstance(src, str) and (src.startswith("rtsp://") or src.startswith("http")):
        return src
    if isinstance(src, str) and src.isdigit():
        return src
    if isinstance(src, int):
        return str(src)

    nvr = store_cfg.get("nvr", {})
    host = nvr.get("host")
    if not host:
        raise ValueError(f"Camera {camera_cfg.get('name')} has no source and store has no nvr.host")
    user = nvr.get("username", "admin")
    pw = nvr.get("password", "")
    port = nvr.get("port", 554)
    channel = camera_cfg.get("channel")
    subtype = 1 if camera_cfg.get("stream", "sub") == "sub" else 0
    return f"rtsp://{user}:{pw}@{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"


def load_config() -> dict | None:
    if not CONFIG_PATH.exists():
        print(f"No {CONFIG_PATH}.")
        print(f"  Copy: cp {EXAMPLE_PATH} {CONFIG_PATH}")
        print(f"  Then edit it to enable the cameras you want to run.")
        return None
    return yaml.safe_load(CONFIG_PATH.read_text())


class Child:
    def __init__(self, store_id: str, store_cfg: dict, camera_cfg: dict):
        self.store_id = store_id
        self.store_cfg = store_cfg
        self.camera_cfg = camera_cfg
        self.camera_id = camera_cfg.get("name") or f"ch{camera_cfg.get('channel', '?')}"
        self.key = f"{store_id}/{self.camera_id}"
        self.url = build_url(camera_cfg, store_cfg)
        self.proc: subprocess.Popen | None = None
        self.log_fh = None
        self.log_path = LOGS_DIR / f"{store_id}_{self.camera_id}.log"

    def start(self) -> None:
        self.log_fh = open(self.log_path, "a", buffering=1)
        env = os.environ.copy()
        env["STORE_ID"] = self.store_id
        env["CAMERA_ID"] = self.camera_id
        env["REID"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        if self.camera_cfg.get("entry_line"):
            env["ENTRY_LINE"] = str(self.camera_cfg["entry_line"])
        if self.camera_cfg.get("exclusion_line"):
            env["EXCLUSION_LINE"] = str(self.camera_cfg["exclusion_line"])
        # Pull-through any CPU-impact env vars from the supervisor.
        for k in ("FRAME_SKIP", "BUSINESS_HOURS"):
            if os.environ.get(k):
                env[k] = os.environ[k]

        # Run pipelines at BELOW NORMAL priority so employees' apps always
        # win the CPU when they need it. Pipeline gets leftovers.
        popen_kwargs: dict = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        else:
            popen_kwargs["preexec_fn"] = lambda: os.nice(10)

        self.proc = subprocess.Popen(
            [sys.executable, str(PIPELINE_SCRIPT), self.url],
            env=env,
            stdout=self.log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            **popen_kwargs,
        )
        printable_url = self.url if "@" not in self.url else self.url.split("@", 1)[-1]
        print(f"[supervisor] started {self.key} pid={self.proc.pid} src={printable_url} log={self.log_path}")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log_fh:
            try:
                self.log_fh.close()
            except Exception:
                pass


def main() -> int:
    cfg = load_config()
    if not cfg:
        return 1
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    children: list[Child] = []
    for store in cfg.get("stores", []):
        if not store.get("enabled", True):
            continue
        for camera in store.get("cameras", []):
            if not camera.get("enabled", True):
                continue
            children.append(Child(store["id"], store, camera))

    if not children:
        print("[supervisor] no enabled cameras in config. Nothing to do.")
        return 0

    for c in children:
        c.start()

    print(f"[supervisor] {len(children)} camera process(es) running. Tail logs with:")
    print(f"  tail -F data/logs/*.log")
    print(f"Ctrl+C to stop all.")

    shutting_down = [False]

    def shutdown(signum, frame):
        if shutting_down[0]:
            return
        shutting_down[0] = True
        print(f"\n[supervisor] caught signal {signum}, terminating children...")
        for c in children:
            c.stop()
        print("[supervisor] all stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(2)
        for c in children:
            if c.is_alive():
                continue
            code = c.proc.returncode if c.proc else None
            print(f"[supervisor] {c.key} exited (code={code}), restarting in {RESTART_DELAY_SEC}s")
            try:
                c.log_fh and c.log_fh.close()
            except Exception:
                pass
            time.sleep(RESTART_DELAY_SEC)
            c.start()


if __name__ == "__main__":
    sys.exit(main())
