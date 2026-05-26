"""
Launch the local dashboard at http://localhost:8000

Usage:
    python scripts/run_dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    import os
    # Bind to all interfaces by default so the dashboard is reachable over
    # Tailscale (or LAN) from other machines. Override with DASHBOARD_HOST
    # if you want to lock it down to localhost only.
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("DASHBOARD_PORT", "8000"))
    print(f"Dashboard starting on {host}:{port}")
    print(f"  Local:     http://localhost:{port}")
    print(f"  Tailscale: http://<this-machine's-tailscale-ip>:{port}")
    print("Ctrl+C to stop.")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)
