# Getting an RTSP URL from a Lorex NVR

The NVR streams video over a protocol called RTSP. Once we have one working URL we can pull video into our scripts.

## What you need

- The NVR's local IP address
- An admin username + password
- The channel number of the camera you want (1, 2, 3, ...)

## Find the NVR's IP

Easiest way: open the **Lorex Cloud** or **Lorex Home** app on your phone, go to the NVR's settings, look for "device info" or "network." You'll see an IP like `192.168.1.108`.

Or from the NVR's monitor: Main Menu → System → Network → TCP/IP.

## Build the URL

Lorex uses one of two URL formats depending on firmware. Try both — one will work.

**Format A (most common):**
```
rtsp://USER:PASS@IP:554/cam/realmonitor?channel=N&subtype=0
```

**Format B (newer firmware):**
```
rtsp://USER:PASS@IP:554/Streaming/Channels/N01
```

Where:
- `USER` — admin username
- `PASS` — admin password
- `IP` — NVR's local IP
- `N` — camera channel number (1, 2, ...)
- `subtype=0` is the high-res main stream; `subtype=1` is a lower-res sub-stream (good for testing — less CPU)

**Example:**
```
rtsp://admin:MyPass123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1
```

## Same network as this Mac?

For the very first test, easiest is to use a camera at the store this Mac is physically at. The Mac and the NVR need to be on the same LAN so the local IP is reachable.

If the NVR is at a remote store, we'll add **Tailscale** in Phase 7 — but skip that for now. Pick the closest store first.

## Test the URL works *before* running my script

VLC is the quickest sanity check.

1. Install: `brew install --cask vlc` (or download from videolan.org)
2. Open VLC → File → Open Network → paste the URL → Open
3. You should see live video within a few seconds

If VLC can play it, our Python script will too.

If VLC can't play it: wrong URL format, wrong credentials, NVR not reachable on the network, or RTSP disabled on the NVR (check NVR settings → Network → Advanced → RTSP, enable port 554).

## Once you have a working URL

Paste it into the test command:

```bash
python scripts/test_stream.py "rtsp://admin:MyPass123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
```

Quotes around the URL are important (the `?` and `&` confuse the shell otherwise).
