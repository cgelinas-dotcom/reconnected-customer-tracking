# Deploying to a Windows mini-PC at a store

Use this for each store's mini PC. ~30 minutes per store the first time, then ~5 min per store after you've done one.

Prerequisites: AnyDesk access to the mini PC. Your Lorex NVR's local IP for that store. NVR admin user/password.

---

## 1. Open PowerShell as Administrator on the mini PC

Right-click the Start button → **Windows PowerShell (Admin)** or **Terminal (Admin)**.

## 2. Run the bootstrap script (installs Python, ffmpeg, Tailscale, Git, Chocolatey)

Paste this single line and hit Enter:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')); choco install -y python311 ffmpeg git tailscale
```

What it does:
- Installs **Chocolatey** (Windows package manager) if missing
- Installs **Python 3.11**, **ffmpeg**, **git**, and **Tailscale** in one go

Takes ~5-10 minutes. You'll see lots of output. Wait for the prompt to come back.

Then close PowerShell completely and re-open it as Admin (so the new PATH takes effect).

## 3. Sign into Tailscale

```powershell
tailscale up
```

It'll print a URL — open it in a browser, log in with the same Tailscale account you use on your Mac. Once logged in, this mini PC joins your Tailnet. (Your Mac and the mini PC can now reach each other.)

Verify:
```powershell
tailscale ip -4
```

Note the IP it prints (something like `100.64.5.20`). That's how you'll reach this mini PC's dashboard from your Mac later.

## 4. Get the project onto the mini PC

For the first store, the simplest is to zip-transfer over AnyDesk:

**On your Mac:**
```bash
cd ~/Desktop
tar --exclude='.venv' --exclude='data/events.sqlite' --exclude='data/out_annotated.mp4' --exclude='data/logs' -czf customer-tracking.tar.gz customer-tracking
```

**Use AnyDesk → File Transfer** to send `customer-tracking.tar.gz` to the mini PC's Desktop.

**On the mini PC PowerShell:**
```powershell
cd $env:USERPROFILE\Desktop
tar -xzf customer-tracking.tar.gz
cd customer-tracking
```

(Windows 10+ ships with `tar` so this works.)

For ongoing updates after the first deployment, switch to `git clone` from a private GitHub repo — much smoother for the remaining 7 stores.

## 5. Set up the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

The `pip install` takes ~5 minutes (PyTorch is big). When done, verify:
```powershell
python -c "import cv2, ultralytics, boxmot; print('OK')"
```

## 6. Configure for this store

Copy the example config and edit it:

```powershell
copy config\stores.example.yaml config\stores.yaml
notepad config\stores.yaml
```

In stores.yaml, fill in this store's section:

```yaml
stores:
  - id: anthem
    name: "Anthem"
    enabled: true
    nvr:
      host: "192.168.0.153"      # the local IP of this store's NVR
      port: 554
      username: "admin"           # NVR admin user
      password: "your-nvr-password"
    cameras:
      - name: "entrance"
        channel: 1                # the channel showing the front door
        stream: "sub"
        enabled: true
        # Add entry_line later once you've picked the coords (see step 8)
```

Save and close Notepad.

## 7. Test one camera connection

```powershell
python scripts\test_stream.py "rtsp://admin:your-password@192.168.0.153:554/cam/realmonitor?channel=1&subtype=1"
```

Should print frame dimensions + `SUCCESS. Phase 0 complete.` If it fails: confirm the NVR's IP, your admin password, and that the mini PC is on the same LAN as the NVR.

## 8. Pick the entry-line coordinates

```powershell
python scripts\pick_entry_line.py "rtsp://admin:your-password@192.168.0.153:554/cam/realmonitor?channel=1&subtype=1"
```

A window opens showing the entrance camera. Click 2 points across the doorway + 1 point on the inside-the-store side. The script prints an `ENTRY_LINE=...` string — copy it. Paste into `config\stores.yaml` under that camera as the `entry_line:` value.

If there's a staff-zone boundary visible in this camera, run the same script again, pick those coords, paste as `exclusion_line:`.

## 9. Run the pipeline — with low CPU impact

```powershell
$env:FRAME_SKIP=3
$env:BUSINESS_HOURS="9-21"
python scripts\run_multi.py
```

`FRAME_SKIP=3` processes every 3rd frame (~10fps from the camera's 30fps), cutting CPU usage to roughly a third. People walk slowly relative to that — tracking still works fine. `BUSINESS_HOURS=9-21` makes the pipeline skip detection between 9pm and 9am (adjust to your hours; e.g. `10-22` for 10am–10pm).

The supervisor also automatically runs camera processes at **BELOW NORMAL priority** so your employees' apps always win the CPU when they're actively using the PC. The pipeline only uses leftover capacity.

Net expected impact on a 12th-gen i5 with 12GB RAM running one camera: **~5–15% of one core average**, briefly higher during foot-traffic spikes. Imperceptible to someone using the PC for normal POS / browsing / paperwork.

Leave this PowerShell window running (or set up the scheduled task in step 12 to auto-start on boot).

## 10. Run the dashboard

Open a SECOND PowerShell window (Admin not needed this time):

```powershell
cd $env:USERPROFILE\Desktop\customer-tracking
.\.venv\Scripts\Activate.ps1
python scripts\run_dashboard.py
```

## 11. Access the dashboard from your Mac

From your Mac browser:

```
http://<tailscale-ip-of-mini-pc>:8000
```

(That's the IP you noted in step 3.)

You can also bookmark it. You now have a live dashboard for the Anthem store accessible from anywhere via Tailscale.

## 12. Make it auto-start on boot

The pipeline + dashboard need to run continuously. Set them as Windows scheduled tasks so they start automatically:

```powershell
$root = "$env:USERPROFILE\Desktop\customer-tracking"
$pyExe = "$root\.venv\Scripts\python.exe"

# Pipeline
schtasks /Create /F /SC ONSTART /RU SYSTEM /TN "CustomerTracking_Pipeline" `
  /TR "$pyExe $root\scripts\run_multi.py" /RL HIGHEST

# Dashboard
schtasks /Create /F /SC ONSTART /RU SYSTEM /TN "CustomerTracking_Dashboard" `
  /TR "$pyExe $root\scripts\run_dashboard.py" /RL HIGHEST
```

## 13. Disable sleep/hibernate

In PowerShell:

```powershell
# Never sleep on AC power
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
# Disable hibernation entirely
powercfg /hibernate off
```

The mini PC will stay awake processing camera feeds 24/7.

---

## You're done with this store

Things that should be true now:
- `tailscale status` shows this mini PC connected
- Pipeline subprocesses are running (visible in Task Manager as multiple `python.exe`)
- `http://<tailscale-ip>:8000` from your Mac shows the dashboard with live stats
- The "Today" panel fills in as people walk through

## For the next 7 stores

Most of the work is one-time. Repeat steps 1–3 + 5 + 7 + 8 + 9 + 10 + 12 + 13. Step 4 becomes `git clone` once you've pushed the repo. Step 6 just adds another entry to the same stores.yaml (or you keep separate stores.yaml per mini PC — your call).

## Troubleshooting

**`python` command not found after install** — close and re-open PowerShell so the new PATH takes effect.

**`pip install` hangs or errors** — wait it out; PyTorch downloads ~2GB. If it actually errors, paste me the last 20 lines.

**`test_stream.py` connection refused** — the mini PC can't reach the NVR. Confirm both are on the same subnet (`192.168.0.x`). Try `ping 192.168.0.153` from the mini PC.

**`tailscale up` doesn't show a URL** — try `tailscale up --hostname=anthem-pc` to give it a memorable name.

**Dashboard shows no data** — check the pipeline PowerShell window for errors. Check `data\events.sqlite` exists. Check `data\logs\*.log` for per-camera errors.
