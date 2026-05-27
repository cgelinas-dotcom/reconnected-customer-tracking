# One-shot setup script for a new store's mini PC.
# AnyDesk into the mini PC, open PowerShell as Administrator, then:
#
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#   iwr -useb https://raw.githubusercontent.com/cgelinas-dotcom/reconnected-customer-tracking/main/scripts/deploy_new_store.ps1 | iex
#
# What this does:
#   1. Installs Chocolatey + Python 3.11 + ffmpeg + git + Tailscale + GitHub CLI
#   2. Clones the customer-tracking repo
#   3. Creates the Python venv and installs all dependencies
#   4. Prompts you for the store's NVR info and creates stores.yaml
#   5. Walks you through picking the entry line
#   6. Creates Windows scheduled tasks to auto-start the pipeline + dashboard on boot
#   7. Disables sleep/hibernate
#
# After this script finishes you'll need to manually:
#   - Sign into Tailscale (the script will pop up the auth URL)
#   - Sign into GitHub CLI (gh auth login)
#   - Refresh the dashboard on your Mac to confirm it's live

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

Step "1/8  Install Chocolatey + Python 3.11 + ffmpeg + git + Tailscale + GitHub CLI"
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}
choco install -y python311 ffmpeg git tailscale gh
# Refresh PATH for this session so freshly-installed tools work without restart
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User") + ";C:\Python311;C:\Python311\Scripts"

Step "2/8  Tailscale: sign in (browser will open)"
& "C:\Program Files\Tailscale\tailscale.exe" up
Read-Host "Press Enter once you've signed into Tailscale in the browser and seen the success page"

Step "3/8  GitHub CLI: sign in"
Write-Host "When prompted: GitHub.com -> HTTPS -> Y -> Login with web browser"
gh auth login

Step "4/8  Clone the repo to the Desktop"
cd "$env:USERPROFILE\Desktop"
if (Test-Path "customer-tracking") {
    Write-Host "customer-tracking folder already exists; pulling latest instead"
    cd customer-tracking
    git pull
} else {
    gh repo clone cgelinas-dotcom/reconnected-customer-tracking customer-tracking
    cd customer-tracking
}

Step "5/8  Python venv + install dependencies (takes ~5 min)"
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -c "import cv2, ultralytics, boxmot; print('Python deps OK')"

Step "6/8  Configure this store"
function Read-Required($prompt) {
    while ($true) {
        $v = Read-Host $prompt
        if ($v -ne "") { return $v }
        Write-Host "  ! value cannot be empty, try again" -ForegroundColor Yellow
    }
}
$storeId  = Read-Required "Store id (e.g. anthem, bullhead, prescott)"
$nvrHost  = Read-Required "NVR local IP (e.g. 192.168.0.153)"
$nvrUser  = Read-Required "NVR admin USERNAME (usually 'admin', sometimes a custom name like 'camerongelinas')"
$nvrPass  = Read-Required "NVR admin password"
$channel  = Read-Required "Channel number of the FRONT-ENTRANCE camera (e.g. 2)"

& .\.venv\Scripts\python.exe -c @"
import yaml
cfg = {'stores': [{'id': '$storeId', 'name': '$storeId', 'enabled': True,
  'nvr': {'host': '$nvrHost', 'port': 554, 'username': '$nvrUser', 'password': '$nvrPass'},
  'cameras': [{'name': 'front', 'channel': int('$channel'), 'stream': 'main', 'enabled': True}]}]}
open('config/stores.yaml', 'w').write(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))
print('wrote config/stores.yaml')
"@

$rtsp = "rtsp://${nvrUser}:${nvrPass}@${nvrHost}:554/cam/realmonitor?channel=${channel}&subtype=0"

Step "6.5  Verify the camera connection before going further"
& .\.venv\Scripts\python.exe scripts\test_stream.py $rtsp
if ($LASTEXITCODE -ne 0) {
    Write-Host "Could not connect to camera. Check the IP / username / password / channel and re-run this script." -ForegroundColor Red
    exit 1
}

Step "7/8  Pick the entry line (a window will open)"
Write-Host "Click 2 points across the doorway threshold + 1 point on the inside-the-store side."
& .\.venv\Scripts\python.exe scripts\pick_entry_line.py $rtsp

$entryLine = Read-Host "Paste the ENTRY_LINE= value (just the numbers, e.g. 1300,500,2700,500,0,100)"
& .\.venv\Scripts\python.exe -c @"
import yaml
cfg = yaml.safe_load(open('config/stores.yaml'))
cfg['stores'][0]['cameras'][0]['entry_line'] = '$entryLine'
open('config/stores.yaml', 'w').write(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))
print('entry_line saved')
"@

Step "8/8  Scheduled tasks (auto-start on boot) + disable sleep"
$root = "$env:USERPROFILE\Desktop\customer-tracking"
$pyExe = "$root\.venv\Scripts\python.exe"
schtasks /Create /F /SC ONSTART /RU SYSTEM /TN "CustomerTracking_Pipeline" /TR "cmd /c set FRAME_SKIP=3 && set BUSINESS_HOURS=9-21 && `"$pyExe`" `"$root\scripts\run_multi.py`"" /RL HIGHEST
schtasks /Create /F /SC ONSTART /RU SYSTEM /TN "CustomerTracking_Dashboard" /TR "`"$pyExe`" `"$root\scripts\run_dashboard.py`"" /RL HIGHEST
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /hibernate off
schtasks /Run /TN "CustomerTracking_Pipeline"
schtasks /Run /TN "CustomerTracking_Dashboard"

Write-Host ""
Write-Host "DONE." -ForegroundColor Green
$tailscaleIp = & "C:\Program Files\Tailscale\tailscale.exe" ip -4
Write-Host "Dashboard for this store:  http://${tailscaleIp}:8000"
Write-Host "Rename this device in https://login.tailscale.com/admin/machines to '${storeId}-minipc' for clarity."
