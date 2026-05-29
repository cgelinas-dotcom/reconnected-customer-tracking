# One-paste DESTRUCTIVE recovery for a stuck mini PC.
#
# Same as recover.ps1 PLUS wipes all re-ID / detection data from the local
# events.sqlite. Use this when:
#   (a) the dashboard restart mechanism is broken (so the over-network wipe
#       button can't reach the endpoint), AND
#   (b) you want to start fresh with empty registries.
#
# AnyDesk into the store, open PowerShell as Administrator, paste:
#
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#   iwr -useb https://raw.githubusercontent.com/cgelinas-dotcom/reconnected-customer-tracking/main/scripts/wipe_and_recover.ps1 | iex
#
# What it does:
#   1. Kills every python.exe (clears port 8000, drops zombie pipeline)
#   2. cd into the repo on Desktop
#   3. git fetch + reset --hard origin/main (gets the latest code)
#   4. Wipes events.sqlite — drops all rows from persons / track_persons /
#      detections / entry_events / visits / employees. Keeps settings.
#   5. Installs/refreshes the watchdog scheduled task
#   6. Starts pipeline + dashboard scheduled tasks
#   7. Waits for port 8000 to come up and prints the Tailscale URL
#
# After this script runs, the store is on the latest code AND has zero
# re-ID data — fresh start.

$ErrorActionPreference = "Continue"

function Step($n, $msg) {
    Write-Host ""
    Write-Host "==> [$n] $msg" -ForegroundColor Cyan
}

Step "1/7" "Kill every python.exe (frees port 8000, drops zombie pipeline)"
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  killing PID $($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

Step "2/7" "Locate repo on the Desktop"
$repo = "$env:USERPROFILE\Desktop\customer-tracking"
if (-not (Test-Path $repo)) {
    Write-Host "Repo not found at $repo — run deploy_new_store.ps1 first." -ForegroundColor Red
    exit 1
}
Set-Location $repo
Write-Host "  repo: $repo"

Step "3/7" "git fetch + reset --hard origin/main (hardened env)"
$env:GIT_TERMINAL_PROMPT = "0"
$env:GCM_INTERACTIVE     = "Never"
$env:GCM_GUI_PROMPT      = "false"
$env:GIT_CONFIG_NOSYSTEM = "1"
$askpassPath = Join-Path $env:TEMP ("ct_askpass_" + [guid]::NewGuid().ToString("N") + ".bat")
Set-Content -Path $askpassPath -Value "@exit /b 1" -Encoding ASCII
$env:GIT_ASKPASS = $askpassPath
$gitArgs = @(
    "-c", "credential.helper=",
    "-c", "credential.modalprompt=false",
    "-c", "credential.guiprompt=false"
)
try {
    & git @gitArgs fetch --prune origin 2>&1 | Write-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "git fetch failed (exit $LASTEXITCODE)" -ForegroundColor Red
        Remove-Item $askpassPath -ErrorAction SilentlyContinue
        exit 1
    }
    & git @gitArgs reset --hard origin/main 2>&1 | Write-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "git reset failed (exit $LASTEXITCODE)" -ForegroundColor Red
        Remove-Item $askpassPath -ErrorAction SilentlyContinue
        exit 1
    }
} finally {
    Remove-Item $askpassPath -ErrorAction SilentlyContinue
}
$head = & git rev-parse --short HEAD
Write-Host "  HEAD is now at: $head"

Step "4/7" "WIPE events.sqlite (persons / detections / entry_events / etc)"
$dbPath = "$repo\data\events.sqlite"
if (-not (Test-Path $dbPath)) {
    Write-Host "  No events.sqlite found (already clean or never created)" -ForegroundColor Yellow
} else {
    $pyExe = "$repo\.venv\Scripts\python.exe"
    if (-not (Test-Path $pyExe)) {
        Write-Host "Python venv not found at $pyExe" -ForegroundColor Red
        exit 1
    }
    $wipePy = @"
import sqlite3, sys
db = sqlite3.connect(r'$dbPath')
tables = ['entry_events', 'track_persons', 'visits', 'detections', 'employees', 'persons']
total = 0
for t in tables:
    try:
        n = db.execute(f'DELETE FROM {t}').rowcount
        print(f'  {t}: {n} rows deleted')
        total += n
    except Exception as e:
        print(f'  {t}: skipped ({e})')
db.commit()
db.close()
# VACUUM in its own connection (cannot run inside a transaction)
db2 = sqlite3.connect(r'$dbPath')
db2.execute('VACUUM')
db2.close()
print(f'TOTAL: {total} rows wiped, db vacuumed')
print('settings table preserved (your tuned threshold etc. still applies)')
"@
    & $pyExe -c $wipePy
}

Step "5/7" "Install/refresh CustomerTracking_Watchdog scheduled task"
$watchdogPs = @"
`$listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if (-not `$listen) {
    Start-ScheduledTask -TaskName 'CustomerTracking_Dashboard'
}
`$pipeAlive = Get-Process python -ErrorAction SilentlyContinue
if (-not `$pipeAlive) {
    Start-ScheduledTask -TaskName 'CustomerTracking_Pipeline'
}
"@
$watchdogPath = "$repo\scripts\watchdog_check.ps1"
Set-Content -Path $watchdogPath -Value $watchdogPs -Encoding ASCII
$watchdogAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$watchdogPath`""
$watchdogTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 2) -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName "CustomerTracking_Watchdog" `
    -Action $watchdogAction -Trigger $watchdogTrigger -Principal $principal -Force | Out-Null
Write-Host "  watchdog installed (checks every 2 min)"

Step "6/7" "Restart pipeline + dashboard scheduled tasks"
Start-ScheduledTask -TaskName "CustomerTracking_Pipeline"
Start-ScheduledTask -TaskName "CustomerTracking_Dashboard"

Step "7/7" "Wait for dashboard on port 8000"
$tsIp = (& "C:\Program Files\Tailscale\tailscale.exe" ip -4 2>$null | Select-Object -First 1)
$deadline = (Get-Date).AddSeconds(30)
$up = $false
while ((Get-Date) -lt $deadline) {
    if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
        $up = $true
        break
    }
    Start-Sleep -Seconds 1
}
Write-Host ""
if ($up) {
    Write-Host "DASHBOARD UP (fresh + on latest code)." -ForegroundColor Green
    if ($tsIp) { Write-Host "  http://${tsIp}:8000" }
    Write-Host "  HEAD: $head"
} else {
    Write-Host "Dashboard still not listening on port 8000 after 30s." -ForegroundColor Red
    Write-Host "  Get-ScheduledTaskInfo -TaskName CustomerTracking_Dashboard"
}
