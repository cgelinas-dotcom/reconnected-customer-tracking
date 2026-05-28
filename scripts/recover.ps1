# One-paste recovery for a stuck mini PC.
#
# AnyDesk into the store, open PowerShell as Administrator, paste this:
#
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#   iwr -useb https://raw.githubusercontent.com/cgelinas-dotcom/reconnected-customer-tracking/main/scripts/recover.ps1 | iex
#
# What it does:
#   1. Kills every python.exe (clears port 8000 + any zombie pipeline)
#   2. cd into the repo on the Desktop
#   3. git fetch + git reset --hard origin/main, with the same hardened env
#      the dashboard uses (so this works even if the local git config has a
#      broken credential helper)
#   4. Installs/refreshes a CustomerTracking_Watchdog scheduled task that
#      checks every 2 min and restarts the dashboard if port 8000 is dead
#   5. Restarts the pipeline + dashboard scheduled tasks
#   6. Waits for port 8000 to come up and prints the Tailscale URL

$ErrorActionPreference = "Continue"

function Step($n, $msg) {
    Write-Host ""
    Write-Host "==> [$n] $msg" -ForegroundColor Cyan
}

Step "1/6" "Kill every python.exe (frees port 8000, drops zombie pipeline)"
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  killing PID $($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

Step "2/6" "Locate repo on the Desktop"
$repo = "$env:USERPROFILE\Desktop\customer-tracking"
if (-not (Test-Path $repo)) {
    Write-Host "Repo not found at $repo — run deploy_new_store.ps1 first." -ForegroundColor Red
    exit 1
}
Set-Location $repo
Write-Host "  repo: $repo"

Step "3/6" "git fetch + reset --hard origin/main (hardened env)"
$env:GIT_TERMINAL_PROMPT = "0"
$env:GCM_INTERACTIVE     = "Never"
$env:GCM_GUI_PROMPT      = "false"
$env:GIT_CONFIG_NOSYSTEM = "1"
# No-op askpass on disk: fails fast on any credential prompt instead of hanging
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
        Write-Host "git fetch failed (exit $LASTEXITCODE) — check network / origin URL" -ForegroundColor Red
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
    Remove-Item $askpass.FullName -ErrorAction SilentlyContinue
}
$head = & git rev-parse --short HEAD
Write-Host "  HEAD is now at: $head"

Step "4/6" "Install/refresh CustomerTracking_Watchdog scheduled task"
# Watchdog body: if port 8000 isn't listening, run the dashboard task.
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

Step "5/6" "Restart pipeline + dashboard scheduled tasks"
Start-ScheduledTask -TaskName "CustomerTracking_Pipeline"
Start-ScheduledTask -TaskName "CustomerTracking_Dashboard"

Step "6/6" "Wait for dashboard on port 8000"
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
    Write-Host "DASHBOARD UP." -ForegroundColor Green
    if ($tsIp) { Write-Host "  http://${tsIp}:8000" }
    Write-Host "  HEAD: $head"
} else {
    Write-Host "Dashboard still not listening on port 8000 after 30s." -ForegroundColor Red
    Write-Host "Check the dashboard scheduled task's last result code:" -ForegroundColor Yellow
    Write-Host "  Get-ScheduledTaskInfo -TaskName CustomerTracking_Dashboard"
}
