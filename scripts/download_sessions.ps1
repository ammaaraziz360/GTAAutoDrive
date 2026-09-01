# Download session folders from remote server to local machine (Windows/PowerShell)
# Usage: .\download_sessions.ps1 -RemoteHost "user@server" -RemotePath "/path/to/frames" -LocalPath ".\data\frames"

param(
    [string]$RemoteHost = "user@your-server.com",
    [string]$RemotePath = "/path/to/remote/data/frames",
    [string]$LocalPath = ".\data\frames"
)

# Create local directory if it doesn't exist
if (-not (Test-Path $LocalPath)) {
    New-Item -ItemType Directory -Path $LocalPath -Force | Out-Null
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Downloading sessions from remote server"
Write-Host "Remote: ${RemoteHost}:${RemotePath}"
Write-Host "Local:  $LocalPath"
Write-Host "============================================" -ForegroundColor Cyan

# Get list of session folders from remote
Write-Host "Fetching session list..."
$sessions = ssh $RemoteHost "ls -d $RemotePath/session-*/ 2>/dev/null | xargs -n1 basename"

if (-not $sessions) {
    Write-Host "No session folders found at $RemotePath" -ForegroundColor Red
    exit 1
}

$sessionList = $sessions -split "`n" | Where-Object { $_ -ne "" }
$total = $sessionList.Count
$count = 0

Write-Host "Found $total session(s)`n"

foreach ($session in $sessionList) {
    $count++
    $remoteSession = "$RemotePath/$session"
    $localSession = Join-Path $LocalPath $session
    
    # Check if already downloaded
    if (Test-Path $localSession) {
        $localCount = (Get-ChildItem "$localSession\*.jpg" -ErrorAction SilentlyContinue | Measure-Object).Count
        $remoteCount = [int](ssh $RemoteHost "ls -1 $remoteSession/*.jpg 2>/dev/null | wc -l")
        
        if ($localCount -eq $remoteCount -and $localCount -gt 0) {
            Write-Host "[$count/$total] $session - Already downloaded ($localCount frames), skipping" -ForegroundColor Green
            continue
        } else {
            Write-Host "[$count/$total] $session - Incomplete (local: $localCount, remote: $remoteCount), re-downloading..." -ForegroundColor Yellow
        }
    } else {
        Write-Host "[$count/$total] $session - Downloading..." -ForegroundColor White
    }
    
    # Download using scp (or rsync if available via WSL)
    # Option 1: Use scp (built into Windows)
    scp -r "${RemoteHost}:${remoteSession}" $LocalPath
    
    # Option 2: Use rsync via WSL (uncomment if preferred)
    # wsl rsync -avz --progress "${RemoteHost}:${remoteSession}/" (wsl wslpath -u $localSession)/
    
    if ($LASTEXITCODE -eq 0) {
        $frameCount = (Get-ChildItem "$localSession\*.jpg" -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "[$count/$total] $session - Done ($frameCount frames)" -ForegroundColor Green
    } else {
        Write-Host "[$count/$total] $session - FAILED" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Download complete!"
Write-Host "Total sessions: $total"
Write-Host "============================================" -ForegroundColor Cyan
