# Investor Sentiment Dashboard - local development
# =============================================================================
# Starts FastAPI (backend\app\main.py) and Vite frontend in new PowerShell windows.
#
# Prerequisites:
#   pip install -r backend\requirements.txt
#   Node.js + npm (runs npm install in frontend if node_modules is missing)
#   DATABASE_URL in .env at repo root or backend\.env for database features
#
# URLs:
#   API:      http://127.0.0.1:8000  (docs at /docs)
#   Frontend: http://localhost:3000 (Vite proxies /api to the backend)
# =============================================================================

$ErrorActionPreference = 'Continue'

$RootDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }

Write-Host ''
Write-Host 'Investor Sentiment Dashboard - dev startup' -ForegroundColor Green
Write-Host "Root: $RootDir" -ForegroundColor DarkGray
Write-Host ''

Write-Host 'Checking backend...' -ForegroundColor Cyan
$BackendMain = Join-Path $RootDir 'backend\app\main.py'
if (-not (Test-Path -LiteralPath $BackendMain)) {
    Write-Host 'backend\app\main.py not found. Run from repo root, or keep start-dev.ps1 at repo root.' -ForegroundColor Red
    exit 1
}

Write-Host 'Checking frontend...' -ForegroundColor Cyan
$FrontendPkg = Join-Path $RootDir 'frontend\package.json'
if (-not (Test-Path -LiteralPath $FrontendPkg)) {
    Write-Host 'frontend\package.json not found.' -ForegroundColor Red
    exit 1
}

# Full path to npm so child windows and PowerShell call operators behave reliably
$NpmCmdInfo = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $NpmCmdInfo) {
    $NpmCmdInfo = Get-Command npm -ErrorAction SilentlyContinue
}
$NpmExePath = if ($NpmCmdInfo) { $NpmCmdInfo.Source } else { $null }

$FrontendNodeModules = Join-Path $RootDir 'frontend\node_modules'
if (-not (Test-Path -LiteralPath $FrontendNodeModules)) {
    if (-not $NpmExePath) {
        Write-Host 'npm not found on PATH. Install Node.js, then run this script again.' -ForegroundColor Red
        exit 1
    }
    Write-Host 'Installing frontend dependencies...' -ForegroundColor Yellow
    Push-Location (Join-Path $RootDir 'frontend')
    try {
        & $NpmExePath install
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'npm install failed. Fix errors above, then run this script again.' -ForegroundColor Red
            Pop-Location
            exit 1
        }
    } finally {
        Pop-Location
    }
}

$BackendPath = Join-Path $RootDir 'backend'
$FrontendPath = Join-Path $RootDir 'frontend'

# Single-line -Command avoids broken parsing when Start-Process passes multiline strings to powershell.exe
$bp = $BackendPath.Replace("'", "''")
$backendOneLiner = "Set-Location -LiteralPath '$bp'; if (Test-Path -LiteralPath '.venv\Scripts\Activate.ps1') { . '.\.venv\Scripts\Activate.ps1' }; `$env:PYTHONPATH='.'; Write-Host 'Backend: http://127.0.0.1:8000' -ForegroundColor Green; python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

Write-Host ''
Write-Host 'Starting backend on http://127.0.0.1:8000 ...' -ForegroundColor Cyan
Start-Process -FilePath 'powershell.exe' -ArgumentList @(
    '-NoExit',
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-Command', $backendOneLiner
)

Write-Host 'Waiting for /health (up to 45s)...' -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 45; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $ready) {
    Write-Host 'Backend not ready on /health yet; frontend will still start. Check the backend window.' -ForegroundColor Yellow
} else {
    Write-Host 'Backend health check OK.' -ForegroundColor Green
}

if (-not $NpmExePath) {
    Write-Host 'npm not found on PATH. Cannot start frontend.' -ForegroundColor Red
    exit 1
}

$fp = $FrontendPath.Replace("'", "''")
$npmEsc = $NpmExePath.Replace("'", "''")
$frontendOneLiner = "Set-Location -LiteralPath '$fp'; `$env:VITE_API_URL='http://localhost:3000/api/v1'; Write-Host 'Frontend: http://localhost:3000' -ForegroundColor Cyan; & '$npmEsc' run dev"

Write-Host 'Starting frontend on http://localhost:3000 ...' -ForegroundColor Cyan
Start-Process -FilePath 'powershell.exe' -ArgumentList @(
    '-NoExit',
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-Command', $frontendOneLiner
)

Write-Host ''
Write-Host 'Both windows should be open.' -ForegroundColor Green
Write-Host '  Frontend:  http://localhost:3000' -ForegroundColor Cyan
Write-Host '  Backend:   http://127.0.0.1:8000' -ForegroundColor Cyan
Write-Host '  API docs:  http://127.0.0.1:8000/docs' -ForegroundColor Cyan
Write-Host ''
Write-Host 'Stop each server with Ctrl+C in its window.' -ForegroundColor Yellow
Write-Host ''
