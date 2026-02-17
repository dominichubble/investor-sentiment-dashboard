# Development Server Startup Script
# Starts both backend and frontend in separate windows

Write-Host "Starting Investor Sentiment Dashboard..." -ForegroundColor Green
Write-Host ""

# Get current directory
$RootDir = Get-Location

# Check if backend exists
Write-Host "Checking backend..." -ForegroundColor Cyan
if (-not (Test-Path "backend\api\main.py")) {
    Write-Host "Backend not found!" -ForegroundColor Red
    exit 1
}

# Check if frontend exists
Write-Host "Checking frontend..." -ForegroundColor Cyan
if (-not (Test-Path "frontend\package.json")) {
    Write-Host "Frontend not found!" -ForegroundColor Red
    exit 1
}

# Check if node_modules exists
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Push-Location frontend
    npm install
    Pop-Location
}

Write-Host ""
Write-Host "✅ Starting services..." -ForegroundColor Green
Write-Host ""

# Start backend in new window
Write-Host "Starting backend on http://localhost:8000" -ForegroundColor Cyan
$BackendPath = Join-Path $RootDir "backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$BackendPath'; Write-Host 'Backend API Server' -ForegroundColor Green; python -m uvicorn api.main:app --reload --host localhost --port 8000"

# Wait for backend to start
Write-Host "Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start frontend in new window
Write-Host "Starting frontend on http://localhost:3000" -ForegroundColor Cyan
$FrontendPath = Join-Path $RootDir "frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$FrontendPath'; Write-Host 'React Frontend' -ForegroundColor Blue; npm run dev"

Write-Host ""
Write-Host "Both servers are starting!" -ForegroundColor Green
Write-Host ""
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C in each window to stop the servers" -ForegroundColor Yellow
