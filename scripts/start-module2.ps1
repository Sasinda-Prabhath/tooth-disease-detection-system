# Start Module 2 API + Frontend (Windows)
# Run from repo root: .\scripts\start-module2.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Module2 = Join-Path $Root "services\module2-impaction-boneloss"
$Frontend = Join-Path $Root "frontend"

Write-Host "Starting Module 2 API on http://localhost:8002"
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$Module2'; if (Test-Path '.venv\Scripts\Activate.ps1') { .\.venv\Scripts\Activate.ps1 }; `$env:PYTHONPATH=(Get-Location).Path; uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload; if ($LASTEXITCODE -ne 0) { Write-Error 'Module 2 API failed to start.' }"
)

Start-Sleep -Seconds 3

Write-Host "Starting Frontend on http://localhost:3000"
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$Frontend'; `$env:VITE_API_BASE_URL='http://localhost:8002'; npm run dev"
)

Write-Host ""
Write-Host "Both services starting in new windows."
Write-Host "Open http://localhost:3000 and upload a panoramic X-ray."
Write-Host "Sample image: services\module2-impaction-boneloss\training\data\raw\synthetic_000.png"
