# Module 2 setup — Windows PowerShell
# Run from repository root: .\scripts\setup-module2.ps1

param(
    [switch]$DockerTrain,
    [switch]$SkipTrain,
    [string]$Dataset = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Module2 = Join-Path $Root "services\module2-impaction-boneloss"

Write-Host "==> Installing shared-lib"
pip install -e (Join-Path $Root "shared-lib")

Write-Host "==> Creating Python venv for Module 2"
Set-Location $Module2
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e (Join-Path $Root "shared-lib")

if (-not $SkipTrain -and $Dataset) {
    if ($DockerTrain) {
        Set-Location $Root
        Write-Host "==> Training models in Docker (recommended — clean TensorFlow env)"
        docker compose -f docker-compose.train.yml run --rm --build module2-train python training/train_all.py --dataset $Dataset --device cpu
        if ($LASTEXITCODE -ne 0) { throw "Docker hybrid training failed; Dataset must be a mounted path relative to /app" }
    } else {
        Write-Host "==> Training hybrid models on the prepared dataset"
        python training\train_all.py --dataset $Dataset
        if ($LASTEXITCODE -ne 0) { throw "Hybrid training failed" }
    }
} else {
    Write-Host "Training skipped. Prepare expert labels using training/HYBRID_TRAINING.md, then pass -Dataset."
}

Write-Host "==> Installing frontend dependencies"
Set-Location (Join-Path $Root "frontend")
if (-not (Test-Path "node_modules")) { npm install }

Write-Host ""
Write-Host "Setup complete!"
Write-Host ""
Write-Host "Option A — Local dev (two terminals):"
Write-Host "  Terminal 1: cd services\module2-impaction-boneloss"
Write-Host "              .\.venv\Scripts\Activate.ps1"
Write-Host "              `$env:PYTHONPATH=(Get-Location).Path"
Write-Host "              uvicorn app.main:app --port 8002 --reload"
Write-Host "  Terminal 2: cd frontend"
Write-Host "              `$env:VITE_API_BASE_URL='http://localhost:8002'"
Write-Host "              npm run dev"
Write-Host "  Open http://localhost:3000"
Write-Host ""
Write-Host "Option B — Docker (single command):"
Write-Host "  docker compose -f docker-compose.module2-only.yml up --build"
Write-Host "  Open http://localhost:3000"
