# Module 2 setup — Windows PowerShell
# Run from repository root: .\scripts\setup-module2.ps1

param(
    [switch]$DockerTrain,
    [switch]$SkipTrain
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

if (-not $SkipTrain) {
    Write-Host "==> Generating bootstrap training data (if missing)"
    if (-not (Test-Path "training\data\raw\synthetic_000.png")) {
        python training\generate_sample_data.py --num-images 40
    }

    if ($DockerTrain) {
        Set-Location $Root
        Write-Host "==> Training models in Docker (recommended — clean TensorFlow env)"
        docker compose -f docker-compose.train.yml up --build
    } else {
        Write-Host "==> Training models locally (quick mode, ~10-20 min)"
        $env:MODULE2_QUICK_TRAIN = "1"
        python training\train_all.py --skip-data --quick
    }
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
