# Module 2 local setup script (Windows PowerShell)
# Run from repository root: .\scripts\setup-module2.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Module2 = Join-Path $Root "services\module2-impaction-boneloss"

Write-Host "==> Setting up shared-lib"
pip install -e (Join-Path $Root "shared-lib")

Write-Host "==> Creating Python venv for Module 2"
Set-Location $Module2
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e (Join-Path $Root "shared-lib")

Write-Host "==> Generating bootstrap training data"
python training\generate_sample_data.py --num-images 40

Write-Host "==> Training all models (may take 15-30 minutes)"
python training\train_all.py

Write-Host "==> Installing frontend dependencies"
Set-Location (Join-Path $Root "frontend")
npm install

Write-Host ""
Write-Host "Setup complete. Start services in two terminals:"
Write-Host "  Terminal 1: cd services\module2-impaction-boneloss; .\.venv\Scripts\Activate.ps1; `$env:PYTHONPATH=(Get-Location).Path; uvicorn app.main:app --port 8002 --reload"
Write-Host "  Terminal 2: cd frontend; `$env:VITE_API_BASE_URL='http://localhost:8002'; npm run dev"
Write-Host "  Open http://localhost:3000"
