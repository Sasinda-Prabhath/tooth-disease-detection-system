param([switch]$ML)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Module2 = Join-Path $Root 'services\module2-impaction-boneloss'
Set-Location $Module2
if (-not (Test-Path '.venv\Scripts\python.exe')) { py -3.11 -m venv .venv }
$Requirements = if ($ML) { 'requirements-ml.txt' } else { 'requirements.txt' }
& .\.venv\Scripts\python.exe -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
Set-Location (Join-Path $Root 'frontend')
npm ci
if ($LASTEXITCODE -ne 0) { throw 'Frontend installation failed.' }
Write-Host 'Setup complete. Follow the Module 2 README for patient data preparation and training.'
