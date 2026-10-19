param([switch]$ML)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Module2 = Join-Path $Root 'services\module2-impaction-boneloss'
Set-Location $Module2
if (-not (Test-Path '.venv\Scripts\python.exe')) { py -3.11 -m venv .venv }
$Requirements = if ($ML) { 'requirements-ml.txt' } else { 'requirements.txt' }
& .\.venv\Scripts\python.exe -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements-fdi.txt
if ($LASTEXITCODE -ne 0) { throw 'FDI runtime installation failed.' }
& .\.venv\Scripts\python.exe -m pip install --no-deps -e ./fdi_review
if ($LASTEXITCODE -ne 0) { throw 'Imported FDI package installation failed.' }
if (-not (Test-Path '.venv-review\Scripts\python.exe')) {
    & .\.venv\Scripts\python.exe -m venv .venv-review
}
& .\.venv\Scripts\python.exe -c "from pathlib import Path; import sysconfig; Path('.venv-review/Lib/site-packages/module2-api.pth').write_text(sysconfig.get_path('purelib') + '\n')"
& .\.venv-review\Scripts\python.exe -m pip install -e ./fdi_review
if ($LASTEXITCODE -ne 0) { throw 'FDI review application installation failed.' }
& (Join-Path $PSScriptRoot 'setup-frontend.ps1')
Set-Location $Root
Write-Host 'Setup complete. Follow the Module 2 README for patient data preparation and training.'
