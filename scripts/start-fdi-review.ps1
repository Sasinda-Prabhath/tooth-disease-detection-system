$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Module2 = Join-Path $Root 'services\module2-impaction-boneloss'
$Python = Join-Path $Module2 '.venv-review\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Run scripts/setup-module2.ps1 first.' }
if (Get-NetTCPConnection -LocalPort 8502 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host 'Port 8502 is already in use. Review app may already be running at http://localhost:8502.'
    exit 0
}
Start-Process -FilePath $Python -WorkingDirectory (Join-Path $Module2 'fdi_review') -WindowStyle Hidden -ArgumentList @('-m', 'streamlit', 'run', 'app.py', '--server.address', '127.0.0.1', '--server.port', '8502', '--server.headless', 'true') -RedirectStandardOutput (Join-Path $Module2 'fdi-review.log') -RedirectStandardError (Join-Path $Module2 'fdi-review-error.log')
Write-Host 'FDI review starting at http://localhost:8502 (numbering, bulk exports, drawing and training review tools).'
