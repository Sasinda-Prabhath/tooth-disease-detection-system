$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Module2 = Join-Path $Root 'services\module2-impaction-boneloss'
$Frontend = Join-Path $Root 'frontend'
$Python = Join-Path $Module2 '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Run scripts/setup-module2.ps1 first.' }
Start-Process -FilePath $Python -WorkingDirectory $Module2 -WindowStyle Hidden -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8002') -RedirectStandardOutput (Join-Path $Module2 'api.log') -RedirectStandardError (Join-Path $Module2 'api-error.log')
Start-Process -FilePath 'cmd.exe' -WorkingDirectory $Frontend -WindowStyle Hidden -ArgumentList @('/c', 'npm run dev -- --host 127.0.0.1') -RedirectStandardOutput (Join-Path $Frontend 'dev.log') -RedirectStandardError (Join-Path $Frontend 'dev-error.log')
Write-Host 'Services starting in background. Open http://localhost:3000. Check api-error.log and dev-error.log if startup fails.'
