$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Frontend = (Resolve-Path (Join-Path $Root 'frontend')).Path
# Windows locks esbuild.exe while Vite is running. Only stop this checkout's
# Node/esbuild processes; other projects and the Python services stay running.
$FrontendProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('node.exe', 'esbuild.exe') -and $_.CommandLine -and
    $_.CommandLine.IndexOf($Frontend + '\', [StringComparison]::OrdinalIgnoreCase) -ge 0
})
if ($FrontendProcesses.Count) {
    Write-Host 'Stopping this project frontend to release dependency files...'
    foreach ($FrontendProcess in $FrontendProcesses) {
        $RunningProcess = Get-Process -Id $FrontendProcess.ProcessId -ErrorAction SilentlyContinue
        if ($RunningProcess) {
            $RunningProcess | Stop-Process -ErrorAction SilentlyContinue
            $RunningProcess.WaitForExit(10000) | Out-Null
        }
    }
}
Push-Location $Frontend
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend installation failed. Resolve the npm error before starting the frontend.' }
} finally {
    Pop-Location
}
if ($FrontendProcesses.Count) {
    & (Join-Path $PSScriptRoot 'start-module2.ps1')
}
