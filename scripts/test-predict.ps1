# Quick API test (Windows PowerShell)
# Run: .\scripts\test-predict.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Sample = Join-Path $Root "services\module2-impaction-boneloss\training\data\raw\synthetic_000.png"

if (-not (Test-Path $Sample)) {
    Write-Error "Sample image not found. Run setup first: .\scripts\setup-module2.ps1 -SkipTrain"
}

Write-Host "Testing health..."
Invoke-RestMethod -Uri "http://127.0.0.1:8002/health" | ConvertTo-Json

Write-Host "`nTesting predict..."
$boundary = [System.Guid]::NewGuid().ToString()
$fileBytes = [System.IO.File]::ReadAllBytes($Sample)
$bodyLines = @(
    "--$boundary",
    'Content-Disposition: form-data; name="file"; filename="synthetic_000.png"',
    "Content-Type: image/png",
    "",
    [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes),
    "--$boundary",
    'Content-Disposition: form-data; name="pixel_spacing_mm"',
    "",
    "0.1",
    "--$boundary--"
)
$body = $bodyLines -join "`r`n"
Invoke-RestMethod -Uri "http://127.0.0.1:8002/predict" -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $body | ConvertTo-Json -Depth 6
