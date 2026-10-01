param([Parameter(Mandatory=$true)][string]$Image)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Image -PathType Leaf)) { throw 'Supply an authorised OPG file.' }
$SessionToken = [Guid]::NewGuid().ToString('N') + [Guid]::NewGuid().ToString('N')
Invoke-RestMethod 'http://localhost:8002/module2/health' | ConvertTo-Json
curl.exe --fail-with-body -H "X-Case-Session: $SessionToken" -F "file=@$Image" 'http://localhost:8002/module2/predict'
