param(
    [string]$LogRoot = '',
    [ValidateRange(1,2147483647)][int]$Tail = 200,
    [switch]$Full
)
$ErrorActionPreference = 'Stop'
if (!$LogRoot) {
    if ($env:CGMLI_LOG_DIR) { $LogRoot = $env:CGMLI_LOG_DIR }
    else { $LogRoot = Join-Path $PSScriptRoot '../../../logs' }
}
$pointer = Join-Path ([IO.Path]::GetFullPath($LogRoot)) 'latest.txt'
if (!(Test-Path -LiteralPath $pointer -PathType Leaf)) {
    throw "No saved run yet: $pointer. Run the rebuilt offline application first."
}
$logFile = (Get-Content -LiteralPath $pointer -Encoding UTF8 -TotalCount 1).Trim()
if (!(Test-Path -LiteralPath $logFile -PathType Leaf)) { throw "Latest log is missing: $logFile" }
Write-Host "Latest log: $logFile"
if ($Full) { Get-Content -LiteralPath $logFile -Encoding UTF8 }
else { Get-Content -LiteralPath $logFile -Encoding UTF8 -Tail $Tail }
