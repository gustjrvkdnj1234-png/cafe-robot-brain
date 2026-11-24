# Stop the Ollama server and its runners, and verify that it is really gone.
#   powershell -ExecutionPolicy Bypass -File v2\scripts\stop_ollama.ps1 [-IncludeApp]
# Stops ollama.exe and llama-server.exe (with their process trees), then checks that no such process
# is left and that 127.0.0.1:11434 no longer answers. Exit code 0 = stopped, 1 = still running.
# -IncludeApp also stops the desktop tray app ("ollama app.exe").
param([switch]$IncludeApp, [int]$TimeoutSec = 15)
$ErrorActionPreference = 'Continue'
function Stop-Tree([int]$id) { cmd.exe /c "taskkill /PID $id /T /F >nul 2>&1" }

$names = @('ollama', 'llama-server')
if ($IncludeApp) { $names += 'ollama app' }

function Get-Left { @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $names -contains $_.Name }) }
function Test-Port {
    try { Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null; $true } catch { $false }
}

$before = Get-Left
foreach ($p in $before) {
    # taskkill /T also ends child runners; fall back to Stop-Process if taskkill refuses
    Stop-Tree $p.Id
    if (-not $p.HasExited) { try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch {} }
}
$deadline = (Get-Date).AddSeconds($TimeoutSec)
do {
    $left = Get-Left
    $up = Test-Port
    if (-not $left -and -not $up) { break }
    foreach ($p in $left) { Stop-Tree $p.Id }
    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $deadline)

if ($left -or $up) {
    $ids = ($left | ForEach-Object { "$($_.Name)#$($_.Id)" }) -join ', '
    Write-Output "WARNING: Ollama is still running (processes: $ids; port 11434 answering: $up)"
    exit 1
}
Write-Output ("Ollama stopped ({0} process(es) ended, port 11434 closed)." -f $before.Count)
exit 0
