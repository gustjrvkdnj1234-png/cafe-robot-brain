# Start `ollama serve` with environment variables set for the serve process only.
# OLLAMA_NOPRUNE=1 is always set, so a server start never deletes blobs.
#
#   powershell -ExecutionPolicy Bypass -File v2\scripts\start_ollama.ps1 -Log logs\ollama_serve_x.log
#   ... -Env LLAMA_ARG_CACHE_RAM=0              # extra NAME=VALUE pairs, comma-separated (default: OLLAMA_IGPU_ENABLE=1 only)
#
# Refuses to start if an Ollama server already answers (its NOPRUNE setting cannot be checked),
# and stops the new server again if its log does not show OLLAMA_NOPRUNE:true.
#
# Warning (Ollama 0.40 on Windows): any `ollama` CLI command run while no server is up launches the
# desktop app, whose server starts WITHOUT OLLAMA_NOPRUNE. Run this script first, then the CLI.
param(
    [Parameter(Mandatory = $true)][string]$Log,
    [string]$Env = '',
    [string]$OllamaHost = 'http://127.0.0.1:11434'
)
$ErrorActionPreference = 'Stop'

function Test-Up {
    try { Invoke-RestMethod "$OllamaHost/api/version" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

if (Test-Up) {
    Write-Error "An Ollama server is already running at $OllamaHost. Stop it first (its OLLAMA_NOPRUNE setting cannot be verified)."
    exit 2
}
$exe = (Get-Command ollama -ErrorAction Stop).Source
$logPath = [IO.Path]::GetFullPath($Log)
New-Item -ItemType Directory -Force (Split-Path $logPath) | Out-Null

$vars = @{ OLLAMA_IGPU_ENABLE = '1' }
foreach ($pair in ($Env -split ',' | Where-Object { $_.Trim() })) {
    $k, $v = $pair.Trim() -split '=', 2
    if (-not $k -or $null -eq $v) { Write-Error "bad -Env entry '$pair' (expected NAME=VALUE)"; exit 4 }
    $vars[$k] = $v
}
$vars['OLLAMA_NOPRUNE'] = '1'   # always, cannot be overridden

# cmd.exe child: variables are set only inside it and inherited by ollama serve
$sets = ($vars.GetEnumerator() | Sort-Object Name | ForEach-Object { "set `"$($_.Name)=$($_.Value)`"" }) -join ' && '
$cmd = "$sets && `"$exe`" serve >> `"$logPath`" 2>&1"
$psi = New-Object System.Diagnostics.ProcessStartInfo 'cmd.exe', "/d /c $cmd"
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$proc = [System.Diagnostics.Process]::Start($psi)

for ($i = 0; $i -lt 60 -and -not (Test-Up); $i++) { Start-Sleep -Milliseconds 500 }
if (-not (Test-Up)) { Write-Error "ollama serve did not start (see $logPath)"; exit 1 }

$cfg = Select-String -Path $logPath -Pattern 'msg="server config"' | Select-Object -Last 1
if (-not $cfg -or $cfg.Line -notmatch 'OLLAMA_NOPRUNE:true') {
    Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Error "server log does not show OLLAMA_NOPRUNE:true; server stopped."
    exit 3
}
$shown = ($vars.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' '
Write-Output "ollama serve started ($shown), log: $logPath"
