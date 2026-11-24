# Start the voice ordering demo (run from anywhere):
#   powershell -ExecutionPolicy Bypass -File v2\voice\app\start_demo.ps1 [-Port 8770] [-Open] [-UseRunning]
#
# 1. Starts `ollama serve` through v2\scripts\start_ollama.ps1 with, for that process only:
#    OLLAMA_IGPU_ENABLE=1, LLAMA_ARG_CACHE_RAM=0 (same as v1) and OLLAMA_NOPRUNE=1 (verified in the log).
#    If an Ollama server is already running it stops, unless -UseRunning is given.
# 2. Starts the app (v2\voice\app\server.py, 127.0.0.1 only). The app loads faster-whisper medium,
#    Supertonic 3 and the Heami worker, and warms up song/gemma4-26b-cafe-64e.
# 3. Prints the URL when everything is ready. Ctrl+C stops the app and the Ollama server it started.
#    To stop Ollama by hand at any time: powershell -ExecutionPolicy Bypass -File v2\scripts\stop_ollama.ps1
# Extra arguments for server.py (e.g. '--history text --layout deploy --max-turns 0') go in -AppArgs.
param([int]$Port = 8770, [switch]$Open, [switch]$UseRunning, [string]$AppArgs = '')
$ErrorActionPreference = 'Stop'
$App = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $App '..\..\..')).Path
$Py = Join-Path $Root '.venv-voice\Scripts\python.exe'
$OllamaLog = Join-Path $Root 'logs\ollama_serve_voice.log'

function Test-Ollama { try { Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null; $true } catch { $false } }

$startedOllama = $false
$server = $null
try {
    if (Test-Ollama) {
        if (-not $UseRunning) {
            Write-Host 'An Ollama server is already running; its OLLAMA_NOPRUNE / LLAMA_ARG_CACHE_RAM settings cannot be verified.'
            Write-Host 'Stop it first, or rerun with -UseRunning to use it as is.'
            exit 2
        }
        Write-Host 'Using the Ollama server that is already running (-UseRunning).'
    } else {
        Write-Host '[1/3] starting Ollama (OLLAMA_IGPU_ENABLE=1, LLAMA_ARG_CACHE_RAM=0, OLLAMA_NOPRUNE=1) ...'
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'v2\scripts\start_ollama.ps1') `
            -Log $OllamaLog -Env 'LLAMA_ARG_CACHE_RAM=0'
        if ($LASTEXITCODE -ne 0) { throw "start_ollama.ps1 failed (exit $LASTEXITCODE)" }
        $startedOllama = $true
    }

    Write-Host '[2/3] starting the app (STT / TTS load + 64e warm-up, about 30-60 s) ...'
    $argv = @("`"$(Join-Path $App 'server.py')`"", '--port', $Port) + @($AppArgs -split '\s+' | Where-Object { $_ })
    $server = Start-Process -FilePath $Py -ArgumentList $argv -NoNewWindow -PassThru
    $url = "http://127.0.0.1:$Port/"
    $ready = $false
    for ($i = 0; $i -lt 240 -and -not $server.HasExited; $i++) {
        Start-Sleep -Milliseconds 1000
        try {
            $h = Invoke-RestMethod "$($url)api/health" -TimeoutSec 3
            if ($h.ready) { $ready = $true; break }
            if ($h.errors.Count) { Write-Host ('  ' + ($h.errors -join ' | ')) }
        } catch {}
    }
    if (-not $ready) { throw 'the app did not become ready (see messages above)' }
    $w = $h.llm_warmup
    Write-Host ("[3/3] ready. 64e warm-up {0} s, on iGPU: {1}, runner {2} GB, GPU shared {3} GB" -f `
        $w.wall_s, $w.full_igpu, $w.runner_private_gb, $w.gpu_shared_gb)
    Write-Host ''
    Write-Host "  >>> $url  (Ctrl+C to stop)"
    Write-Host ''
    if ($Open) { Start-Process $url }
    $server.WaitForExit()
} finally {
    # Every step must run even if an earlier one fails: with 'Stop', a native command's stderr
    # (e.g. taskkill "process not found") aborted this block before Ollama was stopped.
    $ErrorActionPreference = 'Continue'
    # app: the venv python.exe is a launcher with a child interpreter, so end the whole tree
    try {
        $ids = @()
        if ($server) { $ids += $server.Id }
        $ids += Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match 'voice\\app\\server\.py' } | ForEach-Object { $_.ProcessId }
        foreach ($id in ($ids | Select-Object -Unique)) {
            if (Get-Process -Id $id -ErrorAction SilentlyContinue) { cmd.exe /c "taskkill /PID $id /T /F >nul 2>&1" }
        }
    } catch { Write-Host "app stop: $($_.Exception.Message)" }
    if ($startedOllama) {
        # verified stop (checks processes and port 11434, prints a warning if anything is left)
        try {
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'v2\scripts\stop_ollama.ps1') | Write-Host
        } catch { Write-Host "Ollama stop: $($_.Exception.Message)" }
    }
}
