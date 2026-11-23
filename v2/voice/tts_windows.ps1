# Synthesize the TTS test sentences with the built-in Windows Korean voice (no install).
#   -Engine sapi     : System.Speech (SAPI 5, "Microsoft Heami Desktop")
#   -Engine onecore  : Windows.Media.SpeechSynthesis (OneCore, "Microsoft Heami")
# Writes <OutDir>\NN.wav and prints one JSON line with latency and memory.
param(
    [Parameter(Mandatory = $true)][ValidateSet('sapi', 'onecore')][string]$Engine,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [Parameter(Mandatory = $true)][string]$SentencesJson
)
$ErrorActionPreference = 'Stop'
$texts = (Get-Content -Raw -Encoding UTF8 $SentencesJson | ConvertFrom-Json).tts
New-Item -ItemType Directory -Force $OutDir | Out-Null
$me = Get-Process -Id $PID
$baseWs = $me.WorkingSet64
$sw = [Diagnostics.Stopwatch]::StartNew()

if ($Engine -eq 'sapi') {
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'ko-KR' } | Select-Object -First 1
    $synth.SelectVoice($voice.VoiceInfo.Name)
    $voiceName = $voice.VoiceInfo.Name
    $speak = {
        param($text, $path)
        $synth.SetOutputToWaveFile($path)
        $synth.Speak($text)
        $synth.SetOutputToNull()
    }
} else {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
    $synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
    $synth.Voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices |
        Where-Object { $_.Language -eq 'ko-KR' } | Select-Object -First 1
    $voiceName = $synth.Voice.DisplayName
    $asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
    $asTask = $asTask.MakeGenericMethod([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
    $speak = {
        param($text, $path)
        $task = $asTask.Invoke($null, @($synth.SynthesizeTextToStreamAsync($text)))
        $task.Wait()
        $src = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($task.Result)
        $fs = [IO.File]::Create($path)
        $src.CopyTo($fs)
        $fs.Close()
        $src.Dispose()
        $task.Result.Dispose()
    }
}
$loadS = $sw.Elapsed.TotalSeconds

# warm-up (not counted)
$warm = Join-Path $OutDir '_warmup.wav'
$sw.Restart(); & $speak '안녕하세요.' $warm; $warmS = $sw.Elapsed.TotalSeconds
Remove-Item $warm

$items = @()
for ($i = 0; $i -lt $texts.Count; $i++) {
    $path = Join-Path $OutDir ('{0:D2}.wav' -f ($i + 1))
    $sw.Restart()
    & $speak $texts[$i] $path
    $items += [ordered]@{ id = '{0:D2}' -f ($i + 1); latency_s = [math]::Round($sw.Elapsed.TotalSeconds, 3) }
}
$me.Refresh()
[ordered]@{
    engine = $Engine; voice = $voiceName; load_s = [math]::Round($loadS, 2); warmup_s = [math]::Round($warmS, 2)
    peak_wset_bytes = $me.PeakWorkingSet64; base_wset_bytes = $baseWs; items = $items
} | ConvertTo-Json -Compress -Depth 4
