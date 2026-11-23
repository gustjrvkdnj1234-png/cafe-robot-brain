# Persistent Windows SAPI worker (Microsoft Heami Desktop, ko-KR) for the voice demo.
# stdin : one line per request = base64(UTF-8 JSON {"text": ..., "path": ...})
# stdout: "READY" once, then "OK <path>" or "ERR <message>" per request.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'ko-KR' } | Select-Object -First 1
if (-not $voice) { [Console]::Out.WriteLine('ERR no ko-KR SAPI voice'); exit 1 }
$synth.SelectVoice($voice.VoiceInfo.Name)
[Console]::Out.WriteLine('READY ' + $voice.VoiceInfo.Name)
[Console]::Out.Flush()
while ($null -ne ($line = [Console]::In.ReadLine())) {
    try {
        $req = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($line.Trim())) | ConvertFrom-Json
        $synth.SetOutputToWaveFile($req.path)
        $synth.Speak([string]$req.text)
        $synth.SetOutputToNull()
        [Console]::Out.WriteLine('OK ' + $req.path)
    } catch {
        try { $synth.SetOutputToNull() } catch {}
        [Console]::Out.WriteLine('ERR ' + ($_.Exception.Message -replace '\s+', ' '))
    }
    [Console]::Out.Flush()
}
