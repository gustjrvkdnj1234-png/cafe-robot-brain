"""Offline Korean TTS benchmark (CPU only) on the 5 demo replies.

Usage:
    .venv-voice\\Scripts\\python.exe v2\\voice\\bench_tts.py voices     # Supertonic speaker scan
    .venv-voice\\Scripts\\python.exe v2\\voice\\bench_tts.py            # all candidates + report
    .venv-voice\\Scripts\\python.exe v2\\voice\\bench_tts.py --report   # report only

Each candidate runs in its own process, one after another, with outbound
network blocked (sherpa-onnx) or nothing to download (Windows voices).
One warm-up sentence (not counted), then the 5 replies.
Latency = wall time from text to complete waveform (no streaming).
Peak RAM = peak working set of the synthesizing process (PowerShell for the
Windows voices, so it includes PowerShell itself; its base is reported too).
Intelligibility = CER of faster-whisper medium (the best STT here) on each output,
spaces and punctuation removed.
WAVs go to v2/voice/results/tts/<candidate>/NN.wav (not committed).
"""
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
OUT = HERE / "results" / "tts"
THREADS = 8
WARMUP = "안녕하세요. 주문 도와드릴게요."
ST3 = MODELS / "supertonic3-int8" / "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
ST2 = MODELS / "supertonic2-int8" / "sherpa-onnx-supertonic-tts-int8-2026-03-06"
MIMIC = MODELS / "mimic3-ko-kss" / "vits-mimic3-ko_KO-kss_low"
SUPERTONIC_STEPS = 8   # value used in the sherpa-onnx example
VOICE_FILE = OUT / "supertonic_voices.json"

CANDIDATES = {
    "windows-sapi-heami": {"engine": "windows", "mode": "sapi", "label": "Windows SAPI Heami",
                           "download": "download_none"},
    "windows-onecore-heami": {"engine": "windows", "mode": "onecore", "label": "Windows OneCore Heami",
                              "download": "download_none"},
    "supertonic3-f": {"engine": "supertonic", "dir": ST3, "gender": "F", "label": "Supertonic 3 int8 (여성)",
                      "download": "supertonic3-int8"},
    "supertonic3-m": {"engine": "supertonic", "dir": ST3, "gender": "M", "label": "Supertonic 3 int8 (남성)",
                      "download": "supertonic3-int8"},
    "supertonic3-f-steps4": {"engine": "supertonic", "dir": ST3, "gender": "F", "steps": 4,
                             "label": "Supertonic 3 int8 (여성, 4단계)", "download": "supertonic3-int8"},
    "supertonic2-f": {"engine": "supertonic", "dir": ST2, "gender": "F", "label": "Supertonic 2 int8 (여성)",
                      "download": "supertonic2-int8"},
}
# Mimic3 KSS (vits-mimic3-ko_KO-kss_low) was downloaded but is not benchmarked:
# sherpa-onnx 2.0.2 has no built-in espeak-ng frontend for it ("phoneme_codepoints is
# empty, no lexicon, and no frontend available"), and its KSS training data is CC BY-NC-SA.
SKIPPED = {"mimic3-kss": "sherpa-onnx 2.0.2에 espeak-ng 발음 변환기가 없어 합성 불가(외부 음소 변환기 필요). "
                         "학습 데이터 KSS는 CC BY-NC-SA 4.0(비상업)."}


def texts():
    return json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))["tts"]


def block_network():
    real = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise OSError(f"network blocked in benchmark: {address}")
        return real(self, address)
    socket.socket.connect = guarded
    os.environ["HF_HUB_OFFLINE"] = "1"


def supertonic(d):
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
            duration_predictor=str(d / "duration_predictor.int8.onnx"),
            text_encoder=str(d / "text_encoder.int8.onnx"),
            vector_estimator=str(d / "vector_estimator.int8.onnx"),
            vocoder=str(d / "vocoder.int8.onnx"), tts_json=str(d / "tts.json"),
            unicode_indexer=str(d / "unicode_indexer.bin"), voice_style=str(d / "voice.bin")),
        num_threads=THREADS, provider="cpu"))
    return sherpa_onnx.OfflineTts(cfg)


def supertonic_say(tts, text, sid, steps=SUPERTONIC_STEPS):
    import sherpa_onnx
    g = sherpa_onnx.GenerationConfig()
    g.sid = sid
    g.num_steps = steps
    g.speed = 1.0
    g.extra["lang"] = "ko"
    a = tts.generate(text, g)
    return a.samples, a.sample_rate


def median_f0(x, sr):
    """Rough median pitch (Hz) of voiced 40 ms frames via autocorrelation."""
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    n = int(0.04 * sr)
    f0s = []
    for i in range(0, len(x) - n, n):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt((fr ** 2).mean()) < 0.02:
            continue
        ac = np.correlate(fr, fr, "full")[n - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else 0.0


def scan_voices():
    """Synthesize sentence 1 with every Supertonic 3 speaker; label by pitch."""
    import soundfile as sf
    block_network()
    tts = supertonic(ST3)
    n = tts.num_speakers
    rows = []
    out = OUT / "supertonic3_voices"
    out.mkdir(parents=True, exist_ok=True)
    for sid in range(n):
        x, sr = supertonic_say(tts, texts()[0], sid)
        sf.write(out / f"sid{sid}.wav", x, sr)
        f0 = median_f0(x, sr)
        rows.append({"sid": sid, "median_f0_hz": round(f0, 1), "gender_est": "F" if f0 >= 165 else "M"})
        print(rows[-1])
    VOICE_FILE.write_text(json.dumps({"num_speakers": n, "voices": rows}, indent=2), encoding="utf-8")


def pick_sid(gender):
    v = json.loads(VOICE_FILE.read_text(encoding="utf-8"))["voices"]
    return next(r["sid"] for r in v if r["gender_est"] == gender)


def worker(name):
    import psutil
    import soundfile as sf
    block_network()
    cfg = CANDIDATES[name]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if cfg["engine"] == "supertonic":
        tts = supertonic(cfg["dir"])
        sid = pick_sid(cfg["gender"])
        steps = cfg.get("steps", SUPERTONIC_STEPS)
        say = lambda t: supertonic_say(tts, t, sid, steps)  # noqa: E731
        extra = {"sid": sid, "num_steps": steps}
    else:
        import sherpa_onnx
        d = cfg["dir"]
        tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(d / "ko_KO-kss_low.onnx"),
                                                       tokens=str(d / "tokens.txt"),
                                                       data_dir=str(d / "espeak-ng-data")),
            num_threads=THREADS, provider="cpu")))

        def say(t):
            a = tts.generate(t, sid=0, speed=1.0)
            return a.samples, a.sample_rate
        extra = {}
    load_s = time.perf_counter() - t0
    tw = time.perf_counter()
    say(WARMUP)
    warm = time.perf_counter() - tw
    items = []
    for i, text in enumerate(texts(), 1):
        t = time.perf_counter()
        x, sr = say(text)
        lat = time.perf_counter() - t
        sf.write(out / f"{i:02d}.wav", x, sr)
        items.append({"id": f"{i:02d}", "latency_s": round(lat, 3)})
    peak = psutil.Process().memory_info().peak_wset
    (out / "run.json").write_text(json.dumps({"load_s": round(load_s, 2), "warmup_s": round(warm, 2),
                                              "peak_wset_bytes": peak, "items": items, **extra}, indent=2),
                                  encoding="utf-8")


def run_windows(name):
    cfg = CANDIDATES[name]
    out = OUT / name
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(HERE / "tts_windows.ps1"), "-Engine", cfg["mode"], "-OutDir", str(out),
                        "-SentencesJson", str(HERE / "sentences.json")],
                       capture_output=True, text=True, encoding="utf-8", check=True)
    (out / "run.json").write_text(json.dumps(json.loads(r.stdout.strip().splitlines()[-1]), indent=2),
                                  encoding="utf-8")


def intelligibility():
    """CER of faster-whisper medium on every synthesized reply (round-trip check)."""
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    sys.path.insert(0, str(HERE))
    from bench_stt import edit_distance, normalize
    block_network()
    m = WhisperModel(str(MODELS / "fw-medium"), device="cpu", compute_type="int8", cpu_threads=THREADS)
    refs = texts()
    for name in CANDIDATES:
        p = OUT / name / "run.json"
        if not p.exists():
            continue
        run = json.loads(p.read_text(encoding="utf-8"))
        for it, ref in zip(run["items"], refs):
            x, sr = sf.read(OUT / name / f"{it['id']}.wav", dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:  # band-limited FFT resample to 16 kHz
                n = int(round(len(x) * 16000 / sr))
                x = np.fft.irfft(np.fft.rfft(x)[: n // 2 + 1], n) * (n / len(x))
            segs, _ = m.transcribe(x.astype(np.float32), language="ko", beam_size=1, temperature=0.0,
                                   condition_on_previous_text=False, without_timestamps=True)
            hyp = "".join(s.text for s in segs).strip()
            it.update({"duration_s": round(sf.info(OUT / name / f"{it['id']}.wav").duration, 2),
                       "sample_rate": sr, "asr": hyp, "asr_errors": edit_distance(normalize(ref), normalize(hyp)),
                       "ref_chars": len(normalize(ref))})
        p.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, [it["asr_errors"] for it in run["items"]])


# Transcription-format differences that are not pronunciation errors
# (the STT writes numbers as digits and "께요" for "게요"). Lenient column only.
TTS_VARIANTS = {"2잔": "두잔", "3잔": "세잔", "께요": "게요"}


def lenient_errors(ref, hyp):
    sys.path.insert(0, str(HERE))
    from bench_stt import edit_distance, normalize
    h = normalize(hyp)
    for a, b in TTS_VARIANTS.items():
        h = h.replace(a, b)
    return edit_distance(normalize(ref), h)


def report():
    dl = json.loads((HERE / "results" / "downloads.json").read_text(encoding="utf-8"))
    refs = texts()
    lines = ["# TTS 비교 (CPU, 오프라인)", "",
             f"응답 5문장, 예열 1문장 뒤 측정, 스레드 {THREADS}. 지연 = 문장 전체 파형이 나올 때까지(스트리밍 없음). "
             "RTF = 지연 / 음성 길이. 되알아듣기 CER = faster-whisper medium이 합성음을 받아 적은 결과의 CER "
             "(낮을수록 발음이 또렷함, 음질 점수는 아님). 음성 파일은 `results/tts/<후보>/NN.wav`(커밋 안 함).", "",
             "'표기 차이 제외'는 STT가 숫자로 적은 것(2잔·3잔)과 '께요' 표기를 오류로 세지 않은 값.", "",
             "| 후보 | 출력 | 지연 평균 | 지연 최대 | RTF | 최대 RAM | 되알아듣기 CER | 표기 차이 제외 | 다운로드 | 로드 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for name, cfg in CANDIDATES.items():
        p = OUT / name / "run.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        lat = [i["latency_s"] for i in r["items"]]
        dur = [i["duration_s"] for i in r["items"]]
        n_chars = sum(i["ref_chars"] for i in r["items"])
        cer = 100 * sum(i["asr_errors"] for i in r["items"]) / n_chars
        cer_len = 100 * sum(lenient_errors(ref, i["asr"]) for i, ref in zip(r["items"], refs)) / n_chars
        ram = r["peak_wset_bytes"] / 1e9
        ram_s = f"{ram:.2f} GB" + (f" (PowerShell 기본 {r['base_wset_bytes'] / 1e9:.2f})" if "base_wset_bytes" in r else "")
        size = dl[cfg["download"]]["download_bytes"] / 1e9 if cfg["download"] in dl else 0.0
        s = {"latency_mean_s": statistics.mean(lat), "latency_max_s": max(lat), "rtf": sum(lat) / sum(dur),
             "peak_ram_gb": ram, "asr_cer": cer, "asr_cer_lenient": cer_len, "download_gb": size, "load_s": r["load_s"],
             "sample_rate": r["items"][0]["sample_rate"], "sid": r.get("sid")}
        summary[name] = s
        lines.append(f"| {cfg['label']}{' sid ' + str(r['sid']) if 'sid' in r else ''} | {s['sample_rate'] // 1000} kHz | "
                     f"{s['latency_mean_s']:.2f} s | {s['latency_max_s']:.2f} s | {s['rtf']:.3f} | {ram_s} | "
                     f"{cer:.1f}% | {cer_len:.1f}% | {size:.2f} GB | {r['load_s']:.1f} s |")
    lines += [""] + [f"- 제외: {k} — {v}" for k, v in SKIPPED.items()]
    lines += ["", "## 문장별 (지연 s / 길이 s / 되알아듣기)", ""]
    names = [n for n in CANDIDATES if n in summary]
    data = {n: json.loads((OUT / n / "run.json").read_text(encoding="utf-8")) for n in names}
    lines += ["| # | 문장 | " + " | ".join(CANDIDATES[n]["label"] for n in names) + " |",
              "|---|---|" + "---|" * len(names)]
    for k, ref in enumerate(refs):
        cells = []
        for n in names:
            it = data[n]["items"][k]
            mark = "✓" if it["asr_errors"] == 0 else f"({it['asr_errors']}) {it['asr']}"
            cells.append(f"{it['latency_s']:.2f} / {it['duration_s']:.1f} / {mark}")
        lines.append(f"| {k + 1:02d} | {ref} | " + " | ".join(cells) + " |")
    if VOICE_FILE.exists():
        v = json.loads(VOICE_FILE.read_text(encoding="utf-8"))
        lines += ["", f"## Supertonic 3 화자 ({v['num_speakers']}명, 1번 문장, `results/tts/supertonic3_voices/sidN.wav`)", "",
                  "| sid | 중앙 음높이 | 추정 |", "|---:|---:|---|"]
        lines += [f"| {r['sid']} | {r['median_f0_hz']:.0f} Hz | {'여성' if r['gender_est'] == 'F' else '남성'} |"
                  for r in v["voices"]]
    (OUT.parent / "tts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT.parent / "tts_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines[:6 + len(summary)]))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args[:1] == ["--worker"]:
        return worker(args[1])
    if args == ["voices"]:
        return scan_voices()
    if args == ["--intelligibility"]:
        return intelligibility()
    if args != ["--report"]:
        for name in (args or list(CANDIDATES)):
            print(f"== {name}", flush=True)
            t = time.time()
            if CANDIDATES[name]["engine"] == "windows":
                run_windows(name)
            else:
                subprocess.run([sys.executable, "-I", str(Path(__file__)), "--worker", name], check=True)
            print(f"== {name} done in {time.time() - t:.0f} s", flush=True)
        subprocess.run([sys.executable, str(Path(__file__)), "--intelligibility"], check=True)
    report()


if __name__ == "__main__":
    main()
