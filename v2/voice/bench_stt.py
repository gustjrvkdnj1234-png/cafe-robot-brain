"""Offline Korean STT benchmark on the recorded sentences (CPU only).

Usage:
    .venv-voice\\Scripts\\python.exe v2\\voice\\bench_stt.py            # all candidates, then report
    .venv-voice\\Scripts\\python.exe v2\\voice\\bench_stt.py fw-small    # one candidate
    .venv-voice\\Scripts\\python.exe v2\\voice\\bench_stt.py --report    # rebuild stt.md only

Each candidate runs in its own process, one after another. Inside the worker,
outbound network is blocked (only loopback allowed) and HF_HUB_OFFLINE=1.
Same decoding for all: language ko, greedy (beam 1), temperature 0 without
fallback, no timestamps, 8 CPU threads. One warm-up utterance (not counted),
then builtin 01-10 and bluetooth 01-10.

Latency = wall time of one transcription call on a 16 kHz WAV already in memory
(whisper.cpp: HTTP request to a local whisper-server with the model loaded).
Peak RAM = peak working set of the worker process (+ whisper-server).
CER = character edit distance / reference length, after removing spaces and
punctuation (NFC).
"""
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import unicodedata
import uuid
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
REC = HERE / "recordings"
OUT = HERE / "results" / "stt"
DEVICES = ["builtin", "bluetooth"]
THREADS = 8
WHISPERCPP_PORT = 18080

CANDIDATES = {
    "fw-small": {"engine": "faster-whisper", "model": "fw-small", "label": "faster-whisper small (int8)"},
    "fw-medium": {"engine": "faster-whisper", "model": "fw-medium", "label": "faster-whisper medium (int8)"},
    "fw-large-v3-turbo": {"engine": "faster-whisper", "model": "fw-large-v3-turbo",
                          "label": "faster-whisper large-v3-turbo (int8)"},
    "whispercpp-turbo-q5_0": {"engine": "whisper.cpp", "model": "whispercpp-large-v3-turbo-q5_0",
                              "label": "whisper.cpp large-v3-turbo (q5_0)"},
    "sensevoice-int8": {"engine": "sherpa-onnx", "model": "sensevoice-int8",
                        "label": "SenseVoice-Small (sherpa-onnx int8)"},
}


# ---------------------------------------------------------------- scoring
def normalize(s):
    s = unicodedata.normalize("NFC", s)
    return "".join(ch for ch in s if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


# Spelling variants that are not recognition errors ("라테" is the standard spelling,
# the reference uses the common "라떼"). Used only for the lenient CER column.
VARIANTS = {"라테": "라떼"}


def normalize_lenient(s):
    s = normalize(s)
    for a, b in VARIANTS.items():
        s = s.replace(a, b)
    return s


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def utterances():
    items = []
    for dev in DEVICES:
        man = json.loads((REC / dev / "manifest.json").read_text(encoding="utf-8"))
        for key, it in sorted(man["items"].items()):
            items.append({"device": dev, "id": key, "path": str(REC / dev / it["file"]),
                          "ref": it["text"], "duration_s": it["duration_s"]})
    return items


# ---------------------------------------------------------------- worker side
def block_network():
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise OSError(f"network blocked in benchmark: {address}")
        return real_connect(self, address)

    socket.socket.connect = guarded
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    assert sr == 16000 and x.ndim == 1, path
    return x


def make_faster_whisper(model_dir):
    from faster_whisper import WhisperModel
    m = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=THREADS)

    def run(audio, path):
        segs, _ = m.transcribe(audio, language="ko", beam_size=1, best_of=1, temperature=0.0,
                               condition_on_previous_text=False, without_timestamps=True,
                               vad_filter=False)
        return "".join(s.text for s in segs).strip()
    return run, None


def make_sensevoice(model_dir):
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_dir / "model.int8.onnx"), tokens=str(model_dir / "tokens.txt"),
        num_threads=THREADS, language="ko", use_itn=True)

    def run(audio, path):
        s = rec.create_stream()
        s.accept_waveform(16000, audio)
        rec.decode_stream(s)
        return s.result.text.strip()
    return run, None


def make_whispercpp(model_dir):
    exe = MODELS / "whispercpp-bin" / "Release" / "whisper-server.exe"
    model = next(model_dir.glob("*.bin"))
    # whisper.cpp opens the model with narrow fopen, which fails on the non-ASCII repo
    # path; run from the model folder and pass the (ASCII) file name only.
    proc = subprocess.Popen(
        [str(exe), "-m", model.name, "-l", "ko", "-t", str(THREADS), "-bs", "1", "-bo", "1", "-nf",
         "-nt", "--host", "127.0.0.1", "--port", str(WHISPERCPP_PORT)],
        cwd=str(model_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{WHISPERCPP_PORT}/inference"
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{WHISPERCPP_PORT}/", timeout=1)
            break
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("whisper-server exited")
            time.sleep(0.1)

    def run(audio, path):
        boundary = uuid.uuid4().hex
        data = Path(path).read_bytes()
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n".encode(), data, b"\r\n",
            *[f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
              for k, v in {"temperature": "0.0", "temperature_inc": "0.0", "response_format": "json",
                           "language": "ko"}.items()],
            f"--{boundary}--\r\n".encode()])
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode("utf-8"))["text"].strip()
    return run, proc


def worker(name):
    import psutil
    block_network()
    cfg = CANDIDATES[name]
    model_dir = MODELS / cfg["model"]
    t0 = time.perf_counter()
    make = {"faster-whisper": make_faster_whisper, "sherpa-onnx": make_sensevoice,
            "whisper.cpp": make_whispercpp}[cfg["engine"]]
    run, server = make(model_dir)
    load_s = time.perf_counter() - t0
    items = utterances()
    try:
        warm = items[0]
        tw = time.perf_counter()
        run(load_wav(warm["path"]), warm["path"])
        warmup_s = time.perf_counter() - tw
        rows = []
        for it in items:
            audio = load_wav(it["path"])
            t = time.perf_counter()
            hyp = run(audio, it["path"])
            lat = time.perf_counter() - t
            ref_n, hyp_n = normalize(it["ref"]), normalize(hyp)
            err = edit_distance(ref_n, hyp_n)
            rows.append({**{k: it[k] for k in ("device", "id", "ref", "duration_s")}, "hyp": hyp,
                         "errors": err, "ref_chars": len(ref_n), "latency_s": round(lat, 3)})
            print(f"  {it['device']:<9} {it['id']} {lat:5.2f}s err {err:2d}  {hyp}", flush=True)
        me = psutil.Process()
        peak = me.memory_info().peak_wset
        server_peak = 0
        if server is not None:
            server_peak = psutil.Process(server.pid).memory_info().peak_wset
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=30)
    result = {"name": name, **cfg, "threads": THREADS, "load_s": round(load_s, 2),
              "warmup_s": round(warmup_s, 2), "peak_wset_bytes": peak + server_peak,
              "peak_wset_worker_bytes": peak, "peak_wset_server_bytes": server_peak, "items": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- report
def report():
    dl = json.loads((HERE / "results" / "downloads.json").read_text(encoding="utf-8"))
    lines = ["# STT 비교 (CPU, 오프라인)", "",
             "녹음 20개(내장 마이크 10 + 블루투스 10), 같은 디코딩 설정(한국어 고정, greedy, temperature 0, "
             f"스레드 {THREADS}). CER은 띄어쓰기·문장부호를 빼고 글자 단위로 계산.", "",
             "'라테/라떼 같게' 열은 표준 표기 '라테'를 오류로 세지 않은 값(참고용).", "",
             "| 모델 | 엔진 | CER 내장 | CER 블루투스 | CER 전체 | 라테/라떼 같게 | 지연 평균 (내장 / BT) | 지연 최대 | RTF | 최대 RAM | 다운로드 | 로드 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for name, cfg in CANDIDATES.items():
        p = OUT / f"{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        def cer(rows):
            return 100 * sum(x["errors"] for x in rows) / sum(x["ref_chars"] for x in rows)

        def cer_lenient(rows):
            err = sum(edit_distance(normalize_lenient(x["ref"]), normalize_lenient(x["hyp"])) for x in rows)
            return 100 * err / sum(x["ref_chars"] for x in rows)
        by = {d: [x for x in r["items"] if x["device"] == d] for d in DEVICES}
        lat = [x["latency_s"] for x in r["items"]]
        lat_dev = {d: statistics.mean(x["latency_s"] for x in by[d]) for d in DEVICES}
        rtf = sum(lat) / sum(x["duration_s"] for x in r["items"])
        size = dl[cfg["model"]]["download_bytes"] + (dl["whispercpp-bin"]["download_bytes"]
                                                     if cfg["engine"] == "whisper.cpp" else 0)
        s = {"cer_builtin": cer(by["builtin"]), "cer_bluetooth": cer(by["bluetooth"]),
             "cer_all": cer(r["items"]), "cer_all_lenient": cer_lenient(r["items"]),
             "latency_mean_s": statistics.mean(lat),
             "latency_builtin_s": lat_dev["builtin"], "latency_bluetooth_s": lat_dev["bluetooth"],
             "latency_max_s": max(lat), "rtf": rtf, "peak_ram_gb": r["peak_wset_bytes"] / 1e9,
             "download_gb": size / 1e9, "load_s": r["load_s"]}
        summary[name] = s
        lines.append(f"| {cfg['label']} | {cfg['engine']} | {s['cer_builtin']:.1f}% | {s['cer_bluetooth']:.1f}% | "
                     f"**{s['cer_all']:.1f}%** | {s['cer_all_lenient']:.1f}% | {s['latency_mean_s']:.2f} s ({s['latency_builtin_s']:.2f} / "
                     f"{s['latency_bluetooth_s']:.2f}) | {s['latency_max_s']:.2f} s | {s['rtf']:.2f} | "
                     f"{s['peak_ram_gb']:.2f} GB | {s['download_gb']:.2f} GB | {s['load_s']:.1f} s |")
    lines += ["", "## 문장별 인식 결과", ""]
    names = [n for n in CANDIDATES if (OUT / f"{n}.json").exists()]
    data = {n: json.loads((OUT / f"{n}.json").read_text(encoding="utf-8")) for n in names}
    for dev in DEVICES:
        lines += [f"### {dev}", "", "| # | 정답 | " + " | ".join(CANDIDATES[n]["label"] for n in names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for i in range(10):
            rows = [next(x for x in data[n]["items"] if x["device"] == dev and x["id"] == f"{i + 1:02d}")
                    for n in names]
            cells = [("✓ " if r["errors"] == 0 else f"({r['errors']}) ") + r["hyp"].replace("|", "/") for r in rows]
            lines.append(f"| {i + 1:02d} | {rows[0]['ref']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("✓ = 띄어쓰기·문장부호 빼고 정답과 같음, (n) = 틀린 글자 수.")
    (OUT.parent / "stt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT.parent / "stt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines[:5 + len(summary) + 2]))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args[:1] == ["--worker"]:
        return worker(args[1])
    if args != ["--report"]:
        for name in (args or list(CANDIDATES)):
            print(f"== {name}", flush=True)
            t = time.time()
            rc = subprocess.run([sys.executable, "-I", str(Path(__file__)), "--worker", name]).returncode
            print(f"== {name} done rc={rc} in {time.time() - t:.0f} s", flush=True)
    report()


if __name__ == "__main__":
    main()
