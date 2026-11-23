"""Download the STT/TTS candidates into v2/voice/models/<name>/ (one at a time).

Usage: .venv-voice\\Scripts\\python.exe v2\\voice\\download_models.py [name ...]
Writes v2/voice/results/downloads.json (source, bytes downloaded, bytes on disk).
After this step every benchmark runs with HF_HUB_OFFLINE=1 and no network.
"""
import json
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
GH = "https://github.com"

# name -> (kind, source, files/patterns)
CANDIDATES = {
    "fw-small": ("hf", "Systran/faster-whisper-small", None),
    "fw-medium": ("hf", "Systran/faster-whisper-medium", None),
    "fw-large-v3-turbo": ("hf", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", None),
    # The 2025-09-09 "int8" repo is a Cantonese fine-tune (ASLP-lab/WSYue-ASR); it outputs
    # Chinese for Korean speech. Use the original multilingual SenseVoiceSmall export.
    "sensevoice-int8": ("hf", "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                        ["model.int8.onnx", "tokens.txt", "LICENSE", "README.md"]),
    "whispercpp-bin": ("zip", f"{GH}/ggml-org/whisper.cpp/releases/download/b5454/whisper-bin-x64.zip", None),
    "whispercpp-large-v3-turbo-q5_0": ("hf", "ggerganov/whisper.cpp", ["ggml-large-v3-turbo-q5_0.bin"]),
    "supertonic3-int8": ("tar", f"{GH}/k2-fsa/sherpa-onnx/releases/download/tts-models/"
                                "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2", None),
    "supertonic2-int8": ("tar", f"{GH}/k2-fsa/sherpa-onnx/releases/download/tts-models/"
                                "sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2", None),
    "mimic3-ko-kss": ("tar", f"{GH}/k2-fsa/sherpa-onnx/releases/download/tts-models/"
                             "vits-mimic3-ko_KO-kss_low.tar.bz2", None),
}


def dir_bytes(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def fetch(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    names = sys.argv[1:] or list(CANDIDATES)
    log_path = HERE / "results" / "downloads.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    for name in names:
        kind, src, files = CANDIDATES[name]
        out = MODELS / name
        if name in log and out.exists():
            print(f"{name}: already downloaded ({log[name]['download_bytes'] / 1e6:.0f} MB)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if kind == "hf":
            snapshot_download(src, local_dir=out, allow_patterns=files)
            dl = dir_bytes(out)
        else:
            archive = MODELS / f"_{name}{'.zip' if kind == 'zip' else '.tar.bz2'}"
            dl = fetch(src, archive)
            if kind == "zip":
                with zipfile.ZipFile(archive) as z:
                    for m in z.namelist():  # refuse absolute / parent paths
                        if m.startswith(("/", "\\")) or ".." in Path(m).parts:
                            sys.exit(f"unsafe path in {archive}: {m}")
                    z.extractall(out)
            else:
                with tarfile.open(archive, "r:bz2") as t:
                    t.extractall(out, filter="data")
            archive.unlink()
        log[name] = {"kind": kind, "source": src, "files": files, "download_bytes": dl,
                     "disk_bytes": dir_bytes(out), "seconds": round(time.time() - t0, 1)}
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"{name}: {dl / 1e6:.0f} MB in {log[name]['seconds']} s")
    total = sum(v["download_bytes"] for v in log.values())
    print(f"total downloaded: {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
