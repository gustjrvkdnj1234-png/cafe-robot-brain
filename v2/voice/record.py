"""Record the 10 STT test sentences, one at a time (Windows terminal).

Usage (from the repo root):
    .venv-voice\\Scripts\\python.exe v2\\voice\\record.py builtin
    .venv-voice\\Scripts\\python.exe v2\\voice\\record.py bluetooth
    .venv-voice\\Scripts\\python.exe v2\\voice\\record.py --device 14 --label other_mic
    options: --only 3,7 (re-record some), --redo (overwrite existing files)

Keys: Enter = start, Enter = stop, then Enter = save / r = retry / p = play back.
s = skip a sentence, q = quit (already saved files are kept; rerun to resume).

Captures through WASAPI at the device's native rate, mixes to mono and resamples
to 16 kHz, then writes recordings/<label>/NN.wav (PCM 16-bit), NN.txt (exact
reference text) and manifest.json (device info, duration, peak level).
"""
import argparse
import collections
import json
import msvcrt
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

HERE = Path(__file__).resolve().parent
REC_DIR = HERE / "recordings"
TARGET_SR = 16000
PRE_ROLL_S = 0.3
TAIL_S = 0.3
PRESETS = {
    "builtin": ["마이크 배열", "Microphone Array"],
    "bluetooth": ["Buds", "수화기", "Headset", "Hands-Free"],
}


def find_device(spec):
    """Return a WASAPI input device index from an index or a name substring."""
    if spec.isdigit():
        return int(spec)
    keys = PRESETS.get(spec, [spec])
    for i, d in enumerate(sd.query_devices()):
        if (d["max_input_channels"] > 0
                and sd.query_hostapis(d["hostapi"])["name"] == "Windows WASAPI"
                and any(k.lower() in d["name"].lower() for k in keys)):
            return i
    sys.exit(f"입력 장치를 찾지 못했습니다: {spec}. devices.py로 목록을 보고 --device 번호를 쓰세요.")


def lowpass_decimate(x, factor):
    """Windowed-sinc low-pass (cutoff 0.9 * new Nyquist), then keep every factor-th sample."""
    n_taps = 64 * factor + 1
    t = np.arange(n_taps) - (n_taps - 1) / 2
    cutoff = 0.9 / factor
    h = cutoff * np.sinc(cutoff * t) * np.kaiser(n_taps, 8.0)
    h /= h.sum()
    return np.convolve(x, h, mode="same")[::factor]


def to_16k(x, sr):
    if sr == TARGET_SR:
        return x
    if sr % TARGET_SR == 0:
        return lowpass_decimate(x, sr // TARGET_SR)
    # Non-integer ratio (e.g. 44.1 kHz): FFT resampling, band-limited to the new Nyquist.
    n_out = int(round(len(x) * TARGET_SR / sr))
    spec = np.fft.rfft(x)[: n_out // 2 + 1]
    return np.fft.irfft(spec, n_out) * (n_out / len(x))


def dbfs(x):
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return 20 * np.log10(peak) if peak > 0 else -120.0


def read_key():
    """Block until a key is pressed; return it lower-cased ('\\r' for Enter)."""
    while msvcrt.kbhit():
        msvcrt.getwch()
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):  # arrow/function keys send two codes
        msvcrt.getwch()
        return ""
    return ch.lower()


class Recorder:
    def __init__(self, device):
        info = sd.query_devices(device)
        self.device = device
        self.name = info["name"]
        self.sr = int(info["default_samplerate"])
        self.channels = min(2, info["max_input_channels"])
        self.lock = threading.Lock()
        self.pre = collections.deque(maxlen=max(1, int(PRE_ROLL_S * self.sr / 512) + 1))
        self.blocks = []
        self.recording = False
        self.level = -120.0
        self.stream = sd.InputStream(device=device, samplerate=self.sr, channels=self.channels,
                                     dtype="float32", blocksize=512, callback=self._cb)

    def _cb(self, indata, frames, t, status):
        mono = indata.mean(axis=1).copy()
        self.level = dbfs(mono)
        with self.lock:
            if self.recording:
                self.blocks.append(mono)
            else:
                self.pre.append(mono)

    def start(self):
        with self.lock:
            self.blocks = list(self.pre)
            self.recording = True

    def stop(self):
        time.sleep(TAIL_S)
        with self.lock:
            self.recording = False
            blocks, self.blocks = self.blocks, []
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


def output_candidates():
    """WASAPI default output first, then PortAudio's default; None = PortAudio default."""
    cands = []
    for h in sd.query_hostapis():
        if h["name"] == "Windows WASAPI" and h["default_output_device"] >= 0:
            cands.append(h["default_output_device"])
    cands.append(None)
    return cands


def play(audio):
    """Play back a take. Never raises: a Bluetooth headset in call mode can refuse
    to open an output stream (PaErrorCode -9985), which must not stop recording."""
    errors = []
    for dev in output_candidates():
        try:
            extra = sd.WasapiSettings(auto_convert=True) if dev is not None else None
            sd.play(audio, TARGET_SR, device=dev, extra_settings=extra)
            sd.wait()
            return
        except Exception as e:  # PortAudioError, ValueError (unsupported rate), ...
            errors.append(f"{dev}: {e}")
            try:
                sd.stop()
            except Exception:
                pass
    print("  ⚠ 들어 보기 실패 (녹음·저장에는 영향 없음):", " | ".join(errors))


def meter(db):
    n = int(max(0, min(30, (db + 60) / 2)))
    return "#" * n + "." * (30 - n)


def record_one(rec):
    rec.start()
    while not msvcrt.kbhit():
        print(f"\r  ● 녹음 중  [{meter(rec.level)}]  Enter: 멈춤 ", end="", flush=True)
        time.sleep(0.05)
    msvcrt.getwch()
    print()
    return to_16k(rec.stop(), rec.sr).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="STT 시험용 10문장 녹음")
    ap.add_argument("preset", nargs="?", choices=sorted(PRESETS), help="builtin 또는 bluetooth")
    ap.add_argument("--device", help="장치 번호 또는 이름 일부 (devices.py 참고)")
    ap.add_argument("--label", help="저장 폴더 이름 (기본: 프리셋 이름)")
    ap.add_argument("--only", help="다시 녹음할 번호, 예: 3,7")
    ap.add_argument("--redo", action="store_true", help="이미 있는 파일도 다시 녹음")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.preset and not args.device:
        ap.error("builtin / bluetooth 중 하나 또는 --device를 주세요")

    label = args.label or args.preset or "custom"
    device = find_device(args.device or args.preset)
    sentences = json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))["stt"]
    only = {int(s) for s in args.only.split(",")} if args.only else None

    out = REC_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"items": {}}

    rec = Recorder(device)
    manifest.update({"label": label, "device_index": device, "device_name": rec.name,
                     "hostapi": "Windows WASAPI", "native_samplerate": rec.sr,
                     "native_channels": rec.channels, "samplerate": TARGET_SR})
    print(f"장치: [{device}] {rec.name}  ({rec.sr} Hz, {rec.channels}ch → 16 kHz mono)")
    print(f"저장: {out}")
    if label == "bluetooth":
        print("참고: 마이크를 여는 동안 이어폰이 통화 모드로 바뀌어 음악 음질이 떨어질 수 있습니다.")
    print("키: Enter 시작/멈춤 · s 건너뛰기 · q 끝내기\n")

    with rec.stream:
        time.sleep(0.5)  # let the device settle (BT switches profile here)
        for n, text in enumerate(sentences, 1):
            wav = out / f"{n:02d}.wav"
            if only is not None and n not in only:
                continue
            if wav.exists() and not args.redo and only is None:
                print(f"[{n:02d}/10] 이미 있음, 건너뜀 (--redo로 다시)")
                continue
            print(f"[{n:02d}/10]  {text}")
            print("  Enter: 녹음 시작 / s: 건너뛰기 / q: 끝내기")
            k = read_key()
            if k == "q":
                break
            if k == "s":
                continue
            while True:
                audio = record_one(rec)
                dur, peak = len(audio) / TARGET_SR, dbfs(audio)
                warn = ""
                if peak > -1.0:
                    warn = "  ⚠ 너무 큼(클리핑) — 조금 떨어져서 다시"
                elif peak < -30.0:
                    warn = "  ⚠ 소리가 작음 — 마이크 음량/거리 확인"
                elif dur < 0.8:
                    warn = "  ⚠ 너무 짧음"
                print(f"  {dur:.1f}초, 최대 {peak:.1f} dBFS{warn}")
                print("  Enter: 저장하고 다음 / r: 다시 / p: 들어 보기 / q: 저장 안 하고 끝내기")
                k = read_key()
                while k == "p":
                    play(audio)
                    k = read_key()
                if k == "r":
                    continue
                break
            if k == "q":
                break
            sf.write(wav, audio, TARGET_SR, subtype="PCM_16")
            (out / f"{n:02d}.txt").write_text(text, encoding="utf-8")
            manifest["items"][f"{n:02d}"] = {
                "file": wav.name, "text": text, "duration_s": round(dur, 2),
                "peak_dbfs": round(peak, 1), "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }
            man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print("  저장됨\n")

    done = sorted(manifest["items"])
    print(f"\n완료: {len(done)}/10  {', '.join(done)}")
    missing = [f"{i:02d}" for i in range(1, 11) if f"{i:02d}" not in manifest["items"]]
    if missing:
        print(f"남음: {', '.join(missing)}  (같은 명령을 다시 실행하면 이어서 녹음)")


if __name__ == "__main__":
    main()
