"""Check recorded WAV levels (near-silent / clipped) against manifest.json.

Usage: .venv-voice\\Scripts\\python.exe v2\\voice\\check_levels.py
Writes v2/voice/results/levels.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
SILENT_PEAK_DBFS = -30.0
CLIP_LEVEL = 0.999


def db(v):
    return float(20 * np.log10(v)) if v > 0 else -120.0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    for d in sorted((HERE / "recordings").iterdir()):
        man_path = d / "manifest.json"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        print(f"{d.name}: [{man['device_index']}] {man['device_name']} "
              f"{man['native_samplerate']} Hz {man['native_channels']}ch")
        rows = []
        for key, item in sorted(man["items"].items()):
            x, sr = sf.read(d / item["file"], dtype="float32")
            peak = db(float(np.abs(x).max()))
            rms = db(float(np.sqrt(np.mean(x ** 2))))
            # speech-active RMS: loudest 20 ms frames (top 30 %)
            fr = x[: len(x) // 320 * 320].reshape(-1, 320)
            frms = np.sqrt((fr ** 2).mean(axis=1))
            active = db(float(np.mean(np.sort(frms)[-max(1, len(frms) * 3 // 10):])))
            floor = db(float(np.percentile(frms, 10)))
            clipped = int((np.abs(x) >= CLIP_LEVEL).sum())
            text_ok = (d / f"{key}.txt").read_text(encoding="utf-8") == item["text"]
            flags = []
            if peak < SILENT_PEAK_DBFS:
                flags.append("near-silent")
            if clipped:
                flags.append(f"clipped({clipped})")
            if sr != 16000 or x.ndim != 1:
                flags.append("format")
            if not text_ok:
                flags.append("text-mismatch")
            row = {"id": key, "samplerate": sr, "duration_s": round(len(x) / sr, 2),
                   "peak_dbfs": round(peak, 1), "rms_dbfs": round(rms, 1),
                   "speech_rms_dbfs": round(active, 1), "noise_floor_dbfs": round(floor, 1),
                   "snr_est_db": round(active - floor, 1), "clipped_samples": clipped,
                   "manifest_peak_dbfs": item["peak_dbfs"], "recorded_at": item["recorded_at"],
                   "flags": flags}
            rows.append(row)
            print(f"  {key} {row['duration_s']:5.2f}s peak {peak:6.1f} speech {active:6.1f} "
                  f"floor {floor:6.1f} SNR~{row['snr_est_db']:5.1f} dB  {' '.join(flags) or 'ok'}")
        out[d.name] = {"device_name": man["device_name"], "native_samplerate": man["native_samplerate"],
                       "items": rows}
    path = HERE / "results" / "levels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
