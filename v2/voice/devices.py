"""List audio input/output devices and their native sample rates.

Usage:
    .venv-voice\\Scripts\\python.exe v2\\voice\\devices.py

Writes v2/voice/results/devices.json. WASAPI rows are the ones record.py uses
(WASAPI shared mode reports the device's real mix format).
"""
import json
import sys
from pathlib import Path

import sounddevice as sd

OUT = Path(__file__).resolve().parent / "results" / "devices.json"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for i, d in enumerate(sd.query_devices()):
        rows.append({
            "index": i,
            "hostapi": sd.query_hostapis(d["hostapi"])["name"],
            "name": d["name"],
            "max_input_channels": d["max_input_channels"],
            "max_output_channels": d["max_output_channels"],
            "default_samplerate": d["default_samplerate"],
        })
    default_in, default_out = sd.default.device
    print(f"{'idx':>3}  {'host API':<20} {'in':>2} {'out':>3} {'rate':>7}  name")
    for r in rows:
        mark = " *" if r["index"] in (default_in, default_out) else ""
        print(f"{r['index']:>3}  {r['hostapi']:<20} {r['max_input_channels']:>2} "
              f"{r['max_output_channels']:>3} {int(r['default_samplerate']):>7}  {r['name']}{mark}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"default_input": default_in, "default_output": default_out,
                               "devices": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n* = default device. saved {OUT}")


if __name__ == "__main__":
    main()
