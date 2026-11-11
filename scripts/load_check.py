"""Load one model in a given mode and report whether it fits on the iGPU, with peak memory during loading.

    .venv\\Scripts\\python.exe scripts\\load_check.py --model song/gemma4-26b-cafe-98e --mode igpu --out results/step5/load_igpu.json

Unloads every model first, then sends one 1-token chat request (num_ctx 8192, same options as run_exam.py).
Reports `ollama ps` (processor split) and /api/ps size / size_vram, and peak runner / GPU shared / system memory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bench_baseline import api_ps, unload_all  # noqa: E402
from run_exam import MODE_OPTIONS, OPTIONS, MemorySampler  # noqa: E402

HOST = "http://127.0.0.1:11434"
GB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=sorted(MODE_OPTIONS), default="igpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    unload_all()
    mem = MemorySampler(interval_s=1.0).start()
    t0 = time.perf_counter()
    res = {"model": a.model, "mode": a.mode}
    try:
        r = requests.post(f"{HOST}/api/chat", json={
            "model": a.model, "messages": [{"role": "user", "content": "안녕하세요"}], "stream": False, "think": False,
            "keep_alive": "10m", "options": {**OPTIONS, **MODE_OPTIONS[a.mode], "num_ctx": 8192, "num_predict": 1}}, timeout=1800)
        res["http_status"] = r.status_code
        if r.status_code != 200:
            res["error"] = r.text[:500]
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    res["load_s"] = round(time.perf_counter() - t0, 1)
    time.sleep(3)
    res["peak_memory"] = mem.stop()
    ps = {k: v for k, v in api_ps().items() if not k.startswith("_")}
    m = ps.get(a.model) or ps.get(a.model + ":latest") or {}
    res["api_ps"] = {"size_gb": round(m.get("size", 0) / GB, 2), "size_vram_gb": round(m.get("size_vram", 0) / GB, 2),
                     "context_length": m.get("context_length")}
    res["ollama_ps"] = subprocess.run(["ollama", "ps"], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    s, v = m.get("size", 0), m.get("size_vram", 0)
    res["fits_igpu_fully"] = bool(res.get("http_status") == 200 and s and v >= s)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
