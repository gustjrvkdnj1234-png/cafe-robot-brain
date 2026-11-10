"""Step 3: score the original models (and the vending machine) on all exams.

    python scripts/run_step3.py            # run everything, resumes where it stopped
    python scripts/run_step3.py --dry-run  # list the jobs and their state

Each job is one `run_exam.py` call with a fixed --run-dir under results/step3/, so an
interrupted run is resumed case by case. A job whose summary.json already exists is skipped.
Before each model: unload every loaded model, then one warm-up request in the job's mode
(so model loading is not counted in the first case's latency).
Thinking off, temperature 0, seed 42, same system prompt as the smoke test.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bench_baseline import api_ps, unload_all  # noqa: E402
from run_exam import MODE_OPTIONS, OPTIONS  # noqa: E402

HOST = "http://127.0.0.1:11434"
OUT = ROOT / "results" / "step3"
LOG = ROOT / "logs" / "step3_run.log"
PY = sys.executable

EXAMS = {"cafe": "data/exam_cafe.jsonl", "live": "data/exam_live.jsonl", "knowledge": "data/exam_knowledge.jsonl"}
# (contestant, mode, exams). e4b/12b on the iGPU; 26b does not fit the iGPU (Step 1) -> CPU.
CONTESTANTS = [
    ("vending", "rules", ["cafe", "live"]),
    ("gemma4:e4b", "igpu", ["cafe", "live", "knowledge"]),
    ("gemma4:12b", "igpu", ["cafe", "live", "knowledge"]),
    ("gemma4:26b", "cpu", ["cafe", "live", "knowledge"]),
]


def run_dir(model: str, mode: str, exam: str) -> Path:
    return OUT / f"{exam}__{model.replace(':', '-')}__{mode}"


def log(msg: str):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def warm_up(model: str, mode: str):
    t0 = time.perf_counter()
    r = requests.post(f"{HOST}/api/chat", json={
        "model": model, "messages": [{"role": "user", "content": "안녕하세요"}], "stream": False, "think": False,
        "keep_alive": "10m", "options": {**OPTIONS, **MODE_OPTIONS[mode], "num_ctx": 8192, "num_predict": 1},
    }, timeout=1800)
    r.raise_for_status()
    ps = {k: v for k, v in api_ps().items() if not k.startswith("_")}
    log(f"warm-up {model} ({mode}) {time.perf_counter() - t0:.1f}s, loaded: "
        + ", ".join(f"{n} size={m.get('size', 0) / 1024**3:.2f}GB vram={m.get('size_vram', 0) / 1024**3:.2f}GB" for n, m in ps.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(m, mode, e) for m, mode, exams in CONTESTANTS for e in exams]
    if a.dry_run:
        for m, mode, e in jobs:
            d = run_dir(m, mode, e)
            n = len((d / "raw.jsonl").read_text(encoding="utf-8").splitlines()) if (d / "raw.jsonl").exists() else 0
            print(f"{m:12} {mode:5} {e:10} {'done' if (d / 'summary.json').exists() else f'{n} cases so far'}")
        return

    log(f"=== step 3 start ({len(jobs)} jobs) ===")
    current = None
    for m, mode, e in jobs:
        d = run_dir(m, mode, e)
        if (d / "summary.json").exists():
            log(f"skip (done): {m} {mode} {e}")
            continue
        if m != "vending" and current != (m, mode):
            log(f"unloading all models before {m}")
            unload_all()
            warm_up(m, mode)
            current = (m, mode)
        cmd = [PY, str(ROOT / "scripts" / "run_exam.py"), "--model", m, "--exam", EXAMS[e], "--run-dir", str(d)]
        if m != "vending":
            cmd += ["--mode", mode]
        log(f"run: {m} {mode} {e}")
        t0 = time.perf_counter()
        with LOG.open("a", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        log(f"done: {m} {mode} {e} rc={rc} {time.perf_counter() - t0:.0f}s")
        if rc != 0 or not (d / "summary.json").exists():
            log("job failed; stopping (rerun the script to resume)")
            sys.exit(1)
    unload_all()
    log("=== step 3 finished ===")


if __name__ == "__main__":
    main()
