"""Step 5: score a pruned model on all Step 3 exams with the same settings (resumable).

    .venv\\Scripts\\python.exe scripts\\run_step5.py --model song/gemma4-26b-cafe-98e --mode igpu
    .venv\\Scripts\\python.exe scripts\\run_step5.py --model song/gemma4-26b-cafe-98e --mode igpu --dry-run

Same as run_step3.py (thinking off, temperature 0, seed 42, num_ctx 8192, unload everything + one warm-up first),
output under results/step5/<exam>__<model>__<mode>/. A job whose summary.json exists is skipped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_step3  # noqa: E402
from bench_baseline import unload_all  # noqa: E402

OUT = ROOT / "results" / "step5"
run_step3.LOG = ROOT / "logs" / "step5_run.log"
log, warm_up, EXAMS = run_step3.log, run_step3.warm_up, run_step3.EXAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["igpu", "cpu"], default="igpu")
    ap.add_argument("--exams", default="cafe,live,knowledge")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    exams = a.exams.split(",")
    dirs = {e: OUT / f"{e}__{a.model.replace(':', '-').replace('/', '-')}__{a.mode}" for e in exams}
    if a.dry_run:
        for e, d in dirs.items():
            n = len((d / "raw.jsonl").read_text(encoding="utf-8").splitlines()) if (d / "raw.jsonl").exists() else 0
            print(f"{e:10} {'done' if (d / 'summary.json').exists() else f'{n} cases so far'}")
        return
    log(f"=== step 5 start: {a.model} ({a.mode}) {exams} ===")
    warmed = False
    for e in exams:
        d = dirs[e]
        if (d / "summary.json").exists():
            log(f"skip (done): {e}")
            continue
        if not warmed:
            log("unloading all models")
            unload_all()
            warm_up(a.model, a.mode)
            warmed = True
        cmd = [sys.executable, str(ROOT / "scripts" / "run_exam.py"), "--model", a.model, "--exam", EXAMS[e],
               "--run-dir", str(d), "--mode", a.mode]
        log(f"run: {e}")
        t0 = time.perf_counter()
        with run_step3.LOG.open("a", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        log(f"done: {e} rc={rc} {time.perf_counter() - t0:.0f}s")
        if rc != 0 or not (d / "summary.json").exists():
            log("job failed; stopping (rerun to resume)")
            sys.exit(1)
    unload_all()
    log("=== step 5 finished ===")


if __name__ == "__main__":
    main()
