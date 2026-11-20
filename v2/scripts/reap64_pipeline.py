"""Build and evaluate REAP-64: pure REAP keep-set (64 experts/layer), same pipeline as v1 64e. Resumable.

    .venv-prune\\Scripts\\python.exe v2\\scripts\\reap64_pipeline.py            # all steps, finished ones skipped
    .venv-prune\\Scripts\\python.exe v2\\scripts\\reap64_pipeline.py --only plan
    .venv-prune\\Scripts\\python.exe v2\\scripts\\reap64_pipeline.py --keep 98   # also 48 -> v2/results/reap_curve/reap<k>

Keep-set: v1 Step 4 plain REAP ranking (results/step4/reap_prune_plan.json, removed lists for keep 64), i.e. combined
REAP score on the plain v1 statistics, ties by router probability mass. Re-derived here from the v1 stats with
plan_prune.plain_rank and checked against that file.

Pipeline = v1 scripts/overnight_step5.py pruned(): build_pruned.py -> validate_pruned.py (informational) ->
convert_hf_to_gguf.py bf16 -> verify_pruned_files.py -> llama-quantize Q4_K_M with the v1 tensor-type file and q8_0
token embedding, no imatrix -> verify_q4.py -> delete BF16 GGUF and pruned safetensors (config + prune_info.json stay)
-> ollama create with the v1 Modelfile (original gemma4:26b Modelfile, FROM replaced) -> blob hash check -> delete the
local Q4 copy -> iGPU load check -> 5-case smoke -> exams with v1 run_step5.py (outputs redirected to v2).

Safety: the original checkpoint is only read; deletes only inside models/gguf and the REAP-64 folder; >= 20 GB free
disk before every step (plus what the step writes); one job at a time; peak RSS of every step's process tree is
sampled and recorded; a step is stopped if its private memory exceeds 20 GB (RSS, which also counts
memory-mapped file pages, is recorded too).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from plan_prune import plain_rank  # noqa: E402  (v1, read only)

ORIG = (ROOT / "models/gemma-4-26B-A4B-it").resolve()
GGUF_DIR = ROOT / "models/gguf"
MDIR_REL = "models/gemma-4-26B-A4B-it-cafe-64e-reap"
MDIR = ROOT / MDIR_REL
OUT = ROOT / "v2/results/reap64"
PLAN = OUT / "prune_plan_reap64.json"
STATE = OUT / "pipeline_state.json"
LOG = ROOT / "logs/reap64_pipeline.log"
STEP_LOGS = ROOT / "logs/reap64"
PRUNE_PY = str(ROOT / ".venv-prune/Scripts/python.exe")
EXAM_PY = str(ROOT / ".venv/Scripts/python.exe")
LLAMA = ROOT / "tools/llama.cpp-b12283"
CONVERT = LLAMA / "llama.cpp-b12283/convert_hf_to_gguf.py"
QUANTIZE = LLAMA / "bin/llama-quantize.exe"
TENSOR_TYPES = GGUF_DIR / "tensor_types_ollama26b.txt"
MODELFILE_ORIG = GGUF_DIR / "Modelfile.gemma4-26b.orig"
NAME = "song/gemma4-26b-cafe-64e-reap"
V1_NAME = "song/gemma4-26b-cafe-64e"
BF16 = GGUF_DIR / "gemma4-26b-cafe-64e-reap-bf16.gguf"
Q4 = GGUF_DIR / "gemma4-26b-cafe-64e-reap-Q4_K_M.gguf"
HOST = "http://127.0.0.1:11434"
MIN_FREE_GB, MAX_RSS_GB, GB = 20, 20, 1e9
SMOKE_IDS = "O3-1,C1-2,T2-1,D1-2,I3-1"
K = 64


class Stop(Exception):
    pass


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def mark(step, **info):
    s = state()
    s[step] = {"done": datetime.now().isoformat(timespec="seconds"), **info}
    STATE.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")


def free_gb():
    return shutil.disk_usage(ROOT).free / GB


def check_disk(step, need_gb=0.0):
    f = free_gb()
    log(f"[{step}] free disk {f:.1f} GB (step writes ~{need_gb:.0f} GB)")
    if f < MIN_FREE_GB or f - need_gb < MIN_FREE_GB:
        raise Stop(f"disk: {f:.1f} GB free before {step}, step writes ~{need_gb:.0f} GB")


def safe_delete(path: Path):
    p = path.resolve()
    if p == ORIG or ORIG in p.parents:
        raise Stop(f"refusing to delete inside the original checkpoint: {p}")
    if not (p.parent == GGUF_DIR.resolve() or p.parent == MDIR.resolve()):
        raise Stop(f"refusing to delete outside allowed folders: {p}")
    if p.exists():
        size = p.stat().st_size
        p.unlink()
        log(f"deleted {p.relative_to(ROOT.resolve())} ({size / GB:.1f} GB)")


def tree_mem(proc):
    """(rss, private) bytes of the process tree. RSS includes memory-mapped file pages (e.g. GGUFReader's np.memmap of
    a 28 GB file), which the OS can drop; private = the processes' own committed memory. The limit applies to private."""
    rss = priv = 0
    try:
        for p in [proc, *proc.children(recursive=True)]:
            try:
                mi = p.memory_info()
                rss += mi.rss
                priv += getattr(mi, "private", mi.rss)
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return rss, priv


def run(step, cmd, timeout=None, cwd=ROOT):
    """One heavy job; stdout to logs/reap64/<step>.log; peak RSS of the process tree sampled every 0.5 s."""
    STEP_LOGS.mkdir(parents=True, exist_ok=True)
    out = STEP_LOGS / f"{step}.log"
    log(f"[{step}] run: {' '.join(str(c) for c in cmd)[:300]}")
    t0 = time.perf_counter()
    peak = [0, 0]                                   # rss, private
    with out.open("w", encoding="utf-8") as f:
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        ps = psutil.Process(p.pid)
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                rss, priv = tree_mem(ps)
                peak[0], peak[1] = max(peak[0], rss), max(peak[1], priv)
                if peak[1] > MAX_RSS_GB * GB:
                    log(f"[{step}] private memory {peak[1] / GB:.1f} GB > {MAX_RSS_GB} GB, killing")
                    for c in [*ps.children(recursive=True), ps]:
                        try:
                            c.kill()
                        except psutil.Error:
                            pass
                    return
                stop.wait(0.5)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = "timeout"
        stop.set()
        th.join()
    dt = time.perf_counter() - t0
    log(f"[{step}] rc={rc} in {dt / 60:.1f} min, peak private {peak[1] / GB:.2f} GB, peak RSS {peak[0] / GB:.2f} GB")
    if peak[1] > MAX_RSS_GB * GB:
        raise Stop(f"{step}: peak private memory {peak[1] / GB:.1f} GB over limit")
    if rc != 0:
        tail = out.read_text(encoding="utf-8", errors="replace")[-800:].replace("\n", " ")
        raise Stop(f"{step} rc={rc}: ...{tail}")
    return out, {"seconds": round(dt, 1), "peak_private_gb": round(peak[1] / GB, 2), "peak_rss_gb": round(peak[0] / GB, 2)}


# ------------------------------------------------------------------ ollama

def ollama_up():
    try:
        with urllib.request.urlopen(f"{HOST}/api/version", timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def ensure_ollama():
    """Same server settings as v1 Step 5: only OLLAMA_IGPU_ENABLE=1, for the serve process only."""
    if ollama_up():
        return
    log("starting ollama serve with OLLAMA_IGPU_ENABLE=1 (serve process only)")
    env = dict(os.environ, OLLAMA_IGPU_ENABLE="1")
    logf = open(ROOT / "logs/ollama_serve_reap64.log", "a", encoding="utf-8")  # noqa: SIM115
    subprocess.Popen(["ollama", "serve"], cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                     creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    for _ in range(60):
        time.sleep(2)
        if ollama_up():
            return
    raise Stop("ollama server did not start")


def unload_all():
    try:
        with urllib.request.urlopen(f"{HOST}/api/ps", timeout=30) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        for n in names:
            req = urllib.request.Request(f"{HOST}/api/generate", data=json.dumps({"model": n, "keep_alive": 0}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=120).read()
        time.sleep(5)
    except Exception as e:  # noqa: BLE001
        log(f"unload_all: {e!r}")


def sha256(path: Path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(64 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def modelfile_body(name):
    show = subprocess.run(["ollama", "show", "--modelfile", name], capture_output=True, text=True, encoding="utf-8").stdout
    lines = [l for l in show.split("\n") if not l.startswith("#")]
    first_from = next(i for i, l in enumerate(lines) if l.startswith("FROM "))   # the language-model GGUF
    return lines[:first_from] + lines[first_from + 1:], show


# ------------------------------------------------------------------ steps

def make_plan():
    v1plan = json.loads((ROOT / "results/step4/reap_prune_plan.json").read_text(encoding="utf-8"))
    layers, mism = {}, []
    for i in range(30):
        s = json.loads((ROOT / f"results/step4/reap_600/stats_layer{i:02d}.json").read_text(encoding="utf-8"))
        order = plain_rank(s["combined"])
        kept, removed = sorted(order[:K]), sorted(order[K:])
        if set(removed) != set(v1plan["keep"][str(K)][str(i)]):
            mism.append(i)
        layers[str(i)] = {"kept": kept, "removed": removed}
    if mism:
        raise Stop(f"plain REAP keep-set differs from results/step4/reap_prune_plan.json in layers {mism}")
    v1_64 = json.loads((ROOT / f"results/step5/prune_plan_{K}.json").read_text(encoding="utf-8"))["layers"]
    diff = {i: len(set(layers[i]["kept"]) - set(v1_64[i]["kept"])) for i in layers}
    plan = {"keep": K, "num_experts_original": 128, "top_k": 8, "source_stats": "results/step4/reap_600",
            "ranking": "pure REAP: mean g*||f(x)|| over routed tokens on the plain combined v1 stats, "
                       "ties by router prob mass, then index (= results/step4/reap_prune_plan.json keep 64)",
            "checked_against": "results/step4/reap_prune_plan.json (all 30 layers equal)",
            f"experts_differing_from_v1_{K}e_per_layer": diff, "layers": layers}
    OUT.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    log(f"plan written; experts kept by REAP-{K} but not v1 {K}e: total {sum(diff.values())} "
        f"(mean {sum(diff.values()) / 30:.1f}/layer)")


def register():
    ensure_ollama()
    lines = MODELFILE_ORIG.read_text(encoding="utf-8").split("\n")
    assert lines[4].startswith("FROM ") and "a8bdd07a" in lines[4]
    lines[4] = f"FROM ./{Q4.name}"
    mf = GGUF_DIR / f"Modelfile.{NAME.split('/')[-1]}"
    mf.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    shutil.copy2(mf, OUT / mf.name)
    STEP_LOGS.mkdir(parents=True, exist_ok=True)
    with (STEP_LOGS / "register.log").open("w", encoding="utf-8") as f:
        rc = subprocess.run(["ollama", "create", NAME, "-f", mf.name], cwd=GGUF_DIR, stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    if rc != 0:
        raise Stop(f"ollama create rc={rc}")
    body_new, show = modelfile_body(NAME)
    body_v1, _ = modelfile_body(V1_NAME)
    if sorted(body_new) != sorted(body_v1):          # `ollama show` may list PARAMETER lines in a different order
        raise Stop(f"registered Modelfile differs from {V1_NAME} beyond the main FROM line")
    blob = re.search(r"^FROM .*sha256-([0-9a-f]{64})", show, re.M)
    local = sha256(Q4)
    if not blob or blob.group(1) != local:
        raise Stop(f"registered blob {blob.group(1) if blob else None} != local Q4 {local}")
    log(f"registered {NAME}; blob sha256 == local Q4 ({local[:12]}); Modelfile equals {V1_NAME} except FROM")
    safe_delete(Q4)
    return {"blob_sha256": local}


def exams(mode):
    """v1 run_step5.py with its output folder redirected to v2/results/reap64."""
    code = ("import sys; sys.path.insert(0, r'%s'); import run_step5; from pathlib import Path; "
            "run_step5.OUT = Path(r'%s'); run_step5.run_step3.LOG = Path(r'%s'); "
            "sys.argv = ['run_step5', '--model', '%s', '--mode', '%s']; run_step5.main()"
            % (ROOT / "scripts", OUT, ROOT / f"logs/reap{K}_exams.log", NAME, mode))
    return run("exams", [EXAM_PY, "-c", code], timeout=6 * 3600)[1]


def configure(k):
    """Paths for another keep size. k = 64 keeps the original REAP-64 layout (v2/results/reap64)."""
    global K, MDIR_REL, MDIR, OUT, PLAN, STATE, LOG, STEP_LOGS, NAME, V1_NAME, BF16, Q4
    if k == 64:
        return
    K = k
    MDIR_REL = f"models/gemma-4-26B-A4B-it-cafe-{k}e-reap"
    MDIR = ROOT / MDIR_REL
    OUT = ROOT / f"v2/results/reap_curve/reap{k}"
    PLAN = OUT / f"prune_plan_reap{k}.json"
    STATE = OUT / "pipeline_state.json"
    LOG = ROOT / f"logs/reap{k}_pipeline.log"
    STEP_LOGS = ROOT / f"logs/reap{k}"
    NAME = f"song/gemma4-26b-cafe-{k}e-reap"
    V1_NAME = f"song/gemma4-26b-cafe-{k}e"
    BF16 = GGUF_DIR / f"gemma4-26b-cafe-{k}e-reap-bf16.gguf"
    Q4 = GGUF_DIR / f"gemma4-26b-cafe-{k}e-reap-Q4_K_M.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--keep", type=int, default=64, choices=[98, 64, 48])
    ap.add_argument("--continue-after-smoke-fail", action="store_true",
                    help="record a failed 5-case smoke gate but still run the full exams (build already verified)")
    a = ap.parse_args()
    configure(a.keep)
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== reap{K} start (free {free_gb():.1f} GB) ===")
    bf16_gb = 51.6 * (0.115 + 0.885 * K / 128)
    steps = []

    def step(name):
        def deco(fn):
            steps.append((name, fn))
            return fn
        return deco

    @step("plan")
    def _():
        make_plan()
        return {}

    @step("build")
    def _():
        check_disk("build", bf16_gb)
        if MDIR.exists():
            for p in MDIR.glob("*"):
                safe_delete(p)
            MDIR.rmdir()
        return run("build", [PRUNE_PY, "scripts/build_pruned.py", "--plan", PLAN.relative_to(ROOT).as_posix(),
                             "--dst", MDIR_REL])[1]

    @step("validate")
    def _():
        check_disk("validate")
        return run("validate", [PRUNE_PY, "scripts/validate_pruned.py", "--pruned", MDIR_REL, "--n", "8",
                                "--out", (OUT / f"validate_reap{K}.json").relative_to(ROOT).as_posix()])[1]

    @step("convert")
    def _():
        check_disk("convert", bf16_gb)
        safe_delete(BF16)
        return run("convert", [PRUNE_PY, CONVERT, MDIR_REL, "--outtype", "bf16", "--outfile", BF16])[1]

    @step("verify_bf16")
    def _():
        return run("verify_bf16", [PRUNE_PY, "scripts/verify_pruned_files.py", "--pruned", MDIR_REL, "--gguf", BF16])[1]

    @step("quantize")
    def _():
        check_disk("quantize", 17.07 * (0.17 + 0.83 * K / 128))
        safe_delete(Q4)
        info = run("quantize", [QUANTIZE, "--tensor-type-file", TENSOR_TYPES, "--token-embedding-type", "q8_0",
                                BF16, Q4, "Q4_K_M", "8"])[1]
        return {**info, "q4_bytes": Q4.stat().st_size}

    @step("verify_q4")
    def _():
        info = run("verify_q4", [PRUNE_PY, "scripts/verify_q4.py", "--gguf", Q4, "--experts", str(K)])[1]
        safe_delete(BF16)
        for p in MDIR.glob("model-*.safetensors"):
            safe_delete(p)
        return info

    @step("register")
    def _():
        check_disk("register", Q4.stat().st_size / GB if Q4.exists() else 14)
        return register()

    @step("load")
    def _():
        ensure_ollama()
        info = run("load_igpu", [EXAM_PY, "scripts/load_check.py", "--model", NAME, "--mode", "igpu",
                                 "--out", (OUT / "load_igpu.json").relative_to(ROOT).as_posix()], timeout=3600)[1]
        unload_all()
        res = json.loads((OUT / "load_igpu.json").read_text(encoding="utf-8"))
        return {**info, "fits_igpu_fully": res.get("fits_igpu_fully")}

    @step("smoke")
    def _():
        ensure_ollama()
        out, info = run("smoke", [EXAM_PY, "scripts/run_exam.py", "--model", NAME, "--exam", "data/exam_cafe.jsonl",
                                  "--ids", SMOKE_IDS, "--out-dir", (OUT / "smoke").relative_to(ROOT).as_posix(),
                                  "--mode", "igpu"])
        unload_all()
        m = re.search(r"saved: (\S+)", out.read_text(encoding="utf-8", errors="replace"))
        d = ROOT / m.group(1)
        rows = [json.loads(l) for l in (d / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        n = sum(1 for r in rows if r.get("tool_calls"))
        log(f"smoke: {n}/5 cases with tool calls")
        if n < 3 and not a.continue_after_smoke_fail:
            raise Stop(f"smoke: only {n}/5 cases produced tool calls")
        if n < 3:
            log(f"smoke gate failed ({n}/5) but --continue-after-smoke-fail given: running the full exams anyway")
        return {**info, "cases_with_tool_calls": n, "gate_passed": n >= 3}

    @step("exams")
    def _():
        ensure_ollama()
        st = state()
        mode = "igpu" if st["load"].get("fits_igpu_fully") else "cpu"
        return {**exams(mode), "mode": mode}

    try:
        for name, fn in steps:
            if a.only and a.only != name:
                continue
            if name in state():
                continue
            log(f"--- {name} ---")
            mark(name, **fn())
    except Stop as e:
        log(f"STOPPED: {e}")
        sys.exit(1)
    log(f"=== reap{K} end (free {free_gb():.1f} GB) ===")


if __name__ == "__main__":
    main()
