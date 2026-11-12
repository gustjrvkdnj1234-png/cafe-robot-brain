"""Step 5 overnight pipeline: no-imatrix control, then keep = 64 and 48. One heavy job at a time.

    .venv-prune\\Scripts\\python.exe scripts\\overnight_step5.py            # resumes: finished steps are skipped
    .venv-prune\\Scripts\\python.exe scripts\\overnight_step5.py --only control

Safety
- The original checkpoint (models/gemma-4-26B-A4B-it) is only ever read. Every delete goes through safe_delete(),
  which refuses any path inside it (or outside models/gguf and the pruned folders).
- Before every step: free disk must be >= 20 GB, and >= 20 GB + the size the step will write. Otherwise stop everything.
- A failed step is logged (logs/overnight_step5.log + results/step5/overnight_notes.md); the rest of that model is
  skipped and the pipeline moves on to the next model.
- Intermediates (pruned BF16 safetensors, BF16 GGUF) are deleted only after the Q4 file passed verify_q4.py; the local Q4
  copy is deleted only after its SHA-256 equals the blob Ollama registered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG = (ROOT / "models/gemma-4-26B-A4B-it").resolve()
GGUF_DIR = ROOT / "models/gguf"
S5 = ROOT / "results/step5"
LOG = ROOT / "logs/overnight_step5.log"
STEP_LOGS = ROOT / "logs/overnight"
NOTES = S5 / "overnight_notes.md"
STATE = S5 / "overnight_state.json"
PRUNE_PY = str(ROOT / ".venv-prune/Scripts/python.exe")
EXAM_PY = str(ROOT / ".venv/Scripts/python.exe")
LLAMA = ROOT / "tools/llama.cpp-b12283"
CONVERT = LLAMA / "llama.cpp-b12283/convert_hf_to_gguf.py"
QUANTIZE = LLAMA / "bin/llama-quantize.exe"
TENSOR_TYPES = GGUF_DIR / "tensor_types_ollama26b.txt"
MODELFILE_ORIG = GGUF_DIR / "Modelfile.gemma4-26b.orig"
HOST = "http://127.0.0.1:11434"
MIN_FREE_GB = 20
SMOKE_IDS = "O3-1,C1-2,T2-1,D1-2,I3-1"
GB = 1e9


class Stop(Exception):
    """Stop the whole pipeline (disk or safety)."""


class StepFailed(Exception):
    pass


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def note(section, msg):
    """Append a Korean note to overnight_notes.md under '## <section>'."""
    text = NOTES.read_text(encoding="utf-8") if NOTES.exists() else "# 밤사이 실행 기록\n"
    head = f"\n## {section}\n"
    if head not in text:
        text = text.rstrip() + "\n" + head
    text = text.rstrip() + f"\n- {datetime.now():%m-%d %H:%M} {msg}\n"
    NOTES.write_text(text, encoding="utf-8")


def state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def mark(step):
    s = state()
    s[step] = datetime.now().isoformat(timespec="seconds")
    STATE.write_text(json.dumps(s, indent=1), encoding="utf-8")


def free_gb():
    return shutil.disk_usage(ROOT).free / GB


def check_disk(step, need_gb=0.0):
    f = free_gb()
    log(f"[{step}] free disk {f:.1f} GB (step writes ~{need_gb:.0f} GB)")
    if f < MIN_FREE_GB or f - need_gb < MIN_FREE_GB:
        msg = f"디스크 부족으로 중단: {step} 전 여유 {f:.1f} GB, 이 단계가 쓸 크기 약 {need_gb:.0f} GB (기준 {MIN_FREE_GB} GB)"
        note("실패·중단", msg)
        raise Stop(msg)


def safe_delete(path: Path):
    p = path.resolve()
    if p == ORIG or ORIG in p.parents:
        raise Stop(f"refusing to delete inside the original checkpoint: {p}")
    allowed = [GGUF_DIR.resolve()] + [(ROOT / f"models/gemma-4-26B-A4B-it-cafe-{k}e").resolve() for k in (98, 64, 48)]
    if not any(a == p.parent or a in p.parents for a in allowed):
        raise Stop(f"refusing to delete outside allowed folders: {p}")
    if p.exists():
        size = p.stat().st_size
        p.unlink()
        log(f"deleted {p.relative_to(ROOT.resolve())} ({size / GB:.1f} GB)")


def run(step, cmd, timeout=None, env=None):
    """Run one heavy job, output to logs/overnight/<step>.log. Raises StepFailed on non-zero exit."""
    STEP_LOGS.mkdir(parents=True, exist_ok=True)
    out = STEP_LOGS / f"{step}.log"
    log(f"[{step}] run: {' '.join(str(c) for c in cmd)[:300]}")
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as f:
        try:
            rc = subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    dt = time.perf_counter() - t0
    log(f"[{step}] rc={rc} in {dt / 60:.1f} min (log: {out.relative_to(ROOT).as_posix()})")
    if rc != 0:
        tail = out.read_text(encoding="utf-8", errors="replace")[-600:].replace("\n", " ")
        raise StepFailed(f"{step} rc={rc}: ...{tail}")
    return out


# ------------------------------------------------------------------ ollama helpers

def ollama_up():
    try:
        with urllib.request.urlopen(f"{HOST}/api/version", timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def ensure_ollama():
    if ollama_up():
        return
    log("ollama server not responding; starting it with OLLAMA_IGPU_ENABLE=1 (this process only)")
    env = dict(os.environ, OLLAMA_IGPU_ENABLE="1")
    logf = open(ROOT / "logs/ollama_serve_overnight.log", "a", encoding="utf-8")  # noqa: SIM115
    subprocess.Popen(["ollama", "serve"], cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                     creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    for _ in range(60):
        time.sleep(2)
        if ollama_up():
            note("결정", "Ollama 서버가 응답하지 않아 `OLLAMA_IGPU_ENABLE=1`(이 프로세스에만)로 다시 띄웠다.")
            return
    raise StepFailed("ollama server did not start")


def unload_all():
    try:
        with urllib.request.urlopen(f"{HOST}/api/ps", timeout=30) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        for n in names:
            req = urllib.request.Request(f"{HOST}/api/generate", data=json.dumps({"model": n, "keep_alive": 0}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=120).read()
        time.sleep(5)
        if names:
            log(f"unloaded {names}")
    except Exception as e:  # noqa: BLE001
        log(f"unload_all: {e!r}")


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(64 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def register(step, name, q4: Path):
    """Write a Modelfile (original with FROM replaced), ollama create, check blob hash, delete the local Q4 copy."""
    ensure_ollama()
    lines = MODELFILE_ORIG.read_text(encoding="utf-8").split("\n")
    assert lines[4].startswith("FROM ") and "a8bdd07a" in lines[4]
    lines[4] = f"FROM ./{q4.name}"
    mf = GGUF_DIR / f"Modelfile.{name.split('/')[-1]}"
    mf.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    shutil.copy2(mf, S5 / "build" / mf.name)
    STEP_LOGS.mkdir(parents=True, exist_ok=True)
    with (STEP_LOGS / f"{step}.log").open("w", encoding="utf-8") as f:
        rc = subprocess.run(["ollama", "create", name, "-f", mf.name], cwd=GGUF_DIR, stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    if rc != 0:
        raise StepFailed(f"ollama create {name} rc={rc}")
    show = subprocess.run(["ollama", "show", "--modelfile", name], capture_output=True, text=True, encoding="utf-8").stdout
    blob = re.search(r"^FROM .*sha256-([0-9a-f]{64})", show, re.M)
    local = sha256(q4)
    if not blob or blob.group(1) != local:
        raise StepFailed(f"registered blob {blob.group(1) if blob else None} != local Q4 {local}")
    log(f"[{step}] registered {name}; blob sha256 == local Q4")
    safe_delete(q4)
    note("결정", f"`{name}`: Ollama 블롭과 SHA-256이 같은 것을 확인한 뒤 로컬 Q4 복사본({q4.name})을 지웠다(디스크 절약, 등록은 그대로).")


def load_check(step, name, mode, out):
    ensure_ollama()
    run(step, [EXAM_PY, "scripts/load_check.py", "--model", name, "--mode", mode, "--out", out], timeout=3600)
    res = json.loads((ROOT / out).read_text(encoding="utf-8"))
    unload_all()
    return res


# ------------------------------------------------------------------ pipelines

def control():
    tag, name = "control", "song/gemma4-26b-128e-noimat"
    bf16 = GGUF_DIR / "gemma4-26b-128e-bf16.gguf"
    q4 = GGUF_DIR / "gemma4-26b-128e-noimat-Q4_K_M.gguf"
    st = state()
    if f"{tag}:convert" not in st:
        check_disk(f"{tag}:convert", 52)
        safe_delete(bf16)
        run(f"{tag}_convert", [PRUNE_PY, CONVERT, "models/gemma-4-26B-A4B-it", "--outtype", "bf16", "--outfile", bf16])
        mark(f"{tag}:convert")
    if f"{tag}:verify_bf16" not in st:
        check_disk(f"{tag}:verify_bf16")
        run(f"{tag}_verify_bf16", [PRUNE_PY, "scripts/verify_pruned_files.py", "--pruned", "models/gemma-4-26B-A4B-it", "--gguf", bf16])
        mark(f"{tag}:verify_bf16")
    if f"{tag}:quantize" not in st:
        check_disk(f"{tag}:quantize", 18)
        safe_delete(q4)
        run(f"{tag}_quantize", [QUANTIZE, "--tensor-type-file", TENSOR_TYPES, "--token-embedding-type", "q8_0", bf16, q4, "Q4_K_M", "8"])
        mark(f"{tag}:quantize")
    if f"{tag}:verify_q4" not in st:
        check_disk(f"{tag}:verify_q4")
        run(f"{tag}_verify_q4", [PRUNE_PY, "scripts/verify_q4.py", "--gguf", q4, "--experts", "128"])
        mark(f"{tag}:verify_q4")
        safe_delete(bf16)
    if f"{tag}:register" not in st:
        check_disk(f"{tag}:register", 18)
        register(f"{tag}_register", name, q4)
        mark(f"{tag}:register")
    if f"{tag}:load" not in st:
        check_disk(f"{tag}:load")
        res = load_check(f"{tag}_load", name, "cpu", "results/step5/load_128e-noimat_cpu.json")
        log(f"[{tag}] cpu load http {res.get('http_status')}, peak runner {res['peak_memory']['runner_private_gb']} GB")
        mark(f"{tag}:load")
    if f"{tag}:exams" not in st:
        check_disk(f"{tag}:exams")
        ensure_ollama()
        run(f"{tag}_exams", [EXAM_PY, "scripts/run_step5.py", "--model", name, "--mode", "cpu"])
        mark(f"{tag}:exams")


def pruned(k):
    tag, name = f"keep{k}", f"song/gemma4-26b-cafe-{k}e"
    mdir = ROOT / f"models/gemma-4-26B-A4B-it-cafe-{k}e"
    rel_mdir = f"models/gemma-4-26B-A4B-it-cafe-{k}e"
    bf16 = GGUF_DIR / f"gemma4-26b-cafe-{k}e-bf16.gguf"
    q4 = GGUF_DIR / f"gemma4-26b-cafe-{k}e-Q4_K_M.gguf"
    bf16_gb = 51.6 * (0.115 + 0.885 * k / 128)      # experts are 88.5% of the BF16 checkpoint
    st = state()
    if f"{tag}:build" not in st:
        check_disk(f"{tag}:build", bf16_gb)
        if mdir.exists():                                # leftover of an interrupted build: our own output, rebuild
            for p in mdir.glob("*"):
                safe_delete(p)
            mdir.rmdir()
            note("결정", f"`{rel_mdir}`에 중단된 빌드가 남아 있어 지우고 다시 만들었다.")
        run(f"{tag}_build", [PRUNE_PY, "scripts/build_pruned.py", "--plan", f"results/step5/prune_plan_{k}.json", "--dst", rel_mdir])
        mark(f"{tag}:build")
    if f"{tag}:validate" not in st:
        check_disk(f"{tag}:validate")
        try:
            run(f"{tag}_validate", [PRUNE_PY, "scripts/validate_pruned.py", "--pruned", rel_mdir, "--n", "8",
                                    "--out", f"results/step5/validate_{k}e.json"])
        except StepFailed as e:                          # informational only: keep going
            log(f"[{tag}] validation failed, continuing: {e}")
            note("실패·중단", f"{k}개: 은닉 상태 검증 실패(나머지는 계속 진행). {str(e)[:200]}")
        mark(f"{tag}:validate")
    if f"{tag}:convert" not in st:
        check_disk(f"{tag}:convert", bf16_gb)
        safe_delete(bf16)
        run(f"{tag}_convert", [PRUNE_PY, CONVERT, rel_mdir, "--outtype", "bf16", "--outfile", bf16])
        mark(f"{tag}:convert")
    if f"{tag}:verify_bf16" not in st:
        check_disk(f"{tag}:verify_bf16")
        run(f"{tag}_verify_bf16", [PRUNE_PY, "scripts/verify_pruned_files.py", "--pruned", rel_mdir, "--gguf", bf16])
        mark(f"{tag}:verify_bf16")
    if f"{tag}:quantize" not in st:
        check_disk(f"{tag}:quantize", 17.07 * (0.17 + 0.83 * k / 128))
        safe_delete(q4)
        run(f"{tag}_quantize", [QUANTIZE, "--tensor-type-file", TENSOR_TYPES, "--token-embedding-type", "q8_0", bf16, q4, "Q4_K_M", "8"])
        mark(f"{tag}:quantize")
    if f"{tag}:verify_q4" not in st:
        check_disk(f"{tag}:verify_q4")
        run(f"{tag}_verify_q4", [PRUNE_PY, "scripts/verify_q4.py", "--gguf", q4, "--experts", str(k)])
        mark(f"{tag}:verify_q4")
        safe_delete(bf16)
        for p in mdir.glob("model-*.safetensors"):
            safe_delete(p)
    if f"{tag}:register" not in st:
        check_disk(f"{tag}:register", 15)
        register(f"{tag}_register", name, q4)
        mark(f"{tag}:register")
    if f"{tag}:load" not in st:
        check_disk(f"{tag}:load")
        res = load_check(f"{tag}_load_igpu", name, "igpu", f"results/step5/load_{k}e.json")
        fits = res.get("fits_igpu_fully")
        log(f"[{tag}] igpu load http {res.get('http_status')}, fits fully: {fits}")
        if not fits:
            load_check(f"{tag}_load_cpu", name, "cpu", f"results/step5/load_{k}e_cpu.json")
            note("결정", f"{k}개: iGPU에 다 들어가지 않아(HTTP {res.get('http_status')}) CPU 모드로 시험했다.")
        mark(f"{tag}:load")
    mode = "igpu" if json.loads((S5 / f"load_{k}e.json").read_text(encoding="utf-8")).get("fits_igpu_fully") else "cpu"
    if f"{tag}:smoke" not in st:
        check_disk(f"{tag}:smoke")
        ensure_ollama()
        out = run(f"{tag}_smoke", [EXAM_PY, "scripts/run_exam.py", "--model", name, "--exam", "data/exam_cafe.jsonl",
                                   "--ids", SMOKE_IDS, "--out-dir", "results/step5/smoke", "--mode", mode])
        unload_all()
        m = re.search(r"saved: (\S+)", out.read_text(encoding="utf-8", errors="replace"))
        d = ROOT / m.group(1) if m else None
        rows = [json.loads(l) for l in (d / "raw.jsonl").read_text(encoding="utf-8").splitlines()] if d else []
        with_calls = sum(1 for r in rows if r.get("tool_calls"))
        exact = json.loads((d / "summary.json").read_text(encoding="utf-8"))["summary"]["exact"] if d else 0
        log(f"[{tag}] smoke: {with_calls}/5 cases with tool calls, exact {exact}")
        if with_calls < 3:
            note("실패·중단", f"{k}개: 스모크 테스트에서 도구 호출이 {with_calls}/5건뿐이라 전체 시험을 건너뛰었다(기준: 5건 중 3건 이상).")
            raise StepFailed(f"smoke: only {with_calls}/5 cases produced tool calls")
        mark(f"{tag}:smoke")
    if f"{tag}:exams" not in st:
        check_disk(f"{tag}:exams")
        ensure_ollama()
        run(f"{tag}_exams", [EXAM_PY, "scripts/run_step5.py", "--model", name, "--mode", mode])
        mark(f"{tag}:exams")
    if f"{tag}:report" not in st:
        check_disk(f"{tag}:report")
        run(f"{tag}_report", [EXAM_PY, "scripts/report_step5.py", "--keep", str(k)])
        mark(f"{tag}:report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["control", "64", "48", "comparison"])
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    jobs = [("control", control), ("64", lambda: pruned(64)), ("48", lambda: pruned(48))]
    log(f"=== overnight start (free {free_gb():.1f} GB) ===")
    note("결정", "스모크 테스트 통과 기준: 5문항 중 3문항 이상에서 도구 호출이 나오면 전체 시험을 돌린다"
                 "(지시: '도구 호출이 되면'). 모드는 iGPU 로딩 확인에서 `ollama ps` 100% GPU면 iGPU, 아니면 CPU.") \
        if "notes:criteria" not in state() else None
    mark("notes:criteria")
    try:
        for key, fn in jobs:
            if a.only and a.only != key:
                continue
            log(f"--- {key} ---")
            try:
                fn()
            except StepFailed as e:
                log(f"FAILED {key}: {e}")
                note("실패·중단", f"{key}: 실패 후 다음 모델로 넘어감. {str(e)[:300]}")
                unload_all()
            # disk errors raise Stop and end everything
        if not a.only or a.only == "comparison":
            try:
                run("comparison", [EXAM_PY, "scripts/report_comparison.py"])
            except StepFailed as e:
                note("실패·중단", f"comparison.md 생성 실패: {str(e)[:200]}")
    except Stop as e:
        log(f"STOPPED: {e}")
    log(f"=== overnight end (free {free_gb():.1f} GB) ===")


if __name__ == "__main__":
    main()
