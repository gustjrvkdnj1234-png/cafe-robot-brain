"""Check every song/ model registered in Ollama: manifest -> expected blob, blob file integrity,
and one short prompt. Requires a running server started with v2/scripts/start_ollama.ps1
(OLLAMA_NOPRUNE=1); refuses to run otherwise, because on Windows any `ollama` CLI call without a
server launches the desktop app, whose server starts without NOPRUNE.

    .venv-voice\\Scripts\\python.exe v2\\scripts\\verify_ollama_models.py [--no-hash]

Writes v2/results/ollama_models/verify.json and verify.md.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
HOST = "http://127.0.0.1:11434"
BLOBS = Path(os.path.expanduser("~/.ollama/models/blobs"))
OUT = ROOT / "v2/results/ollama_models"
DRAFT, PROJECTOR = "491f38afe0eb215a819c085110255fcebd6c68b4cc8e0f29168324ef3f913e32", \
    "7e78b02d928be76d0db72d23ba011aeae7fcdd9488358fd83741a8fd10b8556d"

# model -> (expected main GGUF blob prefix, where that expectation is recorded, full build?)
EXPECTED = {
    "song/gemma4-26b-cafe-98e": ("061cc70f", "results/step5/overnight_notes.md", True),
    "song/gemma4-26b-cafe-64e": ("1fa88d73", "scripts/report_step6.py BLOB", True),
    "song/gemma4-26b-cafe-48e": ("9a01e444", "scripts/report_step6.py BLOB", True),
    "song/gemma4-26b-cafe-64e-text": ("1fa88d73", "models/gguf/lean/Modelfile.64e-text FROM", False),
    "song/gemma4-26b-cafe-48e-text": ("9a01e444", "models/gguf/lean/Modelfile.48e-text FROM", False),
    "song/gemma4-26b-128e-noimat": (None, "no hash recorded; logs/overnight_step5.log: registered 18:24:35 "
                                          "'blob sha256 == local Q4', local Q4 (17.1 GB) deleted 18:24:37", True),
    "song/gemma4-26b-cafe-98e-reap": ("0da08975", "v2/results/reap_curve/reap98/pipeline_state.json", True),
    "song/gemma4-26b-cafe-64e-reap": ("b92b9962", "v2/results/reap64/pipeline_state.json", True),
    "song/gemma4-26b-cafe-48e-reap": ("a8832fd3", "v2/results/reap_curve/reap48/pipeline_state.json", True),
}
NOPRUNE_LOG = ROOT / "logs/ollama_serve_verify.log"
# v1 ran the 17 GB 128e control CPU-only (load_check --mode cpu); on the iGPU it fails to allocate.
CPU_ONLY = {"song/gemma4-26b-128e-noimat"}


def server_ok():
    try:
        requests.get(f"{HOST}/api/version", timeout=3)
    except requests.RequestException:
        return False
    return True


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hash", action="store_true", help="skip re-hashing blob files")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not server_ok():
        sys.exit("no Ollama server: start it with v2/scripts/start_ollama.ps1 first (never run the CLI without it)")
    hashed = {}
    rows = []
    for name, (exp, source, full) in EXPECTED.items():
        r = {"model": name, "expected_prefix": exp, "expected_source": source}
        mf = subprocess.run(["ollama", "show", "--modelfile", name], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        if mf.returncode != 0:
            r.update(status="not registered", error=mf.stderr.strip()[:200])
            rows.append(r)
            continue
        froms = re.findall(r"^FROM .*sha256-([0-9a-f]{64})", mf.stdout, re.M)
        drafts = re.findall(r"^DRAFT .*sha256-([0-9a-f]{64})", mf.stdout, re.M)
        main_blob = froms[0] if froms else None
        r.update(main_blob=main_blob, from_blobs=froms, draft_blobs=drafts,
                 blob_bytes=(BLOBS / f"sha256-{main_blob}").stat().st_size if main_blob else None)
        checks = {"main_blob_expected": (exp is None) or bool(main_blob and main_blob.startswith(exp)),
                  "main_blob_file_exists": bool(main_blob and (BLOBS / f"sha256-{main_blob}").exists())}
        if full:
            checks["projector_and_draft"] = PROJECTOR in froms and DRAFT in drafts
        else:
            checks["text_only_no_projector"] = len(froms) == 1 and not drafts
        if not a.no_hash and main_blob:
            if main_blob not in hashed:
                t = time.time()
                hashed[main_blob] = sha256_file(BLOBS / f"sha256-{main_blob}")
                print(f"  hashed {main_blob[:12]} in {time.time() - t:.0f} s", flush=True)
            checks["blob_content_matches_name"] = hashed[main_blob] == main_blob
        # one short prompt, then unload (one model in memory at a time)
        try:
            t = time.perf_counter()
            resp = requests.post(f"{HOST}/api/chat", json={
                "model": name, "messages": [{"role": "user", "content": "안녕하세요. 한 문장으로 인사해 주세요."}],
                "stream": False, "think": False, "keep_alive": 0,
                "options": {"temperature": 0, "seed": 42, "num_ctx": 2048, "num_predict": 24,
                            **({"num_gpu": 0} if name in CPU_ONLY else {})}}, timeout=900).json()
            r["mode"] = "cpu" if name in CPU_ONLY else "igpu"
            r["prompt_reply"] = (resp.get("message") or {}).get("content", "").strip()
            r["prompt_wall_s"] = round(time.perf_counter() - t, 1)
            r["prompt_error"] = resp.get("error")
            checks["prompt_ok"] = bool(r["prompt_reply"]) and not resp.get("error")
        except requests.RequestException as e:
            r["prompt_error"] = str(e)[:200]
            checks["prompt_ok"] = False
        r["checks"] = checks
        r["status"] = "registered OK" if all(checks.values()) else "failed"
        rows.append(r)
        print(f"{name}: {r['status']}  blob {main_blob and main_blob[:12]}  {checks}  "
              f"reply {r.get('prompt_reply', '')[:40]!r} ({r.get('prompt_wall_s')} s)", flush=True)
    noprune = None
    if NOPRUNE_LOG.exists():
        cfg = [l for l in NOPRUNE_LOG.read_text(encoding="utf-8", errors="replace").splitlines() if 'msg="server config"' in l]
        noprune = bool(cfg) and "OLLAMA_NOPRUNE:true" in cfg[-1]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify.json").write_text(json.dumps({"server_noprune": noprune, "models": rows}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    L = ["# Ollama song/ 모델 확인", "", f"서버 OLLAMA_NOPRUNE:true 확인: {noprune}", "",
         "| 모델 | 상태 | 주 GGUF 블롭 | 크기 | 기대값 출처 | 확인 항목 | 짧은 응답 (모드) |", "|---|---|---|---:|---|---|---|"]
    for r in rows:
        r.setdefault("mode", "-")
        ck = ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in (r.get("checks") or {}).items())
        L.append(f"| `{r['model']}` | {r['status']} | `{(r.get('main_blob') or '-')[:12]}` | "
                 f"{(r.get('blob_bytes') or 0) / 1e9:.2f} GB | {r['expected_source']} | {ck} | "
                 f"{(r.get('prompt_reply') or r.get('prompt_error') or '')[:30].replace(chr(10), ' ')} "
                 f"({r.get('prompt_wall_s', '-')} s, {r['mode']}) |")
    L += ["", "- 128e-noimat은 v1처럼 CPU 전용(`num_gpu 0`)으로 확인했다. iGPU로는 Vulkan 버퍼 할당 실패(메모리 부족)로 뜨지 않는다."]
    (OUT / "verify.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
