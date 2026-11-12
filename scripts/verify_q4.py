"""Check a quantized GGUF before its BF16 intermediates are deleted.

    .venv-prune\\Scripts\\python.exe scripts\\verify_q4.py --gguf models/gguf/x-Q4_K_M.gguf --experts 64

1. Every tensor name and type equals the Ollama gemma4:26b blob (the recipe we copy); shapes equal except the expert axis.
2. expert_count metadata equals --experts; file covers the last tensor.
3. llama.cpp loads it and generates a short non-empty reply (CPU, 4 threads, greedy).
Exit code 0 only if all pass.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools/llama.cpp-b12283/llama.cpp-b12283/gguf-py"))
import gguf  # noqa: E402

REF = Path.home() / ".ollama/models/blobs/sha256-a8bdd07a49b895083bb1420bb895ad9ab3e3b8f7d5cc42c7582ca0fa765eede2"
BIN = ROOT / "tools/llama.cpp-b12283/bin/llama-completion.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--experts", type=int, required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    p = ROOT / a.gguf
    bad = 0
    o, n = gguf.GGUFReader(REF), gguf.GGUFReader(p)
    ot = {t.name: (t.tensor_type.name, [int(x) for x in t.shape]) for t in o.tensors}
    nt = {t.name: (t.tensor_type.name, [int(x) for x in t.shape]) for t in n.tensors}
    if set(ot) != set(nt):
        print("tensor names differ:", sorted(set(ot) ^ set(nt))[:10])
        bad += 1
    for k in set(ot) & set(nt):
        if ot[k][0] != nt[k][0]:
            print(f"type differs {k}: {ot[k][0]} vs {nt[k][0]}")
            bad += 1
        so, sn = ot[k][1], nt[k][1]
        if so != sn and not (so[:-1] == sn[:-1] and so[-1] == 128 and sn[-1] == a.experts) \
                and not (len(so) == 1 and so[0] == 128 and sn[0] == a.experts):
            print(f"shape differs {k}: {so} vs {sn}")
            bad += 1
    kv = {f.name: f for f in n.fields.values()}
    ec = kv["gemma4.expert_count"].contents()
    if ec != a.experts:
        print(f"expert_count {ec} != {a.experts}")
        bad += 1
    last = max(t.data_offset + t.n_bytes for t in n.tensors)
    if os.path.getsize(p) < last:
        print("file shorter than last tensor")
        bad += 1
    print(f"metadata/types: {len(nt)} tensors, expert_count {ec}, {'OK' if not bad else f'{bad} problems'}", flush=True)
    del o, n
    prompt = "<|turn>user\n아메리카노 한 잔 주세요.<turn|>\n<|turn>model\n"
    r = subprocess.run([str(BIN), "-m", str(p), "-p", prompt, "-n", "16", "--temp", "0", "-t", "4", "-c", "512", "-no-cnv"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    out = r.stdout
    reply = out.split("<|turn>model", 1)[-1].strip() if "<|turn>model" in out else out.strip()
    ok = r.returncode == 0 and len(reply) > 0
    print(f"llama.cpp load+generate: rc={r.returncode}, reply={reply[:80]!r} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        print(r.stderr[-1500:])
        bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
