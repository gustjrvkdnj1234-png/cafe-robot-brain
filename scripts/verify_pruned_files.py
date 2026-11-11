"""Check the data (not only sizes) of the pruned checkpoint and its BF16 GGUF, e.g. after a crash.

    .venv-prune\\Scripts\\python.exe scripts\\verify_pruned_files.py --pruned models/gemma-4-26B-A4B-it-cafe-98e \
        --gguf models/gguf/gemma4-26b-cafe-98e-bf16.gguf

A. Pruned safetensors vs original: every output tensor's bytes must equal the kept byte ranges of the source tensor
   (SHA-256), and each shard's size must equal header + data length.
B. BF16 GGUF vs pruned safetensors: every GGUF tensor stored as BF16 must be byte-identical to some pruned tensor;
   F32 tensors must be exact BF16 values whose BF16 bytes equal a pruned tensor. Generated tensors with no source
   (rope_freqs) are listed separately. File size must cover the last tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_pruned import _Null, copy_ranges, ranges, read_header, sliced_kind  # noqa: E402

GENERATED = {"rope_freqs.weight"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/gemma-4-26B-A4B-it")
    ap.add_argument("--pruned", required=True)
    ap.add_argument("--gguf")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    src, dst = ROOT / a.src, ROOT / a.pruned
    info = json.loads((dst / "prune_info.json").read_text(encoding="utf-8"))
    keep = {int(k): v for k, v in info["kept_experts"].items()}

    # ---- A
    sidx = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    sh = {s: read_header(src / s) for s in set(sidx.values())}
    didx = json.loads((dst / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    bad, hashes, n = 0, {}, 0
    for shard in sorted(set(didx.values())):
        base, h = read_header(dst / shard)
        end = max(m["data_offsets"][1] for k, m in h.items() if k != "__metadata__")
        size = (dst / shard).stat().st_size
        if size != base + end:
            print(f"A SIZE {shard}: {size} != {base + end}")
            bad += 1
        with open(dst / shard, "rb") as f:
            for name, meta in h.items():
                if name == "__metadata__":
                    continue
                o0, o1 = meta["data_offsets"]
                hd = hashlib.sha256()
                copy_ranges(f, _Null(), [(base + o0, o1 - o0)], hd)
                s_shard = sidx[name]
                sbase, shdr = sh[s_shard]
                layer, _ = sliced_kind(name)
                if layer is not None:
                    rng = ranges(shdr[name], sbase, keep[layer])
                else:
                    so0, so1 = shdr[name]["data_offsets"]
                    rng = [(sbase + so0, so1 - so0)]
                hs = hashlib.sha256()
                with open(src / s_shard, "rb") as sf:
                    copy_ranges(sf, _Null(), rng, hs)
                if hd.hexdigest() != hs.hexdigest():
                    print(f"A DATA MISMATCH {name}")
                    bad += 1
                hashes.setdefault(hd.hexdigest(), name)
                n += 1
        print(f"A {shard}: checked", flush=True)
    print(f"A: {n} tensors, {'OK' if not bad else f'{bad} problems'}", flush=True)

    # ---- B
    if a.gguf:
        sys.path.insert(0, str(ROOT / "tools/llama.cpp-b12283/llama.cpp-b12283/gguf-py"))
        import gguf
        path = ROOT / a.gguf
        r = gguf.GGUFReader(path)
        last = max(t.data_offset + t.n_bytes for t in r.tensors)
        size = path.stat().st_size
        bbad, matched, generated = 0, 0, []
        if size < last:
            print(f"B SIZE: file {size} < last tensor end {last}")
            bbad += 1
        for t in r.tensors:
            raw = np.asarray(t.data)
            tt = t.tensor_type.name
            if tt == "BF16":
                b = raw.tobytes()
            elif tt == "F32":
                u = raw.astype(np.float32, copy=False).view(np.uint32)
                if t.name not in GENERATED and np.any(u & 0xFFFF):
                    print(f"B F32 not exact bf16: {t.name}")
                    bbad += 1
                    continue
                b = (u >> 16).astype(np.uint16).tobytes()
            else:
                print(f"B unexpected type {tt}: {t.name}")
                bbad += 1
                continue
            hx = hashlib.sha256(b).hexdigest()
            if hx in hashes:
                matched += 1
            elif t.name in GENERATED:
                generated.append(t.name)
            else:
                print(f"B NO MATCH {t.name} ({tt}, {list(map(int, t.shape))})")
                bbad += 1
        print(f"B: {len(r.tensors)} tensors, {matched} identical to a pruned tensor, generated {generated}; "
              f"{'OK' if not bbad else f'{bbad} problems'}")
        bad += bbad
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
