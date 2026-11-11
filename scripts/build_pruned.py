"""Step 5a.2: write a pruned copy of the HF checkpoint (original files are only read).

    .venv-prune\\Scripts\\python.exe scripts\\build_pruned.py --plan results/step5/prune_plan_98.json \
        --src models/gemma-4-26B-A4B-it --dst models/gemma-4-26B-A4B-it-cafe-98e

Per layer the expert dimension (dim 0) of these tensors is sliced to the kept experts (ascending original index):
    experts.gate_up_proj [E, 2*I, H], experts.down_proj [E, H, I], router.proj.weight [E, H], router.per_expert_scale [E]
Everything else is copied byte for byte. config.json gets num_experts = keep; top_k_experts stays 8.
Streaming: each expert block is a contiguous byte range, so data is copied in chunks (no tensor is held whole).
Afterwards every output tensor is checked against the source (SHA-256 of the kept byte ranges).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "model.language_model.layers."
SLICED = ("experts.gate_up_proj", "experts.down_proj", "router.proj.weight", "router.per_expert_scale")
SHARD_BYTES = 5 * 1024 ** 3
CHUNK = 64 * 1024 ** 2
COPY_FILES = ("chat_template.jinja", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json")


def read_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def sliced_kind(name):
    if not name.startswith(PREFIX):
        return None, None
    layer = int(name[len(PREFIX):].split(".")[0])
    rest = name[len(PREFIX):].split(".", 1)[1]
    return (layer, rest) if rest in SLICED else (None, None)


def ranges(meta, base, keep):
    """Byte ranges in the source file for the kept rows along dim 0."""
    a, b = meta["data_offsets"]
    E = meta["shape"][0]
    row = (b - a) // E
    assert row * E == b - a
    out, i = [], 0
    while i < len(keep):                       # merge consecutive experts into one range
        j = i
        while j + 1 < len(keep) and keep[j + 1] == keep[j] + 1:
            j += 1
        out.append((base + a + keep[i] * row, (keep[j] - keep[i] + 1) * row))
        i = j + 1
    return out


def copy_ranges(src_f, dst_f, rngs, h=None):
    n = 0
    for off, size in rngs:
        src_f.seek(off)
        left = size
        while left:
            buf = src_f.read(min(CHUNK, left))
            if not buf:
                raise IOError("short read")
            dst_f.write(buf)
            if h is not None:
                h.update(buf)
            left -= len(buf)
            n += len(buf)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--src", default="models/gemma-4-26B-A4B-it")
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    src, dst = ROOT / a.src, ROOT / a.dst
    if src.resolve() == dst.resolve():
        sys.exit("dst must differ from src")
    plan = json.loads((ROOT / a.plan).read_text(encoding="utf-8"))
    keep_n = plan["keep"]
    keep = {int(i): sorted(v["kept"]) for i, v in plan["layers"].items()}
    assert all(len(v) == keep_n and len(set(v)) == keep_n for v in keep.values())
    dst.mkdir(parents=True, exist_ok=False)

    idx = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    headers = {s: read_header(src / s) for s in sorted(set(idx["weight_map"].values()))}
    # all tensors in source order (shard, offset)
    tensors = []
    for s, (base, h) in headers.items():
        for name, meta in h.items():
            if name != "__metadata__":
                tensors.append((s, meta["data_offsets"][0], name, meta, base))
    tensors.sort()

    # plan output layout
    out_items = []                                  # (name, dtype, shape, nbytes, src_shard, src_ranges)
    for s, _, name, meta, base in tensors:
        layer, kind = sliced_kind(name)
        if layer is not None:
            assert meta["shape"][0] == 128, (name, meta["shape"])
            rng = ranges(meta, base, keep[layer])
            shape = [keep_n] + meta["shape"][1:]
        else:
            o0, o1 = meta["data_offsets"]
            rng = [(base + o0, o1 - o0)]
            shape = meta["shape"]
        out_items.append((name, meta["dtype"], shape, sum(r[1] for r in rng), s, rng))
    shards, cur, size = [], [], 0
    for it in out_items:
        if cur and size + it[3] > SHARD_BYTES:
            shards.append(cur)
            cur, size = [], 0
        cur.append(it)
        size += it[3]
    shards.append(cur)
    names = [f"model-{k + 1:05d}-of-{len(shards):05d}.safetensors" for k in range(len(shards))]

    weight_map, total, t0 = {}, 0, time.time()
    src_files = {s: open(src / s, "rb") for s in headers}
    digests = {}
    try:
        for fname, items in zip(names, shards):
            header, off = {"__metadata__": {"format": "pt"}}, 0
            for name, dt, shape, nb, _, _ in items:
                header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
                off += nb
                weight_map[name] = fname
            hb = json.dumps(header, separators=(",", ":")).encode()
            hb += b" " * (-len(hb) % 8)
            with open(dst / fname, "wb") as out:
                out.write(struct.pack("<Q", len(hb)))
                out.write(hb)
                for name, dt, shape, nb, s, rng in items:
                    h = hashlib.sha256()
                    assert copy_ranges(src_files[s], out, rng, h) == nb
                    digests[name] = h.hexdigest()
            total += off
            print(f"{time.strftime('%H:%M:%S')} wrote {fname} ({off / 1e9:.2f} GB, {len(items)} tensors)", flush=True)
    finally:
        for f in src_files.values():
            f.close()

    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2),
                                                      encoding="utf-8")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    assert cfg["text_config"]["num_experts"] == 128 and cfg["text_config"]["top_k_experts"] == 8
    cfg["text_config"]["num_experts"] = keep_n
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    for f in COPY_FILES:
        shutil.copy2(src / f, dst / f)
    (dst / "prune_info.json").write_text(json.dumps({"source": a.src, "plan": a.plan, "keep": keep_n,
                                                     "kept_experts": {str(k): v for k, v in keep.items()}}, indent=1),
                                         encoding="utf-8")

    # verify: re-read every output tensor and compare hash with what was copied from source
    bad = 0
    for fname in names:
        base, h = read_header(dst / fname)
        with open(dst / fname, "rb") as f:
            for name, meta in h.items():
                if name == "__metadata__":
                    continue
                o0, o1 = meta["data_offsets"]
                hh = hashlib.sha256()
                copy_ranges(f, _Null(), [(base + o0, o1 - o0)], hh)
                if hh.hexdigest() != digests[name]:
                    print("MISMATCH", name)
                    bad += 1
    print(f"done: {len(weight_map)} tensors, {total / 1e9:.2f} GB in {time.time() - t0:.0f}s; verify {'OK' if not bad else f'{bad} BAD'}")
    sys.exit(1 if bad else 0)


class _Null:
    def write(self, b):
        pass


if __name__ == "__main__":
    main()
