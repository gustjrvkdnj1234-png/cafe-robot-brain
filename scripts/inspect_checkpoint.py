"""Read config.json and safetensors headers (no weights loaded) and summarize an MoE checkpoint.

    python scripts/inspect_checkpoint.py --dir models/gemma-4-26B-A4B-it
    python scripts/inspect_checkpoint.py --repo google/gemma-4-26B-A4B-it --rev <sha>   # HTTP range requests only

A safetensors file starts with an 8-byte little-endian header length, then a JSON header
{tensor_name: {dtype, shape, data_offsets}}. Only those bytes are read.
Output: results/step4/checkpoint_inspect.json
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
         "F8_E4M3": 1, "F8_E5M2": 1}


def _get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "checkpoint-inspector"})
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class Source:
    def __init__(self, a):
        self.dir = Path(a.dir) if a.dir else None
        self.base = None if self.dir else f"https://huggingface.co/{a.repo}/resolve/{a.rev}/"

    def text(self, name: str) -> str:
        return (self.dir / name).read_text(encoding="utf-8") if self.dir else _get(self.base + name).decode("utf-8")

    def header(self, name: str) -> dict:
        if self.dir:
            with open(self.dir / name, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                return json.loads(f.read(n))
        n = struct.unpack("<Q", _get(self.base + name, (0, 7)))[0]
        return json.loads(_get(self.base + name, (8, 8 + n - 1)))


def numel(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--repo", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--rev", default="main")
    ap.add_argument("--out", default="results/step4/checkpoint_inspect.json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    src = Source(a)

    config = json.loads(src.text("config.json"))
    index = json.loads(src.text("model.safetensors.index.json"))
    shards = sorted(set(index["weight_map"].values()))
    tensors = {}
    for sh in shards:
        h = src.header(sh)
        for k, v in h.items():
            if k == "__metadata__":
                continue
            tensors[k] = {"dtype": v["dtype"], "shape": v["shape"], "shard": sh,
                          "bytes": v["data_offsets"][1] - v["data_offsets"][0]}

    # group by layer
    layer_re = re.compile(r"(.*\.layers\.)(\d+)\.(.*)")
    per_layer = defaultdict(dict)   # language-model decoder layers only
    vision_params = 0
    other = {}
    for k, v in tensors.items():
        m = layer_re.match(k)
        if m and "language_model" in m.group(1):
            per_layer[int(m.group(2))][m.group(3)] = v
        elif m:
            vision_params += numel(v["shape"])
        else:
            other[k] = v
    out = {
        "config": config,
        "n_tensors": len(tensors),
        "total_params": sum(numel(v["shape"]) for v in tensors.values()),
        "total_bytes": sum(v["bytes"] for v in tensors.values()),
        "dtypes": sorted({v["dtype"] for v in tensors.values()}),
        "n_layers_in_weights": len(per_layer),
        "vision_tower_layer_params": vision_params,
        "layer_prefixes": sorted({layer_re.match(k).group(1) for k in tensors if layer_re.match(k)}),
        "non_layer_tensors": {k: {"shape": v["shape"], "dtype": v["dtype"], "params": numel(v["shape"])} for k, v in other.items()},
        "layers": {str(i): {k: {"shape": v["shape"], "dtype": v["dtype"], "params": numel(v["shape"])} for k, v in sorted(d.items())}
                   for i, d in sorted(per_layer.items())},
    }
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"tensors {len(tensors)}, params {out['total_params'] / 1e9:.3f} B, bytes {out['total_bytes'] / 1e9:.2f} GB, dtypes {out['dtypes']}")
    print("layer prefixes:", out["layer_prefixes"], "layers:", len(per_layer))
    print("non-layer tensors:")
    for k, v in out["non_layer_tensors"].items():
        print(f"  {k} {v['shape']} {v['dtype']}")
    print(f"saved {p.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
