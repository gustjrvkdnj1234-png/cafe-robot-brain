"""Parameter breakdown of the checkpoint and size estimates when keeping K of 128 experts per layer.

    python scripts/size_estimates.py      # reads results/step4/checkpoint_inspect.json
Q4 uses the effective bits/weight of Ollama's gemma4:26b Q4_K_M file (measured, see --q4-file-bytes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default="128,98,64,48")
    ap.add_argument("--q4-file-bytes", type=int, default=17074419072, help="Ollama gemma4:26b Q4_K_M model blob")
    ap.add_argument("--q4-params", type=int, default=25233142046, help="general.parameter_count of that GGUF")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    d = json.loads((ROOT / "results/step4/checkpoint_inspect.json").read_text(encoding="utf-8"))
    tc = d["config"].get("text_config", d["config"])
    E = tc["num_experts"]

    groups = {"experts": 0, "router": 0, "dense_mlp": 0, "attention": 0, "norms_scalars": 0}
    for layer in d["layers"].values():
        for k, v in layer.items():
            if k.startswith("experts."):
                groups["experts"] += v["params"]
            elif k.startswith("router."):
                groups["router"] += v["params"]
            elif k.startswith("mlp."):
                groups["dense_mlp"] += v["params"]
            elif k.startswith("self_attn.") and k.endswith("proj.weight"):
                groups["attention"] += v["params"]
            else:
                groups["norms_scalars"] += v["params"]
    nl = d["non_layer_tensors"]
    groups["embedding (tied with LM head)"] = nl["model.language_model.embed_tokens.weight"]["params"]
    groups["vision tower + projector"] = d["vision_tower_layer_params"] + sum(
        v["params"] for k, v in nl.items() if "vision" in k)
    groups["final norm"] = nl["model.language_model.norm.weight"]["params"]
    total = d["total_params"]
    assert sum(groups.values()) == total, (sum(groups.values()), total)

    per_expert = groups["experts"] / (E * len(d["layers"]))
    router_per_expert = (tc["hidden_size"] + 1)  # proj row + per_expert_scale
    text_total = total - groups["vision tower + projector"]
    q4_bpw = a.q4_file_bytes * 8 / a.q4_params

    est = []
    for k in [int(x) for x in a.keep.split(",")]:
        removed = (E - k) * len(d["layers"]) * (per_expert + router_per_expert)
        tot = total - removed
        txt = text_total - removed
        active = (text_total - groups["experts"]) + tc["top_k_experts"] * per_expert * len(d["layers"])
        est.append({"experts_kept": k, "total_params_B": round(tot / 1e9, 3), "text_params_B": round(txt / 1e9, 3),
                    "bf16_GB": round(tot * 2 / 1e9, 2), "q4_text_GB": round(txt * q4_bpw / 8 / 1e9, 2),
                    "active_params_B": round(active / 1e9, 3)})
    out = {"groups": groups, "total_params": total, "text_params": text_total, "params_per_expert": per_expert,
           "experts_share": groups["experts"] / total,
           "embedding_share": groups["embedding (tied with LM head)"] / total,
           "q4_effective_bits_per_weight": round(q4_bpw, 3), "estimates": est}
    (ROOT / "results/step4/size_estimates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for g, v in groups.items():
        print(f"{g:32} {v / 1e9:8.3f} B  {v / total * 100:5.1f}%")
    print(f"params per expert {per_expert / 1e6:.3f} M, Q4 bpw {q4_bpw:.2f}")
    for e in est:
        print(e)


if __name__ == "__main__":
    main()
