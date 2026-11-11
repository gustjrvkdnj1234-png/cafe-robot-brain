"""Step 5a.3: compare hidden states of a pruned checkpoint with the original, layer by layer on CPU (no full load).

    .venv-prune\\Scripts\\python.exe scripts\\validate_pruned.py --pruned models/gemma-4-26B-A4B-it-cafe-98e \
        --n 8 --out results/step5/validate_98e.json

For each layer i, three runs over the same 8 calibration cases (prefix tree, as in reap_stats.py):
  orig      original layer on the original model's hidden states  (reference chain)
  pruned    pruned layer on the pruned model's hidden states       (error accumulates over layers)
  isolated  pruned layer on the original model's hidden states     (error of this layer alone)
Cosine similarity per token (mean / 1st percentile / min), over all tokens and over case tokens only.
After the last layer: final norm + tied embedding head, top-1 next-token agreement on case tokens.

Baseline for scale: --pruned <original dir> --pruned-dtype bfloat16 compares the original model in float32 with
the same model computed in bfloat16, i.e. drift from rounding alone (no pruning).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from reap_stats import (EMBED_KEY, Checkpoint, PeakRAM, build_tree, make_layer, pick_cases,  # noqa: E402
                        render_cases, run_layer_on_tree)


def cos_stats(a, b, segs, kinds_only=None):
    vals = []
    for s in segs:
        c = F.cosine_similarity(a[s.sid], b[s.sid], dim=-1)
        if kinds_only:
            m = torch.tensor([k == kinds_only for k in s.kinds])
            c = c[m]
        vals.append(c)
    v = torch.cat(vals)
    return {"mean": round(v.mean().item(), 6), "p01": round(v.quantile(0.01).item(), 6), "min": round(v.min().item(), 6),
            "n": int(v.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="models/gemma-4-26B-A4B-it")
    ap.add_argument("--pruned", required=True)
    ap.add_argument("--calib", default="data/calib_cafe.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pruned-dtype", choices=["float32", "bfloat16"], default="float32",
                    help="compute dtype of the second model (bfloat16 + original dir = rounding-only baseline)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(a.threads)
    mem = PeakRAM()
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    cfg_o = AutoConfig.from_pretrained(ROOT / a.orig).text_config
    cfg_p = AutoConfig.from_pretrained(ROOT / a.pruned).text_config
    for c in (cfg_o, cfg_p):
        c._attn_implementation = "sdpa"
    print(f"experts: original {cfg_o.num_experts}, pruned {cfg_p.num_experts}, top-k {cfg_p.top_k_experts}", flush=True)
    tok = AutoTokenizer.from_pretrained(ROOT / a.orig)
    rows = [json.loads(l) for l in (ROOT / a.calib).read_text(encoding="utf-8").splitlines() if l.strip()]
    chosen = pick_cases(rows, a.n, a.seed)
    cases = render_cases(tok, chosen)
    tops, segs = build_tree(cases)
    print(f"{len(cases)} cases, {sum(len(s.tokens) for s in segs):,} tokens in the tree", flush=True)
    ck_o, ck_p = Checkpoint(ROOT / a.orig), Checkpoint(ROOT / a.pruned)
    W = ck_o.get(EMBED_KEY)
    scale = torch.tensor(cfg_o.hidden_size ** 0.5).to(W.dtype)
    st_o = {s.sid: (W[torch.tensor(s.tokens)] * scale).float() for s in segs}
    del W
    st_p = {k: v.clone() for k, v in st_o.items()}
    rot = Gemma4TextRotaryEmbedding(cfg_o)
    dtype = torch.float32
    dtype_p = getattr(torch, a.pruned_dtype)
    res = {"orig": a.orig, "pruned": a.pruned, "pruned_dtype": a.pruned_dtype, "case_ids": [c["id"] for c in cases], "layers": {}}
    t_all = time.perf_counter()
    for i in range(cfg_o.num_hidden_layers):
        t0 = time.perf_counter()
        lo = make_layer(cfg_o, i, ck_o.layer_state(i), dtype)
        new_o = run_layer_on_tree(lo, i, cfg_o, rot, tops, st_o, {}, dtype)
        del lo
        lp = make_layer(cfg_p, i, ck_p.layer_state(i), dtype_p)
        iso = run_layer_on_tree(lp, i, cfg_p, rot, tops, st_o, {}, dtype_p)
        new_p = run_layer_on_tree(lp, i, cfg_p, rot, tops, st_p, {}, dtype_p)
        del lp
        gc.collect()
        r = {"isolated_all": cos_stats(new_o, iso, segs), "isolated_case": cos_stats(new_o, iso, segs, "C"),
             "chain_all": cos_stats(new_o, new_p, segs), "chain_case": cos_stats(new_o, new_p, segs, "C"),
             "seconds": round(time.perf_counter() - t0, 1)}
        res["layers"][str(i)] = r
        st_o, st_p = new_o, new_p
        del iso
        print(f"layer {i:2d}: isolated mean {r['isolated_all']['mean']:.5f} min {r['isolated_all']['min']:.4f} | "
              f"chain mean {r['chain_all']['mean']:.5f} (case {r['chain_case']['mean']:.5f}) min {r['chain_all']['min']:.4f} "
              f"| {r['seconds']}s, RSS peak {mem.gb():.1f} GB", flush=True)

    # final norm + head: top-1 next-token agreement on case tokens (positions whose next token is a case token)
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
    norm = Gemma4RMSNorm(cfg_o.hidden_size, eps=cfg_o.rms_norm_eps)
    norm.load_state_dict({"weight": ck_o.get("model.language_model.norm.weight").float()})
    W = ck_o.get(EMBED_KEY).float()
    agree = total = 0
    kl_sum = 0.0
    with torch.no_grad():
        for s in segs:
            m = torch.tensor([k == "C" for k in s.kinds])
            if not m.any():
                continue
            cap = cfg_o.final_logit_softcapping
            lo = torch.tanh(norm(st_o[s.sid][m]) @ W.T / cap) * cap
            lp = torch.tanh(norm(st_p[s.sid][m]) @ W.T / cap) * cap
            agree += (lo.argmax(-1) == lp.argmax(-1)).sum().item()
            total += int(m.sum())
            kl_sum += F.kl_div(F.log_softmax(lp, -1), F.log_softmax(lo, -1), log_target=True, reduction="sum").item()
    res["final"] = {"top1_agreement_case_tokens": round(agree / total, 4), "n_tokens": total,
                    "mean_kl_orig_to_pruned": round(kl_sum / total, 5), "note": "logits with final softcapping"}
    res["peak_rss_gb"] = round(mem.gb(), 2)
    res["total_s"] = round(time.perf_counter() - t_all, 1)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"final: top-1 agreement {agree}/{total} = {agree / total:.4f}, mean KL {kl_sum / total:.5f}; "
          f"peak RSS {mem.gb():.1f} GB, {res['total_s']}s")


if __name__ == "__main__":
    main()
