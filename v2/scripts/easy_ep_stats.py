"""Collect EASY-EP statistics (Dong et al., arXiv:2504.06792) for Gemma 4 26B-A4B with the v1 layer-by-layer CPU method.

    # check against the v1 pilot (8 cases, layers 0-1)
    .venv-prune\\Scripts\\python.exe v2\\scripts\\easy_ep_stats.py --n 8 --layers 0-1 --out <scratch dir>
    # full run, same 600 calibration prompts as v1 (resumes after the last finished layer)
    .venv-prune\\Scripts\\python.exe v2\\scripts\\easy_ep_stats.py --n 600 --layers all --out v2/results/easy_ep/stats

Everything except the instrumentation is v1 code imported read-only from scripts/reap_stats.py: case selection
(seed 42), chat-template rendering, prefix tree, token weights (fixed prompt counted once, case tokens per case),
one decoder layer in RAM at a time, float32.

EASY-EP importance of expert i: I_i = sum_t g_t,i * ||e_i(x_t)|| * s_t with s_t = 1 - cos(h_t, h~_t).
Gemma 4 runs the dense MLP and the routed experts in parallel from the same pre-FFN residual, so three variants:
  A (default)  h  = routed-expert input  = pre_feedforward_layernorm_2(residual)
               h~ = h + sum_k g_k * e_k(h)          (routed-expert output before post_feedforward_layernorm_2)
  B            h  = residual (pre-FFN hidden state, also the router input)
               h~ = residual + post_feedforward_layernorm_2(sum_k g_k * e_k(h))
  C            h  = residual
               h~ = decoder layer output (residual + whole FFN block incl. dense MLP); layer_scalar > 0 does not
                    change the cosine
Per expert and group (fixed / case / combined): weighted count, sum g*||e|| (both must equal v1), and sum
g*||e||*s_t for A, B, C. Per group also the weighted mean of s_t for each variant.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from reap_stats import (EMBED_KEY, LAYER_PREFIX, Checkpoint, PeakRAM, build_tree, make_layer,  # noqa: E402
                        parse_layers, pick_cases, render_cases, run_layer_on_tree)

VARIANTS = ("A", "B", "C")
FIELDS = ("count", "sum_gate_norm", *(f"easy_{v}" for v in VARIANTS))


class EasyStats:
    def __init__(self, E):
        self.d = {g: {f: torch.zeros(E, dtype=torch.float64) for f in FIELDS} for g in ("fixed", "case")}
        self.tok = {g: 0.0 for g in ("fixed", "case")}
        self.s_sum = {g: {v: 0.0 for v in VARIANTS} for g in ("fixed", "case")}

    def to_json(self):
        out = {}
        for g in ("fixed", "case", "combined"):
            if g == "combined":
                d = {f: self.d["fixed"][f] + self.d["case"][f] for f in FIELDS}
                tok = self.tok["fixed"] + self.tok["case"]
                ss = {v: self.s_sum["fixed"][v] + self.s_sum["case"][v] for v in VARIANTS}
            else:
                d, tok, ss = self.d[g], self.tok[g], self.s_sum[g]
            out[g] = {"weighted_tokens": tok, **{f: [round(float(x), 6) for x in d[f]] for f in FIELDS},
                      "s_mean": {v: (ss[v] / tok if tok else 0.0) for v in VARIANTS}}
        return out


def instrument(layer, stats: EasyStats, ctx: dict):
    """Same expert computation as v1; additionally captures the tensors needed for s_t and accumulates after the
    whole decoder layer has run (variant C needs the layer output)."""
    experts, router = layer.experts, layer.router
    orig_router_forward, orig_layer_forward = router.forward, layer.forward

    def router_forward(hidden_states):
        ctx["resid"] = hidden_states                       # pre-FFN residual, flattened [T, H]
        return orig_router_forward(hidden_states)

    def experts_forward(hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        toks, exps, gns = [], [], []
        for e in torch.unique(top_k_index).tolist():
            top_k_pos, token_idx = torch.where((top_k_index == e).T)
            x = hidden_states[token_idx]
            gate, up = F.linear(x, experts.gate_up_proj[e]).chunk(2, dim=-1)
            h = F.linear(experts.act_fn(gate) * up, experts.down_proj[e])     # e_i(x), before gate weighting
            g = top_k_weights[token_idx, top_k_pos]
            final.index_add_(0, token_idx, (h * g[:, None]).to(final.dtype))
            toks.append(token_idx)
            exps.append(torch.full_like(token_idx, e))
            gns.append(g.double() * h.double().norm(dim=-1))
        ctx["x_in"], ctx["moe_raw"] = hidden_states, final
        ctx["rec"] = (torch.cat(toks), torch.cat(exps), torch.cat(gns))
        return final

    def capture_moe_out(_m, _inp, output):
        ctx["moe_out"] = output

    def s_of(h, h2):
        return 1.0 - F.cosine_similarity(h.double(), h2.double(), dim=-1)

    def layer_forward(*args, **kwargs):
        out = orig_layer_forward(*args, **kwargs)
        H = ctx["x_in"].shape[-1]
        resid = ctx["resid"].reshape(-1, H)
        s = {"A": s_of(ctx["x_in"], ctx["x_in"] + ctx["moe_raw"]),
             "B": s_of(resid, resid + ctx["moe_out"].reshape(-1, H)),
             "C": s_of(resid, out[0].reshape(-1, H))}
        tw, is_fixed = ctx["w"], ctx["is_fixed"]
        tok, exp, gn = ctx["rec"]
        for grp, mtok in (("fixed", is_fixed), ("case", ~is_fixed)):
            if mtok.any():
                stats.tok[grp] += float(tw[mtok].sum())
                for v in VARIANTS:
                    stats.s_sum[grp][v] += float((tw[mtok] * s[v][mtok]).sum())
            m = mtok[tok]
            if not m.any():
                continue
            w, e_, g_ = tw[tok[m]], exp[m], gn[m]
            d = stats.d[grp]
            d["count"].index_add_(0, e_, w)
            d["sum_gate_norm"].index_add_(0, e_, w * g_)
            for v in VARIANTS:
                d[f"easy_{v}"].index_add_(0, e_, w * g_ * s[v][tok[m]])
        for k in ("x_in", "moe_raw", "moe_out", "resid", "rec"):
            ctx.pop(k, None)
        return out

    router.forward = router_forward
    experts.forward = experts_forward
    layer.post_feedforward_layernorm_2.register_forward_hook(capture_moe_out)
    layer.forward = layer_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/gemma-4-26B-A4B-it")
    ap.add_argument("--calib", default="data/calib_cafe.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--out", default="v2/results/easy_ep/stats")
    ap.add_argument("--v1-meta", default="results/step4/reap_600/meta.json",
                    help="the run must use exactly the same cases as this v1 run (skip check with '')")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dtype = torch.float32
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    mem = PeakRAM()
    t_start = time.perf_counter()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    model_dir = ROOT / a.model_dir
    config = AutoConfig.from_pretrained(model_dir).text_config
    config._attn_implementation = "sdpa"
    if getattr(config, "hidden_size_per_layer_input", 0):
        sys.exit("per-layer inputs present: variant C would include them; not handled")
    tok = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(l) for l in (ROOT / a.calib).read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = render_cases(tok, pick_cases(rows, a.n, a.seed))
    ids = [c["id"] for c in cases]
    if a.v1_meta:
        v1 = json.loads((ROOT / a.v1_meta).read_text(encoding="utf-8"))
        if v1["case_ids"] != ids:
            sys.exit("case selection differs from the v1 run")
    tops, segs = build_tree(cases)
    n_nodes = sum(len(s.tokens) for s in segs)
    print(f"{len(cases)} cases, {n_nodes:,} tokens computed ({len(segs)} segments)", flush=True)
    layers = parse_layers(a.layers, config.num_hidden_layers)
    ckpt = Checkpoint(model_dir)

    meta_path, state_path = out / "meta.json", out / "states_after_layer.pt"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta and meta.get("case_ids") != ids:
        sys.exit(f"{out} holds a run with different cases")
    done = meta.get("done_layers", [])
    if done and state_path.exists():
        states = torch.load(state_path)
        print(f"resume after layer {done[-1]}", flush=True)
    else:
        W = ckpt.get(EMBED_KEY)
        scale = torch.tensor(config.hidden_size ** 0.5).to(W.dtype)
        states = {s.sid: (W[torch.tensor(s.tokens)] * scale).float() for s in segs}
        del W
        done = []
    rotary = Gemma4TextRotaryEmbedding(config)
    meta.update({"n_cases": len(cases), "seed": a.seed, "dtype": "float32", "threads": torch.get_num_threads(),
                 "case_ids": ids, "tokens_computed": n_nodes, "segments": len(segs), "variants": {
                     "A": "h = pre_feedforward_layernorm_2(residual) (routed-expert input); h~ = h + sum g*e(h) [default]",
                     "B": "h = residual; h~ = residual + post_feedforward_layernorm_2(sum g*e)",
                     "C": "h = residual; h~ = decoder layer output (incl. dense MLP)"},
                 "layers": meta.get("layers", {})})
    for i in layers:
        if i in done:
            continue
        mem.reset()
        t0 = time.perf_counter()
        layer = make_layer(config, i, ckpt.layer_state(i), dtype)
        if not bool((layer.layer_scalar > 0).all()):
            sys.exit(f"layer {i}: layer_scalar <= 0, variant C cosine would flip")
        t_load = time.perf_counter() - t0
        stats, ctx = EasyStats(config.num_experts), {}
        instrument(layer, stats, ctx)
        t1 = time.perf_counter()
        states = run_layer_on_tree(layer, i, config, rotary, tops, states, ctx, dtype)
        t_fwd = time.perf_counter() - t1
        del layer, ctx
        gc.collect()
        (out / f"easy_layer{i:02d}.json").write_text(json.dumps(stats.to_json()), encoding="utf-8")
        torch.save(states, state_path)
        done.append(i)
        meta["layers"][str(i)] = {"load_s": round(t_load, 2), "forward_s": round(t_fwd, 2), "peak_rss_gb": round(mem.gb(), 2)}
        meta["done_layers"] = done
        meta["peak_rss_gb"] = round(max(v["peak_rss_gb"] for v in meta["layers"].values()), 2)
        meta["elapsed_s_this_session"] = round(time.perf_counter() - t_start, 1)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"layer {i:2d}: load {t_load:.1f}s, forward {t_fwd:.1f}s, peak RSS {mem.gb():.2f} GB", flush=True)
    meta["total_s"] = round(time.perf_counter() - t_start, 1)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
