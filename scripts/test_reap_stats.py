"""Self-test for reap_stats.py on a tiny random Gemma 4 text model (no checkpoint needed).

    .venv-prune\\Scripts\\python.exe scripts\\test_reap_stats.py

Checks
1. Prefix-tree forward (segments + KV caches) gives the same hidden states as transformers' own
   Gemma4TextModel run on each case alone (sliding-window layer with a small window included).
2. Weighted statistics from the tree equal a brute-force run of every case alone, where each case's
   case-tokens weigh 1 and each fixed token weighs 1/(k * cases sharing that computed token).
"""
from __future__ import annotations

import random
import sys
from collections import UserDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reap_stats import Stats, build_tree, instrument, run_layer_on_tree  # noqa: E402


def tiny_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    c = Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=24, moe_intermediate_size=8, num_experts=8, top_k_experts=2,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        global_head_dim=16, num_global_key_value_heads=1, sliding_window=6,
        # more than one sliding layer, like the real model
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        enable_moe_block=True, hidden_size_per_layer_input=0,
        num_kv_shared_layers=0, max_position_embeddings=512, vocab_size_per_layer_input=64,
    )
    c._attn_implementation = "sdpa"
    return c


def synthetic_cases(rnd):
    A = [rnd.randrange(3, 64) for _ in range(20)]           # fixed system text (longer than the window, like the real prompt)
    B = [rnd.randrange(3, 64) for _ in range(12)]           # fixed tool declarations
    orders = [[], [11, 12], [13], [11, 14, 15]]
    cases = []
    for k in range(10):
        order = orders[k % len(orders)]
        conv = [rnd.randrange(3, 64) for _ in range(rnd.randrange(2, 6))]
        if k in (2, 6):                                     # two cases with the same order and a shared conv start
            conv = [20, 21] + conv
        ids = A + order + B + conv
        kinds = ["F"] * len(A) + ["C"] * len(order) + ["F"] * len(B) + ["C"] * len(conv)
        cases.append({"id": f"s{k}", "category": "x", "ids": ids, "kinds": kinds})
    return cases


def main():
    torch.manual_seed(0)
    rnd = random.Random(0)
    config = tiny_config()
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4TextRotaryEmbedding
    model = Gemma4TextModel(config).eval()
    with torch.no_grad():                                   # make the norms/scales non-trivial
        for n, p in model.named_parameters():
            if p.dim() == 1:
                p.add_(torch.randn_like(p) * 0.3)
    cases = synthetic_cases(rnd)
    tops, segs = build_tree(cases)
    rotary = Gemma4TextRotaryEmbedding(config)

    # --- tree run (embeddings exactly like the model)
    states = {s.sid: model.embed_tokens(torch.tensor(s.tokens)).float() for s in segs}
    tree_stats = []
    for i, layer in enumerate(model.layers):
        st = Stats(config.num_experts)
        ctx = {}
        orig = (layer.router.forward, layer.experts.forward)
        instrument(layer, st, ctx)
        states = run_layer_on_tree(layer, i, config, rotary, tops, states, ctx, torch.float32)
        layer.router.forward, layer.experts.forward = orig
        tree_stats.append(st)

    # map each case to its path of segments
    def path_of(ci):
        out = []
        for top in tops:
            stack = [(top, [top])]
            while stack:
                seg, p = stack.pop()
                if ci in seg.cases and not any(ci in ch.cases for ch in seg.children):
                    return p
                stack += [(ch, p + [ch]) for ch in seg.children if ci in ch.cases]
        return out

    # --- 1. hidden states vs transformers
    max_err = 0.0
    for ci, c in enumerate(cases):
        p = path_of(ci)
        tree_h = torch.cat([states[s.sid] for s in p])
        with torch.no_grad():
            ref = model(input_ids=torch.tensor([c["ids"]]), use_cache=False).last_hidden_state[0]
        mine = model.norm(tree_h)
        max_err = max(max_err, float((mine - ref).abs().max()))
    print(f"1. hidden states, max abs diff vs Gemma4TextModel: {max_err:.2e}")
    ok1 = max_err < 1e-4

    # --- 2. stats vs brute force
    ctx_count = {}
    for s in segs:
        for fi in s.fixed_idx:
            if fi >= 0:
                ctx_count[fi] = ctx_count.get(fi, 0) + 1
    ok2 = True
    with torch.no_grad():                                   # reference layer inputs, before any instrumentation
        all_hs = [model(input_ids=torch.tensor([c["ids"]]), use_cache=False, output_hidden_states=True).hidden_states
                  for c in cases]
    for i, layer in enumerate(model.layers):
        st = Stats(config.num_experts)
        ctx = {}
        orig = (layer.router.forward, layer.experts.forward)
        instrument(layer, st, ctx)
        for ci, c in enumerate(cases):
            p = path_of(ci)
            # per-token weights for this case alone
            w, fixed = [], []
            for s in p:
                for k, fi in zip(s.kinds, s.fixed_idx):
                    w.append(1.0 / (ctx_count[fi] * len(s.cases)) if k == "F" else 1.0)
                    fixed.append(k == "F")
            h = all_hs[ci][i]                               # input of layer i for this case alone
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
            lt = config.layer_types[i]
            pos = torch.arange(h.shape[1]).unsqueeze(0)
            mask_fn = create_causal_mask if lt == "full_attention" else create_sliding_window_causal_mask
            mask = mask_fn(config=config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos)
            ctx["w"] = torch.tensor(w, dtype=torch.float64)
            ctx["is_fixed"] = torch.tensor(fixed)
            with torch.no_grad():
                layer(h, None, shared_kv_states=UserDict(), position_embeddings=rotary(h, pos, lt), attention_mask=mask,
                      position_ids=pos, past_key_values=None)
        layer.router.forward, layer.experts.forward = orig
        a, b = tree_stats[i].to_json(), st.to_json()
        for g in ("fixed", "case", "combined"):
            for f in ("count", "sum_gate", "sum_norm", "sum_gate_norm", "prob_mass", "reap"):
                d = max(abs(x - y) for x, y in zip(a[g][f], b[g][f]))
                if d > 1e-3:
                    ok2 = False
                    print(f"   layer {i} {g}.{f} differs by {d:.2e}")
        print(f"2. layer {i}: tree vs brute-force stats {'match' if ok2 else 'DIFFER'}; "
              f"weighted tokens fixed {a['fixed']['weighted_tokens']:.2f} (fixed prompt length 32), "
              f"case {a['case']['weighted_tokens']:.0f}")
    print("SELF-TEST", "PASS" if ok1 and ok2 else "FAIL")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
