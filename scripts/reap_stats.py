"""Collect REAP statistics (router gate x expert output norm) for Gemma 4 26B-A4B, one decoder layer at a time, on CPU.

    # pilot: 8 cases, first 2 layers
    .venv-prune\\Scripts\\python.exe scripts\\reap_stats.py --n 8 --layers 0-1 --out results/step4/pilot
    # full run (resumes after the last finished layer)
    .venv-prune\\Scripts\\python.exe scripts\\reap_stats.py --n 600 --layers all --out results/step4/reap_600

How it works
- Every calibration case is rendered with the checkpoint's chat template exactly as the robot sees it:
  system prompt (menu, rules, current order) + tool declarations + conversation + the expected tool call.
- The token sequences of all cases are put in a prefix tree. Each tree segment (a run of tokens with the same
  history) is computed once per layer; children reuse the parent's key/value cache. Results are identical to
  running every case on its own (checked by --self-test), but ~16x fewer tokens for 600 cases.
- Only one decoder layer's weights are in RAM at a time (read straight from the safetensors file by offset).
  Hidden states of all tree segments are kept between layers and saved after each layer, so a run can resume.

Weighting (what each token counts for)
- "fixed" tokens: text that is identical in every request (system instructions + tool declarations). The fixed
  prompt as a whole counts ONCE: a fixed token that ends up computed in k different contexts (e.g. tool
  declarations after k different order states) gets weight 1/k in each.
- "case" tokens: current order state + conversation + expected answer. Weight = number of cases sharing that
  tree segment (usually 1).
Statistics are kept for fixed, case and combined (= fixed + case).

Per expert and group: weighted routed-token count, sum of gate weight g (the weight actually applied, i.e.
renormalized top-k prob x per_expert_scale), sum of ||f(x)||_2 (expert output before weighting), sum of
g*||f(x)||, max g*||f(x)||, and the weighted router probability mass over all tokens.
REAP saliency = sum(g*||f||) / count (mean over the tokens routed to that expert).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import struct
import sys
import threading
import time
from collections import UserDict
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cafe_common import build_system_prompt, load_tools  # noqa: E402

LAYER_PREFIX = "model.language_model.layers.{}."
EMBED_KEY = "model.language_model.embed_tokens.weight"
ORDER_MARK = "[현재 주문 내역]\n"
ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


# ------------------------------------------------------------------ memory / timing

class PeakRAM:
    """Peak RSS of this process, sampled in a background thread."""

    def __init__(self, interval=0.2):
        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            self._stop.wait(0.2)

    def reset(self):
        self.peak = self.proc.memory_info().rss

    def gb(self):
        self.peak = max(self.peak, self.proc.memory_info().rss)
        return self.peak / 1024**3


# ------------------------------------------------------------------ checkpoint reading

class Checkpoint:
    """Reads single tensors from safetensors shards by byte offset (no full load)."""

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        idx = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.where = idx["weight_map"]
        self.headers = {}

    def _header(self, shard: str):
        if shard not in self.headers:                     # read lazily: a shard is only opened when needed
            with open(self.dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.headers[shard] = (8 + n, json.loads(f.read(n)))
        return self.headers[shard]

    def get(self, name: str) -> torch.Tensor:
        shard = self.where[name]
        base, h = self._header(shard)
        meta = h[name]
        a, b = meta["data_offsets"]
        with open(self.dir / shard, "rb") as f:
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
        if len(buf) != b - a:
            raise IOError(f"{name}: short read ({len(buf)} of {b - a} bytes) - is {shard} fully downloaded?")
        return torch.frombuffer(buf, dtype=ST_DTYPES[meta["dtype"]]).reshape(meta["shape"])

    def layer_state(self, i: int) -> dict[str, torch.Tensor]:
        p = LAYER_PREFIX.format(i)
        return {k[len(p):]: self.get(k) for k in self.where if k.startswith(p)}


# ------------------------------------------------------------------ inputs and prefix tree

def render_cases(tok, rows):
    """Token ids + per-token kind ('F' fixed / 'C' case) for each calibration case."""
    tools = load_tools()
    out = []
    for r in rows:
        msgs = [{"role": "system", "content": build_system_prompt(r["order_state"])}] + r["conversation"] + [
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in r["expected"]]}]
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
        ref = tok.apply_chat_template(msgs, tools=tools, tokenize=True)
        ref = list(ref["input_ids"] if hasattr(ref, "keys") else ref)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        if ids != ref:
            raise RuntimeError(f"{r['id']}: tokenizing the rendered text differs from apply_chat_template")
        # case spans: order state (after ORDER_MARK until the first tool declaration) and everything after the system turn
        o1 = text.index(ORDER_MARK) + len(ORDER_MARK)
        o2 = text.index("<|tool>", o1)
        sys_end = text.index("<turn|>", text.rindex("<tool|>")) + len("<turn|>")
        kinds = ["C" if (o1 <= s < o2) or s >= sys_end else "F" for s, _ in enc["offset_mapping"]]
        out.append({"id": r["id"], "category": r["category"], "ids": ids, "kinds": kinds})
    # fixed-token sequences must be identical across cases (so "one copy of the fixed prompt" is well defined)
    fixed0 = [t for t, k in zip(out[0]["ids"], out[0]["kinds"]) if k == "F"]
    for c in out[1:]:
        if [t for t, k in zip(c["ids"], c["kinds"]) if k == "F"] != fixed0:
            raise RuntimeError(f"{c['id']}: fixed prompt tokens differ from {out[0]['id']}")
    return out


class Segment:
    __slots__ = ("sid", "tokens", "kinds", "fixed_idx", "start", "cases", "children", "parent", "weights")

    def __init__(self, sid, start, parent):
        self.sid, self.start, self.parent = sid, start, parent
        self.tokens, self.kinds, self.fixed_idx, self.cases, self.children = [], [], [], set(), []
        self.weights = None


def build_tree(cases):
    """Compressed prefix tree. Each segment: tokens shared by exactly the same set of cases."""
    # plain trie first
    root = {"ch": {}, "cases": set()}
    for ci, c in enumerate(cases):
        cur = root
        fidx = 0
        for t, k in zip(c["ids"], c["kinds"]):
            node = cur["ch"].get(t)
            if node is None:
                node = cur["ch"][t] = {"ch": {}, "cases": set(), "tok": t, "kind": k, "fidx": fidx if k == "F" else -1}
            node["cases"].add(ci)
            if k == "F":
                fidx += 1
            cur = node
    segs = []

    def make(node, start, parent):
        seg = Segment(len(segs), start, parent)
        segs.append(seg)
        n = node
        while True:
            seg.tokens.append(n["tok"]); seg.kinds.append(n["kind"]); seg.fixed_idx.append(n["fidx"])
            seg.cases = n["cases"]
            if len(n["ch"]) == 1 and next(iter(n["ch"].values()))["cases"] == n["cases"]:
                n = next(iter(n["ch"].values()))
                continue
            break
        for child in n["ch"].values():
            seg.children.append(make(child, start + len(seg.tokens), seg))
        return seg

    sys.setrecursionlimit(100000)
    tops = [make(n, 0, None) for n in root["ch"].values()]
    # weights
    contexts_per_fixed = {}
    for s in segs:
        for fi in s.fixed_idx:
            if fi >= 0:
                contexts_per_fixed[fi] = contexts_per_fixed.get(fi, 0) + 1
    for s in segs:
        s.weights = torch.tensor([1.0 / contexts_per_fixed[fi] if k == "F" else float(len(s.cases))
                                  for k, fi in zip(s.kinds, s.fixed_idx)], dtype=torch.float64)
    return tops, segs


# ------------------------------------------------------------------ statistics

class Stats:
    FIELDS = ("count", "sum_gate", "sum_norm", "sum_gate_norm", "max_gate_norm", "prob_mass", "raw_routed")

    def __init__(self, E):
        self.d = {g: {f: torch.zeros(E, dtype=torch.float64) for f in self.FIELDS} for g in ("fixed", "case")}
        self.tokens = {"fixed": 0.0, "case": 0.0}

    def to_json(self):
        out = {}
        for g in ("fixed", "case", "combined"):
            if g == "combined":
                d = {f: (self.d["fixed"][f] + self.d["case"][f]) if f != "max_gate_norm"
                     else torch.maximum(self.d["fixed"][f], self.d["case"][f]) for f in self.FIELDS}
                tok = self.tokens["fixed"] + self.tokens["case"]
            else:
                d, tok = self.d[g], self.tokens[g]
            reap = torch.where(d["count"] > 0, d["sum_gate_norm"] / d["count"].clamp_min(1e-12), torch.zeros_like(d["count"]))
            out[g] = {"weighted_tokens": tok, **{f: [round(float(x), 6) for x in d[f]] for f in self.FIELDS},
                      "reap": [round(float(x), 6) for x in reap],
                      "frequency": [round(float(x / tok), 6) if tok else 0.0 for x in d["count"]]}
        return out


def instrument(layer, stats: Stats, ctx: dict):
    """Replace experts/router forward with versions that compute the same thing and record statistics.
    ctx['w'] (weights per token), ctx['is_fixed'] (bool per token) are set before each call."""
    experts, router = layer.experts, layer.router
    orig_router_forward = router.forward

    def router_forward(hidden_states):
        probs, w, idx = orig_router_forward(hidden_states)
        tw = ctx["w"]
        for g, mask in (("fixed", ctx["is_fixed"]), ("case", ~ctx["is_fixed"])):
            if mask.any():
                stats.d[g]["prob_mass"] += (probs[mask].double() * tw[mask, None]).sum(0)
                stats.tokens[g] += float(tw[mask].sum())
        return probs, w, idx

    def experts_forward(hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        tw, is_fixed = ctx["w"], ctx["is_fixed"]
        for e in torch.unique(top_k_index).tolist():
            top_k_pos, token_idx = torch.where((top_k_index == e).T)
            x = hidden_states[token_idx]
            gate, up = F.linear(x, experts.gate_up_proj[e]).chunk(2, dim=-1)
            h = F.linear(experts.act_fn(gate) * up, experts.down_proj[e])     # f_e(x), before gate weighting
            g = top_k_weights[token_idx, top_k_pos]
            final.index_add_(0, token_idx, (h * g[:, None]).to(final.dtype))
            norm = h.double().norm(dim=-1)
            gd = g.double()
            wt = tw[token_idx]
            for grp, m in (("fixed", is_fixed[token_idx]), ("case", ~is_fixed[token_idx])):
                if m.any():
                    s = stats.d[grp]
                    s["count"][e] += float(wt[m].sum())
                    s["sum_gate"][e] += float((wt[m] * gd[m]).sum())
                    s["sum_norm"][e] += float((wt[m] * norm[m]).sum())
                    s["sum_gate_norm"][e] += float((wt[m] * gd[m] * norm[m]).sum())
                    s["max_gate_norm"][e] = max(float(s["max_gate_norm"][e]), float((gd[m] * norm[m]).max()))
                    s["raw_routed"][e] += float(m.sum())
        return final

    router.forward = router_forward
    experts.forward = experts_forward


# ------------------------------------------------------------------ forward over the tree

def make_layer(config, i, state, dtype):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(config, i)
    missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
    if unexpected or missing:
        raise RuntimeError(f"layer {i}: missing {missing}, unexpected {unexpected}")
    layer = layer.to(dtype)
    layer.eval()
    return layer


def run_layer_on_tree(layer, i, config, rotary, tops, states, stats_ctx, dtype):
    """Run one decoder layer over every tree segment (DFS with KV caches). Returns new states dict sid -> tensor."""
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    layer_type = config.layer_types[i]
    mask_fn = create_causal_mask if layer_type == "full_attention" else create_sliding_window_causal_mask
    new_states = {}

    def visit(seg, cache):
        h = states[seg.sid].to(dtype).unsqueeze(0)                      # [1, L, H]
        L = h.shape[1]
        pos = torch.arange(seg.start, seg.start + L).unsqueeze(0)
        pe = rotary(h, pos, layer_type)
        mask = mask_fn(config=config, inputs_embeds=h, attention_mask=None, past_key_values=cache, position_ids=pos,
                       layer_idx=i)                                   # the cache's own layer decides the kv length
        stats_ctx["w"] = seg.weights
        stats_ctx["is_fixed"] = torch.tensor([k == "F" for k in seg.kinds])
        with torch.no_grad():
            out = layer(h, None, shared_kv_states=UserDict(), position_embeddings=pe, attention_mask=mask,
                        position_ids=pos, past_key_values=cache)
        new_states[seg.sid] = out[0].float()
        for k, child in enumerate(seg.children):
            visit(child, cache if k == len(seg.children) - 1 else copy.deepcopy(cache))

    for top in tops:
        visit(top, DynamicCache(config=config))
    return new_states


# ------------------------------------------------------------------ main

def pick_cases(rows, n, seed):
    if n >= len(rows):
        return rows
    rnd = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r)
    for v in by.values():
        rnd.shuffle(v)
    cats, out, k = sorted(by), [], 0
    while len(out) < n:
        c = cats[k % len(cats)]
        if by[c]:
            out.append(by[c].pop())
        k += 1
    return out


def parse_layers(s, n_layers):
    if s == "all":
        return list(range(n_layers))
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/gemma-4-26B-A4B-it")
    ap.add_argument("--calib", default="data/calib_cafe.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", default="0-1")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="results/step4/pilot")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.threads:
        torch.set_num_threads(a.threads)
    dtype = getattr(torch, a.dtype)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    mem = PeakRAM()
    t_start = time.perf_counter()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    model_dir = ROOT / a.model_dir
    config = AutoConfig.from_pretrained(model_dir).text_config
    config._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(l) for l in (ROOT / a.calib).read_text(encoding="utf-8").splitlines() if l.strip()]
    chosen = pick_cases(rows, a.n, a.seed)
    cases = render_cases(tok, chosen)
    tops, segs = build_tree(cases)
    n_nodes = sum(len(s.tokens) for s in segs)
    n_alone = sum(len(c["ids"]) for c in cases)
    print(f"{len(cases)} cases, {n_alone:,} tokens if run alone, {n_nodes:,} computed in the tree "
          f"({len(segs)} segments), fixed tokens per case {cases[0]['kinds'].count('F')}", flush=True)
    layers = parse_layers(a.layers, config.num_hidden_layers)
    ckpt = Checkpoint(model_dir)

    meta_path = out / "meta.json"
    state_path = out / "states_after_layer.pt"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta and meta.get("case_ids") != [c["id"] for c in cases]:
        sys.exit(f"{out} holds a run with different cases; use another --out")
    done_layers = meta.get("done_layers", [])

    if done_layers and state_path.exists():
        states = torch.load(state_path)
        print(f"resume after layer {done_layers[-1]}", flush=True)
    else:
        t0 = time.perf_counter()
        W = ckpt.get(EMBED_KEY)
        scale = torch.tensor(config.hidden_size ** 0.5).to(W.dtype)      # same bf16 rounding as the model
        states = {s.sid: (W[torch.tensor(s.tokens)] * scale).float() for s in segs}
        del W
        meta.update({"embed_s": round(time.perf_counter() - t0, 2)})
        done_layers = []

    rotary = Gemma4TextRotaryEmbedding(config)
    meta.update({
        "model_dir": a.model_dir, "n_cases": len(cases), "seed": a.seed, "dtype": a.dtype,
        "threads": torch.get_num_threads(), "case_ids": [c["id"] for c in cases],
        "categories": sorted({c["category"] for c in cases}),
        "tokens_if_alone": n_alone, "tokens_computed": n_nodes, "segments": len(segs),
        "fixed_tokens_per_case": cases[0]["kinds"].count("F"),
        "case_tokens_mean": round(sum(c["kinds"].count("C") for c in cases) / len(cases), 1),
        "layers": meta.get("layers", {}),
    })
    for i in layers:
        if i in done_layers:
            continue
        mem.reset()
        t0 = time.perf_counter()
        layer = make_layer(config, i, ckpt.layer_state(i), dtype)
        t_load = time.perf_counter() - t0
        stats = Stats(config.num_experts)
        ctx = {}
        instrument(layer, stats, ctx)
        t1 = time.perf_counter()
        states = run_layer_on_tree(layer, i, config, rotary, tops, states, ctx, dtype)
        t_fwd = time.perf_counter() - t1
        del layer, ctx
        gc.collect()          # the instrumented forwards close over the layer (reference cycle): free it now
        js = stats.to_json()
        js["per_expert_scale"] = [round(float(x), 6) for x in ckpt.get(LAYER_PREFIX.format(i) + "router.per_expert_scale").float()]
        (out / f"stats_layer{i:02d}.json").write_text(json.dumps(js), encoding="utf-8")
        torch.save(states, state_path)
        done_layers.append(i)
        meta["layers"][str(i)] = {"type": config.layer_types[i], "load_s": round(t_load, 2), "forward_s": round(t_fwd, 2),
                                  "tokens_per_s": round(n_nodes / t_fwd, 1), "peak_rss_gb": round(mem.gb(), 2)}
        meta["done_layers"] = done_layers
        meta["peak_rss_gb"] = round(max(v["peak_rss_gb"] for v in meta["layers"].values()), 2)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"layer {i:2d} ({config.layer_types[i]}): load {t_load:.1f}s, forward {t_fwd:.1f}s "
              f"({n_nodes / t_fwd:.0f} tok/s), peak RSS {mem.gb():.2f} GB", flush=True)
    meta["total_s"] = round(time.perf_counter() - t_start, 1)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
