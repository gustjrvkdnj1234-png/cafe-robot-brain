"""Step 6a: fair latency / memory benchmark on 30 cafe cases (one model + mode per call).

    .venv\\Scripts\\python.exe scripts\\bench_latency.py --model gemma4:e4b --mode igpu
    .venv\\Scripts\\python.exe scripts\\bench_latency.py --model gemma4:e4b --mode igpu --n 5 --out-dir results/step6/pilot

Server must run with (process-only env):
    OLLAMA_IGPU_ENABLE=1      iGPU allowed (CPU mode = num_gpu 0, same as Step 1/3)
    LLAMA_ARG_CACHE_RAM=0     llama-server prompt cache off -> RAM does not grow with old prompts
    LLAMA_ARG_SWA_FULL=1      full-size SWA KV, so the single slot can reuse the system-prompt prefix

Per run: unload all -> warm-up (system prompt + tools, empty order) -> 30 orders, streamed, sequential.
Same request as run_exam.py (system prompt, tools, temperature 0, seed 42, thinking off, num_ctx 8192, num_predict 512).
Cases: every 4th line of data/exam_cafe.jsonl (30 cases, all 12 categories).
Output: <out-dir>/<model>__<mode>.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bench_baseline import api_ps, gpu_memory, runner_memory, system_memory, unload_all  # noqa: E402
from cafe_common import build_system_prompt, load_tools  # noqa: E402
from run_exam import MODE_OPTIONS, OPTIONS, slug, tool_calls_of  # noqa: E402
from score import load_jsonl  # noqa: E402

HOST = "http://127.0.0.1:11434"
NUM_CTX, NUM_PREDICT, KEEP = 8192, 512, "30m"


class Series:
    """Memory time series (every interval_s) in a background thread."""

    def __init__(self, interval_s=2.0):
        self.interval_s, self.rows, self.phase = interval_s, [], "load"
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            try:
                d = {**runner_memory(), **gpu_memory(), **system_memory()}
                d.pop("runner_names", None)
                self.rows.append({"t": round(time.perf_counter() - t0, 1), "phase": self.phase,
                                  **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}})
            except Exception as e:  # noqa: BLE001
                self.rows.append({"t": round(time.perf_counter() - t0, 1), "phase": self.phase, "error": str(e)})
            self._stop.wait(self.interval_s)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=10)


def stream_chat(model, mode, messages, tools, num_predict):
    payload = {"model": model, "messages": messages, "stream": True, "think": False, "keep_alive": KEEP,
               "tools": tools,
               "options": {**OPTIONS, **MODE_OPTIONS[mode], "num_ctx": NUM_CTX, "num_predict": num_predict}}
    t0 = time.perf_counter()
    first = None
    content, calls, final = [], [], None
    with requests.post(f"{HOST}/api/chat", json=payload, stream=True, timeout=1800) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                raise RuntimeError(d["error"])
            m = d.get("message") or {}
            if first is None and (m.get("content") or m.get("thinking") or m.get("tool_calls")):
                first = time.perf_counter() - t0
            content.append(m.get("content") or "")
            calls += tool_calls_of(m)
            if d.get("done"):
                final = d
    wall = time.perf_counter() - t0
    ns = lambda k: (final.get(k) or 0) / 1e9  # noqa: E731
    ec, es = final.get("eval_count") or 0, ns("eval_duration")
    return {
        "wall_s": wall, "first_chunk_s": first,
        "load_s": ns("load_duration"), "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_s": ns("prompt_eval_duration"), "eval_count": ec, "eval_s": es,
        # server-side time to first token: load + prompt processing + one decode step
        "ttft_s": ns("load_duration") + ns("prompt_eval_duration") + (es / ec if ec else 0),
        "tok_s": ec / es if es else None,
        "tool_calls": calls, "content": "".join(content), "done_reason": final.get("done_reason"),
    }


def fits(model):
    want = model if ":" in model else model + ":latest"
    for name, m in api_ps().items():
        if name == want:
            return {"size": m.get("size"), "size_vram": m.get("size_vram"),
                    "full_gpu": bool(m.get("size") and m.get("size_vram", 0) >= m["size"])}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=sorted(MODE_OPTIONS), required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out-dir", default="results/step6/latency")
    ap.add_argument("--server-env", default="OLLAMA_IGPU_ENABLE=1,LLAMA_ARG_CACHE_RAM=0,LLAMA_ARG_SWA_FULL=1",
                    help="comma list, recorded in the output (must match how ollama serve was started)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cases = load_jsonl(ROOT / "data/exam_cafe.jsonl")[::4][: a.n]
    tools = load_tools()
    out = ROOT / a.out_dir / f"{slug(a.model)}__{a.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"model": a.model, "mode": a.mode, "started": datetime.now().isoformat(timespec="seconds"),
           "settings": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, **OPTIONS, **MODE_OPTIONS[a.mode],
                        "think": False, "server_env": a.server_env.split(",")},
           "case_ids": [c["id"] for c in cases]}
    unload_all()
    time.sleep(3)
    ser = Series().start()
    try:
        warm_msgs = [{"role": "system", "content": build_system_prompt(None)}, {"role": "user", "content": "안녕하세요"}]
        res["warmup"] = {k: v for k, v in stream_chat(a.model, a.mode, warm_msgs, tools, 1).items()
                         if k not in ("tool_calls", "content")}
        res["placement"] = fits(a.model)
        print(f"warm-up: load {res['warmup']['load_s']:.1f}s, prompt {res['warmup']['prompt_eval_count']} tok "
              f"in {res['warmup']['prompt_eval_s']:.1f}s, placement {res['placement']}", flush=True)
        if a.mode == "igpu" and not (res["placement"] or {}).get("full_gpu"):
            res["skipped"] = "not fully on iGPU"
            print("not fully on iGPU -> skip", flush=True)
            return
        ser.phase = "orders"
        res["cases"] = []
        for c in cases:
            msgs = [{"role": "system", "content": build_system_prompt(c.get("order_state"))}] + c["conversation"]
            r = stream_chat(a.model, a.mode, msgs, tools, NUM_PREDICT)
            r["id"] = c["id"]
            res["cases"].append(r)
            print(f"{c['id']}: ttft {r['ttft_s']:.2f}s total {r['wall_s']:.2f}s prompt_eval {r['prompt_eval_count']} tok "
                  f"{r['prompt_eval_s']:.2f}s, {r['eval_count']} tok {r['tok_s'] or 0:.1f} tok/s", flush=True)
        ser.phase = "done"
        res["placement_after"] = fits(a.model)
    except Exception as e:  # noqa: BLE001
        res["error"] = str(e)
        print("ERROR", e, flush=True)
    finally:
        time.sleep(2)
        ser.stop()
        res["memory_series"] = ser.rows
        res["finished"] = datetime.now().isoformat(timespec="seconds")
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        unload_all()
        print("saved", out.relative_to(ROOT))
        cs = res.get("cases") or []
        if cs:
            print("median ttft", round(statistics.median(x["ttft_s"] for x in cs), 2),
                  "median total", round(statistics.median(x["wall_s"] for x in cs), 2))


if __name__ == "__main__":
    main()
