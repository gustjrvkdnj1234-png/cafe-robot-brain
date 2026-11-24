"""Run a typed multi-turn script through a running demo (one session) and record each turn,
including how much of the prompt llama-server reused from its cache (from the Ollama log).

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\run_script.py --label after [--port 8770]

Writes v2/voice/results/app_script_<label>.json and .md.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

VOICE = Path(__file__).resolve().parents[1]
ROOT = VOICE.parents[1]
OLLAMA_LOG = ROOT / "logs" / "ollama_serve_voice.log"
SCRIPT = ["커피 한 잔 주세요", "아메리카노요", "아이스로요", "디저트 뭐 있어요?", "아아 사이즈 뭐 있어요?",
          "아아 두잔", "카페라떼 반샷만요", "주문할게요"]


def log_tasks(offset: int) -> list:
    """llama-server tasks written after `offset`: total prompt tokens, reused (cached) and evaluated."""
    text = OLLAMA_LOG.read_bytes()[offset:].decode("utf-8", errors="replace")
    tasks, cur = [], None
    for line in text.splitlines():
        m = re.search(r"task (\d+) \| new prompt.*task\.n_tokens = (\d+)", line)
        if m:
            cur = {"task": int(m[1]), "n_tokens": int(m[2]), "cached": []}
            tasks.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"cached n_tokens = (\d+)", line)
        if m:
            cur["cached"].append(int(m[1]))
        m = re.search(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", line)
        if m:
            cur["eval_ms"], cur["eval_tokens"] = float(m[1]), int(m[2])
    return [t for t in tasks if "eval_tokens" in t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--settings", help='JSON for /api/settings before the run, e.g. {"history":"text","layout":"deploy","max_turns":0}')
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    base = f"http://127.0.0.1:{a.port}"
    if a.settings:
        print("settings:", requests.post(f"{base}/api/settings", json=json.loads(a.settings), timeout=10).json())
    h = requests.get(f"{base}/api/health", timeout=5).json()
    if not h.get("ready"):
        sys.exit(f"app not ready: {h}")
    offset = OLLAMA_LOG.stat().st_size
    sess = requests.post(f"{base}/api/new_session", json={}, timeout=5).json()["session"]
    rows = []
    for text in SCRIPT:
        t = time.perf_counter()
        d = requests.post(f"{base}/api/turn/text", json={"session": sess, "text": text}, timeout=600).json()
        rows.append({"input": text, "tool_calls": d["tool_calls"], "content": d["content"], "reply": d["reply"],
                     "kinds": d["kinds"], "order": d["order"], "timings": d["timings"],
                     "client_s": round(time.perf_counter() - t, 3)})
        tt = d["timings"]
        print(f"{text} | {json.dumps(d['tool_calls'], ensure_ascii=False)[:110]} | content={d['content'][:40]!r} | "
              f"{d['reply'][:50]} | prompt {tt.get('prompt_tokens')} tok {tt.get('prompt_s')} s | llm {tt.get('llm_s')}",
              flush=True)
    time.sleep(1)
    tasks = log_tasks(offset)
    if len(tasks) == len(rows):
        for r, t in zip(rows, tasks):
            r["cache"] = {"n_tokens": t["n_tokens"], "reused": t["cached"][0] if t["cached"] else 0,
                          "evaluated": t["eval_tokens"], "eval_s": round(t["eval_ms"] / 1000, 3)}
    else:
        print(f"note: {len(tasks)} llama-server tasks for {len(rows)} turns; cache columns left empty")
    free = sum("free_text" in r["kinds"] for r in rows)
    res = {"label": a.label, "settings": h.get("settings"), "free_text_turns": free, "turns": rows}
    out = VOICE / "results"
    (out / f"app_script_{a.label}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    s = h.get("settings") or {}
    L = [f"# 대화 스크립트 ({a.label})", "",
         f"- 설정: 기록 형식 `{s.get('history')}`, 배치 `{s.get('layout')}`, 기록 상한 {s.get('max_turns')}턴"
         f"(넘으면 {s.get('trim_to')}턴으로 줄임)", f"- 도구 없이 말로만 답한 턴: {free}/{len(rows)}", "",
         "| # | 손님 | 도구 호출 | content | 답 | 프롬프트 토큰 (재사용 / 새로 계산) | prompt_s | llm_s |",
         "|---|---|---|---|---|---|---:|---:|"]
    for i, r in enumerate(rows, 1):
        calls = "; ".join(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})" for c in r["tool_calls"]) or "(없음)"
        c = r.get("cache")
        cache = f"{c['n_tokens']} ({c['reused']} / {c['evaluated']})" if c else str(r["timings"].get("prompt_tokens"))
        L.append(f"| {i} | {r['input']} | `{calls}` | {r['content'].strip().replace(chr(10), ' ') or '-'} | {r['reply']} | "
                 f"{cache} | {r['timings'].get('prompt_s')} | {r['timings'].get('llm_s')} |")
    (out / f"app_script_{a.label}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:4]))


if __name__ == "__main__":
    main()
