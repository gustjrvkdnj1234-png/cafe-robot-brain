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
SCRIPTS = {
    "s8": ["커피 한 잔 주세요", "아메리카노요", "아이스로요", "디저트 뭐 있어요?", "아아 사이즈 뭐 있어요?",
           "아아 두잔", "카페라떼 반샷만요", "주문할게요"],
    "s9": ["커피 한 잔 주세요", "아메리카노요", "아이스로요", "디저트 뭐 있어요?", "마카롱 있어요?",
           "아아 사이즈 뭐 있어요?", "큰 걸로 바꿔 주세요", "카페라떼 반샷만요", "주문할게요"],
}


def from_log(prefix: str) -> list:
    """Transcripts of one real session (session id prefix), in order, from the local turn logs (not committed)."""
    out = []
    for p in sorted((VOICE / "app" / "logs").glob("turns_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if str(d.get("session", "")).startswith(prefix) and (d.get("transcript") or "").strip():
                out.append(d["transcript"])
    return out


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
    ap.add_argument("--script", choices=sorted(SCRIPTS), default="s8")
    ap.add_argument("--from-log", help="replay the transcripts of this session id (prefix) from app/logs instead")
    ap.add_argument("--warmup", action="store_true", help="one throwaway turn first (e.g. after a mode switch)")
    ap.add_argument("--settings", help='JSON for /api/settings before the run, e.g. {"history":"text","layout":"deploy","max_turns":0}')
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    base = f"http://127.0.0.1:{a.port}"
    if a.settings:
        print("settings:", requests.post(f"{base}/api/settings", json=json.loads(a.settings), timeout=10).json())
    h = requests.get(f"{base}/api/health", timeout=5).json()
    if not h.get("ready"):
        sys.exit(f"app not ready: {h}")
    if a.warmup:
        w = requests.post(f"{base}/api/new_session", json={}, timeout=5).json()["session"]
        requests.post(f"{base}/api/turn/text", json={"session": w, "text": "안녕하세요"}, timeout=900)
        requests.post(f"{base}/api/new_session", json={"previous": w}, timeout=5)
    offset = OLLAMA_LOG.stat().st_size
    sess = requests.post(f"{base}/api/new_session", json={}, timeout=5).json()["session"]
    rows = []
    texts = from_log(a.from_log) if a.from_log else SCRIPTS[a.script]
    if not texts:
        sys.exit(f"no transcripts for session {a.from_log}")
    for text in texts:
        t = time.perf_counter()
        d = requests.post(f"{base}/api/turn/text", json={"session": sess, "text": text}, timeout=600).json()
        rows.append({"input": text, "tool_calls": d["tool_calls"], "content": d["content"], "reply": d["reply"],
                     "kinds": d["kinds"], "order": d["order"], "timings": d["timings"],
                     "confirmed": d.get("confirmed"), "phase": d.get("phase"), "dine_option": d.get("dine_option"),
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
    h = requests.get(f"{base}/api/health", timeout=5).json()
    res = {"label": a.label, "script": f"log:{a.from_log}" if a.from_log else a.script, "settings": h.get("settings"), "free_text_turns": free, "turns": rows}
    out = VOICE / "results"
    (out / f"app_script_{a.label}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    s = h.get("settings") or {}
    L = [f"# 대화 스크립트 ({a.label})", "",
         f"- 설정: 기록 형식 `{s.get('history')}`, 배치 `{s.get('layout')}`, 기록 상한 {s.get('max_turns')}턴"
         f"(넘으면 {s.get('trim_to')}턴으로 줄임), 설정 `{s.get('config')}`, 모드 `{s.get('mode')}`, num_ctx {s.get('num_ctx')}",
         f"- 도구 없이 말로만 답한 턴: {free}/{len(rows)}", "",
         "| # | 손님 | 도구 호출 | 답 | 주문 | 확정 상태 | 프롬프트 토큰 (재사용 / 새로 계산) | prompt_s | gen_s | llm_s | total_s |",
         "|---|---|---|---|---|---|---|---:|---:|---:|---:|"]
    for i, r in enumerate(rows, 1):
        calls = "; ".join(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})" for c in r["tool_calls"]) or "(없음)"
        c = r.get("cache")
        cache = f"{c['n_tokens']} ({c['reused']} / {c['evaluated']})" if c else str(r["timings"].get("prompt_tokens"))
        state = r["phase"] or ("확정" if r["confirmed"] else "주문 중")
        state = {"ordering": "주문 중", "awaiting": "확인 대기", "confirmed": "확정"}.get(state, state)
        if r["confirmed"] and r.get("dine_option"):
            state += f" ({'포장' if r['dine_option'] == 'takeout' else '매장'})"
        reply = r["reply"].replace(chr(10), " / ")
        L.append(f"| {i} | {r['input']} | `{calls}` | {reply} | {' / '.join(r['order']) or '-'} | {state} | "
                 f"{cache} | {r['timings'].get('prompt_s')} | {r['timings'].get('gen_s')} | {r['timings'].get('llm_s')} | "
                 f"{r['timings'].get('total_s')} |")
    (out / f"app_script_{a.label}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:4]))


if __name__ == "__main__":
    main()
