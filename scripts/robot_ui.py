"""Step 7: local robot comparison page for the blog.

    .venv\\Scripts\\python.exe scripts\\robot_ui.py            -> http://127.0.0.1:8765/
    .venv\\Scripts\\python.exe scripts\\robot_ui.py --port 8800

Replay mode reads the recorded Step 3 / Step 5 exam runs (results/step3, results/step5) at request time.
Live mode sends one typed order to Ollama with the Step 6c deploy prompt layout (order state last),
same options as the exams (temperature 0, seed 42, thinking off, num_ctx 8192, num_predict 512).
Exports (GIF / PNG frames made in the browser) are saved under ui/exports/<case>/ (gitignored).
Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bench_baseline import gpu_memory, runner_memory  # noqa: E402
from bench_latency import build_messages  # noqa: E402
from cafe_common import KO_DINE, defaults, items_by_name, load_tools, normalize_add  # noqa: E402
from run_exam import MODE_OPTIONS, OPTIONS, tool_calls_of  # noqa: E402
from score import load_jsonl, score_cafe_case_lenient  # noqa: E402
from vending_machine import VendingMachine  # noqa: E402

HOST = "http://127.0.0.1:11434"
UI = ROOT / "ui"
EXPORTS = UI / "exports"
RES = ROOT / "results"

ROBOTS = [
    {"key": "vending", "name": "자판기", "sub": "규칙 프로그램", "model": None, "mode": "rules",
     "dir": "step3/{exam}__vending__rules"},
    {"key": "e4b", "name": "E4B", "sub": "작은 모델 · iGPU", "model": "gemma4:e4b", "mode": "igpu",
     "dir": "step3/{exam}__gemma4-e4b__igpu"},
    {"key": "12b", "name": "12B", "sub": "중간 모델 · iGPU", "model": "gemma4:12b", "mode": "igpu",
     "dir": "step3/{exam}__gemma4-12b__igpu"},
    {"key": "26b", "name": "26B 원본", "sub": "큰 모델 · CPU", "model": "gemma4:26b", "mode": "cpu",
     "dir": "step3/{exam}__gemma4-26b__cpu"},
    {"key": "64e", "name": "64e", "sub": "26B 가지치기 · iGPU", "model": "song/gemma4-26b-cafe-64e", "mode": "igpu",
     "dir": "step5/{exam}__song-gemma4-26b-cafe-64e__igpu"},
]
# preselected "dramatic" cases for the blog (반샷만, 따뜻한 아이스, 녹차라떼 fallback, 캐리어, 아샷추, 시럽)
# Badges use the fair Step 6a measurement (same laptop, 30 cafe cases, prompt cache fixed), not the per-case
# Step 3/5 times, which had the prompt re-processing problem and the growing cache.
FAIR = {"e4b": "gemma4-e4b__igpu", "12b": "gemma4-12b__igpu", "26b": "gemma4-26b__cpu",
        "64e": "song-gemma4-26b-cafe-64e__igpu"}
PICKS = ["L09", "L04", "D1-2", "L18", "L13", "L14"]
CATEGORY_KO = {
    "basic": "기본", "options": "옵션", "slang": "줄임말", "multi_item": "여러 메뉴", "off_menu": "없는 메뉴",
    "impossible": "안 되는 옵션", "ambiguous": "애매한 말", "conditional": "조건부", "call_staff": "직원 호출",
    "modify_cancel": "변경·취소", "takeout": "포장", "unavailable": "품절·불가",
}
EXAMS = {"cafe": ROOT / "data/exam_cafe.jsonl", "live": ROOT / "data/exam_live.jsonl"}
EXAM_KO = {"cafe": "카페 시험", "live": "실전 시험"}


# ---------- tool calls -> plain Korean speech ----------

def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def shots_ko(v) -> str:
    v = _num(v)
    n = abs(v)
    whole, half = int(n), (n - int(n)) >= 0.5
    amount = ("반 개" if whole == 0 else f"{whole}개 반") if half else f"{whole}개"
    return f"샷 {amount} {'추가' if v > 0 else '빼고'}"


def item_ko(args: dict) -> str:
    a = normalize_add(args)
    item = items_by_name().get(a["menu"])
    head = str(a["menu"])
    if item and len(item["temperatures"]) > 1 and a.get("temperature") in ("hot", "iced"):
        head = ("아이스 " if a["temperature"] == "iced" else "따뜻한 ") + head
    elif not item and a.get("temperature") == "iced":
        head = "아이스 " + head
    opts = []
    if a.get("size") == "large":
        opts.append("큰 사이즈")
    if _num(a.get("shots_delta")):
        opts.append(shots_ko(a["shots_delta"]))
    if a.get("sweetness") == "less":
        opts.append("덜 달게")
    elif a.get("sweetness") == "more":
        opts.append("더 달게")
    if a.get("milk") == "oat":
        opts.append("오트우유")
    unit = "개" if item and item["category"] == "dessert" else "잔"
    return head + (f"({', '.join(opts)})" if opts else "") + f" {a.get('quantity') or 1}{unit}"


def changes_ko(ch: dict) -> str:
    out = []
    for k, v in (ch or {}).items():
        if k == "menu":
            out.append(f"{v}(으)로")
        elif k == "temperature":
            out.append({"iced": "아이스로", "hot": "따뜻하게"}.get(v, str(v)))
        elif k == "size":
            out.append({"large": "큰 사이즈로", "regular": "보통 사이즈로"}.get(v, str(v)))
        elif k == "shots_delta":
            out.append(shots_ko(v) if _num(v) else "샷 기본으로")
        elif k == "sweetness":
            out.append({"less": "덜 달게", "more": "더 달게", "normal": "보통 당도로"}.get(v, str(v)))
        elif k == "milk":
            out.append({"oat": "오트우유로", "regular": "일반 우유로"}.get(v, str(v)))
        elif k == "quantity":
            out.append(f"{v}잔으로")
        else:
            out.append(f"{k}={v}")
    return ", ".join(out) or "(바뀐 것 없음)"


def _obj(word: str) -> str:
    """word + object particle 을/를 (by the final consonant of the last Hangul syllable)."""
    ch = word.strip()[-1:]
    if "가" <= ch <= "힣":
        return word + ("을" if (ord(ch) - 0xAC00) % 28 else "를")
    return word + "을(를)"


def _line_name(order_state, idx) -> str:
    """'1번 아이스 아메리카노를' (object form, used before 뺐어요 / 바꿨어요)."""
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return _obj(f"{idx}번 메뉴")
    if order_state and 1 <= i <= len(order_state):
        short = re.sub(r"\(.*\)", "", item_ko(order_state[i - 1]).rsplit(" ", 1)[0]).strip()
        return _obj(f"{i}번 {short}")
    return _obj(f"없는 {i}번 메뉴")


REASON_KO = {"ambiguous": "되묻기", "impossible": "안 되는 옵션", "off_menu": "없는 메뉴"}


def speech(calls: list, content: str = "", order_state: list | None = None) -> list[dict]:
    """Each line: kind (add/modify/remove/ask/staff/confirm/talk/silent), text, optional note."""
    lines = []
    for c in calls or []:
        name, a = c.get("name"), c.get("arguments") or {}
        if name == "add_item":
            lines.append({"kind": "add", "text": f"{item_ko(a)} 담았어요"})
        elif name == "modify_item":
            lines.append({"kind": "modify", "text": f"{_line_name(order_state, a.get('item_index'))} "
                                                    f"{changes_ko(a.get('changes'))} 바꿨어요"})
        elif name == "remove_item":
            lines.append({"kind": "remove", "text": f"{_line_name(order_state, a.get('item_index'))} 뺐어요"})
        elif name == "ask_clarification":
            lines.append({"kind": "ask", "text": str(a.get("question") or "…"),
                          "note": REASON_KO.get(a.get("reason"), str(a.get("reason") or ""))})
        elif name == "call_staff":
            lines.append({"kind": "staff", "text": "직원 불러드릴게요!", "note": f"메모: {a.get('reason') or ''}".strip()})
        elif name == "confirm_order":
            opt = a.get("dine_option") or defaults()["dine_option"]
            lines.append({"kind": "confirm", "text": f"주문 확정할게요! ({KO_DINE.get(opt, opt)})"})
        else:
            lines.append({"kind": "talk", "text": f"{name}?", "note": "없는 도구"})
    if not lines:
        content = re.sub(r"\s+", " ", (content or "")).strip()
        if content:
            lines.append({"kind": "talk", "text": content[:140] + ("…" if len(content) > 140 else ""),
                          "note": "말만 하고 도구는 안 썼어요"})
        else:
            lines.append({"kind": "silent", "text": "……", "note": "아무 응답도 없었어요"})
    return lines


# ---------- replay data ----------

_cache: dict = {}


def _by_id(path: Path) -> dict:
    if path not in _cache:
        _cache[path] = {r["id"]: r for r in load_jsonl(path)} if path.exists() else {}
    return _cache[path]


def fair_numbers(key: str) -> dict:
    """Median time per order and steady memory (median of max(runner private, GPU shared) while ordering)."""
    if ("fair", key) in _cache:
        return _cache[("fair", key)]
    import statistics
    if key == "vending":  # rule program, not part of 6a: median of its own cafe exam run (no cache involved)
        rows = load_jsonl(RES / "step3/cafe__vending__rules/raw.jsonl")
        out = {"wall_s": statistics.median(r["wall_s"] for r in rows), "memory_gb": None}
    else:
        from report_step6 import mem
        res = json.loads((RES / "step6/latency" / f"{FAIR[key]}.json").read_text(encoding="utf-8"))
        out = {"wall_s": statistics.median(c["wall_s"] for c in res["cases"]), "memory_gb": mem(res)["foot_med"]}
    out = {"wall_s": round(out["wall_s"], 3), "memory_gb": None if out["memory_gb"] is None else round(out["memory_gb"], 1)}
    _cache[("fair", key)] = out
    return out


def reason_only(exam: str, case: dict, rb: dict) -> bool:
    """Wrong only because of the ask_clarification reason code (strict 0, lenient 1)."""
    d = RES / rb["dir"].format(exam=exam)
    sc, raw = _by_id(d / "scores.jsonl").get(case["id"]), _by_id(d / "raw.jsonl").get(case["id"])
    if not sc or not raw or sc["exact"]:
        return False
    return bool(score_cafe_case_lenient(case, raw.get("tool_calls") or [])["exact"])


def case_index() -> list[dict]:
    out = []
    for exam, path in EXAMS.items():
        for c in load_jsonl(path):
            marks = ""
            for rb in ROBOTS:
                s = _by_id(RES / rb["dir"].format(exam=exam) / "scores.jsonl").get(c["id"])
                marks += "?" if s is None else ("○" if s["exact"] else "△" if reason_only(exam, c, rb) else "✗")
            last = [m["content"] for m in c["conversation"] if m["role"] == "user"][-1]
            out.append({"id": c["id"], "exam": exam, "exam_ko": EXAM_KO[exam],
                        "category": CATEGORY_KO.get(c["category"], c["category"]), "text": last,
                        "marks": marks, "pick": c["id"] in PICKS})
    out.sort(key=lambda r: (not r["pick"], PICKS.index(r["id"]) if r["pick"] else 0, r["exam"] != "live", r["id"]))
    return out


def replay(case_id: str) -> dict:
    for exam, path in EXAMS.items():
        case = _by_id(path).get(case_id)
        if case:
            break
    else:
        raise KeyError(case_id)
    st = case.get("order_state") or []
    robots = []
    for rb in ROBOTS:
        d = RES / rb["dir"].format(exam=exam)
        raw, sc = _by_id(d / "raw.jsonl").get(case_id), _by_id(d / "scores.jsonl").get(case_id)
        r = {k: rb[k] for k in ("key", "name", "sub", "mode")}
        if raw is None:
            r.update(missing=True)
        else:
            r.update(lines=speech(raw.get("tool_calls"), raw.get("content"), st),
                     correct=bool(sc and sc["exact"]), partial=sc and sc.get("partial"),
                     reason_only=reason_only(exam, case, rb), **fair_numbers(rb["key"]))
        robots.append(r)
    return {
        "id": case_id, "exam": exam, "exam_ko": EXAM_KO[exam],
        "category": CATEGORY_KO.get(case["category"], case["category"]),
        "conversation": case["conversation"],
        "order_state": [item_ko(it) for it in st],
        "expected": speech(case.get("expected"), "", st),
        "robots": robots,
    }


# ---------- live mode ----------

_vm = None


def apply_calls(state: list, calls: list) -> list:
    state = [dict(x) for x in state]
    for c in calls:
        a = c.get("arguments") or {}
        try:
            if c["name"] == "add_item":
                state.append(normalize_add(a))
            elif c["name"] == "modify_item":
                i = int(a.get("item_index")) - 1
                if 0 <= i < len(state):
                    state[i] = normalize_add({**state[i], **(a.get("changes") or {})})
            elif c["name"] == "remove_item":
                i = int(a.get("item_index")) - 1
                if 0 <= i < len(state):
                    state.pop(i)
        except (TypeError, ValueError):
            pass
    return state


def live(body: dict) -> dict:
    global _vm
    rb = next(r for r in ROBOTS if r["key"] == body["robot"])
    text = str(body["text"]).strip()
    history = body.get("history") or []
    state = body.get("order_state") or []
    mode = body.get("mode") or rb["mode"]
    t0 = time.perf_counter()
    info = {}
    if rb["model"] is None:
        _vm = _vm or VendingMachine()
        users = [m["content"] for m in history if m["role"] == "user"] + [text]
        calls, content = _vm.predict_text(text, state, users), ""
        wall = time.perf_counter() - t0
        mem = None
    else:
        msgs = build_messages("deploy", state, history + [{"role": "user", "content": text}])
        payload = {"model": rb["model"], "messages": msgs, "stream": False, "think": False, "keep_alive": "10m",
                   "tools": load_tools(),
                   "options": {**OPTIONS, **MODE_OPTIONS[mode], "num_ctx": 8192, "num_predict": 512}}
        r = requests.post(f"{HOST}/api/chat", json=payload, timeout=1800)
        wall = time.perf_counter() - t0
        if r.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        d = r.json()
        msg = d.get("message") or {}
        calls, content = tool_calls_of(msg), msg.get("content") or ""
        info = {"load_s": round((d.get("load_duration") or 0) / 1e9, 2),
                "prompt_tokens": d.get("prompt_eval_count"), "gen_tokens": d.get("eval_count")}
        rm, gm = runner_memory(), gpu_memory()
        mem = round(max(rm.get("runner_private_gb") or 0, gm.get("gpu_shared_gb") or 0), 1)
    lines = speech(calls, content, state)
    new_state = apply_calls(state, calls)
    return {"robot": rb["key"], "mode": mode, "lines": lines, "tool_calls": calls, "wall_s": round(wall, 3),
            "memory_gb": mem, **info, "order_state": new_state, "order_state_ko": describe_state(new_state),
            "assistant_text": " ".join(x["text"] for x in lines)}


def describe_state(state: list) -> list[str]:
    return [item_ko(it) for it in state]


# ---------- export ----------

def save_export(body: dict) -> dict:
    case = re.sub(r"[^A-Za-z0-9_-]", "_", str(body.get("case") or "live"))[:40]
    out = EXPORTS / case
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in body["files"]:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", f["name"])[:80]
        if not name.endswith((".gif", ".png")):
            continue
        (out / name).write_bytes(base64.b64decode(f["data"].split(",", 1)[-1]))
        saved.append(name)
    return {"dir": str(out.relative_to(ROOT)).replace("\\", "/"), "files": saved}


# ---------- http ----------

TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".png": "image/png", ".gif": "image/gif", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/cases":
                return self._json({"cases": case_index(), "robots": [{k: r[k] for k in ("key", "name", "sub", "mode")}
                                                                      for r in ROBOTS]})
            if u.path == "/api/replay":
                return self._json(replay(q["case"][0]))
            rel = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
            p = (UI / rel).resolve()
            if UI.resolve() not in p.parents or not p.is_file():
                return self._send(404, b"not found", "text/plain")
            return self._send(200, p.read_bytes(), TYPES.get(p.suffix, "application/octet-stream"))
        except KeyError as e:
            return self._json({"error": f"unknown case {e}"}, 404)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if u.path == "/api/live":
                return self._json(live(body))
            if u.path == "/api/state":
                return self._json({"lines": describe_state(body.get("order_state") or [])})
            if u.path == "/api/export":
                return self._json(save_export(body))
            return self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"로봇 비교 페이지: http://127.0.0.1:{a.port}/  (Ctrl+C로 종료)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
