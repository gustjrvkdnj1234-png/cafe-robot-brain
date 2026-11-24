"""Regression check: does the demo config (demo prompt + inform tool) keep ordering accuracy?

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\demo_regression.py run --config v1|demo --exam cafe|live
    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\demo_regression.py report

Same model and request options as the v1 exams (song/gemma4-26b-cafe-64e, iGPU, thinking off,
temperature 0, seed 42, num_ctx 8192, num_predict 512), deploy prompt layout as in the app
(order state right before the last customer turn). Needs a running Ollama (start_ollama.ps1).

Scoring = v1 strict (score_cafe_case) and lenient, after removing inform calls, except for
pure menu questions, where calling inform (and nothing that changes the order) also counts as correct.
Pure menu question (automatic, no hand labels): every expected call is ask_clarification, the last
customer turn is a question, and it has no ordering verb.
Live outputs keep only ids, categories, tool calls and scores (no question text).
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

APP = Path(__file__).resolve().parent
ROOT = APP.parents[2]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "scripts"))
import engines  # noqa: E402
from score import load_jsonl, score_cafe_case, score_cafe_case_lenient  # noqa: E402

EXAMS = {"cafe": ROOT / "data/exam_cafe.jsonl", "live": ROOT / "data/exam_live.jsonl"}
OUT = ROOT / "v2/voice/results/demo_regression"
V1 = {"cafe": ROOT / "results/step5/cafe__song-gemma4-26b-cafe-64e__igpu",
      "live": ROOT / "results/step5/live__song-gemma4-26b-cafe-64e__igpu"}
QUESTION = re.compile(r"(\?|？|있어요|있나요|있어|되나요|돼요|되죠|돼|뭐예요|뭐에요|뭐야|뭐|어때요|나요|까요|가요)\s*[.!~]*\s*$")
ORDER_VERB = re.compile(r"(주세요|줘|주실|주시|할게요|할께요|부탁|드릴게|갈게요|담아|바꿔|빼|추가|하나요$|로요|으로요|잔요|잔이요|개요|개이요)")
CHANGES_ORDER = {"add_item", "modify_item", "remove_item", "confirm_order"}


def last_user(case) -> str:
    return [m["content"] for m in case["conversation"] if m["role"] == "user"][-1]


def is_menu_question(case) -> bool:
    exp = case.get("expected") or []
    if not exp or any(c.get("name") != "ask_clarification" for c in exp):
        return False
    t = last_user(case).strip()
    return bool(QUESTION.search(t)) and not ORDER_VERB.search(t)


def score(case, calls) -> dict:
    no_inform = [c for c in calls if c.get("name") != "inform"]
    strict = bool(score_cafe_case(case, no_inform)["exact"])
    lenient = bool(score_cafe_case_lenient(case, no_inform)["exact"])
    mq = is_menu_question(case)
    if mq and any(c.get("name") == "inform" for c in calls) and not any(c.get("name") in CHANGES_ORDER for c in calls):
        strict = lenient = True
    return {"strict": strict, "lenient": lenient, "menu_question": mq,
            "inform": any(c.get("name") == "inform" for c in calls)}


def messages_for(case) -> list:
    head, state_msg = engines.split_prompt(case.get("order_state") or [])
    conv = case["conversation"]
    i = max(k for k, m in enumerate(conv) if m["role"] == "user")
    return [head] + conv[:i] + [state_msg] + conv[i:]


FREE_TEXT_ARGS = {"ask_clarification": ("question",), "call_staff": ("reason",), "inform": ("item",)}


def redact(calls: list) -> list:
    """Live exam: drop free-text arguments (they may quote the customer's words)."""
    out = []
    for c in calls:
        args = {k: ("…" if k in FREE_TEXT_ARGS.get(c.get("name"), ()) else v) for k, v in (c.get("arguments") or {}).items()}
        out.append({"name": c.get("name"), "arguments": args})
    return out


def run(config: str, exam: str, mode: str = "igpu"):
    engines.RUN.update(config=config, mode=mode, num_ctx=8192, num_predict=512)
    llm = engines.LLM()
    cases = load_jsonl(EXAMS[exam])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"{config}__{exam}.jsonl" if mode == "igpu" else f"{config}__{exam}__{mode}.jsonl")
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    with open(path, "a", encoding="utf-8") as f:
        for k, c in enumerate(cases, 1):
            if c["id"] in done:
                continue
            r = llm.chat(messages_for(c))
            s = score(c, r["tool_calls"])
            rec = {"id": c["id"], "category": c["category"],
                   "tool_calls": r["tool_calls"] if exam == "cafe" else redact(r["tool_calls"]),
                   "content": r["content"] if exam == "cafe" else ("(text)" if r["content"].strip() else ""),
                   **s, "wall_s": round(r["wall_s"], 2), "prompt_tokens": r["prompt_tokens"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{config} {exam} {k}/{len(cases)}] {c['id']} strict={int(s['strict'])} "
                  f"{'MQ ' if s['menu_question'] else ''}{'INFORM ' if s['inform'] else ''}"
                  f"{json.dumps(r['tool_calls'], ensure_ascii=False)[:90]}", flush=True)


def stored_v1(exam) -> dict:
    """The stored v1 Step 5 run (exam layout), re-scored with the same rule."""
    cases = {c["id"]: c for c in load_jsonl(EXAMS[exam])}
    raw = {r["id"]: r for r in load_jsonl(V1[exam] / "raw.jsonl")}
    return {i: {"id": i, "category": cases[i]["category"], **score(cases[i], raw[i].get("tool_calls") or [])}
            for i in cases}


def report():
    L = ["# 데모 설정 회귀 확인 (64e, iGPU, 카페 120 + 실전 20)", "",
         "- 같은 모델·옵션(thinking 끔, temperature 0, seed 42, num_ctx 8192, num_predict 512), 앱과 같은 배포용 배치.",
         "- v1 설정 = 실험 프롬프트·도구 그대로, 데모 설정 = 데모 규칙 + inform 도구. 둘 다 이번에 같은 서버에서 실행.",
         "- 채점 = v1 엄격/관대 채점. inform 호출은 빼고 채점하되, 순수 메뉴 질문(기대 답이 되묻기뿐이고, 마지막 말이 "
         "질문이며 주문 동사가 없는 문항, 자동 판정)은 inform만 불러도 정답.",
         "- 참고: 5단계 저장 결과(시험용 배치)도 같은 규칙으로 다시 채점.", ""]
    summary = {}
    for exam in ("cafe", "live"):
        runs = {}
        for cfg, fname in (("v1", f"v1__{exam}.jsonl"), ("demo", f"demo__{exam}.jsonl"), ("demo_cpu", f"demo__{exam}__cpu.jsonl")):
            p = OUT / fname
            if p.exists():
                runs[cfg] = {r["id"]: r for r in load_jsonl(p)}
        runs["v1_stored"] = stored_v1(exam)
        n = len(runs["v1_stored"])
        L += [f"## {'카페 120' if exam == 'cafe' else '실전 20'}", "",
              "| 실행 | 엄격 정답 | 관대 정답 | inform 호출 문항 | 순수 메뉴 질문 (그중 inform) |", "|---|---:|---:|---:|---:|"]
        for name, label in (("v1_stored", "5단계 저장 결과 (시험용 배치)"), ("v1", "v1 설정 (이번)"), ("demo", "데모 설정 (이번)"),
                            ("demo_cpu", "데모 설정, CPU 전용 (이번)")):
            if name not in runs:
                continue
            rs = runs[name].values()
            st, le = sum(r["strict"] for r in rs), sum(r["lenient"] for r in rs)
            mq = [r for r in rs if r["menu_question"]]
            summary[f"{exam}_{name}"] = {"strict": st, "lenient": le, "n": len(runs[name])}
            L.append(f"| {label} | {st}/{len(runs[name])} ({100 * st / len(runs[name]):.1f}%) | {le}/{len(runs[name])} | "
                     f"{sum(r['inform'] for r in rs)} | {len(mq)} ({sum(r['inform'] for r in mq)}) |")
        if "v1" in runs and "demo" in runs:
            cats = sorted({r["category"] for r in runs["v1"].values()})
            L += ["", "| 유형 | 문항 | v1 설정 엄격 | 데모 설정 엄격 | 데모 CPU 엄격 | 5단계 저장 엄격 |", "|---|---:|---:|---:|---:|---:|"]
            for cat in cats:
                ids = [i for i, r in runs["v1"].items() if r["category"] == cat]
                L.append(f"| {cat} | {len(ids)} | {sum(runs['v1'][i]['strict'] for i in ids)} | "
                         f"{sum(runs['demo'][i]['strict'] for i in ids if i in runs['demo'])} | "
                         f"{sum(runs['demo_cpu'][i]['strict'] for i in ids if i in runs.get('demo_cpu', {})) if 'demo_cpu' in runs else '-'} | "
                         f"{sum(runs['v1_stored'][i]['strict'] for i in ids)} |")
            changed = [i for i in runs["v1"] if i in runs["demo"] and runs["v1"][i]["strict"] != runs["demo"][i]["strict"]]
            L += ["", f"결과가 바뀐 문항 (v1 설정 → 데모 설정, 엄격): {len(changed)}개", ""]
            if changed:
                L += ["| id | 유형 | v1 → 데모 | v1 설정 도구 호출 | 데모 설정 도구 호출 |", "|---|---|---|---|---|"]
                for i in changed:
                    a, b = runs["v1"][i], runs["demo"][i]
                    fmt = lambda r: "; ".join(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})"  # noqa: E731
                                              for c in r["tool_calls"]) or "(없음)"
                    L.append(f"| {i} | {a['category']} | {'O' if a['strict'] else 'X'} → {'O' if b['strict'] else 'X'} | "
                             f"`{fmt(a)[:160]}` | `{fmt(b)[:160]}` |")
            L.append("")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT.parent / "demo_regression.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--config", choices=["v1", "demo"])
    ap.add_argument("--exam", choices=["cafe", "live"])
    ap.add_argument("--mode", choices=["igpu", "cpu"], default="igpu")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "run":
        try:
            requests.get(f"{engines.OLLAMA}/api/version", timeout=3)
        except requests.RequestException:
            sys.exit("Ollama is not running: start it with v2/scripts/start_ollama.ps1")
        t = time.time()
        run(a.config, a.exam, a.mode)
        print(f"done in {time.time() - t:.0f} s")
    else:
        report()


if __name__ == "__main__":
    main()
