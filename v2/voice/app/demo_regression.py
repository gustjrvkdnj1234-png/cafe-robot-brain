"""Regression check: does the demo config (demo prompt + inform tool) keep ordering accuracy?

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\demo_regression.py run --config v1|demo --exam cafe|live
    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\demo_regression.py report
    ... run --config demo --exam cafe --mode cpu --tag r2     # 2026-11-25 demo config (A1-D1) -> demo_r2__cafe__cpu.jsonl

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


READ_ONLY = ("inform", "read_order")   # demo-only tools that never change the order


def score(case, calls) -> dict:
    no_inform = [c for c in calls if c.get("name") not in READ_ONLY]
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


def run(config: str, exam: str, mode: str = "igpu", tag: str = ""):
    engines.RUN.update(config=config, mode=mode, num_ctx=8192, num_predict=512)
    llm = engines.LLM()
    cases = load_jsonl(EXAMS[exam])
    OUT.mkdir(parents=True, exist_ok=True)
    name = config + (f"_{tag}" if tag else "")
    path = OUT / (f"{name}__{exam}.jsonl" if mode == "igpu" else f"{name}__{exam}__{mode}.jsonl")
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


TEMP_WORDS = re.compile(r"따뜻|따듯|뜨거|뜨겁|뜨끈|핫|아이스|차가|차갑|시원|얼음|아아|뜨아|아바라|아이스아메|hot|ice", re.I)


def temp_unstated(case) -> bool:
    """A3 applies: an expected add_item is a hot-or-iced item and no customer turn says a temperature."""
    from cafe_common import canonical_menu, items_by_name
    both = [c for c in case.get("expected") or [] if c.get("name") == "add_item" and
            len((items_by_name().get(canonical_menu(str((c.get("arguments") or {}).get("menu", "")))) or {})
                .get("temperatures", [])) > 1]
    said = " ".join(m["content"] for m in case["conversation"] if m["role"] == "user")
    return bool(both) and not TEMP_WORDS.search(said)


def app_would_ask(case, calls) -> bool:
    """The app safety net would hold one of these add_item calls and ask the temperature."""
    import replies
    said = " ".join(m["content"] for m in case["conversation"] if m["role"] == "user")
    return any(c["name"] == "add_item" and replies.needs_temperature(c.get("arguments") or {}, bool(TEMP_WORDS.search(said)))
               for c in calls)


def asks_temperature(exam, calls) -> bool:
    """ask_clarification about the temperature (live: question text is not stored, so any ask_clarification)."""
    asks = [c for c in calls if c["name"] == "ask_clarification"]
    if exam == "live":
        return bool(asks)
    qs = [str((c.get("arguments") or {}).get("question", "")) for c in asks]
    return any(TEMP_WORDS.search(q) or "온도" in q for q in qs)


def report_r2(tag="r2", title_cfg=""):
    """2026-11-25 demo config (A1 phase line, A3 temperature question, B1 read_order) vs the 11-24 demo config, CPU."""
    cases = {e: {c["id"]: c for c in load_jsonl(EXAMS[e])} for e in EXAMS}
    L = [f"# 데모 설정 회귀 확인 2 (2026-11-25, 64e, CPU 전용): {title_cfg}", "",
         f"- 비교: 11-24 데모 설정(`demo__*__cpu.jsonl`) → 11-25 데모 설정(`demo_{tag}__*__cpu.jsonl`). 같은 모델·옵션·배포용 배치.",
         "- 11-25 변경: A3 온도 묻기 규칙(따뜻한/아이스 둘 다 되는 메뉴에 온도를 말하지 않으면 묻기), read_order 도구, "
         "온도 기본값 문구. 시험 문항에는 주문 단계(확인 대기 / 확정) 줄이 붙지 않는다.",
         "- 채점: v1 엄격 채점, inform·read_order 호출은 빼고 센다(순수 메뉴 질문 예외는 그대로).",
         "- **의도된 변화(A3)**: 기대 답에 따뜻한/아이스 둘 다 되는 메뉴의 add_item이 있고, 손님이 어느 턴에서도 온도를 말하지 "
         "않았으며(자동 판정), 틀린 이유가 온도 질문인 문항. prompt: 모델이 온도를 되물음(카페는 질문에 온도 말이 있을 때만, "
         "실전은 질문 문장을 저장하지 않아 ask_clarification이면). app: 모델 호출은 채점 그대로이고, 앱 안전장치가 붙잡아 "
         "온도를 물었을 add_item이 있는 문항을 따로 센다(이 경우 모델 답은 v1 기본값대로라 대부분 정답).", ""]
    summary = {}
    for exam in ("cafe", "live"):
        old_p, new_p = OUT / f"demo__{exam}__cpu.jsonl", OUT / f"demo_{tag}__{exam}__cpu.jsonl"
        if not (old_p.exists() and new_p.exists()):
            continue
        old = {r["id"]: r for r in load_jsonl(old_p)}
        new = {r["id"]: r for r in load_jsonl(new_p)}
        ids = [i for i in cases[exam] if i in old and i in new]
        by_design, other = [], []
        for i in ids:
            c, o, n = cases[exam][i], old[i], new[i]
            if temp_unstated(c) and ((not n["strict"] and asks_temperature(exam, n["tool_calls"])) or
                                     (tag == "r2app" and app_would_ask(c, n["tool_calls"]))):
                by_design.append(i)
            elif o["strict"] != n["strict"]:
                other.append(i)
        st_old, st_new = sum(old[i]["strict"] for i in ids), sum(new[i]["strict"] for i in ids)
        bd_was_ok = [i for i in by_design if old[i]["strict"]]
        summary[exam] = {"n": len(ids), "old": st_old, "new": st_new, "by_design": len(by_design),
                         "by_design_was_correct": len(bd_was_ok),
                         "new_plus_by_design": st_new + sum(not new[i]["strict"] for i in by_design),
                         "other_changes": len(other),
                         "read_order_calls": sum(any(x["name"] == "read_order" for x in new[i]["tool_calls"]) for i in ids)}
        title = "카페 120" if exam == "cafe" else "실전 20"
        L += [f"## {title}", "",
              "| 실행 | 엄격 정답 |", "|---|---:|",
              f"| 11-24 데모 설정, CPU | {st_old}/{len(ids)} |",
              f"| 11-25 데모 설정, CPU | {st_new}/{len(ids)} |",
              f"| 11-25, 의도된 변화(A3)를 정답으로 셀 때 | {st_new + sum(not new[i]['strict'] for i in by_design)}/{len(ids)} |", "",
              f"- 의도된 변화(A3): {len(by_design)}개 (그중 11-24에 정답이던 것 {len(bd_was_ok)}개)",
              f"- 그 밖에 결과가 바뀐 문항: {len(other)}개",
              f"- read_order를 부른 문항: {summary[exam]['read_order_calls']}개", ""]
        cats = sorted({cases[exam][i]["category"] for i in ids})
        L += ["| 유형 | 문항 | 11-24 엄격 | 11-25 엄격 | 의도된 변화(A3) |", "|---|---:|---:|---:|---:|"]
        for cat in cats:
            ci = [i for i in ids if cases[exam][i]["category"] == cat]
            L.append(f"| {cat} | {len(ci)} | {sum(old[i]['strict'] for i in ci)} | {sum(new[i]['strict'] for i in ci)} | "
                     f"{sum(i in by_design for i in ci)} |")
        L.append("")
        fmt = lambda r: "; ".join(  # noqa: E731
            (f"{x['name']}({json.dumps(x['arguments'], ensure_ascii=False)})" if exam == "cafe" else x["name"])
            for x in r["tool_calls"]) or "(없음)"
        if by_design:
            L += [f"의도된 변화(A3) 목록 ({title})", "", "| id | 유형 | 11-24 | 11-25 도구 호출 |", "|---|---|---|---|"]
            for i in by_design:
                L.append(f"| {i} | {cases[exam][i]['category']} | {'O' if old[i]['strict'] else 'X'} | `{fmt(new[i])[:150]}` |")
            L.append("")
        if other:
            L += [f"그 밖에 바뀐 문항 ({title})", "", "| id | 유형 | 11-24 → 11-25 | 11-24 도구 호출 | 11-25 도구 호출 |",
                  "|---|---|---|---|---|"]
            for i in other:
                o, n = old[i], new[i]
                L.append(f"| {i} | {cases[exam][i]['category']} | {'O' if o['strict'] else 'X'} → {'O' if n['strict'] else 'X'} | "
                         f"`{fmt(o)[:150]}` | `{fmt(n)[:150]}` |")
            L.append("")
    if not summary:
        return
    (OUT / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT.parent / f"demo_regression_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--config", choices=["v1", "demo"])
    ap.add_argument("--exam", choices=["cafe", "live"])
    ap.add_argument("--mode", choices=["igpu", "cpu"], default="igpu")
    ap.add_argument("--tag", default="", help="suffix for the output file (a later version of the demo config)")
    ap.add_argument("--a3", choices=["prompt", "app"], default="app",
                    help="temperature question from the demo prompt rule (prompt) or from the app only (app)")
    ap.add_argument("--no-read-order", action="store_true", help="demo config without the read_order tool and rule")
    ap.add_argument("--no-read-order-rule", action="store_true", help="keep the read_order tool, drop its prompt rule")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "run":
        try:
            requests.get(f"{engines.OLLAMA}/api/version", timeout=3)
        except requests.RequestException:
            sys.exit("Ollama is not running: start it with v2/scripts/start_ollama.ps1")
        t = time.time()
        engines.demo_config.A3["mode"] = a.a3
        engines.demo_config.READ_ORDER["on"] = not a.no_read_order
        engines.demo_config.READ_ORDER["rule"] = not a.no_read_order_rule
        run(a.config, a.exam, a.mode, a.tag)
        print(f"done in {time.time() - t:.0f} s")
    else:
        report()
        report_r2("r2", "프롬프트 규칙 + 앱 안전장치 (A3 = prompt)")
        report_r2("r2app", "앱 안전장치만 (A3 = app)")


if __name__ == "__main__":
    main()
