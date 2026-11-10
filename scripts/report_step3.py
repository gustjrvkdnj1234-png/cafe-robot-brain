"""Build the Step 3 report from results/step3/.

    python scripts/report_step3.py                 # -> results/step3_summary.md, results/step3/per_case.csv
    python scripts/report_step3.py --candidates    # print failure candidates (to pick blog examples)

Picked failures for the report live in results/step3/failure_picks.json:
    [{"contestant": "gemma4:e4b", "exam": "cafe", "id": "D1-2", "note": "한국어 한 줄 설명"}, ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_step3 import CONTESTANTS, EXAMS, run_dir  # noqa: E402
from score import load_jsonl  # noqa: E402

OUT_MD = ROOT / "results" / "step3_summary.md"
OUT_CSV = ROOT / "results" / "step3" / "per_case.csv"
PICKS = ROOT / "results" / "step3" / "failure_picks.json"

CAT_KO = {
    "basic": "기본 주문", "options": "옵션", "slang": "줄임말", "multi_item": "여러 개 주문",
    "modify_cancel": "변경·취소", "impossible": "불가능한 조합", "ambiguous": "모호함", "off_menu": "메뉴에 없음",
    "call_staff": "직원 호출", "takeout": "포장", "conditional": "조건부 요청", "unavailable": "제공 안 하는 옵션",
}
MODE_KO = {"rules": "규칙(CPU)", "igpu": "iGPU", "cpu": "CPU 전용"}


def label(model: str) -> str:
    return "자판기" if model == "vending" else model


def load_run(model, mode, exam):
    d = run_dir(model, mode, exam)
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    return {
        "meta": s["meta"], "summary": s["summary"],
        "raw": {r["id"]: r for r in load_jsonl(d / "raw.jsonl")},
        "scores": {r["id"]: r for r in load_jsonl(d / "scores.jsonl")},
    }


def verdict(sc: dict) -> str:
    if "source" in sc:  # knowledge
        return "correct" if sc["exact"] else "wrong"
    if sc["exact"]:
        return "correct"
    return "partial" if sc["partial"] > 0 else "wrong"


def pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


def gb(x):
    return "-" if x is None else f"{x:.1f}"


def calls_ko(calls) -> str:
    if not calls:
        return "(도구 호출 없음)"
    out = []
    for c in calls:
        args = c.get("arguments") or {}
        if c.get("name") == "ask_clarification":
            out.append(f"ask_clarification({args.get('reason')}) “{args.get('question', '')}”")
        elif c.get("name") == "call_staff":
            out.append(f"call_staff “{args.get('reason', '')}”")
        else:
            out.append(f"{c.get('name')}({', '.join(f'{k}={v}' for k, v in args.items())})")
    return " + ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    exams = {k: {c["id"]: c for c in load_jsonl(ROOT / v)} for k, v in EXAMS.items()}
    runs = {(m, e): load_run(m, mode, e) for m, mode, es in CONTESTANTS for e in es}
    contestants = [(m, mode, es) for m, mode, es in CONTESTANTS]

    if a.candidates:
        for (m, e), r in runs.items():
            if not r or e == "knowledge":
                continue
            for cid, sc in r["scores"].items():
                if sc["exact"]:
                    continue
                case = exams[e][cid]
                raw = r["raw"][cid]
                print(f"{label(m):11} {e:5} {cid:6} {case['category']:13} partial={sc['partial']:.2f} | "
                      f"{case['conversation'][-1]['content']} | 정답: {calls_ko(case['expected'])} | "
                      f"답: {calls_ko(raw.get('tool_calls'))}{' | 텍스트: ' + raw['content'][:80] if raw.get('content') else ''}")
        return

    # ---- per-case CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contestant", "mode", "exam", "id", "category", "verdict", "partial", "latency_s", "load_s",
                    "prompt_tokens", "generated_tokens", "error"])
        for m, mode, es in contestants:
            for e in es:
                r = runs[(m, e)]
                if not r:
                    continue
                for cid, sc in r["scores"].items():
                    raw = r["raw"][cid]
                    w.writerow([m, mode, e, cid, sc.get("category") or sc.get("source"), verdict(sc),
                                sc.get("partial", sc.get("exact")), round(raw.get("wall_s") or 0, 3),
                                round(raw.get("load_s") or 0, 3), raw.get("prompt_eval_count"), raw.get("eval_count"),
                                raw.get("error", "")])

    L = ["# 3단계: 원본 모델 기준 점수", ""]
    any_meta = next((r["meta"] for r in runs.values() if r and r["meta"].get("ollama_version")), {})
    started = min((r["meta"]["started"] for r in runs.values() if r), default="-")
    finished = max((r["meta"]["finished"] for r in runs.values() if r), default="-")
    L += [
        f"- 실행: {started} ~ {finished}, Ollama {any_meta.get('ollama_version', '-')}",
        "- 공통 조건: thinking 끔, temperature 0, seed 42, num_ctx 8192, 스모크 테스트와 같은 시스템 프롬프트(`run_exam.py --show-prompt`)",
        "- 모드: e4b·12b는 iGPU(서버 `OLLAMA_IGPU_ENABLE=1`), 26b는 iGPU에 안 들어가서 CPU 전용(`num_gpu=0`). 모델을 바꿀 때마다 다른 모델을 모두 내리고, 준비 요청 1번으로 로딩을 끝낸 뒤 측정.",
        "- 자판기는 고정 버전(README의 해시 3개)으로 실행. 지식 시험은 모델만 본다.",
        "- 점수는 모두 '정답(exact)' 비율. 부분 점수와 문항별 결과는 `results/step3/per_case.csv`.",
        "- 주의: 자판기의 카페 시험 점수는 그대로 믿기 어렵다. 자판기와 카페 시험을 같은 사람(코딩 에이전트)이 같은 규칙을 보고 만들었기 때문이다. 사람이 쓴 실전 시험이 더 공정한 비교다.",
        "- 실전 L09(\"카페라떼 반샷만 넣어주세요\")는 '반 샷 빼기(-0.5)'가 정답이다. 메뉴에 기본 샷 수가 없어서 '반 샷 추가(+0.5)'로 읽을 여지도 있으며, 26b를 뺀 모두가 +0.5로 답했다.",
        "",
    ]

    # ---- main table
    L += ["## 종합", "",
          "| 참가자 | 모드 | 카페 시험 (120) | 실전 시험 (20) | 지식 시험 (300) | 무조건 D | 주문당 평균 응답 시간 | 최대 메모리 |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for m, mode, es in contestants:
        cafe, live, know = runs.get((m, "cafe")), runs.get((m, "live")), runs.get((m, "knowledge"))
        lat = [r["wall_s"] for x in (cafe, live) if x for r in x["raw"].values() if r.get("wall_s") is not None]
        mems = [x["meta"].get("peak_memory_this_session") for x in (cafe, live, know) if x and x["meta"].get("peak_memory_this_session")]
        if mems:
            mem = (f"러너 {gb(max(x['runner_private_gb'] for x in mems))} GB"
                   + (f" / GPU 공유 {gb(max(x['gpu_shared_gb'] for x in mems))} GB" if mode == "igpu" else ""))
        else:
            mem = "- (규칙 기반)" if m == "vending" else "-"
        def fmt_t(t):
            return f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s"
        lat_s = "-" if not lat else f"{fmt_t(statistics.mean(lat))} (중앙값 {fmt_t(statistics.median(lat))})"
        L.append(f"| {label(m)} | {MODE_KO[mode]} | {pct(cafe['summary']['exact']) if cafe else '-'} | "
                 f"{pct(live['summary']['exact']) if live else '-'} | "
                 f"{pct(know['summary']['accuracy']) if know else ('-' if 'knowledge' in es else '해당 없음')} | "
                 f"{pct(know['summary']['always_D_baseline']) if know else '-'} | {lat_s} | {mem} |")
    know_any = next((runs[(m, 'knowledge')] for m, _, es in contestants if 'knowledge' in es and runs.get((m, 'knowledge'))), None)
    L += ["",
          "- 주문당 평균 응답 시간: 카페 시험 + 실전 시험 140문항의 요청 한 번(프롬프트 처리 + 생성) 평균.",
          "- 최대 메모리: 실행 중 3초마다 잰 Ollama 러너(llama-server) private 메모리의 최댓값. iGPU 모드는 GPU 공유 메모리 카운터도 함께 적었다(둘은 겹치는 부분이 있어 더하면 안 된다).",
          ]
    if know_any:
        bs = know_any["summary"]["by_source"]
        L.append("- 지식 시험 세부(정답률 / 무조건 D): " + ", ".join(
            f"{k} " + " · ".join(f"{label(m)} {pct(runs[(m, 'knowledge')]['summary']['by_source'][k]['accuracy'])}"
                                 for m, _, es in contestants if 'knowledge' in es and runs.get((m, 'knowledge')))
            + f" (무조건 D {pct(v['always_D_baseline'])})" for k, v in bs.items()))
    L.append("")

    # ---- partial scores and speed
    L += ["### 부분 점수와 속도", "", "| 참가자 | 카페 부분 점수 | 실전 부분 점수 | 도구 이름 일치(카페) | 평균 생성 토큰(카페) | 생성 속도 tok/s |", "|---|---:|---:|---:|---:|---:|"]
    for m, mode, es in contestants:
        cafe, live = runs.get((m, "cafe")), runs.get((m, "live"))
        toks = [r["eval_count"] for r in cafe["raw"].values() if r.get("eval_count")] if cafe else []
        L.append(f"| {label(m)} | {pct(cafe['summary']['partial']) if cafe else '-'} | {pct(live['summary']['partial']) if live else '-'} | "
                 f"{pct(cafe['summary']['tool_names_match']) if cafe else '-'} | {f'{statistics.mean(toks):.0f}' if toks else '-'} | "
                 f"{cafe['meta'].get('mean_gen_tps', '-') if cafe else '-'} |")
    L.append("")

    # ---- by category
    cats = list(CAT_KO)
    L += ["## 카페 시험 유형별 정답률", "", "| 유형 | 문항 | " + " | ".join(label(m) for m, _, _ in contestants) + " |",
          "|---|---:|" + "---:|" * len(contestants)]
    for c in cats:
        n = sum(1 for x in exams["cafe"].values() if x["category"] == c)
        row = [f"{CAT_KO[c]} (`{c}`)", str(n)]
        for m, _, _ in contestants:
            r = runs.get((m, "cafe"))
            row.append(pct(r["summary"]["by_category"].get(c, {}).get("exact")) if r else "-")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ---- paraphrase consistency
    groups = {}
    for x in exams["cafe"].values():
        groups.setdefault(x["paraphrase_group"], []).append(x["id"])
    L += ["## 바꿔 말하기 일관성", "",
          f"같은 뜻을 3~4가지로 바꿔 쓴 묶음 {len(groups)}개 중, 묶음 안의 문장을 **모두** 맞힌 묶음의 비율.", "",
          "| 참가자 | 모두 맞힘 | 일부만 맞힘 | 모두 틀림 |", "|---|---:|---:|---:|"]
    for m, _, _ in contestants:
        r = runs.get((m, "cafe"))
        if not r:
            continue
        allc = some = none = 0
        for ids in groups.values():
            k = sum(r["scores"][i]["exact"] for i in ids)
            allc += k == len(ids)
            none += k == 0
            some += 0 < k < len(ids)
        L.append(f"| {label(m)} | {allc}/{len(groups)} ({allc / len(groups) * 100:.1f}%) | {some} | {none} |")
    L.append("")

    # ---- live exam per case
    sym = {"correct": "✅", "partial": "△", "wrong": "❌"}
    L += ["## 실전 시험 문항별 결과", "", "✅ 정답, △ 부분 점수, ❌ 오답", "",
          "| 문항 | 유형 | 손님 말 | " + " | ".join(label(m) for m, _, _ in contestants) + " |",
          "|---|---|---|" + ":-:|" * len(contestants)]
    for cid, case in exams["live"].items():
        row = [cid, CAT_KO.get(case["category"], case["category"]), case["conversation"][-1]["content"].replace("|", "\\|")]
        for m, _, _ in contestants:
            r = runs.get((m, "live"))
            row.append(sym[verdict(r["scores"][cid])] if r and cid in r["scores"] else "-")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ---- picked failures
    if PICKS.exists():
        picks = json.loads(PICKS.read_text(encoding="utf-8"))
        L += [f"## 눈에 띄는 실패 {len(picks)}선", ""]
        for k, p in enumerate(picks, 1):
            r = runs[(p["contestant"], p["exam"])]
            case = exams[p["exam"]][p["id"]]
            raw = r["raw"][p["id"]]
            conv = case["conversation"]
            L.append(f"### {k}. {label(p['contestant'])} · {'실전' if p['exam'] == 'live' else '카페'} {p['id']} ({CAT_KO.get(case['category'], case['category'])})")
            L.append("")
            if case.get("order_state"):
                L.append("- 현재 주문: " + ", ".join(x["menu"] + (f"({x.get('temperature')})" if x.get("temperature") else "") for x in case["order_state"]))
            for t in conv[:-1]:
                L.append(f"- 이전 대화 ({'손님' if t['role'] == 'user' else '로봇'}): {t['content']}")
            L.append(f"- 손님: “{conv[-1]['content']}”")
            L.append(f"- 정답: {calls_ko(case['expected'])}")
            got = calls_ko(raw.get("tool_calls"))
            if raw.get("content") and not raw.get("tool_calls"):
                got += f" — 대신 텍스트로 답함: “{raw['content'][:150]}”"
            L.append(f"- 실제: {got}")
            L.append(f"- 해설: {p['note']}")
            L.append("")

    # ---- anomalies
    L += ["## 이상 징후", ""]
    rows = []
    for m, mode, es in contestants:
        for e in es:
            r = runs.get((m, e))
            if not r:
                rows.append(f"- {label(m)} · {e}: **결과 없음**")
                continue
            mt = r["meta"]
            raws = list(r["raw"].values())
            length = sum(1 for x in raws if x.get("done_reason") == "length")
            text_only = sum(1 for x in raws if not x.get("tool_calls") and (x.get("content") or "").strip()) if e != "knowledge" else 0
            unparsed = r["summary"].get("unparsed") if e == "knowledge" else None
            mem = mt.get("peak_memory_this_session") or {}
            pf = (mem.get("pagefile_used_gb") or 0) - (mem.get("pagefile_start_gb") or 0)
            notes = []
            if mt.get("errors"):
                notes.append(f"오류 {mt['errors']}건")
            if mt.get("empty_outputs"):
                notes.append(f"빈 출력(도구 호출도 텍스트도 없음) {mt['empty_outputs']}건")
            if text_only:
                notes.append(f"도구 대신 텍스트만 {text_only}건")
            if length:
                notes.append(f"길이 제한으로 잘림 {length}건")
            if unparsed:
                notes.append(f"알파벳을 못 뽑은 답 {unparsed}건")
            if mem and pf > 0.5:
                notes.append(f"페이지 파일 {pf:.1f} GB 증가(스왑 의심)")
            if mem and mem.get("sys_avail_min_gb") is not None and mem["sys_avail_min_gb"] < 2:
                notes.append(f"가용 RAM 최저 {mem['sys_avail_min_gb']:.1f} GB")
            if m == "vending" and not mt.get("vending_machine_frozen"):
                notes.append("**자판기 해시 불일치**")
            vram = [x for x in mt.get("api_ps_after") or [] if x.get("name") == m]
            if vram and mode == "cpu" and (vram[0].get("size_vram") or 0) > 0:
                notes.append("CPU 모드인데 VRAM 사용 표시")
            if vram and mode == "igpu" and (vram[0].get("size_vram") or 0) == 0:
                notes.append("iGPU 모드인데 VRAM 사용 0")
            rows.append(f"- {label(m)} · {e}: " + (", ".join(notes) if notes else "없음"))
    L += rows + [""]

    # ---- measured context for the anomalies above (numbers checked against the server log)
    def prompt_stats(m, e):
        r = runs.get((m, e))
        if not r:
            return None
        raws = [x for x in r["raw"].values() if x.get("prompt_eval_count")]
        if not raws:
            return None
        return (statistics.mean(x["prompt_eval_count"] for x in raws),
                statistics.mean(x["prompt_eval_s"] for x in raws),
                statistics.mean(x["wall_s"] for x in raws),
                sum(1 for x in raws if x["wall_s"] > 30))
    L += ["### 이상 징후 해석", ""]
    ps = {m: prompt_stats(m, "cafe") for m, _, es in contestants if m != "vending"}
    if all(ps.values()):
        tok = statistics.mean(v[0] for v in ps.values())
        L.append(f"- **응답 시간 대부분은 프롬프트 처리다.** 시스템 프롬프트(메뉴·규칙 전체)가 약 {tok:,.0f} 토큰이고, 서버 로그에 "
                 "\"forcing full prompt re-processing ... (likely due to SWA ...)\"가 반복된다. 즉 같은 시스템 프롬프트라도 요청마다 처음부터 다시 처리하는 경우가 많다. "
                 "카페 시험 기준 프롬프트 처리 평균: " + ", ".join(f"{label(m)} {v[1]:.1f} s / 전체 {v[2]:.1f} s" for m, v in ps.items()) + ".")
        if ps.get("gemma4:26b"):
            L.append(f"- 26b(CPU)는 응답 시간이 둘로 갈린다. 프롬프트 캐시가 맞으면 몇 초, 안 맞으면 프롬프트 전체를 CPU로 다시 처리해 70~80초가 걸린다(카페 시험에서 30초 넘은 문항 {ps['gemma4:26b'][3]}개). 그래서 평균보다 중앙값이 실제 체감에 가깝다.")
        L.append("- 12b(iGPU)가 26b(CPU)보다 느린 이유: 12b는 매 요청 프롬프트 처리에 8초 안팎이 든다. 26b는 1단계에서도 CPU 생성 속도가 12b보다 빨랐다(14.4 vs 7.8 tok/s).")
    L += [
        "- **메모리 최댓값에는 llama-server의 프롬프트 캐시가 들어 있다.** Ollama 0.40.15가 띄우는 llama-server는 처리한 프롬프트 상태를 RAM에 최대 8 GB(8192 MiB)까지 저장한다"
        "(로그: \"prompt cache is enabled, size limit: 8192 MiB\", 실행 중 \"cache state: 90 prompts, 8085 MiB\"). 지식 시험처럼 서로 다른 프롬프트가 많으면 이 캐시가 차서 러너 메모리가 커진다. "
        "모델 자체가 차지하는 메모리는 1단계 측정값(e4b 8.5 GB, 12b 11.3 GB, 26b 19.3 GB)과 카페 시험 초반 값에 가깝다.",
        "- 26b 지식 시험 중 가용 RAM이 0.1 GB까지 떨어지고 페이지 파일이 7 GB 넘게 늘었다(스왑). 스왑은 속도만 떨어뜨리고 답 내용은 바꾸지 않는다. 주문당 응답 시간은 카페·실전 시험만으로 계산했다.",
        "- 지식 시험의 '길이 제한으로 잘림'은 알파벳 한 글자만 답하라는 지시를 어기고 설명을 늘어놓다 16토큰에서 잘린 경우다.",
        "- 채점기 약점: 12b가 \"A, B, C, D\"(kmmlu-082), \"제시된 문제의 보기(A, B, C, D)가\"(kmmlu-053)처럼 답한 2건은 첫 글자 A를 답으로 뽑았다. 두 문항 모두 정답이 C라서 점수가 부풀지는 않았다.",
        "",
    ]

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"saved: {OUT_MD.relative_to(ROOT).as_posix()}, {OUT_CSV.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
