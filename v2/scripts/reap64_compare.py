"""Compare REAP-64 with v1 64e (contribution sum + protection) and the 26B original (light, no model).

    .venv\\Scripts\\python.exe v2\\scripts\\reap64_compare.py

Inputs (read only): v2/results/reap64/{cafe,live,knowledge}__song-gemma4-26b-cafe-64e-reap__<mode>/,
results/step5/*__song-gemma4-26b-cafe-64e__igpu/, results/step3/*__gemma4-26b__cpu/.
Strict score = the stored v1 score (re-scored here and checked); lenient = score.score_cafe_case_lenient (v1).
McNemar: exact two-sided binomial test on discordant items (v1 report_comparison.mcnemar_exact).
Outputs: v2/results/reap64/{comparison.md, comparison.json}. No question text is written (ids, categories and
tool calls only; knowledge items by id only).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from report_comparison import mcnemar_exact  # noqa: E402  (v1, read only)
from score import load_jsonl, score_cafe_case, score_cafe_case_lenient  # noqa: E402

OUT = ROOT / "v2/results/reap64"
EXAMS = {"cafe": ROOT / "data/exam_cafe.jsonl", "live": ROOT / "data/exam_live.jsonl"}


def find(base, exam, slug):
    hits = sorted(base.glob(f"{exam}__{slug}__*"))
    assert len(hits) == 1, (base, exam, slug, hits)
    return hits[0]


MODELS = {
    "reap64": ("REAP-64", lambda e: find(OUT, e, "song-gemma4-26b-cafe-64e-reap")),
    "v1_64e": ("v1 64e (기여 합 + 보호)", lambda e: ROOT / f"results/step5/{e}__song-gemma4-26b-cafe-64e__igpu"),
    "orig26b": ("26B 원본", lambda e: ROOT / f"results/step3/{e}__gemma4-26b__cpu"),
}


def pct(x):
    return f"{x * 100:.1f}%"


def calls_str(calls):
    if not calls:
        return "(도구 호출 없음)"
    return "; ".join(f"{c.get('name')}({', '.join(f'{k}={v}' for k, v in (c.get('arguments') or {}).items())})"
                     for c in calls)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cases = {e: {c["id"]: c for c in load_jsonl(f)} for e, f in EXAMS.items()}
    R = {}
    for m, (_, d) in MODELS.items():
        for e in ("cafe", "live", "knowledge"):
            p = d(e)
            R[(m, e)] = {"dir": p, "summary": json.loads((p / "summary.json").read_text(encoding="utf-8")),
                         "scores": {r["id"]: r for r in load_jsonl(p / "scores.jsonl")},
                         "raw": {r["id"]: r for r in load_jsonl(p / "raw.jsonl")}}
        for e in ("cafe", "live"):
            r = R[(m, e)]
            r["lenient"] = {}
            for cid, case in cases[e].items():
                pred = r["raw"][cid]["tool_calls"]
                if score_cafe_case(case, pred)["exact"] != r["scores"][cid]["exact"]:
                    raise SystemExit(f"strict rescoring differs from stored score: {m} {e} {cid}")
                r["lenient"][cid] = score_cafe_case_lenient(case, pred)["exact"]

    res = {"scores": {}, "mcnemar": {}, "diff_items": {}, "meta": {}}
    for m in MODELS:
        c, l, k = R[(m, "cafe")], R[(m, "live")], R[(m, "knowledge")]
        res["scores"][m] = {
            "cafe_strict": c["summary"]["summary"]["exact"], "cafe_lenient": sum(c["lenient"].values()) / len(c["lenient"]),
            "live_strict": l["summary"]["summary"]["exact"], "live_lenient": sum(l["lenient"].values()) / len(l["lenient"]),
            "knowledge": k["summary"]["summary"]["accuracy"],
            "kmmlu": k["summary"]["summary"]["by_source"]["kmmlu"]["accuracy"],
            "mmlu": k["summary"]["summary"]["by_source"]["mmlu"]["accuracy"],
            "cafe_by_category": {cat: v["exact"] for cat, v in c["summary"]["summary"]["by_category"].items()},
            "mode": c["summary"]["meta"]["mode"]}
        res["meta"][m] = {e: {"started": R[(m, e)]["summary"]["meta"].get("started"),
                              "finished": R[(m, e)]["summary"]["meta"].get("finished"),
                              "options": R[(m, e)]["summary"]["meta"].get("options"),
                              "think": R[(m, e)]["summary"]["meta"].get("think")} for e in ("cafe", "live", "knowledge")}

    def mcn(a, b, e, key):
        if key == "lenient":
            sa, sb = R[(a, e)]["lenient"], R[(b, e)]["lenient"]
        else:
            sa = {i: v["exact"] for i, v in R[(a, e)]["scores"].items()}
            sb = {i: v["exact"] for i, v in R[(b, e)]["scores"].items()}
        ids = sorted(set(sa) & set(sb))
        a_only = sum(1 for i in ids if sa[i] and not sb[i])
        b_only = sum(1 for i in ids if sb[i] and not sa[i])
        both = sum(1 for i in ids if sa[i] and sb[i])
        return {"n": len(ids), "both": both, "a_only": a_only, "b_only": b_only,
                "neither": len(ids) - both - a_only - b_only, "p": mcnemar_exact(a_only, b_only)}

    for a, b in (("reap64", "v1_64e"), ("reap64", "orig26b"), ("v1_64e", "orig26b")):
        for e in ("cafe", "live", "knowledge"):
            for key in (("strict", "lenient") if e != "knowledge" else ("strict",)):
                res["mcnemar"][f"{a}|{b}|{e}|{key}"] = mcn(a, b, e, key)

    # items where the two 64e models differ (strict correctness or tool calls)
    for e in ("cafe", "live"):
        rows = []
        for cid, case in cases[e].items():
            ra, rb = R[("reap64", e)], R[("v1_64e", e)]
            ca, cb = ra["raw"][cid]["tool_calls"], rb["raw"][cid]["tool_calls"]
            ea, eb = ra["scores"][cid]["exact"], rb["scores"][cid]["exact"]
            if ea == eb and ca == cb:
                continue
            rows.append({"id": cid, "category": case.get("category"), "reap64_exact": ea, "v1_64e_exact": eb,
                         "reap64_lenient": ra["lenient"][cid], "v1_64e_lenient": rb["lenient"][cid],
                         "orig26b_exact": R[("orig26b", e)]["scores"][cid]["exact"],
                         "reap64_calls": ca, "v1_64e_calls": cb,
                         **({"expected": case["expected"]} if e == "cafe" else {})})
        res["diff_items"][e] = rows
    k = [(i, R[("reap64", "knowledge")]["scores"][i], R[("v1_64e", "knowledge")]["scores"][i])
         for i in R[("reap64", "knowledge")]["scores"]]
    res["diff_items"]["knowledge"] = {
        "differ_pred": sum(1 for _, a, b in k if a["pred"] != b["pred"]),
        "reap64_only_correct": [i for i, a, b in k if a["exact"] and not b["exact"]],
        "v1_64e_only_correct": [i for i, a, b in k if b["exact"] and not a["exact"]]}
    (OUT / "comparison.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res)
    print("saved", (OUT / "comparison.md").relative_to(ROOT))


def write_md(res):
    S, M = res["scores"], res["mcnemar"]
    lab = {m: v[0] for m, v in MODELS.items()}
    L = ["# REAP-64 vs v1 64e vs 26B 원본", "",
         "- REAP-64: 층마다 순수 REAP(배정된 토큰의 g·‖f(x)‖ 평균, v1 통계, 보호·가중 없음) 상위 64개 전문가. "
         "v1 64e: 50:50 가중 기여 합 + 바쁜 전문가 보호. 둘 다 같은 원본·같은 변환·같은 Q4_K_M 텐서 형식, imatrix 없음.",
         "- 시험 설정은 v1과 같다(thinking 끔, temperature 0, seed 42, num_ctx 8192, 같은 시스템 프롬프트·도구, "
         "Ollama 서버는 `OLLAMA_IGPU_ENABLE=1`만). 엄격 = 공식 점수, 관대 = 되묻기 이유 코드만 무시.",
         f"- 실행 모드: REAP-64 {S['reap64']['mode']}, v1 64e {S['v1_64e']['mode']}, 26B 원본 {S['orig26b']['mode']}"
         "(26B 원본은 iGPU에 들어가지 않아 v1에서 CPU로 잼).", "",
         "## 점수", "",
         "| 모델 | 카페 엄격 (120) | 카페 관대 | 실전 엄격 (20) | 실전 관대 | 지식 (300) | KMMLU (200) | MMLU (100) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m, s in S.items():
        L.append(f"| {lab[m]} | {pct(s['cafe_strict'])} | {pct(s['cafe_lenient'])} | {pct(s['live_strict'])} | "
                 f"{pct(s['live_lenient'])} | {pct(s['knowledge'])} | {pct(s['kmmlu'])} | {pct(s['mmlu'])} |")
    cats = sorted(S["reap64"]["cafe_by_category"])
    L += ["", "### 카페 유형별 엄격 정답률", "", "| 유형 | " + " | ".join(lab[m] for m in S) + " |",
          "|---|" + "---:|" * len(S)]
    for c in cats:
        L.append(f"| {c} | " + " | ".join(pct(S[m]["cafe_by_category"].get(c, 0)) for m in S) + " |")
    L += ["", "## McNemar 정확 검정 (양쪽, 한쪽만 맞힌 문항 수)", "",
          "| 비교 (A vs B) | 시험 | 채점 | 둘 다 | A만 | B만 | 둘 다 틀림 | p |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, v in M.items():
        a, b, e, sc = key.split("|")
        L.append(f"| {lab[a]} vs {lab[b]} | {e} | {'엄격' if sc == 'strict' else '관대'} | {v['both']} | {v['a_only']} | "
                 f"{v['b_only']} | {v['neither']} | {v['p']:.3f} |")
    for e in ("cafe", "live"):
        allrows = res["diff_items"][e]
        rows = [r for r in allrows if (r["reap64_exact"], r["reap64_lenient"]) != (r["v1_64e_exact"], r["v1_64e_lenient"])]
        same = [r for r in allrows if r not in rows]
        L += ["", f"## 두 64e 모델의 채점 결과가 다른 문항: {'카페' if e == 'cafe' else '실전'} ({len(rows)}개)", "",
              "- ○ = 엄격 정답, △ = 관대로만 정답, ✗ = 오답."
              + (" 실전 문항은 문장과 정답을 적지 않는다(사람이 쓴 시험)." if e == "live" else ""), ""]
        if rows:
            hdr = "| 문항 | 유형 | REAP-64 | v1 64e | 26B 원본 | REAP-64 호출 | v1 64e 호출 |" + (" 정답 |" if e == "cafe" else "")
            L += [hdr, "|---|---|:-:|:-:|:-:|---|---|" + ("---|" if e == "cafe" else "")]
            mk = lambda x, l: "○" if x else ("△" if l else "✗")  # noqa: E731
            for r in rows:
                line = (f"| {r['id']} | {r['category']} | {mk(r['reap64_exact'], r['reap64_lenient'])} | "
                        f"{mk(r['v1_64e_exact'], r['v1_64e_lenient'])} | {'○' if r['orig26b_exact'] else '✗'} | "
                        f"`{calls_str(r['reap64_calls'])}` | `{calls_str(r['v1_64e_calls'])}` |")
                if e == "cafe":
                    line += f" `{calls_str(r['expected'])}` |"
                L.append(line)
        else:
            L.append("- 없음")
        worse = Counter(r["category"] for r in rows if r["v1_64e_exact"] and not r["reap64_exact"])
        better = Counter(r["category"] for r in rows if r["reap64_exact"] and not r["v1_64e_exact"])
        L += ["", f"- REAP-64만 틀린 문항의 유형: {', '.join(f'{c} {n}' for c, n in worse.most_common()) or '없음'}",
              f"- v1 64e만 틀린 문항의 유형: {', '.join(f'{c} {n}' for c, n in better.most_common()) or '없음'}"]
        by = {}
        for r in same:
            by.setdefault(r["category"], []).append(r["id"])
        L += [f"- 채점 결과는 같지만 도구 호출이 다른 문항 {len(same)}개(문구·추가 인자 차이, 호출 전체는 `comparison.json`): "
              + ("; ".join(f"{c} {', '.join(ids)}" for c, ids in sorted(by.items())) or "없음")]
    kd = res["diff_items"]["knowledge"]
    L += ["", "## 지식 시험", "",
          f"- 두 64e 모델의 답이 다른 문항 {kd['differ_pred']}개, REAP-64만 맞힘 {len(kd['reap64_only_correct'])}개, "
          f"v1 64e만 맞힘 {len(kd['v1_64e_only_correct'])}개 (문항 번호는 `comparison.json`)."]
    (OUT / "comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
