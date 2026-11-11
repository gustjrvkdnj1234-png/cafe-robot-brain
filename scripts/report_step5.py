"""Build the Step 5a report (keep = 98 experts) next to the Step 3 table.

    .venv\\Scripts\\python.exe scripts\\report_step5.py   # -> results/step5/keep98_summary.md, results/step5/per_case_98e.csv
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from report_step3 import CAT_KO, MODE_KO, calls_ko, gb, pct, verdict  # noqa: E402
from run_step3 import CONTESTANTS, EXAMS, run_dir  # noqa: E402
from score import load_jsonl  # noqa: E402

NEW = "song/gemma4-26b-cafe-98e"
NEW_MODE = "igpu"
S5 = ROOT / "results" / "step5"
OUT_MD = S5 / "keep98_summary.md"   # includes results/step5/keep98_interpretation.md at the end if it exists
OUT_CSV = S5 / "per_case_98e.csv"
LABEL = {"vending": "자판기", NEW: "26b 가지치기 98개"}


def label(m):
    return LABEL.get(m, m)


def new_dir(exam):
    return S5 / f"{exam}__{NEW.replace(':', '-').replace('/', '-')}__{NEW_MODE}"


def load(d: Path):
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    return {"meta": s["meta"], "summary": s["summary"],
            "raw": {r["id"]: r for r in load_jsonl(d / "raw.jsonl")},
            "scores": {r["id"]: r for r in load_jsonl(d / "scores.jsonl")}}


def fmt_t(t):
    return f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exams = {k: {c["id"]: c for c in load_jsonl(ROOT / v)} for k, v in EXAMS.items()}
    contestants = list(CONTESTANTS) + [(NEW, NEW_MODE, ["cafe", "live", "knowledge"])]
    runs = {}
    for m, mode, es in contestants:
        for e in es:
            runs[(m, e)] = load(new_dir(e)) if m == NEW else load(run_dir(m, mode, e))
    missing = [f"{m} {e}" for (m, e), r in runs.items() if r is None]

    plan = json.loads((S5 / "prune_plan_98.json").read_text(encoding="utf-8"))
    val = json.loads((S5 / "validate_98e.json").read_text(encoding="utf-8"))
    base = json.loads((S5 / "validate_baseline_bf16.json").read_text(encoding="utf-8"))
    load_igpu = json.loads((S5 / "load_igpu.json").read_text(encoding="utf-8"))

    # ---- CSV for the new model
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exam", "id", "category", "verdict", "partial", "verdict_26b_original", "latency_s", "prompt_tokens",
                    "generated_tokens", "error"])
        for e in ("cafe", "live", "knowledge"):
            r, o = runs.get((NEW, e)), runs.get(("gemma4:26b", e))
            if not r:
                continue
            for cid, sc in r["scores"].items():
                raw = r["raw"][cid]
                w.writerow([e, cid, sc.get("category") or sc.get("source"), verdict(sc), sc.get("partial", sc.get("exact")),
                            verdict(o["scores"][cid]) if o and cid in o["scores"] else "", round(raw.get("wall_s") or 0, 3),
                            raw.get("prompt_eval_count"), raw.get("eval_count"), raw.get("error", "")])

    nm = runs[(NEW, "cafe")]["meta"] if runs.get((NEW, "cafe")) else {}
    L = ["# 5a단계: 전문가 98개만 남긴 26b (카페 전용 가지치기)", ""]
    if missing:
        L += [f"> **미완성:** 결과 없음 — {', '.join(missing)}", ""]
    L += [
        f"- 모델: `{NEW}` (Ollama). 원본 `gemma-4-26B-A4B-it`의 30개 층에서 층마다 전문가 128개 중 98개만 남겼다(30개 제거, 토큰당 사용 전문가 8개는 그대로).",
        "- 남길 전문가 고르기: 4b단계 통계(카페 보정 문항 600개)로 순위를 매겼다. "
        "규칙 A — 층의 토큰 30% 이상이 거쳐 가는 '바쁜 전문가'는 항상 남김. "
        "규칙 B — 고정 프롬프트(시스템 지시문 + 도구 설명)와 문항별 토큰의 가중치를 50:50으로 맞춤. "
        "순위는 REAP 점수(쓰일 때의 평균)가 아니라 **기여 합**(모든 토큰에 대한 '게이트 × 출력 노름'의 합)으로 매겼다. 자세한 비교는 `results/step5/prune_plans.md`.",
        f"- 시험 조건: 3단계와 같다(thinking 끔, temperature 0, seed 42, num_ctx 8192, 같은 시스템 프롬프트). 모드는 **iGPU**(서버 `OLLAMA_IGPU_ENABLE=1`, 이 프로세스에만 설정). "
        f"실행: {nm.get('started', '-')} ~ {runs[(NEW, 'knowledge')]['meta'].get('finished', '-') if runs.get((NEW, 'knowledge')) else '-'}.",
        "- 원본 26b는 iGPU에 들어가지 않아 3단계에서 CPU로 돌렸다. 그래서 **점수는 바로 비교할 수 있지만, 속도와 메모리는 하드웨어가 달라 바로 비교하면 안 된다.**",
        "",
    ]

    # ---- main table
    L += ["## 종합 (3단계 표 + 새 모델)", "",
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
        lat_s = "-" if not lat else f"{fmt_t(statistics.mean(lat))} (중앙값 {fmt_t(statistics.median(lat))})"
        name = f"**{label(m)}**" if m == NEW else label(m)
        L.append(f"| {name} | {MODE_KO[mode]} | {pct(cafe['summary']['exact']) if cafe else '-'} | "
                 f"{pct(live['summary']['exact']) if live else '-'} | "
                 f"{pct(know['summary']['accuracy']) if know else ('-' if 'knowledge' in es else '해당 없음')} | "
                 f"{pct(know['summary']['always_D_baseline']) if know else '-'} | {lat_s} | {mem} |")
    L += ["", "- 주문당 평균 응답 시간: 카페 + 실전 140문항, 요청 한 번(프롬프트 처리 + 생성)의 평균.",
          "- 최대 메모리: 3초마다 잰 러너(llama-server) private 메모리의 최댓값. iGPU 모드는 GPU 공유 메모리도 함께 적었다(겹치므로 더하면 안 된다). "
          "러너 메모리에는 llama-server의 프롬프트 캐시(최대 8 GB)가 들어 있다(3단계 보고서 참고).", ""]
    srcs = ["kmmlu", "mmlu"]
    L.append("- 지식 시험 세부: " + "; ".join(
        f"{s} " + " · ".join(f"{label(m)} {pct(runs[(m, 'knowledge')]['summary']['by_source'][s]['accuracy'])}"
                             for m, _, es in contestants if 'knowledge' in es and runs.get((m, 'knowledge')))
        for s in srcs if runs.get((NEW, "knowledge"))))
    L.append("")

    L += ["### 부분 점수와 속도", "", "| 참가자 | 카페 부분 점수 | 실전 부분 점수 | 도구 이름 일치(카페) | 평균 생성 토큰(카페) | 생성 속도 tok/s | 프롬프트 처리 평균(카페) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for m, mode, es in contestants:
        cafe, live = runs.get((m, "cafe")), runs.get((m, "live"))
        toks = [r["eval_count"] for r in cafe["raw"].values() if r.get("eval_count")] if cafe else []
        pe = [r["prompt_eval_s"] for r in cafe["raw"].values() if r.get("prompt_eval_s") is not None] if cafe else []
        L.append(f"| {label(m)} | {pct(cafe['summary']['partial']) if cafe else '-'} | {pct(live['summary']['partial']) if live else '-'} | "
                 f"{pct(cafe['summary']['tool_names_match']) if cafe else '-'} | {f'{statistics.mean(toks):.0f}' if toks else '-'} | "
                 f"{cafe['meta'].get('mean_gen_tps', '-') if cafe else '-'} | {f'{statistics.mean(pe):.1f} s' if pe else '-'} |")
    L.append("")

    # ---- by category
    L += ["## 카페 시험 유형별 정답률", "", "| 유형 | 문항 | " + " | ".join(label(m) for m, _, _ in contestants) + " |",
          "|---|---:|" + "---:|" * len(contestants)]
    for c in CAT_KO:
        n = sum(1 for x in exams["cafe"].values() if x["category"] == c)
        row = [f"{CAT_KO[c]} (`{c}`)", str(n)]
        for m, _, _ in contestants:
            r = runs.get((m, "cafe"))
            row.append(pct(r["summary"]["by_category"].get(c, {}).get("exact")) if r else "-")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ---- paraphrase
    groups = {}
    for x in exams["cafe"].values():
        groups.setdefault(x["paraphrase_group"], []).append(x["id"])
    L += ["## 바꿔 말하기 일관성", "", f"같은 뜻을 바꿔 쓴 묶음 {len(groups)}개 중 묶음 안의 문장을 모두 맞힌 비율.", "",
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

    # ---- live per case
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

    # ---- changes vs original 26b
    L += ["## 원본 26b와 결과가 달라진 문항", ""]
    for e, name in (("cafe", "카페 시험"), ("live", "실전 시험")):
        r, o = runs.get((NEW, e)), runs.get(("gemma4:26b", e))
        if not (r and o):
            continue
        rows = []
        for cid, sc in r["scores"].items():
            v_new, v_old = verdict(sc), verdict(o["scores"][cid])
            if v_new != v_old:
                case = exams[e][cid]
                rows.append(f"| {cid} | {CAT_KO.get(case['category'], case['category'])} | {case['conversation'][-1]['content'].replace('|', '\\|')} | "
                            f"{sym[v_old]} → {sym[v_new]} | {calls_ko(case['expected'])} | {calls_ko(o['raw'][cid].get('tool_calls'))} | "
                            f"{calls_ko(r['raw'][cid].get('tool_calls'))}{' — 텍스트: ' + r['raw'][cid]['content'][:80].replace('|', '/') if r['raw'][cid].get('content') and not r['raw'][cid].get('tool_calls') else ''} |")
        better = sum(1 for x in rows if "→ ✅" in x)
        L += [f"**{name}:** {len(rows)}문항 달라짐 (나아짐 {better}, 나빠지거나 부분 점수로 바뀜 {len(rows) - better})", ""]
        if rows:
            L += ["| 문항 | 유형 | 손님 말 | 원본 → 가지치기 | 정답 | 원본 답 | 가지치기 답 |", "|---|---|---|:-:|---|---|---|"] + rows
        L.append("")
    r, o = runs.get((NEW, "knowledge")), runs.get(("gemma4:26b", "knowledge"))
    if r and o:
        up = sum(1 for k, sc in r["scores"].items() if sc["exact"] and not o["scores"][k]["exact"])
        down = sum(1 for k, sc in r["scores"].items() if not sc["exact"] and o["scores"][k]["exact"])
        L += [f"**지식 시험:** 원본이 틀리고 가지치기가 맞힌 문항 {up}개, 원본이 맞히고 가지치기가 틀린 문항 {down}개.", ""]

    # ---- build facts
    gg = json.loads((ROOT / "models/gemma-4-26B-A4B-it-cafe-98e/config.json").read_text(encoding="utf-8"))["text_config"]
    pm = load_igpu["peak_memory"]
    q4 = ROOT / "models/gguf/gemma4-26b-cafe-98e-Q4_K_M.gguf"
    q4_gb = q4.stat().st_size / 1e9 if q4.exists() else None
    L += ["## 만든 과정과 크기", "",
          "| 항목 | 원본 26b | 가지치기 98개 |", "|---|---:|---:|",
          f"| 층당 전문가 / 토큰당 사용 | 128 / 8 | {gg['num_experts']} / {gg['top_k_experts']} |",
          "| 파라미터 (`ollama show`) | 25.2B | 19.9B |",
          "| BF16 체크포인트 | 51.6 GB | 40.9 GB |",
          f"| Q4_K_M GGUF | 17.07 GB | {f'{q4_gb:.2f} GB' if q4_gb else '-'} |",
          "| Ollama 등록 크기 (`ollama list`, 비전 프로젝터·초안 모델 포함) | 18 GB | 15 GB |",
          f"| iGPU에 다 들어가나 | 아니요 (1단계: 로딩 시 HTTP 500) | **예** (`ollama ps` 100% GPU) |",
          "",
          f"- iGPU 로딩(num_ctx 8192): {load_igpu['load_s']} s. 로딩 중 최대: 러너 private {pm['runner_private_gb']} GB, GPU 공유 {pm['gpu_shared_gb']} GB, "
          f"시스템 사용 {pm['sys_used_gb']} GB / 32 GB(가용 최저 {pm['sys_avail_min_gb']} GB), 페이지 파일 증가 없음. "
          "`ollama ps`의 SIZE(1.0 GB)는 iGPU 공유 메모리 모드에서 실제보다 작게 나온다(1단계 참고).",
          "- 순서: 원본 safetensors에서 전문가 텐서(`experts.gate_up_proj`, `experts.down_proj`)와 라우터(`router.proj.weight`, `router.per_expert_scale`)의 전문가 축만 잘라 새 폴더에 썼다(원본은 읽기만 함, 64 MB 단위로 스트리밍). "
          "→ llama.cpp b12283 변환기로 BF16 GGUF → `llama-quantize`로 Q4_K_M → Ollama 등록.",
          "- 양자화는 원본 `gemma4:26b`와 텐서별 형식을 똑같이 맞췄다(658개 텐서 모두 같음: 어텐션·밀집 MLP·임베딩 Q8_0, 전문가 gate_up Q4_K(9·15층 Q5_K), down Q5_1(29층 Q8_0), 라우터 F32). "
          "**다른 점 하나:** 원본은 Ollama가 C4 영어 텍스트로 만든 중요도 행렬(imatrix)을 써서 양자화했는데, 이 파일은 공개되지 않아 가지치기 모델은 imatrix 없이 양자화했다. "
          "그래서 점수 차이에는 가지치기 효과와 imatrix 유무가 섞여 있다.",
          "- 템플릿·파라미터·비전 프로젝터·초안(DRAFT) 모델은 `ollama show --modelfile gemma4:26b` 그대로 썼다(FROM 한 줄만 바꿈).",
          "- 중간 파일(가지치기 BF16 safetensors, BF16 GGUF)은 다음 파일을 검증한 뒤 지웠다. 정전 뒤 두 파일의 내용을 원본과 바이트 단위로 대조해 손상이 없음을 확인했다(`scripts/verify_pruned_files.py`).",
          ""]

    # ---- validation
    def row(i, d):
        x = d["layers"][str(i)]
        return x["isolated_all"]["mean"], x["chain_all"]["mean"], x["chain_case"]["mean"]
    L += ["## 은닉 상태 검증 (전체 모델을 올리지 않고 층 단위로)", "",
          f"보정 문항 8개(토큰 6,156개)를 원본과 가지치기 모델에 층마다 통과시켜 코사인 유사도를 쟀다(float32, CPU). 비교 기준으로 **원본을 bf16으로 계산했을 때**(가지치기 없음, 반올림 오차만)의 값도 같이 쟀다.", "",
          "- 이 층만: 원본의 입력을 가지치기 층에 넣었을 때 (그 층 하나의 오차)",
          "- 누적: 가지치기 모델을 처음부터 따로 돌렸을 때 (앞 층의 오차가 쌓임)", "",
          "| 층 | 가지치기: 이 층만 | 가지치기: 누적 (전체 토큰) | 가지치기: 누적 (문항별 토큰) | bf16 기준: 누적 (전체) | bf16 기준: 누적 (문항별) |",
          "|---:|---:|---:|---:|---:|---:|"]
    for i in (0, 4, 5, 9, 11, 14, 17, 20, 23, 26, 29):
        a, b, c = row(i, val)
        _, bb, bc = row(i, base)
        L.append(f"| {i} | {a:.5f} | {b:.4f} | {c:.4f} | {bb:.4f} | {bc:.4f} |")
    vf, bf = val["final"], base["final"]
    L += ["",
          f"| 마지막 출력 (문항별 토큰 {vf['n_tokens']}개) | 가지치기 98개 | bf16 기준 |", "|---|---:|---:|",
          f"| 다음 토큰 1순위 일치 | {vf['top1_agreement_case_tokens'] * 100:.1f}% | {bf['top1_agreement_case_tokens'] * 100:.1f}% |",
          f"| 평균 KL(원본 ‖ 비교 대상) | {vf['mean_kl_orig_to_pruned']:.3f} | {bf['mean_kl_orig_to_pruned']:.3f} |",
          "",
          "- 층 하나만 보면 가지치기 오차는 거의 없다(대부분 0.9999 이상). 누적 값은 층을 지나며 떨어진다. **bf16 반올림만 해도 같은 모양으로 떨어지고**, 가지치기는 그보다 조금 더 떨어진다(26층 전체 토큰 0.894 vs 0.936, 문항별 토큰 0.974 vs 0.986). "
          "이 모델은 작은 차이가 라우팅(어느 전문가 8개를 고를지)을 바꾸고, 그 차이가 뒤 층으로 커지는 성질이 있다. 일부 토큰의 코사인이 음수까지 내려가는 것도 bf16 기준에서 똑같이 보인다.",
          f"- 다음 토큰 1순위 일치는 bf16 기준과 같은 수준이고, KL은 bf16 기준의 약 {vf['mean_kl_orig_to_pruned'] / bf['mean_kl_orig_to_pruned']:.1f}배다.",
          ""]

    interp = S5 / "keep98_interpretation.md"      # hand-written interpretation, kept out of the generated part
    L += [interp.read_text(encoding="utf-8").strip() if interp.exists() else "<!-- INTERPRETATION -->", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"saved {OUT_MD.relative_to(ROOT).as_posix()}; missing: {missing}")


if __name__ == "__main__":
    main()
