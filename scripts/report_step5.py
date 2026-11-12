"""Build the Step 5 report for one pruned model (keep = 98 / 64 / 48 experts) next to the Step 3 table.

    .venv\\Scripts\\python.exe scripts\\report_step5.py --keep 98   # -> results/step5/keep98_summary.md, per_case_98e.csv
    .venv\\Scripts\\python.exe scripts\\report_step5.py --keep 64

The model's mode (igpu / cpu) is read from its load check (load_igpu.json for 98, load_<k>e.json otherwise).
Rows for the no-imatrix control (song/gemma4-26b-128e-noimat, CPU) and the larger pruned models are added when
their results exist. A hand-written interpretation is appended from keep<k>_interpretation.md if present.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from report_step3 import CAT_KO, MODE_KO, calls_ko, gb, pct, verdict  # noqa: E402
from run_step3 import CONTESTANTS, EXAMS, run_dir  # noqa: E402
from score import load_jsonl  # noqa: E402

S5 = ROOT / "results" / "step5"
CONTROL = "song/gemma4-26b-128e-noimat"
KEEPS = (98, 64, 48)
ALL = ["cafe", "live", "knowledge"]
LABEL = {"vending": "자판기", CONTROL: "26b 대조군 (128개, imatrix 없음)",
         **{f"song/gemma4-26b-cafe-{k}e": f"26b 가지치기 {k}개" for k in KEEPS}}
SYM = {"correct": "✅", "partial": "△", "wrong": "❌"}


def label(m):
    return LABEL.get(m, m)


def step5_dir(model, mode, exam):
    return S5 / f"{exam}__{model.replace(':', '-').replace('/', '-')}__{mode}"


def load_file(k):
    return S5 / ("load_igpu.json" if k == 98 else f"load_{k}e.json")


def model_mode(k):
    f = load_file(k)
    if not f.exists():
        return None
    return "igpu" if json.loads(f.read_text(encoding="utf-8")).get("fits_igpu_fully") else "cpu"


def load(d: Path):
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    return {"meta": s["meta"], "summary": s["summary"],
            "raw": {r["id"]: r for r in load_jsonl(d / "raw.jsonl")},
            "scores": {r["id"]: r for r in load_jsonl(d / "scores.jsonl")}}


def fmt_t(t):
    return f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s"


def ollama_facts(model):
    """parameters (ollama show), listed size (ollama list), main GGUF blob size."""
    def run(*args):
        return subprocess.run(["ollama", *args], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    out = {"params": None, "listed": None, "q4_gb": None}
    m = re.search(r"parameters\s+(\S+)", run("show", model))
    out["params"] = m.group(1) if m else None
    for line in run("list").splitlines():
        parts = line.split()
        if parts and parts[0] in (model, model + ":latest"):
            out["listed"] = " ".join(parts[2:4])
    m = re.search(r"^FROM (.+)$", run("show", "--modelfile", model), re.M)
    if m and Path(m.group(1).strip()).exists():
        out["q4_gb"] = Path(m.group(1).strip()).stat().st_size / 1e9
    return out


def mem_text(runs, m, mode):
    mems = [runs[(m, e)]["meta"].get("peak_memory_this_session") for e in ALL
            if runs.get((m, e)) and runs[(m, e)]["meta"].get("peak_memory_this_session")]
    if not mems:
        return "- (규칙 기반)" if m == "vending" else "-"
    return (f"러너 {gb(max(x['runner_private_gb'] for x in mems))} GB"
            + (f" / GPU 공유 {gb(max(x['gpu_shared_gb'] for x in mems))} GB" if mode == "igpu" else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=98, choices=KEEPS)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    K = a.keep
    NEW = f"song/gemma4-26b-cafe-{K}e"
    NEW_MODE = model_mode(K) or "igpu"
    OUT_MD = S5 / f"keep{K}_summary.md"
    OUT_CSV = S5 / f"per_case_{K}e.csv"
    exams = {k: {c["id"]: c for c in load_jsonl(ROOT / v)} for k, v in EXAMS.items()}

    contestants = list(CONTESTANTS)
    if (step5_dir(CONTROL, "cpu", "cafe") / "summary.json").exists():
        contestants.append((CONTROL, "cpu", ALL))
    for k in KEEPS:
        if k > K and model_mode(k):
            contestants.append((f"song/gemma4-26b-cafe-{k}e", model_mode(k), ALL))
    contestants.append((NEW, NEW_MODE, ALL))
    runs = {}
    for m, mode, es in contestants:
        for e in es:
            runs[(m, e)] = load(step5_dir(m, mode, e)) if m.startswith("song/") else load(run_dir(m, mode, e))
    missing = [f"{m} {e}" for (m, e), r in runs.items() if r is None and m in (NEW, CONTROL)]

    val_f = S5 / f"validate_{K}e.json"
    val = json.loads(val_f.read_text(encoding="utf-8")) if val_f.exists() else None
    base = json.loads((S5 / "validate_baseline_bf16.json").read_text(encoding="utf-8"))
    load_chk = json.loads(load_file(K).read_text(encoding="utf-8")) if load_file(K).exists() else None

    # ---- CSV for the new model
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exam", "id", "category", "verdict", "partial", "verdict_26b_original", "verdict_control_noimat",
                    "latency_s", "prompt_tokens", "generated_tokens", "error"])
        for e in ALL:
            r, o, c = runs.get((NEW, e)), runs.get(("gemma4:26b", e)), runs.get((CONTROL, e))
            if not r:
                continue
            for cid, sc in r["scores"].items():
                raw = r["raw"][cid]
                w.writerow([e, cid, sc.get("category") or sc.get("source"), verdict(sc), sc.get("partial", sc.get("exact")),
                            verdict(o["scores"][cid]) if o and cid in o["scores"] else "",
                            verdict(c["scores"][cid]) if c and cid in c["scores"] else "",
                            round(raw.get("wall_s") or 0, 3), raw.get("prompt_eval_count"), raw.get("eval_count"), raw.get("error", "")])

    nm = runs[(NEW, "cafe")]["meta"] if runs.get((NEW, "cafe")) else {}
    kn = runs[(NEW, "knowledge")]["meta"] if runs.get((NEW, "knowledge")) else {}
    mode_txt = ("**iGPU**(서버 `OLLAMA_IGPU_ENABLE=1`, 이 프로세스에만 설정)" if NEW_MODE == "igpu"
                else "**CPU 전용**(`num_gpu=0`, iGPU에 다 들어가지 않음)")
    L = [f"# 5단계: 전문가 {K}개만 남긴 26b (카페 전용 가지치기)", ""]
    if missing:
        L += [f"> **미완성:** 결과 없음 — {', '.join(missing)}", ""]
    L += [
        f"- 모델: `{NEW}` (Ollama). 원본 `gemma-4-26B-A4B-it`의 30개 층에서 층마다 전문가 128개 중 {K}개만 남겼다"
        f"({128 - K}개 제거, 토큰당 사용 전문가 8개는 그대로).",
        "- 남길 전문가 고르기: 4b단계 통계(카페 보정 문항 600개)로 순위를 매겼다. "
        "규칙 A — 층의 토큰 30% 이상이 거쳐 가는 '바쁜 전문가'는 항상 남김. "
        "규칙 B — 고정 프롬프트(시스템 지시문 + 도구 설명)와 문항별 토큰의 가중치를 50:50으로 맞춤. "
        "순위는 REAP 점수(쓰일 때의 평균)가 아니라 **기여 합**(모든 토큰에 대한 '게이트 × 출력 노름'의 합)으로 매겼다. 자세한 비교는 `results/step5/prune_plans.md`.",
        f"- 시험 조건: 3단계와 같다(thinking 끔, temperature 0, seed 42, num_ctx 8192, 같은 시스템 프롬프트). 모드는 {mode_txt}. "
        f"실행: {nm.get('started', '-')} ~ {kn.get('finished', '-')}.",
        "- 원본 26b는 iGPU에 들어가지 않아 3단계에서 CPU로 돌렸다. 모드가 다르면 **점수는 바로 비교할 수 있지만, 속도와 메모리는 하드웨어가 달라 바로 비교하면 안 된다.**",
        "- 대조군 `song/gemma4-26b-128e-noimat`: 원본 128개 모델을 가지치기 모델과 똑같이(텐서별 형식 같음, imatrix 없음) 양자화한 것(CPU). "
        "원본 26b와 대조군의 차이는 imatrix 효과, 대조군과 가지치기 모델의 차이는 가지치기 효과로 볼 수 있다.",
        "",
    ]

    # ---- main table
    L += ["## 종합 (3단계 표 + 5단계 모델)", "",
          "| 참가자 | 모드 | 카페 시험 (120) | 실전 시험 (20) | 지식 시험 (300) | 무조건 D | 주문당 평균 응답 시간 | 최대 메모리 |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for m, mode, es in contestants:
        cafe, live, know = runs.get((m, "cafe")), runs.get((m, "live")), runs.get((m, "knowledge"))
        if not cafe and m != NEW:
            continue
        lat = [r["wall_s"] for x in (cafe, live) if x for r in x["raw"].values() if r.get("wall_s") is not None]
        lat_s = "-" if not lat else f"{fmt_t(statistics.mean(lat))} (중앙값 {fmt_t(statistics.median(lat))})"
        name = f"**{label(m)}**" if m == NEW else label(m)
        L.append(f"| {name} | {MODE_KO[mode]} | {pct(cafe['summary']['exact']) if cafe else '-'} | "
                 f"{pct(live['summary']['exact']) if live else '-'} | "
                 f"{pct(know['summary']['accuracy']) if know else ('-' if 'knowledge' in es else '해당 없음')} | "
                 f"{pct(know['summary']['always_D_baseline']) if know else '-'} | {lat_s} | {mem_text(runs, m, mode)} |")
    L += ["", "- 주문당 평균 응답 시간: 카페 + 실전 140문항, 요청 한 번(프롬프트 처리 + 생성)의 평균.",
          "- 최대 메모리: 3초마다 잰 러너(llama-server) private 메모리의 최댓값. iGPU 모드는 GPU 공유 메모리도 함께 적었다(겹치므로 더하면 안 된다). "
          "러너 메모리에는 llama-server의 프롬프트 캐시(최대 8 GB)가 들어 있다(3단계 보고서 참고).", ""]
    know_ms = [m for m, _, es in contestants if runs.get((m, "knowledge"))]
    if know_ms:
        L.append("- 지식 시험 세부: " + "; ".join(
            f"{s} " + " · ".join(f"{label(m)} {pct(runs[(m, 'knowledge')]['summary']['by_source'][s]['accuracy'])}" for m in know_ms)
            for s in ("kmmlu", "mmlu")))
        L.append("")

    L += ["### 부분 점수와 속도", "",
          "| 참가자 | 카페 부분 점수 | 실전 부분 점수 | 도구 이름 일치(카페) | 평균 생성 토큰(카페) | 생성 속도 tok/s | 프롬프트 처리 평균(카페) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for m, mode, es in contestants:
        cafe, live = runs.get((m, "cafe")), runs.get((m, "live"))
        if not cafe:
            continue
        toks = [r["eval_count"] for r in cafe["raw"].values() if r.get("eval_count")]
        pe = [r["prompt_eval_s"] for r in cafe["raw"].values() if r.get("prompt_eval_s") is not None]
        L.append(f"| {label(m)} | {pct(cafe['summary']['partial'])} | {pct(live['summary']['partial']) if live else '-'} | "
                 f"{pct(cafe['summary']['tool_names_match'])} | {f'{statistics.mean(toks):.0f}' if toks else '-'} | "
                 f"{cafe['meta'].get('mean_gen_tps', '-')} | {f'{statistics.mean(pe):.1f} s' if pe else '-'} |")
    L.append("")

    shown = [(m, mode, es) for m, mode, es in contestants if runs.get((m, "cafe"))]
    # ---- by category
    L += ["## 카페 시험 유형별 정답률", "", "| 유형 | 문항 | " + " | ".join(label(m) for m, _, _ in shown) + " |",
          "|---|---:|" + "---:|" * len(shown)]
    for c in CAT_KO:
        n = sum(1 for x in exams["cafe"].values() if x["category"] == c)
        row = [f"{CAT_KO[c]} (`{c}`)", str(n)]
        for m, _, _ in shown:
            row.append(pct(runs[(m, "cafe")]["summary"]["by_category"].get(c, {}).get("exact")))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ---- paraphrase
    groups = {}
    for x in exams["cafe"].values():
        groups.setdefault(x["paraphrase_group"], []).append(x["id"])
    L += ["## 바꿔 말하기 일관성", "", f"같은 뜻을 바꿔 쓴 묶음 {len(groups)}개 중 묶음 안의 문장을 모두 맞힌 비율.", "",
          "| 참가자 | 모두 맞힘 | 일부만 맞힘 | 모두 틀림 |", "|---|---:|---:|---:|"]
    for m, _, _ in shown:
        r = runs[(m, "cafe")]
        allc = some = none = 0
        for ids in groups.values():
            k = sum(r["scores"][i]["exact"] for i in ids)
            allc += k == len(ids)
            none += k == 0
            some += 0 < k < len(ids)
        L.append(f"| {label(m)} | {allc}/{len(groups)} ({allc / len(groups) * 100:.1f}%) | {some} | {none} |")
    L.append("")

    # ---- live per case
    lshown = [(m, mode, es) for m, mode, es in contestants if runs.get((m, "live"))]
    L += ["## 실전 시험 문항별 결과", "", "✅ 정답, △ 부분 점수, ❌ 오답", "",
          "| 문항 | 유형 | 손님 말 | " + " | ".join(label(m) for m, _, _ in lshown) + " |",
          "|---|---|---|" + ":-:|" * len(lshown)]
    for cid, case in exams["live"].items():
        row = [cid, CAT_KO.get(case["category"], case["category"]), case["conversation"][-1]["content"].replace("|", "\\|")]
        for m, _, _ in lshown:
            r = runs[(m, "live")]
            row.append(SYM[verdict(r["scores"][cid])] if cid in r["scores"] else "-")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ---- changes vs original 26b and vs the no-imatrix control
    for ref, ref_name in (("gemma4:26b", "원본 26b"), (CONTROL, "대조군(128개, imatrix 없음)")):
        if not runs.get((ref, "cafe")):
            continue
        L += [f"## 결과가 달라진 문항: {ref_name} 대비", ""]
        for e, name in (("cafe", "카페 시험"), ("live", "실전 시험")):
            r, o = runs.get((NEW, e)), runs.get((ref, e))
            if not (r and o):
                continue
            rows = []
            for cid, sc in r["scores"].items():
                v_new, v_old = verdict(sc), verdict(o["scores"][cid])
                if v_new == v_old:
                    continue
                case = exams[e][cid]
                said = case["conversation"][-1]["content"].replace("|", "\\|")
                got = calls_ko(r["raw"][cid].get("tool_calls"))
                if r["raw"][cid].get("content") and not r["raw"][cid].get("tool_calls"):
                    got += " — 텍스트: " + r["raw"][cid]["content"][:80].replace("|", "/")
                rows.append(f"| {cid} | {CAT_KO.get(case['category'], case['category'])} | {said} | {SYM[v_old]} → {SYM[v_new]} | "
                            f"{calls_ko(case['expected'])} | {calls_ko(o['raw'][cid].get('tool_calls'))} | {got} |")
            better = sum(1 for x in rows if "→ ✅" in x)
            L += [f"**{name}:** {len(rows)}문항 달라짐 (나아짐 {better}, 나빠지거나 부분 점수로 바뀜 {len(rows) - better})", ""]
            if rows:
                L += ["| 문항 | 유형 | 손님 말 | 기준 → 가지치기 | 정답 | 기준 답 | 가지치기 답 |", "|---|---|---|:-:|---|---|---|"] + rows
            L.append("")
        r, o = runs.get((NEW, "knowledge")), runs.get((ref, "knowledge"))
        if r and o:
            up = sum(1 for k, sc in r["scores"].items() if sc["exact"] and not o["scores"][k]["exact"])
            down = sum(1 for k, sc in r["scores"].items() if not sc["exact"] and o["scores"][k]["exact"])
            L += [f"**지식 시험:** 기준이 틀리고 가지치기가 맞힌 문항 {up}개, 기준이 맞히고 가지치기가 틀린 문항 {down}개.", ""]

    # ---- build facts
    mdir = ROOT / f"models/gemma-4-26B-A4B-it-cafe-{K}e"
    gg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))["text_config"]
    bf16_gb = json.loads((mdir / "model.safetensors.index.json").read_text(encoding="utf-8"))["metadata"]["total_size"] / 1e9
    fx = ollama_facts(NEW)
    if load_chk and load_chk.get("fits_igpu_fully"):
        fit_txt = "**예** (`ollama ps` 100% GPU)"
    elif load_chk:
        fit_txt = "아니요 (" + ("HTTP " + str(load_chk.get("http_status")) if load_chk.get("http_status") != 200 else "일부만 GPU") + ")"
    else:
        fit_txt = "-"
    q4_txt = f"{fx['q4_gb']:.2f} GB" if fx["q4_gb"] else "-"
    L += ["## 만든 과정과 크기", "",
          f"| 항목 | 원본 26b | 가지치기 {K}개 |", "|---|---:|---:|",
          f"| 층당 전문가 / 토큰당 사용 | 128 / 8 | {gg['num_experts']} / {gg['top_k_experts']} |",
          f"| 파라미터 (`ollama show`) | 25.2B | {fx['params'] or '-'} |",
          f"| BF16 체크포인트 | 51.6 GB | {bf16_gb:.1f} GB |",
          f"| Q4_K_M GGUF | 17.07 GB | {q4_txt} |",
          f"| Ollama 등록 크기 (`ollama list`, 비전 프로젝터·초안 모델 포함) | 18 GB | {fx['listed'] or '-'} |",
          f"| iGPU에 다 들어가나 | 아니요 (1단계: 로딩 시 HTTP 500) | {fit_txt} |",
          ""]
    for f in ([load_file(K)] + ([S5 / f"load_{K}e_cpu.json"] if (S5 / f"load_{K}e_cpu.json").exists() else [])):
        if not f.exists():
            continue
        lc = json.loads(f.read_text(encoding="utf-8"))
        pm = lc["peak_memory"]
        pf = (pm.get("pagefile_used_gb") or 0) - (pm.get("pagefile_start_gb") or 0)
        L.append(f"- {'iGPU' if lc['mode'] == 'igpu' else 'CPU'} 로딩 확인(num_ctx 8192): {lc['load_s']} s, HTTP {lc.get('http_status')}. "
                 f"로딩 중 최대: 러너 private {pm['runner_private_gb']} GB, GPU 공유 {pm['gpu_shared_gb']} GB, "
                 f"시스템 사용 {pm['sys_used_gb']} GB / 32 GB(가용 최저 {pm['sys_avail_min_gb']} GB), 페이지 파일 증가 {pf:.1f} GB.")
    L += ["- `ollama ps`의 SIZE는 iGPU 공유 메모리 모드에서 실제보다 작게 나온다(1단계 참고).",
          "- 순서: 원본 safetensors에서 전문가 텐서(`experts.gate_up_proj`, `experts.down_proj`)와 라우터(`router.proj.weight`, `router.per_expert_scale`)의 전문가 축만 잘라 새 폴더에 썼다(원본은 읽기만 함, 64 MB 단위로 스트리밍). "
          "→ llama.cpp b12283 변환기로 BF16 GGUF → `llama-quantize`로 Q4_K_M → Ollama 등록.",
          "- 양자화는 원본 `gemma4:26b`와 텐서별 형식을 똑같이 맞췄다(658개 텐서 모두 같음: 어텐션·밀집 MLP·임베딩 Q8_0, 전문가 gate_up Q4_K(9·15층 Q5_K), down Q5_1(29층 Q8_0), 라우터 F32). "
          "원본은 Ollama가 C4 영어 텍스트로 만든 중요도 행렬(imatrix)을 썼는데 이 파일은 공개되지 않아, 가지치기 모델과 대조군은 imatrix 없이 양자화했다.",
          "- 템플릿·파라미터·비전 프로젝터·초안(DRAFT) 모델은 `ollama show --modelfile gemma4:26b` 그대로 썼다(FROM 한 줄만 바꿈).",
          "- 중간 파일(가지치기 BF16 safetensors, BF16 GGUF)은 내용을 원본과 대조하고(`scripts/verify_pruned_files.py`) Q4 파일을 검증한 뒤(`scripts/verify_q4.py`) 지웠다.",
          ""]

    # ---- validation
    def row(i, d):
        x = d["layers"][str(i)]
        return x["isolated_all"]["mean"], x["chain_all"]["mean"], x["chain_case"]["mean"]
    if val is None:
        L += ["## 은닉 상태 검증", "", "- 결과 없음 (검증이 실패했거나 실행되지 않음)", ""]
    else:
        L += ["## 은닉 상태 검증 (전체 모델을 올리지 않고 층 단위로)", "",
              "보정 문항 8개(토큰 6,156개)를 원본과 가지치기 모델에 층마다 통과시켜 코사인 유사도를 쟀다(float32, CPU). "
              "비교 기준으로 **원본을 bf16으로 계산했을 때**(가지치기 없음, 반올림 오차만)의 값도 같이 쟀다.", "",
              "- 이 층만: 원본의 입력을 가지치기 층에 넣었을 때 (그 층 하나의 오차)",
              "- 누적: 가지치기 모델을 처음부터 따로 돌렸을 때 (앞 층의 오차가 쌓임)", "",
              "| 층 | 가지치기: 이 층만 | 가지치기: 누적 (전체 토큰) | 가지치기: 누적 (문항별 토큰) | bf16 기준: 누적 (전체) | bf16 기준: 누적 (문항별) |",
              "|---:|---:|---:|---:|---:|---:|"]
        for i in (0, 4, 5, 9, 11, 14, 17, 20, 23, 26, 29):
            a_, b, c = row(i, val)
            _, bb, bc = row(i, base)
            L.append(f"| {i} | {a_:.5f} | {b:.4f} | {c:.4f} | {bb:.4f} | {bc:.4f} |")
        vf, bf = val["final"], base["final"]
        _, v26, c26 = row(26, val)
        _, b26, bc26 = row(26, base)
        L += ["",
              f"| 마지막 출력 (문항별 토큰 {vf['n_tokens']}개) | 가지치기 {K}개 | bf16 기준 |", "|---|---:|---:|",
              f"| 다음 토큰 1순위 일치 | {vf['top1_agreement_case_tokens'] * 100:.1f}% | {bf['top1_agreement_case_tokens'] * 100:.1f}% |",
              f"| 평균 KL(원본 ‖ 비교 대상) | {vf['mean_kl_orig_to_pruned']:.3f} | {bf['mean_kl_orig_to_pruned']:.3f} |",
              "",
              f"- 누적 값은 층을 지나며 떨어지고, bf16 반올림만 해도 같은 모양으로 떨어진다. 26층 기준 가지치기 {v26:.3f} vs bf16 {b26:.3f}(전체 토큰), "
              f"{c26:.3f} vs {bc26:.3f}(문항별 토큰).",
              f"- 다음 토큰 1순위 일치: 가지치기 {vf['top1_agreement_case_tokens'] * 100:.1f}% vs bf16 {bf['top1_agreement_case_tokens'] * 100:.1f}%. "
              f"KL은 bf16 기준의 약 {vf['mean_kl_orig_to_pruned'] / bf['mean_kl_orig_to_pruned']:.1f}배.",
              ""]

    interp = S5 / f"keep{K}_interpretation.md"      # hand-written interpretation, kept out of the generated part
    L += [interp.read_text(encoding="utf-8").strip() if interp.exists() else "<!-- INTERPRETATION -->", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"saved {OUT_MD.relative_to(ROOT).as_posix()}; missing: {missing}")


if __name__ == "__main__":
    main()
