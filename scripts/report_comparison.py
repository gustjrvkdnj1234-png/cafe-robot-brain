"""One Korean table for all contestants (Step 3 + Step 5): results/step5/comparison.md

    .venv\\Scripts\\python.exe scripts\\report_comparison.py

Rows: vending machine, E4B, 12B, original 26B, 128-expert no-imatrix control, pruned 98 / 64 / 48.
A hand-written note is appended from results/step5/comparison_notes.md if present.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from report_step3 import MODE_KO, pct  # noqa: E402
from report_step5 import ALL, CONTROL, KEEPS, S5, fmt_t, label, load, mem_text, model_mode, ollama_facts, step5_dir  # noqa: E402
from run_step3 import CONTESTANTS, run_dir  # noqa: E402

SIZE_STEP3 = {"gemma4:e4b": "6.6 GB", "gemma4:12b": "8.0 GB", "gemma4:26b": "18 GB"}   # `ollama list`


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [(m, mode, es) for m, mode, es in CONTESTANTS]
    rows.append((CONTROL, "cpu", ALL))
    for k in KEEPS:
        rows.append((f"song/gemma4-26b-cafe-{k}e", model_mode(k) or "igpu", ALL))
    runs = {}
    for m, mode, es in rows:
        for e in es:
            runs[(m, e)] = load(step5_dir(m, mode, e)) if m.startswith("song/") else load(run_dir(m, mode, e))

    L = ["# 전체 비교: 원본 모델, 대조군, 가지치기 모델", "",
         "- 모든 모델 공통: thinking 끔, temperature 0, seed 42, num_ctx 8192, 같은 시스템 프롬프트. 점수는 '정답(exact)' 비율.",
         "- 대조군: 원본 128개 모델을 가지치기 모델과 같은 방식(텐서별 형식 같음, imatrix 없음)으로 양자화한 것. "
         "원본 26b와 대조군의 차이 = imatrix 효과, 대조군과 가지치기 모델의 차이 = 가지치기 효과.",
         "- 모드가 다르면(iGPU / CPU) 속도와 메모리는 바로 비교할 수 없다. 점수는 비교할 수 있다.",
         "- 최대 메모리: 시험 중 3초마다 잰 러너(llama-server) private 메모리의 최댓값(프롬프트 캐시 최대 8 GB 포함). iGPU는 GPU 공유 메모리도 함께 적었다.",
         "",
         "| 참가자 | 모드 | 카페 (120) | 실전 (20) | 지식 (300) | 주문당 응답 시간 중앙값 | 최대 메모리 | 모델 크기 (`ollama list`) |",
         "|---|---|---:|---:|---:|---:|---|---:|"]
    for m, mode, es in rows:
        cafe, live, know = runs.get((m, "cafe")), runs.get((m, "live")), runs.get((m, "knowledge"))
        if m.startswith("song/") and not any((cafe, live, know)):
            L.append(f"| {label(m)} | - | 결과 없음 | | | | | |")
            continue
        lat = [r["wall_s"] for x in (cafe, live) if x for r in x["raw"].values() if r.get("wall_s") is not None]
        size = SIZE_STEP3.get(m) or (ollama_facts(m)["listed"] if m.startswith("song/") else "-")
        L.append(f"| {label(m)} | {MODE_KO[mode]} | {pct(cafe['summary']['exact']) if cafe else '-'} | "
                 f"{pct(live['summary']['exact']) if live else '-'} | "
                 f"{pct(know['summary']['accuracy']) if know else ('해당 없음' if 'knowledge' not in es else '-')} | "
                 f"{fmt_t(statistics.median(lat)) if lat else '-'} | {mem_text(runs, m, mode)} | {size or '-'} |")
    L += ["", "- 지식 시험 세부 (kmmlu / mmlu):"]
    for m, mode, es in rows:
        know = runs.get((m, "knowledge"))
        if know:
            bs = know["summary"]["by_source"]
            L.append(f"  - {label(m)}: {pct(bs['kmmlu']['accuracy'])} / {pct(bs['mmlu']['accuracy'])}")
    L.append("")
    notes = S5 / "comparison_notes.md"
    if notes.exists():
        L += [notes.read_text(encoding="utf-8").strip(), ""]
    (S5 / "comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("saved results/step5/comparison.md")


if __name__ == "__main__":
    main()
