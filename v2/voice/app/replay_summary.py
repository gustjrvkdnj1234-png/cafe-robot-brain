"""Compare replays of one real session (run_script.py --from-log ... --label replay_*).

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\replay_summary.py
Writes v2/voice/results/replay_f68fd4.md (medians per run + per-turn speed).
"""
import json
import statistics as S
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
RUNS = [("replay_before", "전 (11-24 코드)"), ("replay_final", "후 (11-25 기본: append, A3=app)"),
        ("replay_after_append", "후 append, A3=prompt"), ("replay_after_compact", "후 compact(D1), A3=prompt")]


def med(turns, k):
    v = [t["timings"].get(k) for t in turns if t["timings"].get(k) is not None]
    return S.median(v) if v else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = {lab: json.loads((RES / f"app_script_{lab}.json").read_text(encoding="utf-8")) for lab, _ in RUNS}
    L = ["# 실제 세션 f68fd4 다시 돌리기 (글자 15턴, CPU 전용)", "",
         "같은 15개 인식 결과를 같은 순서로 넣었다(`run_script.py --from-log f68fd4`). 턴별 표는 "
         "`app_script_replay_*.md`.", "",
         "| 실행 | llm_s 중앙값 | prompt_s | gen_s | 새로 계산한 토큰 | 프롬프트 토큰 | tts_s | total_s | 말로만 답함 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lab, name in RUNS:
        T = data[lab]["turns"]
        ev = [t["cache"]["evaluated"] for t in T if t.get("cache")]
        L.append(f"| {name} | **{med(T, 'llm_s'):.2f}** | {med(T, 'prompt_s'):.2f} | {med(T, 'gen_s'):.2f} | "
                 f"{S.median(ev) if ev else '-'} | {med(T, 'prompt_tokens')} | {med(T, 'tts_s'):.2f} | "
                 f"{med(T, 'total_s'):.2f} | {data[lab]['free_text_turns']} |")
    L += ["", "- tts_s: 11-25 코드는 같은 문장을 다시 합성하지 않는다(캐시). 두 번째 이후 실행은 앞 실행과 같은 답이 많아 "
          "tts_s가 낮게 나온다. 처음 듣는 문장의 합성은 약 0.9–1.5초.", "",
          "| # | " + " | ".join(f"{n} llm_s / 새 토큰" for _, n in RUNS) + " |", "|---|" + "---:|" * len(RUNS)]
    for i in range(len(data[RUNS[0][0]]["turns"])):
        cells = []
        for lab, _ in RUNS:
            t = data[lab]["turns"][i]
            cells.append(f"{t['timings'].get('llm_s')} / {(t.get('cache') or {}).get('evaluated', '-')}")
        L.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    (RES / "replay_f68fd4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
