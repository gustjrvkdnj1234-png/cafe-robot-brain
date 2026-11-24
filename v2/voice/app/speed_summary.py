"""Summarize the speed runs (run_script.py --label speed_<mode>_<ctx>_r<k>) into a table.

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\speed_summary.py
Writes v2/voice/results/app_speed.md and app_speed.json.
"""
import json
import statistics as S
import sys
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    groups = defaultdict(list)
    for p in sorted(RES.glob("app_script_speed_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d["settings"]
        groups[(s["mode"], s["num_ctx"])].append(d)
    rows, out = [], {}
    for (mode, ctx), runs in sorted(groups.items(), key=lambda kv: (kv[0][0] != "igpu", -kv[0][1])):
        turns = [t for r in runs for t in r["turns"]]
        med = lambda k: S.median(t["timings"][k] for t in turns if t["timings"].get(k) is not None)  # noqa: E731
        per_run = [S.median(t["timings"]["llm_s"] for t in r["turns"]) for r in runs]
        big = max(t["timings"].get("prompt_tokens") or 0 for t in turns)
        same = all([c["name"] for c in t["tool_calls"]] == [c["name"] for c in runs[0]["turns"][i]["tool_calls"]]
                   for r in runs for i, t in enumerate(r["turns"]))
        out[f"{mode}_{ctx}"] = {"runs": len(runs), "llm_s": med("llm_s"), "prompt_s": med("prompt_s"), "gen_s": med("gen_s"),
                                "per_run_llm_s": per_run, "max_prompt_tokens": big, "same_tools_all_runs": same}
        rows.append(f"| {'iGPU' if mode == 'igpu' else 'CPU 전용'} | {ctx} | {len(runs)} | **{med('llm_s'):.2f} s** | "
                    f"{med('prompt_s'):.2f} s | {med('gen_s'):.2f} s | {' / '.join(f'{x:.2f}' for x in per_run)} | "
                    f"{big} | {'같음' if same else '다름'} |")
    L = ["# 64e 속도: iGPU vs CPU 전용, num_ctx 8192 vs 4096", "",
         "8턴 글자 스크립트(s8)를 설정마다 3번, 같은 서버·데모 설정·기록 방식(tools / append / 6→3)으로. "
         "모드를 바꾼 뒤 버리는 예열 1턴. 중앙값은 3번 × 8턴 = 24턴 전체.", "",
         "| 모드 | num_ctx | 실행 | llm_s 중앙값 | prompt_s 중앙값 | gen_s 중앙값 | 실행별 llm_s 중앙값 | 최대 프롬프트 토큰 | 도구 이름 (3번 비교) |",
         "|---|---:|---:|---:|---:|---:|---|---:|---|"] + rows
    (RES / "app_speed.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RES / "app_speed.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
