"""D3 / final speed summary: time to first audio vs. all audio, typed replay runs and voice self-tests.

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\first_audio_summary.py
Reads results/app_script_<label>.json and results/app_selftest_<label>.json for the labels below (missing ones are
skipped) and writes results/first_audio.md.
"""
import json
import statistics as S
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
REPLAYS = [("replay_d3_off", "D3 끔 (전체 답 합성 후 보냄)"), ("replay_d3_on", "D3 켬 (첫 문장 먼저)"),
           ("replay_stt_d3", "최종 (추천 STT + D3)")]
SELFTESTS = [("before", "전: STT medium, D3 끔"), ("d3", "STT medium, D3 켬"), ("final", "최종: 추천 STT + D3"),
             ("all_before", "녹음 20개, 전: medium, D3 끔"), ("all_final", "녹음 20개, 최종")]
TARGET = {"text": 4.0, "audio": 6.0}


sys.path.insert(0, str(Path(__file__).resolve().parent))
from tts_text import sentences as tts_sentences  # noqa: E402


def med(v):
    v = [x for x in v if x is not None]
    return S.median(v) if v else None


def f(v):
    return "-" if v is None else f"{v:.2f}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    L = ["# 첫 소리까지 시간 (CPU 전용)", "",
         "- 첫 소리 = 요청을 받은 때부터 첫 문장의 소리가 준비된 때까지(서버). 브라우저는 이때 재생을 시작한다. "
         "D3를 끄면 답 전체를 합성한 뒤라서 첫 소리 = 서버 합계.",
         "- 서버 합계 = 모든 문장의 소리가 준비된 때까지. 클라이언트 값은 같은 PC에서 HTTP로 잰 값.",
         "- 각 실행 전에 TTS 캐시를 비우고 고정 답만 다시 합성해 두었다(앱을 막 켠 상태와 같음).",
         f"- 목표: 글자 턴 첫 소리 ≤ {TARGET['text']:.0f}초, 음성 턴 ≤ {TARGET['audio']:.0f}초.", "",
         "## 글자 15턴 (세션 f68fd4 다시 돌리기)", "",
         "| 실행 | 64e 중앙값 | TTS 중앙값 (첫 문장 / 나머지) | 64e 끝 → 첫 소리 | **첫 소리 중앙값** | 서버 합계 중앙값 | 첫 소리 ≤ 4초 턴 | 클라이언트 첫 소리 | 두 문장 이상 턴 | 그 턴들: 64e 끝 → 첫 소리 / 모든 소리 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    out = {}
    for lab, name in REPLAYS:
        p = RES / f"app_script_{lab}.json"
        if not p.exists():
            continue
        T = json.loads(p.read_text(encoding="utf-8"))["turns"]
        t = [x["timings"] for x in T]
        fa = [x.get("first_audio_s", x.get("total_s")) for x in t]
        row = {"llm": med([x.get("llm_s") for x in t]), "tts": med([x.get("tts_s") for x in t]),
               "tts_rest": med([x.get("tts_rest_s") for x in t]), "first": med(fa), "total": med([x["total_s"] for x in t]),
               "under": sum(v is not None and v <= TARGET["text"] for v in fa), "n": len(T),
               "client_first": med([x.get("client_first_audio_s") for x in T]),
               # time after the model answered (or the fast path) until the first audio: reply + TTS of piece 1
               "after_llm": med([a - (x.get("llm_s") or 0) for a, x in zip(fa, t) if a is not None]),
               "multi": sum(len(tts_sentences(x["reply"])) > 1 for x in T),
               "multi_after": med([a - (x["timings"].get("llm_s") or 0) for a, x in zip(fa, T)
                                   if len(tts_sentences(x["reply"])) > 1]),
               "multi_total_after": med([x["timings"]["total_s"] - (x["timings"].get("llm_s") or 0) for x in T
                                         if len(tts_sentences(x["reply"])) > 1])}
        out[lab] = row
        L.append(f"| {name} | {f(row['llm'])} | {f(row['tts'])} / {f(row['tts_rest'])} | {f(row['after_llm'])} | "
                 f"**{f(row['first'])}** | "
                 f"{f(row['total'])} | {row['under']}/{row['n']} | {f(row['client_first'])} | "
                 f"{row['multi']} | {f(row['multi_after'])} / {f(row['multi_total_after'])} |")
    L += ["", "## 자체 시험 (글자 5턴 + 음성)", "",
          "| 실행 | 글자 첫 소리 중앙값 | 글자 서버 합계 | 음성 첫 소리 중앙값 | 음성 서버 합계 | STT 중앙값 | 음성 CER | 음성 턴 수 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    sys.path.insert(0, str(RES.parent))
    import bench_stt as B
    for lab, name in SELFTESTS:
        p = RES / f"app_selftest_{lab}.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))["turns"]

        def m(kind, key):
            return med([r["timings"].get(key, r["timings"].get("total_s")) for r in rows if r["kind"] == kind])
        au = [r for r in rows if r["kind"] == "audio"]
        cer = 100 * sum(B.edit_distance(B.normalize(r["reference"]), B.normalize(r["transcript"])) for r in au) / \
            sum(len(B.normalize(r["reference"])) for r in au)
        row = {"text_first": m("text", "first_audio_s"), "text_total": m("text", "total_s"),
               "audio_first": m("audio", "first_audio_s"), "audio_total": m("audio", "total_s"),
               "stt": m("audio", "stt_s"), "cer": cer, "n_audio": len(au)}
        out[f"selftest_{lab}"] = row
        L.append(f"| {name} | {f(row['text_first'])} | {f(row['text_total'])} | **{f(row['audio_first'])}** | "
                 f"{f(row['audio_total'])} | {f(row['stt'])} | {cer:.1f}% | {len(au)} |")
    (RES / "first_audio.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RES / "first_audio.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
