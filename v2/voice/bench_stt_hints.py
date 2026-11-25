"""STT speed and accuracy with menu-word hints (C2) and short-clip settings (D2), CPU only.

    .venv-voice\\Scripts\\python.exe v2\\voice\\bench_stt_hints.py              # all models, then report
    .venv-voice\\Scripts\\python.exe v2\\voice\\bench_stt_hints.py fw-small     # one model
    .venv-voice\\Scripts\\python.exe v2\\voice\\bench_stt_hints.py --report

Same 20 recordings, scoring (CER without spaces / punctuation) and base decoding as bench_stt.py
(language ko, beam 1, temperature 0 without fallback, no timestamps, 8 threads, int8, network blocked).
Each model runs in its own process; inside it every setting gets one warm-up clip (not counted) and
then the 20 clips. Settings:
  hints:  none | prompt (initial_prompt = hint text) | hotwords (hotwords = hint text)
  window: full (faster-whisper pads every clip to the 30 s encoder window) |
          short (the mel features are padded only to the clip length + 1 s, by replacing pad_or_trim;
                 not an official option, Whisper was trained on 30 s windows)
  vad:    off | on (vad_filter=True, Silero VAD drops silence before decoding)
Also: two non-speech clips (1.5 s silence, 1.5 s low noise) per setting, to see whether a hint makes the
model write words that were never said (hint text leaking into the output).
Hint text = menu.json item names + nicknames (short aliases / ones implying a temperature, no spelling
variants) + option words (+ a few spoken option words).
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import bench_stt as B  # noqa: E402

OUT = HERE / "results" / "stt_hints"
MODELS = {"fw-small": "small", "fw-medium": "medium", "fw-large-v3-turbo": "large-v3-turbo"}
SETTINGS = [  # (name, hints, window, vad)
    ("base", "none", "full", False),
    ("prompt", "prompt", "full", False),
    ("hotwords", "hotwords", "full", False),
    ("vad", "none", "full", True),
    ("prompt+vad", "prompt", "full", True),
    ("short", "none", "short", False),
    ("prompt+short", "prompt", "short", False),
]
sys.path.insert(0, str(HERE / "app"))
from stt_hints import hint_text  # noqa: E402  (same hint text as the app)


def nonspeech():
    rng = np.random.default_rng(0)
    return {"silence": np.zeros(24000, dtype=np.float32),
            "noise": (rng.standard_normal(24000) * 0.003).astype(np.float32)}


def worker(name):
    B.block_network()
    import faster_whisper.transcribe as FT
    from faster_whisper import WhisperModel
    orig_pad = FT.pad_or_trim
    t0 = time.perf_counter()
    m = WhisperModel(str(B.MODELS / name), device="cpu", compute_type="int8", cpu_threads=B.THREADS)
    load_s = time.perf_counter() - t0
    hints = hint_text()
    n_hint_tokens = len(m.hf_tokenizer.encode(" " + hints).ids)
    items = B.utterances()
    audio = {it["path"]: B.load_wav(it["path"]) for it in items}
    ns = nonspeech()
    res = {"name": name, "load_s": round(load_s, 2), "hint_text": hints, "hint_tokens": n_hint_tokens, "settings": {}}
    for sname, hint, window, vad in SETTINGS:
        def pad(array, length=3000, *, axis=-1, _w=window):
            if _w == "short":   # features of one clip: frames = 100 per second; keep 1 s of padding
                length = min(3000, array.shape[axis] + 100)
            return orig_pad(array, length, axis=axis)
        FT.pad_or_trim = pad
        kw = dict(language="ko", beam_size=1, best_of=1, temperature=0.0, condition_on_previous_text=False,
                  without_timestamps=True, vad_filter=vad)
        if hint == "prompt":
            kw["initial_prompt"] = hints
        elif hint == "hotwords":
            kw["hotwords"] = hints

        def run(x):
            segs, _ = m.transcribe(x, **kw)
            return "".join(s.text for s in segs).strip()
        try:
            run(audio[items[0]["path"]])                    # warm-up
        except Exception as e:  # noqa: BLE001
            res["settings"][sname] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"  {sname}: ERROR {e}", flush=True)
            continue
        rows = []
        for it in items:
            t = time.perf_counter()
            hyp = run(audio[it["path"]])
            lat = time.perf_counter() - t
            ref_n, hyp_n = B.normalize(it["ref"]), B.normalize(hyp)
            rows.append({**{k: it[k] for k in ("device", "id", "ref", "duration_s")}, "hyp": hyp,
                         "errors": B.edit_distance(ref_n, hyp_n), "ref_chars": len(ref_n), "latency_s": round(lat, 3)})
        non = {k: run(v) for k, v in ns.items()}
        cer = 100 * sum(r["errors"] for r in rows) / sum(r["ref_chars"] for r in rows)
        med = statistics.median(r["latency_s"] for r in rows)
        print(f"  {sname:<13} CER {cer:5.1f}%  median {med:.2f}s  non-speech {non}", flush=True)
        res["settings"][sname] = {"hints": hint, "window": window, "vad": vad, "items": rows, "nonspeech": non}
    FT.pad_or_trim = orig_pad
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


def report():
    L = ["# STT 힌트·짧은 창 비교 (CPU, 오프라인, 녹음 20개)", "",
         "기본 디코딩은 bench_stt.py와 같다(한국어 고정, beam 1, temperature 0, 타임스탬프 없음, 스레드 8, int8). "
         "지연 = 메모리에 올린 16 kHz 녹음 한 개의 인식 시간, 중앙값. CER = 띄어쓰기·문장부호 빼고 글자 단위.", "",
         "- 힌트(prompt) = `initial_prompt`, 힌트(hotwords) = `hotwords`. 힌트 문장은 menu.json의 메뉴 이름, 짧은 줄임말(3글자 이하이거나 온도를 뜻하는 것, 표기 변형 제외), 옵션 말에 "
         "말로 하는 옵션 말(반샷, 샷 추가, 큰 걸로, 테이크아웃, 포장, 매장, 캐리어, 덜 달게, 더 달게)을 더했다.",
         "- 주의: 녹음 20개의 문장에 힌트 단어(캐리어, 아바라, 반샷 등)가 들어 있다. 힌트가 맞히는 단어를 고른 셈이라, "
         "힌트로 줄어든 CER은 이 20개에서 유리하게 나온 값이다.",
         "- short = 인코더에 30초 대신 녹음 길이 + 1초만 넣음(pad_or_trim 교체, 공식 옵션 아님). vad = `vad_filter=True`.",
         "- 말 없음: 1.5초 무음 / 1.5초 약한 잡음을 넣었을 때 나온 글자(빈칸이 정상).", ""]
    files = {n: OUT / f"{n}.json" for n in MODELS}
    data = {n: json.loads(p.read_text(encoding="utf-8")) for n, p in files.items() if p.exists()}
    summary = {}
    L += ["| 모델 | 설정 | CER 전체 | CER 내장 | CER BT | 지연 중앙값 | 지연 최대 | 말 없음: 무음 / 잡음 |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for n, d in data.items():
        for sname, *_ in SETTINGS:
            s = d["settings"].get(sname)
            if not s:
                continue
            if "error" in s:
                L.append(f"| {MODELS[n]} | {sname} | 실패 | | | | | {s['error'][:80]} |")
                continue
            rows = s["items"]
            def cer(rs):
                return 100 * sum(r["errors"] for r in rs) / sum(r["ref_chars"] for r in rs)
            lat = [r["latency_s"] for r in rows]
            v = {"cer": cer(rows), "cer_builtin": cer([r for r in rows if r["device"] == "builtin"]),
                 "cer_bt": cer([r for r in rows if r["device"] == "bluetooth"]),
                 "lat_median": statistics.median(lat), "lat_max": max(lat), "nonspeech": s["nonspeech"]}
            summary[f"{n}/{sname}"] = v
            ns = " / ".join((repr(s["nonspeech"][k][:24]) + ("…" if len(s["nonspeech"][k]) > 24 else "")) if s["nonspeech"][k] else "(빈칸)" for k in ("silence", "noise"))
            L.append(f"| {MODELS[n]} | {sname} | **{v['cer']:.1f}%** | {v['cer_builtin']:.1f}% | {v['cer_bt']:.1f}% | "
                     f"**{v['lat_median']:.2f} s** | {v['lat_max']:.2f} s | {ns} |")
    if data:
        any_d = next(iter(data.values()))
        L += ["", f"힌트 문장 ({any_d['hint_tokens']} 토큰):", "", f"> {any_d['hint_text']}", ""]
    # per-utterance table for the settings that matter
    show = [(n, s) for n in data for s in ("base", "prompt", "prompt+short") if s in data[n]["settings"]
            and "error" not in data[n]["settings"][s]]
    if show:
        L += ["## 문장별 인식 결과", "", "| 장치 | # | 정답 | " + " | ".join(f"{MODELS[n]} {s}" for n, s in show) + " |",
              "|---|---|---|" + "---|" * len(show)]
        base_rows = data[show[0][0]]["settings"][show[0][1]]["items"]
        for k, r0 in enumerate(base_rows):
            cells = []
            for n, s in show:
                r = data[n]["settings"][s]["items"][k]
                cells.append(("✓ " if r["errors"] == 0 else f"({r['errors']}) ") + r["hyp"].replace("|", "/"))
            L.append(f"| {r0['device']} | {r0['id']} | {r0['ref']} | " + " | ".join(cells) + " |")
        L += ["", "✓ = 띄어쓰기·문장부호 빼고 정답과 같음, (n) = 틀린 글자 수."]
    (OUT.parent / "stt_hints.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT.parent / "stt_hints_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L[:12 + len(summary) + 2]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args[:1] == ["--worker"]:
        return worker(args[1])
    if args == ["--hint"]:
        return print(hint_text())
    if args != ["--report"]:
        for name in (args or list(MODELS)):
            print(f"== {name}", flush=True)
            t = time.time()
            rc = subprocess.run([sys.executable, "-I", str(Path(__file__)), "--worker", name]).returncode
            print(f"== {name} done rc={rc} in {time.time() - t:.0f} s", flush=True)
    report()


if __name__ == "__main__":
    main()
