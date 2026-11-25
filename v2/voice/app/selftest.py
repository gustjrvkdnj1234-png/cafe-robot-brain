"""Self-test against a running demo (start_demo.ps1): 5 typed turns (STT skipped) as one conversation,
then 2 of the builtin-mic recordings through the whole pipeline, then one Heami preview.

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\selftest.py [--port 8770]

Writes v2/voice/results/app_selftest.json and app_selftest.md. Uses the current voice setting.
"""
import argparse
import base64
import io
import json
import statistics
import sys
import time
from pathlib import Path

import requests
import soundfile as sf

VOICE = Path(__file__).resolve().parents[1]
OUT = VOICE / "results"
TYPED = [  # own test sentences (not from any exam file)
    "아이스 아메리카노 큰 걸로 두 잔이랑 치즈케이크 하나요",
    "녹차라떼 있으면 녹차라떼, 없으면 그냥 카페라떼로 주세요",
    "첫 번째 아메리카노에 샷 하나 추가해 주세요",
    "디카페인으로도 되나요?",
    "이제 주문할게요, 포장해 주세요",
]
RECORDINGS = ["builtin/01.wav", "builtin/07.wav"]


def audio_seconds(data_url):
    if not data_url:
        return None
    raw = base64.b64decode(data_url.split(",", 1)[1])
    info = sf.info(io.BytesIO(raw))
    return round(info.duration, 2), info.samplerate


def post_turn(url, stream, **kw):
    """Returns (turn, client first-audio s, client all-audio s); NDJSON when the server streams (D3)."""
    t = time.perf_counter()
    r = requests.post(url, timeout=600, stream=True, **kw)
    r.raise_for_status()
    if "ndjson" not in r.headers.get("content-type", ""):
        d = r.json()
        w = time.perf_counter() - t
        return d, w, w
    d, first = None, None
    for line in r.iter_lines():
        if not line:
            continue
        m = json.loads(line)
        if d is None:
            d, first = m, time.perf_counter() - t
        elif m.get("done"):
            d["timings"] = m["timings"]
    return d, first, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--stream", type=int, choices=[0, 1], default=None, help="D3 on/off (default: server setting)")
    ap.add_argument("--label", default="", help="suffix for the result files")
    ap.add_argument("--recordings", choices=["two", "all"], default="two",
                    help="two = builtin 01 and 07 in one session (the usual self-test); all = all 20, each a new session")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    base = f"http://127.0.0.1:{a.port}"
    h = requests.get(f"{base}/api/health", timeout=5).json()
    if not h.get("ready"):
        sys.exit(f"app not ready: {h}")
    voice = requests.get(f"{base}/api/config", timeout=5).json()
    stream = a.stream if a.stream is not None else int(bool(h.get("settings", {}).get("stream_tts")))
    rows = []

    def record(kind, inp, d, wall, first):
        dur = audio_seconds(d.get("audio"))
        row = {"kind": kind, "input": inp, "transcript": d["transcript"], "tool_calls": d["tool_calls"],
               "reply": d["reply"], "kinds": d["kinds"], "order": d["order"], "confirmed": d["confirmed"],
               "dine_option": d["dine_option"], "staff_called": d["staff_called"], "timings": d["timings"],
               "client_wall_s": round(wall, 3), "client_first_audio_s": round(first, 3),
               "reply_audio_s": dur and dur[0], "reply_sr": dur and dur[1],
               "error": d.get("error")}
        rows.append(row)
        t = d["timings"]
        print(f"[{kind}] {inp}\n   heard: {d['transcript']}\n   calls: {json.dumps(d['tool_calls'], ensure_ascii=False)}"
              f"\n   reply: {d['reply']}\n   order: {d['order']}  confirmed={d['confirmed']} staff={d['staff_called']}"
              f"\n   STT {t['stt_s']} LLM {t['llm_s']} TTS {t['tts_s']} first audio {t.get('first_audio_s')} "
              f"total {t['total_s']} (client {wall:.2f})"
              f"{'  ERROR ' + d['error'] if d.get('error') else ''}", flush=True)

    sess = requests.post(f"{base}/api/new_session", json={}, timeout=5).json()["session"]
    for text in TYPED:
        d, first, wall = post_turn(f"{base}/api/turn/text", stream, params={"stream": stream},
                                   json={"session": sess, "text": text})
        record("text", text, d, wall, first)

    sess = requests.post(f"{base}/api/new_session", json={}, timeout=5).json()["session"]
    recs = RECORDINGS if a.recordings == "two" else [f"{d}/{k:02d}.wav" for d in ("builtin", "bluetooth") for k in range(1, 11)]
    for rel in recs:
        if a.recordings == "all":   # every recording as the first turn of a new customer
            sess = requests.post(f"{base}/api/new_session", json={"previous": sess}, timeout=5).json()["session"]
        p = VOICE / "recordings" / rel
        man = json.loads((p.parent / "manifest.json").read_text(encoding="utf-8"))
        ref = man["items"][p.stem]["text"]
        d, first, wall = post_turn(f"{base}/api/turn/audio", stream, params={"session": sess, "stream": stream},
                                   data=p.read_bytes(), headers={"Content-Type": "audio/wav"})
        record("audio", f"{rel} (정답: {ref})", d, wall, first)
        rows[-1]["reference"] = ref

    t = time.perf_counter()
    hp = requests.post(f"{base}/api/tts/preview", json={"engine": "heami"}, timeout=60).json()
    heami = {"tts_s": hp["tts_s"], "audio": audio_seconds(hp["audio"]), "voice": hp["voice"]}
    print("heami preview:", heami, flush=True)

    res = {"voice": voice, "health": h, "stream": stream, "turns": rows, "heami_preview": heami}
    OUT.mkdir(parents=True, exist_ok=True)
    name = "app_selftest" + (f"_{a.label}" if a.label else "")
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    def med(key, kind=None):
        v = [r["timings"][key] for r in rows if r["timings"].get(key) is not None and (kind is None or r["kind"] == kind)]
        return statistics.median(v) if v else None

    L = ["# 음성 주문 앱 자체 시험", "",
         f"- 목소리: {voice}, 64e 예열 {h['llm_warmup']['wall_s']} s, iGPU 전부 올라감: {h['llm_warmup']['full_igpu']}",
         "- 글자 5턴은 한 대화(같은 세션), " + ("녹음 2개는 새 세션." if a.recordings == "two" else
                                         "녹음 20개는 하나씩 새 손님(새 세션)의 첫 턴."),
         f"- STT: {h.get('settings', {}).get('stt')}, 첫 문장 먼저 보내기(D3): {'켬' if stream else '끔'}. "
         "첫 소리 = 요청부터 첫 소리 준비까지(서버), 서버 합계 = 모든 소리 준비까지.", "",
         "| # | 입력 | 인식 | 도구 호출 | 답 | STT | 64e | TTS (첫 문장) | 첫 소리 | 서버 합계 | 답 길이 |",
         "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rows, 1):
        t = r["timings"]
        calls = "; ".join(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})" for c in r["tool_calls"]) or "(없음)"
        f = lambda v: "-" if v is None else f"{v:.2f} s"  # noqa: E731
        L.append(f"| {i} | {r['input']} | {r['transcript'] if r['kind'] == 'audio' else '(글자)'} | `{calls}` | {r['reply']} | "
                 f"{f(t['stt_s'])} | {f(t['llm_s'])} | {f(t['tts_s'])} | {f(t.get('first_audio_s'))} | {f(t['total_s'])} | "
                 f"{r['reply_audio_s']} s |")
    L += ["", f"- 마지막 주문 내역: {rows[len(TYPED) - 1]['order']}, 확정 {rows[len(TYPED) - 1]['confirmed']}"
               f" ({rows[len(TYPED) - 1]['dine_option']})",
          f"- 중앙값: 64e {med('llm_s'):.2f} s, TTS {med('tts_s'):.2f} s, STT {med('stt_s', 'audio'):.2f} s, "
          f"서버 합계 (글자) {med('total_s', 'text'):.2f} s / (음성) {med('total_s', 'audio'):.2f} s, "
          f"첫 소리 (글자) {med('first_audio_s', 'text'):.2f} s / (음성) {med('first_audio_s', 'audio'):.2f} s",
          f"- Heami 미리 듣기: 합성 {heami['tts_s']} s, 길이 {heami['audio'][0]} s ({heami['audio'][1]} Hz)"]
    sys.path.insert(0, str(VOICE))
    import bench_stt as B   # same CER as the STT benchmark
    au = [r for r in rows if r["kind"] == "audio"]
    err = sum(B.edit_distance(B.normalize(r["reference"]), B.normalize(r["transcript"])) for r in au)
    L.append(f"- 음성 {len(au)}개 인식 CER {100 * err / sum(len(B.normalize(r['reference'])) for r in au):.1f}%")
    (OUT / f"{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-3:]))


if __name__ == "__main__":
    main()
