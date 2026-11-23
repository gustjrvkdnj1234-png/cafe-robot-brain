"""STT, LLM and TTS engines for the voice demo (all local, CPU / iGPU)."""
from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

APP = Path(__file__).resolve().parent
VOICE = APP.parent
ROOT = VOICE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_latency import build_messages  # noqa: E402  (v1 Step 6c deploy layout)
from cafe_common import load_tools  # noqa: E402
from run_exam import MODE_OPTIONS, OPTIONS, tool_calls_of  # noqa: E402

THREADS = 8
OLLAMA = "http://127.0.0.1:11434"
MODEL = "song/gemma4-26b-cafe-64e"
LLM_OPTIONS = {**OPTIONS, **MODE_OPTIONS["igpu"], "num_ctx": 8192, "num_predict": 512}  # same as the v1 exams
KEEP_ALIVE = "60m"
ST3 = VOICE / "models" / "supertonic3-int8" / "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
FW_MEDIUM = VOICE / "models" / "fw-medium"


# ---------------------------------------------------------------- STT
class STT:
    def __init__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(FW_MEDIUM), device="cpu", compute_type="int8", cpu_threads=THREADS)
        self.transcribe(np.zeros(16000, dtype=np.float32))  # warm-up

    def transcribe(self, audio: np.ndarray) -> str:
        segs, _ = self.model.transcribe(audio, language="ko", beam_size=1, best_of=1, temperature=0.0,
                                        condition_on_previous_text=False, without_timestamps=True,
                                        vad_filter=False)
        return "".join(s.text for s in segs).strip()

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        """Any container PyAV can read (browser webm/opus, wav) -> 16 kHz mono float32.
        (faster_whisper.decode_audio passes metadata_errors=, which PyAV 19 no longer accepts.)"""
        import av
        out = []
        with av.open(io.BytesIO(data), mode="r") as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in container.decode(stream):
                for f in resampler.resample(frame):
                    out.append(f.to_ndarray().reshape(-1))
            for f in resampler.resample(None):  # flush
                out.append(f.to_ndarray().reshape(-1))
        if not out:
            return np.zeros(0, dtype=np.float32)
        return (np.concatenate(out).astype(np.float32) / 32768.0)


# ---------------------------------------------------------------- LLM
class LLM:
    def __init__(self):
        self.tools = load_tools()

    def chat(self, order_state: list, history: list, text: str) -> dict:
        msgs = build_messages("deploy", order_state, history + [{"role": "user", "content": text}])
        payload = {"model": MODEL, "messages": msgs, "stream": False, "think": False, "keep_alive": KEEP_ALIVE,
                   "tools": self.tools, "options": LLM_OPTIONS}
        t0 = time.perf_counter()
        r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=600)
        wall = time.perf_counter() - t0
        if r.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        d = r.json()
        msg = d.get("message") or {}
        return {"tool_calls": tool_calls_of(msg), "content": msg.get("content") or "", "wall_s": wall,
                "load_s": (d.get("load_duration") or 0) / 1e9,
                "prompt_tokens": d.get("prompt_eval_count"), "prompt_s": (d.get("prompt_eval_duration") or 0) / 1e9,
                "gen_tokens": d.get("eval_count"), "gen_s": (d.get("eval_duration") or 0) / 1e9,
                "done_reason": d.get("done_reason")}

    def warmup(self) -> dict:
        """Same as v1 bench_latency: deploy layout, empty order, one token."""
        msgs = build_messages("deploy", None, [{"role": "user", "content": "안녕하세요"}])
        t0 = time.perf_counter()
        r = requests.post(f"{OLLAMA}/api/chat", json={"model": MODEL, "messages": msgs, "stream": False, "think": False,
                                                      "keep_alive": KEEP_ALIVE, "tools": self.tools,
                                                      "options": {**LLM_OPTIONS, "num_predict": 1}}, timeout=900)
        r.raise_for_status()
        wall = round(time.perf_counter() - t0, 2)
        ps = requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", [])
        m = next((x for x in ps if x.get("name", "").startswith(MODEL)), {})
        # /api/ps "size" for this model is 1.25 GB (same value in v1 Step 5), so also report real memory
        from bench_baseline import gpu_memory, runner_memory
        rm, gm = runner_memory(), gpu_memory()
        return {"wall_s": wall, "size": m.get("size"), "size_vram": m.get("size_vram"),
                "full_igpu": bool(m.get("size") and m.get("size_vram", 0) >= m["size"]),
                "runner_private_gb": round(rm.get("runner_private_gb") or 0, 2),
                "gpu_shared_gb": round(gm.get("gpu_shared_gb") or 0, 2)}


# ---------------------------------------------------------------- TTS
def wav_bytes(samples, sr) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class Supertonic:
    def __init__(self):
        import sherpa_onnx
        d = ST3
        self._sherpa = sherpa_onnx
        self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor=str(d / "duration_predictor.int8.onnx"),
                text_encoder=str(d / "text_encoder.int8.onnx"),
                vector_estimator=str(d / "vector_estimator.int8.onnx"),
                vocoder=str(d / "vocoder.int8.onnx"), tts_json=str(d / "tts.json"),
                unicode_indexer=str(d / "unicode_indexer.bin"), voice_style=str(d / "voice.bin")),
            num_threads=THREADS, provider="cpu")))
        self.num_speakers = self.tts.num_speakers
        self.say("안녕하세요.", 0, 4)  # warm-up

    def say(self, text: str, sid: int, steps: int) -> bytes:
        g = self._sherpa.GenerationConfig()
        g.sid, g.num_steps, g.speed = int(sid), int(steps), 1.0
        g.extra["lang"] = "ko"
        a = self.tts.generate(text, g)
        return wav_bytes(a.samples, a.sample_rate)


class Heami:
    """Windows SAPI 'Microsoft Heami Desktop' through one long-running PowerShell process."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(APP / "heami_worker.ps1")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        first = self.proc.stdout.readline().strip()
        if not first.startswith("READY"):
            raise RuntimeError(f"Heami worker failed: {first}")
        self.voice = first[6:]
        self.tmp = Path(tempfile.mkdtemp(prefix="heami_"))

    def say(self, text: str) -> bytes:
        with self.lock:
            path = self.tmp / "out.wav"
            req = base64.b64encode(json.dumps({"text": text, "path": str(path)}, ensure_ascii=False)
                                   .encode("utf-8")).decode("ascii")
            self.proc.stdin.write(req + "\n")
            self.proc.stdin.flush()
            resp = self.proc.stdout.readline().strip()
            if not resp.startswith("OK"):
                raise RuntimeError(f"Heami: {resp}")
            return path.read_bytes()

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()
