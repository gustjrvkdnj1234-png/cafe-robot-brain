"""Local voice ordering demo: push-to-talk -> faster-whisper medium -> 64e (Ollama) -> Supertonic 3 / Heami.

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\server.py [--port 8770]

Normally started by start_demo.ps1 (which starts Ollama with the right environment first).
Binds to 127.0.0.1 only. One turn at a time (a global lock), sessions kept in memory.
Logs each turn to v2/voice/app/logs/turns_YYYYMMDD.jsonl (not committed, no audio saved).
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import engines  # noqa: E402
import replies  # noqa: E402

LOGS = APP / "logs"
CONFIG = APP / "config.local.json"
DEFAULT_VOICE = {"engine": "supertonic", "sid": 0, "steps": 4}
MAX_AUDIO_BYTES = 10 * 1024 * 1024

E: dict = {}
STATUS = {"stt": False, "tts": False, "heami": False, "llm": False, "llm_warmup": None, "errors": []}
SESSIONS: dict = {}
TURN_LOCK = threading.Lock()


# ---------------------------------------------------------------- config
def load_voice() -> dict:
    try:
        v = json.loads(CONFIG.read_text(encoding="utf-8"))
        return clean_voice({**DEFAULT_VOICE, **v})
    except (OSError, ValueError):
        return dict(DEFAULT_VOICE)


def clean_voice(v: dict) -> dict:
    eng = v.get("engine") if v.get("engine") in ("supertonic", "heami") else "supertonic"
    try:
        sid = min(9, max(0, int(v.get("sid", 0))))
    except (TypeError, ValueError):
        sid = 0
    steps = 8 if str(v.get("steps")) == "8" else 4
    return {"engine": eng, "sid": sid, "steps": steps}


def save_voice(v: dict):
    CONFIG.write_text(json.dumps(v, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- startup
def warm_llm():
    try:
        STATUS["llm_warmup"] = E["llm"].warmup()
        STATUS["llm"] = True
    except Exception as e:  # noqa: BLE001
        STATUS["errors"].append(f"LLM warm-up: {e}")


@asynccontextmanager
async def lifespan(app):
    t0 = time.perf_counter()
    E["llm"] = engines.LLM()
    threading.Thread(target=warm_llm, daemon=True).start()   # loads the model on the iGPU meanwhile
    E["stt"] = engines.STT()
    STATUS["stt"] = True
    E["tts"] = engines.Supertonic()
    STATUS["tts"] = True
    try:
        E["heami"] = engines.Heami()
        STATUS["heami"] = True
    except Exception as e:  # noqa: BLE001
        STATUS["errors"].append(f"Heami: {e}")
    STATUS["startup_s"] = round(time.perf_counter() - t0, 1)
    print(f"engines ready in {STATUS['startup_s']} s (LLM warm-up continues in background)", flush=True)
    yield
    if "heami" in E:
        E["heami"].close()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------- core
# Conversation settings (command-line flags; defaults = the fixed behaviour).
SETTINGS = {"history": "tools",   # tools = assistant tool_calls + tool result messages; text = old plain text
            "layout": "append",   # append = append-only prompt; deploy = v1 6c layout (first app version)
            "max_turns": 20,      # keep at most this many past turns (0 = no cap) ...
            "trim_to": 10}        # ... and when exceeded cut back to this many. Trimming forces the history to be
                                  # recomputed (404 tokens = 10.9 s on CPU at 6->3), so the cap is set above a usual order.


def new_session() -> dict:
    return {"state": [], "turns": [], "confirmed": False, "dine_option": None, "staff": [], "n": 0, "ordered": False}


def assistant_messages(calls: list, r: dict) -> list:
    """The finished turn as chat messages for the next request."""
    if SETTINGS["history"] == "text":
        return [{"role": "assistant", "content": r["legacy_history"]}]
    if not calls:
        return [{"role": "assistant", "content": r["spoken"]}]
    msgs = [{"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": c["name"], "arguments": c.get("arguments") or {}}} for c in calls]}]
    for k, (c, res) in enumerate(zip(calls, r["tool_results"])):
        if k == len(calls) - 1:
            res = {**res, "robot_said": r["spoken"]}
        msgs.append({"role": "tool", "tool_name": c["name"], "content": json.dumps(res, ensure_ascii=False)})
    return msgs


def get_session(sid: str) -> dict:
    if not sid or len(sid) > 64:
        raise HTTPException(400, "bad session id")
    return SESSIONS.setdefault(sid, new_session())


def synthesize(text: str, voice: dict) -> tuple[bytes, float]:
    t = time.perf_counter()
    if voice["engine"] == "heami" and "heami" in E:
        data = E["heami"].say(text)
    else:
        data = E["tts"].say(text, voice["sid"], voice["steps"])
    return data, time.perf_counter() - t


def sample_rate(data_url):
    """Sample rate of the WAV sent to the browser (no resampling in the app)."""
    if not data_url:
        return None
    import io
    import soundfile as sf
    return sf.info(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).samplerate


def log_turn(rec: dict):
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / f"turns_{datetime.now():%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_turn(session_id: str, text: str, t_start: float, input_kind: str, stt_s: float | None,
             audio_s: float | None) -> dict:
    s = get_session(session_id)
    voice = load_voice()
    llm_info, error, n_msgs = None, None, None
    if not text.strip():
        calls, content, llm_s = [], "", None
        r = {"state": s["state"], "spoken": replies.T()["no_speech"], "kinds": ["no_speech"], "confirmed": False,
             "dine_option": None, "staff_reason": None, "tool_results": [], "legacy_history": "", "blocked": None}
    else:
        msgs = engines.build_prompt(SETTINGS["layout"], s["state"], s["turns"], text)
        n_msgs = len(msgs)
        state_msg = engines.split_prompt(s["state"])[1]
        try:
            llm_info = E["llm"].chat(msgs)
            STATUS["llm"] = True
            calls, content, llm_s = llm_info["tool_calls"], llm_info["content"], llm_info["wall_s"]
        except Exception as e:  # noqa: BLE001
            calls, content, llm_s, error = [], "", None, f"LLM: {e}"
        r = replies.build_reply(calls, content, s["state"], first_item_possible=not s["ordered"])
        if "add" in r["kinds"]:
            s["ordered"] = True
        s["turns"].append({"state_msg": state_msg, "user": {"role": "user", "content": text},
                           "assistant": assistant_messages(calls, r)})
        if SETTINGS["max_turns"] and len(s["turns"]) > SETTINGS["max_turns"]:
            s["turns"] = s["turns"][-SETTINGS["trim_to"]:]
    s["state"] = r["state"]
    if r["confirmed"]:
        s["confirmed"], s["dine_option"] = True, r["dine_option"]
    if r["staff_reason"]:
        s["staff"].append(r["staff_reason"])
    s["n"] += 1
    audio, tts_s = None, None
    try:
        wav, tts_s = synthesize(r["spoken"], voice)
        audio = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
    except Exception as e:  # noqa: BLE001
        error = (error + " / " if error else "") + f"TTS: {e}"
    total = time.perf_counter() - t_start
    timings = {"stt_s": stt_s, "llm_s": llm_s, "tts_s": tts_s, "total_s": total, "audio_in_s": audio_s}
    if llm_info:
        timings.update({k: llm_info[k] for k in ("load_s", "prompt_tokens", "prompt_s", "gen_tokens", "gen_s")})
    timings = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in timings.items()}
    timings["prompt_messages"] = n_msgs
    timings["history_turns"] = len(s["turns"])
    out = {"session": session_id, "turn": s["n"], "input": input_kind, "transcript": text, "settings": dict(SETTINGS),
           "tool_calls": calls, "content": content, "reply": r["spoken"], "kinds": r["kinds"],
           "order": replies.order_lines(s["state"]), "order_state": s["state"], "confirmed": s["confirmed"],
           "dine_option": s["dine_option"], "staff_called": bool(r["staff_reason"]),
           "staff_reason": r["staff_reason"], "timings": timings, "voice": voice, "error": error}
    log_turn({"time": datetime.now().isoformat(timespec="seconds"),
              **{k: v for k, v in out.items() if k not in ("order_state",)},
              "history_messages": s["turns"][-1]["assistant"] if text.strip() and s["turns"] else None,
              "blocked_free_text": r.get("blocked"), "run": dict(engines.RUN),
              "done_reason": llm_info and llm_info.get("done_reason")})
    out["audio"] = audio
    out["tts_sample_rate"] = sample_rate(audio)
    return out


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(APP / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/app.js")
def app_js():
    return FileResponse(APP / "static" / "app.js", media_type="text/javascript; charset=utf-8",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    llm_up = False
    try:
        llm_up = requests.get(f"{engines.OLLAMA}/api/version", timeout=2).ok
    except requests.RequestException:
        pass
    return {**STATUS, "ollama_up": llm_up, "model": engines.MODEL, "settings": {**SETTINGS, **engines.RUN},
            "ready": STATUS["stt"] and STATUS["tts"] and STATUS["llm"],
            "speakers": getattr(E.get("tts"), "num_speakers", 10)}


@app.get("/api/config")
def get_config():
    return load_voice()


@app.post("/api/config")
async def set_config(req: Request):
    v = clean_voice({**load_voice(), **(await req.json())})
    save_voice(v)
    return v


@app.post("/api/settings")
async def set_settings(req: Request):
    """Change the conversation settings at run time (used for same-server A/B comparisons).
    Applies to turns that follow; existing sessions keep their stored history."""
    body = await req.json()
    with TURN_LOCK:
        if body.get("history") in ("tools", "text"):
            SETTINGS["history"] = body["history"]
        if body.get("layout") in ("append", "deploy"):
            SETTINGS["layout"] = body["layout"]
        for k in ("max_turns", "trim_to"):
            if k in body:
                SETTINGS[k] = max(0, int(body[k]))
        if body.get("config") in ("demo", "v1"):
            engines.RUN["config"] = body["config"]
        if body.get("mode") in ("igpu", "cpu"):
            engines.RUN["mode"] = body["mode"]
        if "num_ctx" in body:
            engines.RUN["num_ctx"] = int(body["num_ctx"])
    return {**SETTINGS, **engines.RUN}


@app.post("/api/new_session")
async def new_session_route(req: Request):
    """New customer: drop the previous session completely (order, history, staff/confirm flags)."""
    try:
        old = (await req.json()).get("previous")
    except ValueError:
        old = None
    with TURN_LOCK:
        if old:
            SESSIONS.pop(str(old), None)
        sid = uuid.uuid4().hex
        SESSIONS[sid] = new_session()
    return {"session": sid, "reset": bool(old)}


@app.post("/api/turn/text")
async def turn_text(req: Request):
    t0 = time.perf_counter()
    body = await req.json()
    text = str(body.get("text") or "").strip()[:300]
    with TURN_LOCK:
        return JSONResponse(run_turn(str(body.get("session")), text, t0, "text", None, None))


@app.post("/api/turn/audio")
async def turn_audio(req: Request, session: str):
    t0 = time.perf_counter()
    data = await req.body()
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(400, "empty or too large audio")
    with TURN_LOCK:
        ts = time.perf_counter()
        try:
            audio = engines.STT.decode(data)
            text = E["stt"].transcribe(audio)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"audio decode/STT failed: {e}") from e
        stt_s = time.perf_counter() - ts
        return JSONResponse(run_turn(session, text, t0, "audio", stt_s, len(audio) / 16000))


@app.post("/api/tts/preview")
async def tts_preview(req: Request):
    body = await req.json()
    voice = clean_voice({**load_voice(), **body})
    text = str(body.get("text") or "네! 아이스 아메리카노 한 잔 추가했어요. 더 필요하신 거 있으세요?")[:200]
    with TURN_LOCK:
        wav, s = synthesize(text, voice)
    return {"audio": "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii"), "tts_s": round(s, 3),
            "voice": voice}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--history", choices=["tools", "text"], default=SETTINGS["history"],
                    help="tools = assistant tool_calls + tool results (default); text = first version's plain text")
    ap.add_argument("--layout", choices=["append", "deploy"], default=SETTINGS["layout"],
                    help="append = append-only prompt (default); deploy = v1 6c layout (first version)")
    ap.add_argument("--max-turns", type=int, default=SETTINGS["max_turns"], help="0 = keep all past turns")
    ap.add_argument("--trim-to", type=int, default=SETTINGS["trim_to"])
    ap.add_argument("--config", choices=["demo", "v1"], default=engines.RUN["config"],
                    help="demo = demo prompt + inform tool (default); v1 = experiment prompt/tools")
    ap.add_argument("--mode", choices=["igpu", "cpu"], default=engines.RUN["mode"], help="cpu = num_gpu 0")
    ap.add_argument("--num-ctx", type=int, default=engines.RUN["num_ctx"])
    a = ap.parse_args()
    engines.RUN.update(config=a.config, mode=a.mode, num_ctx=a.num_ctx)
    SETTINGS.update(history=a.history, layout=a.layout, max_turns=a.max_turns, trim_to=min(a.trim_to, a.max_turns or a.trim_to))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
