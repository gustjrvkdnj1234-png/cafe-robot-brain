"""Repeat-run regression: is the 11-25 drop (cafe -2 / live -2) larger than run-to-run noise?

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\repeat_regression.py run --reps 3 --old-tree <worktree of 3b9b8e4>
    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\repeat_regression.py report

Configs (all CPU only, same model and options as demo_regression.py, deploy layout):
  v1        = v1 experiment prompt + tools (current code, v1 files are frozen)
  demo1124  = the 11-24 demo config, run from a git worktree of commit 3b9b8e4 (its own demo_regression.py)
  demo1125  = the current default (11-25 demo config, A3 = app)
Order: rep 1 (v1, demo1124, demo1125), rep 2 (...), rep 3 (...), each config cafe 120 then live 20, so
time drift does not line up with one config. Every run is resumable (finished ids are skipped).

Power: before and after every run the AC line state, battery %, active power plan and the effective power
mode overlay are written to demo_regression/repeat_power.jsonl. A run is not started on battery.
All runs are re-scored with the current scorer (inform / read_order calls removed).
"""
import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parents[2]
sys.path.insert(0, str(APP))
import demo_regression as DR  # noqa: E402

OUT = DR.OUT
PY = ROOT / ".venv-voice" / "Scripts" / "python.exe"
CONFIGS = [("v1", "v1 설정"), ("demo1124", "11-24 데모 설정"), ("demo1125", "11-25 기본 (A3=app)")]
EXTRA = [("demo1125_noro", "11-25 기본, read_order 도구·규칙 뺌"),
         ("demo1125_norule", "11-25 기본, read_order 도구는 두고 규칙만 뺌")]


def run_file(cfg: str, rep: int, exam: str) -> Path:
    return OUT / f"{cfg}_rep{rep}__{exam}__cpu.jsonl"


# ---------------------------------------------------------------- power state
class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


OVERLAYS = {"00000000-0000-0000-0000-000000000000": "balanced (default)",
            "961cc777-2547-4f9d-8174-7d86181b8a7a": "best power efficiency",
            "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "better performance",
            "ded574b5-45a0-4f42-8737-46345c09c238": "best performance"}


def power() -> dict:
    s = SYSTEM_POWER_STATUS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "ac": s.ACLineStatus == 1, "battery_pct": s.BatteryLifePercent}
    try:
        g = ctypes.c_byte * 16
        buf = g()
        if ctypes.windll.powrprof.PowerGetEffectiveOverlayScheme(ctypes.byref(buf)) == 0:
            ov = str(uuid.UUID(bytes_le=bytes(b & 0xFF for b in buf)))
            out["power_mode"] = OVERLAYS.get(ov, ov)
    except (AttributeError, OSError):
        out["power_mode"] = "unknown"
    try:
        p = ctypes.c_void_p()
        if ctypes.windll.powrprof.PowerGetActiveScheme(None, ctypes.byref(p)) == 0:
            out["plan"] = str(uuid.UUID(bytes_le=ctypes.string_at(p, 16)))
            ctypes.windll.kernel32.LocalFree(p)
    except (AttributeError, OSError):
        pass
    return out


def log_power(event: str, **kw) -> dict:
    rec = {"event": event, **kw, **power()}
    with open(OUT / "repeat_power.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


# ---------------------------------------------------------------- runs
def run_one(cfg: str, rep: int, exam: str, old_tree: Path):
    dst = run_file(cfg, rep, exam)
    if dst.exists() and sum(1 for _ in open(dst, encoding="utf-8")) >= (120 if exam == "cafe" else 20):
        return
    p = log_power("start", config=cfg, rep=rep, exam=exam)
    if not p["ac"]:
        sys.exit(f"on battery ({p}); plug in the charger and rerun (finished runs are kept)")
    t = time.time()
    if cfg == "demo1124":
        tree_out = old_tree / "v2/voice/results/demo_regression"
        tmp = tree_out / f"demo__{exam}__cpu.jsonl"      # the 11-24 script has no --tag: move its output after the run
        if tmp.exists():
            tmp.unlink()                                  # (worktree copy of the committed 11-24 file, or a stale part)
        if dst.exists():
            shutil.copy(dst, tmp)                         # resume a partial run
        cmd = [str(PY), str(old_tree / "v2/voice/app/demo_regression.py"), "run", "--config", "demo", "--exam", exam,
               "--mode", "cpu"]
        rc = subprocess.run(cmd).returncode
        shutil.copy(tmp, dst)
    else:
        config = "v1" if cfg == "v1" else "demo"
        tag = f"{cfg.replace('demo', '')}_rep{rep}".lstrip("_") if cfg != "v1" else f"rep{rep}"
        cmd = [str(PY), str(APP / "demo_regression.py"), "run", "--config", config, "--exam", exam, "--mode", "cpu",
               "--tag", tag, "--a3", "app"] + (["--no-read-order"] if cfg.endswith("_noro") else []) + \
            (["--no-read-order-rule"] if cfg.endswith("_norule") else [])
        rc = subprocess.run(cmd).returncode
        made = OUT / f"{config}_{tag}__{exam}__cpu.jsonl"
        if made != dst:
            made.replace(dst)
    log_power("end", config=cfg, rep=rep, exam=exam, rc=rc, wall_min=round((time.time() - t) / 60, 1))
    if rc != 0:
        sys.exit(f"run failed: {cfg} rep{rep} {exam} rc={rc}")


def run(reps: int, old_tree: Path, configs: list):
    for rep in range(1, reps + 1):
        for cfg in configs:
            for exam in ("cafe", "live"):
                print(f"==== {cfg} rep{rep} {exam}", flush=True)
                run_one(cfg, rep, exam, old_tree)


# ---------------------------------------------------------------- report
def load_run(cfg, rep, exam, cases):
    p = run_file(cfg, rep, exam)
    if not p.exists():
        return None
    rows = {r["id"]: r for r in DR.load_jsonl(p)}
    if len(rows) < len(cases):
        return None
    return {i: {**r, **DR.score(cases[i], r["tool_calls"])} for i, r in rows.items()}


def calls_key(r) -> str:
    return json.dumps(r["tool_calls"], ensure_ascii=False, sort_keys=True)


def report():
    cases = {e: {c["id"]: c for c in DR.load_jsonl(DR.EXAMS[e])} for e in DR.EXAMS}
    L = ["# 반복 회귀 시험 (CPU 전용, 64e, 카페 120 + 실전 20, 설정마다 3번)", "",
         "- 같은 모델·옵션(thinking 끔, temperature 0, seed 42, num_ctx 8192), 배포용 배치, 충전기 연결.",
         "- 순서: 1회차(v1 → 11-24 → 11-25) → 2회차 → 3회차. 채점은 모두 지금 채점기(inform·read_order 호출은 빼고 v1 엄격).",
         "- 11-24 데모 설정은 그 커밋(3b9b8e4)의 코드를 그대로 돌렸다. 11-25 기본과 시험용 프롬프트 차이는 read_order 도구와 "
         "규칙 2줄뿐이다(이 둘을 빼면 글자까지 같다).", ""]
    summary = {}
    power_rows = [json.loads(x) for x in open(OUT / "repeat_power.jsonl", encoding="utf-8")] \
        if (OUT / "repeat_power.jsonl").exists() else []
    for exam in ("cafe", "live"):
        title = "카페 120" if exam == "cafe" else "실전 20"
        n = len(cases[exam])
        L += [f"## {title}", "", "| 설정 | 1회 | 2회 | 3회 | 평균 | 최소–최대 | 회차마다 바뀐 문항 | 도구 호출이 달라진 문항 |",
              "|---|---:|---:|---:|---:|---:|---|---:|"]
        runs = {}
        for cfg, label in CONFIGS + EXTRA:
            rs = [load_run(cfg, k, exam, cases[exam]) for k in (1, 2, 3)]
            rs = [r for r in rs if r]
            if not rs:
                continue
            runs[cfg] = rs
            st = [sum(r[i]["strict"] for i in r) for r in rs]
            flips = sorted(i for i in cases[exam] if len({r[i]["strict"] for r in rs}) > 1)
            call_diff = sum(len({calls_key(r[i]) for r in rs}) > 1 for i in cases[exam])
            mean = sum(st) / len(st)
            summary[f"{exam}_{cfg}"] = {"strict": st, "mean": mean, "min": min(st), "max": max(st), "flips": flips,
                                        "calls_differ": call_diff}
            cells = [str(x) for x in st] + ["-"] * (3 - len(st))
            fl = ", ".join(f"{i}({''.join('O' if r[i]['strict'] else 'X' for r in rs)})" for i in flips) or "없음"
            L.append(f"| {label} | " + " | ".join(cells) + f" | {mean:.1f} | {min(st)}–{max(st)} | {fl} | {call_diff} |")
        L.append("")
        if "demo1124" in runs and "demo1125" in runs:
            a, b = runs["demo1124"], runs["demo1125"]
            always_a = {i for i in cases[exam] if all(r[i]["strict"] for r in a)}
            never_a = {i for i in cases[exam] if not any(r[i]["strict"] for r in a)}
            lost = sorted(i for i in always_a if not any(r[i]["strict"] for r in b))
            gained = sorted(i for i in never_a if all(r[i]["strict"] for r in b))
            summary[f"{exam}_lost_every_run"] = lost
            summary[f"{exam}_gained_every_run"] = gained
            L += [f"- 11-24에서 3번 다 정답 → 11-25에서 3번 다 오답: {', '.join(lost) or '없음'}",
                  f"- 11-24에서 3번 다 오답 → 11-25에서 3번 다 정답: {', '.join(gained) or '없음'}", ""]
            if exam == "cafe" and lost:
                L += ["| id | 유형 | 11-24 도구 호출 (1회) | 11-25 도구 호출 (1회) |", "|---|---|---|---|"]
                fmt = lambda r: "; ".join(f"{x['name']}({json.dumps(x['arguments'], ensure_ascii=False)})"  # noqa: E731
                                          for x in r["tool_calls"]) or "(없음)"
                for i in lost:
                    L.append(f"| {i} | {cases[exam][i]['category']} | `{fmt(a[0][i])[:150]}` | `{fmt(b[0][i])[:150]}` |")
                L.append("")
            if exam == "live" and lost:
                L += ["| id | 유형 | 11-24 도구 (1회) | 11-25 도구 (1회) |", "|---|---|---|---|"]
                for i in lost:
                    L.append(f"| {i} | {cases[exam][i]['category']} | {', '.join(x['name'] for x in a[0][i]['tool_calls']) or '(없음)'} | "
                             f"{', '.join(x['name'] for x in b[0][i]['tool_calls']) or '(없음)'} |")
                L.append("")
    if power_rows:
        ac = all(p["ac"] for p in power_rows)
        modes = sorted({p.get("power_mode", "?") for p in power_rows})
        plans = sorted({p.get("plan", "?") for p in power_rows})
        mins = [p["wall_min"] for p in power_rows if p["event"] == "end"]
        L += ["## 전원 상태", "", f"- 시작·끝 기록 {len(power_rows)}번: 충전기 연결 {'항상' if ac else '아님(확인 필요)'}, "
              f"전원 모드 {', '.join(modes)}, 전원 계획 {', '.join(plans)}.",
              f"- 실행 시간: 시험 한 번(카페 120 또는 실전 20)에 {min(mins):.1f}–{max(mins):.1f}분." if mins else "", ""]
        summary["power"] = {"always_ac": ac, "modes": modes, "plans": plans}
    (OUT / "summary_repeat.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT.parent / "demo_regression_repeat.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "power"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--old-tree", type=Path, help="git worktree of commit 3b9b8e4 (11-24 demo config)")
    ap.add_argument("--configs", default="v1,demo1124,demo1125")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "power":
        print(json.dumps(power(), ensure_ascii=False))
    elif a.cmd == "run":
        cfgs = a.configs.split(",")
        if "demo1124" in cfgs and not (a.old_tree and (a.old_tree / "v2/voice/app/demo_regression.py").exists()):
            sys.exit("--old-tree must point to a worktree of 3b9b8e4")
        run(a.reps, a.old_tree, cfgs)
    else:
        report()


if __name__ == "__main__":
    main()
