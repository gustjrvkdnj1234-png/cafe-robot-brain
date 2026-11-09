"""1단계 기준 모델 벤치마크: CPU 전용 vs iGPU 모드의 생성 속도와 메모리 측정.

사전 조건: iGPU를 쓰려면 서버를 OLLAMA_IGPU_ENABLE=1 로 띄워야 한다.
CPU 모드는 같은 서버에서 요청 옵션 num_gpu=0 으로 강제한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\bench_baseline.py
    .venv\\Scripts\\python.exe scripts\\bench_baseline.py --models gemma4:e4b --modes igpu cpu
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
HOST = "http://127.0.0.1:11434"
GB = 1024**3

DEFAULT_MODELS = ["gemma4:e4b", "gemma4:12b", "gemma4:26b"]
DEFAULT_MODES = ["igpu", "cpu"]
PROMPTS = {
    "ko": "카페 아르바이트생이 손님 주문을 받을 때 주의할 점을 5가지 설명해줘.",
    "en": "Explain 5 things a cafe part-time worker should be careful about when taking customer orders.",
}
BASE_OPTIONS = {"temperature": 0, "num_predict": 256, "num_ctx": 8192, "seed": 42}
MODE_OPTIONS = {"igpu": {}, "cpu": {"num_gpu": 0}}
REPEATS = 3
TIMEOUT_S = 1800


# ---------- 측정 도우미 ----------

def ns_to_s(ns):
    return None if ns is None else ns / 1e9


def tok_per_s(count, dur_ns):
    if not count or not dur_ns:
        return None
    return count / (dur_ns / 1e9)


def ollama_ps() -> dict:
    """`ollama ps` 텍스트를 파싱해 {모델명: {size, processor, raw}}."""
    try:
        out = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, encoding="utf-8", timeout=30
        ).stdout
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return {}
    header = lines[0]
    cols = ["NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "RUNNER", "UNTIL"]
    order = sorted(((c, header.find(c)) for c in cols if header.find(c) >= 0), key=lambda kv: kv[1])
    result = {}
    for line in lines[1:]:
        row = {}
        for i, (c, s) in enumerate(order):
            e = order[i + 1][1] if i + 1 < len(order) else None
            row[c] = (line[s:e] if e else line[s:]).strip()
        result[row.get("NAME") or line.split()[0]] = {
            "size": row.get("SIZE"),
            "processor": row.get("PROCESSOR"),
            "raw": line.strip(),
        }
    return result


def api_ps() -> dict:
    try:
        r = requests.get(f"{HOST}/api/ps", timeout=30)
        r.raise_for_status()
        return {m["name"]: m for m in r.json().get("models", [])}
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}


def runner_memory() -> dict:
    """ollama serve 의 자식 프로세스(llama-server 등) 메모리 합계."""
    rss = priv = 0
    names = []
    for p in psutil.process_iter(["name"]):
        if p.info["name"] != "ollama.exe":
            continue
        try:
            for c in p.children(recursive=True):
                if c.name().lower() == "conhost.exe":
                    continue
                mi = c.memory_info()
                rss += mi.rss
                priv += getattr(mi, "private", 0)
                names.append(c.name())
        except psutil.Error:
            pass
    return {"runner_rss_gb": rss / GB, "runner_private_gb": priv / GB, "runner_names": names}


def gpu_memory() -> dict:
    """Windows 성능 카운터로 GPU 어댑터 공유/전용 메모리 사용량(가장 큰 어댑터)."""
    ps = (
        "$c=Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage','\\GPU Adapter Memory(*)\\Dedicated Usage' "
        "-ErrorAction SilentlyContinue; $c.CounterSamples | ForEach-Object { $_.Path + '|' + $_.CookedValue }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception as e:  # noqa: BLE001
        return {"gpu_mem_error": str(e)}
    shared, dedicated = {}, {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        path, val = line.rsplit("|", 1)
        try:
            v = float(val)
        except ValueError:
            continue
        adapter = path.split("(")[1].split(")")[0] if "(" in path else path
        (shared if path.lower().endswith("shared usage") else dedicated)[adapter] = v
    return {
        "gpu_shared_gb": (max(shared.values()) / GB) if shared else None,
        "gpu_dedicated_gb": (max(dedicated.values()) / GB) if dedicated else None,
    }


def system_memory() -> dict:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "sys_used_gb": (vm.total - vm.available) / GB,
        "sys_avail_gb": vm.available / GB,
        "pagefile_used_gb": sw.used / GB,
    }


# ---------- 모델 제어 ----------

def unload(model: str) -> None:
    try:
        requests.post(f"{HOST}/api/generate", json={"model": model, "keep_alive": 0}, timeout=120)
    except Exception:  # noqa: BLE001
        pass
    for _ in range(60):
        if model not in api_ps():
            break
        time.sleep(1)
    time.sleep(2)  # 메모리 반환 여유


def unload_all() -> None:
    for name in [n for n in api_ps() if not n.startswith("_")]:
        unload(name)


def generate(model: str, prompt: str, options: dict) -> dict:
    t0 = time.perf_counter()
    r = requests.post(
        f"{HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": options},
        timeout=TIMEOUT_S,
    )
    wall = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    d["_wall_s"] = wall
    return d


# ---------- 실행 ----------

def run(models, modes) -> list[dict]:
    records: list[dict] = []
    unload_all()
    for model in models:
        for mode in modes:
            options = {**BASE_OPTIONS, **MODE_OPTIONS[mode]}
            print(f"\n=== {model} [{mode}] ===", flush=True)
            idle = system_memory()
            first_call = True
            for lang, prompt in PROMPTS.items():
                for i in range(REPEATS):
                    rec = {
                        "model": model,
                        "mode": mode,
                        "lang": lang,
                        "run": i + 1,
                        "cold": first_call,          # 모델·모드별 첫 호출(로딩 포함)
                        "first_of_prompt": i == 0,   # 프롬프트 캐시 없는 회차
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "sys_used_gb_before_load": idle["sys_used_gb"],
                    }
                    try:
                        d = generate(model, prompt, options)
                        response = d.get("response") or ""
                        thinking = d.get("thinking") or ""
                        rec.update(
                            {
                                "ok": True,
                                "load_s": ns_to_s(d.get("load_duration")),
                                "total_s": ns_to_s(d.get("total_duration")),
                                "wall_s": d["_wall_s"],
                                "prompt_eval_count": d.get("prompt_eval_count"),
                                "prompt_eval_s": ns_to_s(d.get("prompt_eval_duration")),
                                "prompt_tps": tok_per_s(d.get("prompt_eval_count"), d.get("prompt_eval_duration")),
                                "eval_count": d.get("eval_count"),
                                "eval_s": ns_to_s(d.get("eval_duration")),
                                "gen_tps": tok_per_s(d.get("eval_count"), d.get("eval_duration")),
                                "done_reason": d.get("done_reason"),
                                "thinking_chars": len(thinking),
                                "response_chars": len(response),
                                "response_empty": not response.strip(),
                                "thinking_head": thinking[:200],
                                "response_head": response[:200],
                            }
                        )
                    except Exception as e:  # noqa: BLE001
                        rec.update({"ok": False, "error": repr(e)})

                    ps_txt = ollama_ps().get(model, {})
                    ps_api = api_ps().get(model, {})
                    size_b, vram_b = ps_api.get("size"), ps_api.get("size_vram")
                    rec.update(
                        {
                            "ps_size": ps_txt.get("size"),
                            "ps_processor": ps_txt.get("processor"),
                            "ps_raw": ps_txt.get("raw"),
                            "ps_mem_gb": size_b / GB if size_b else None,
                            "ps_vram_gb": vram_b / GB if vram_b is not None else None,
                            "ps_gpu_ratio": (vram_b / size_b) if size_b and vram_b is not None else None,
                            **runner_memory(),
                            **gpu_memory(),
                            **system_memory(),
                        }
                    )
                    rec["sys_delta_gb"] = rec["sys_used_gb"] - rec["sys_used_gb_before_load"]
                    records.append(rec)
                    first_call = False
                    if rec["ok"]:
                        print(
                            f"  [{lang} #{i+1}{' cold' if rec['cold'] else ''}] "
                            f"load {rec['load_s']:.2f}s | prompt {rec['prompt_tps'] or 0:.1f} tok/s | "
                            f"gen {rec['gen_tps'] or 0:.1f} tok/s ({rec['eval_count']} tok) | "
                            f"ps {rec['ps_size']} {rec['ps_processor']} | "
                            f"runner {rec['runner_private_gb']:.2f}GB | gpu-shared {fmt(rec.get('gpu_shared_gb'),2)}GB | "
                            f"sys avail {rec['sys_avail_gb']:.1f}GB pagefile {rec['pagefile_used_gb']:.1f}GB",
                            flush=True,
                        )
                    else:
                        print(f"  [{lang} #{i+1}] ERROR {rec['error']}", flush=True)
            unload(model)
    return records


# ---------- 요약/출력 ----------

def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def mx(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def pct(x):
    return "-" if x is None else f"{x*100:.0f}%"


def summarize(records) -> list[dict]:
    out = []
    keys = dict.fromkeys((r["model"], r["mode"]) for r in records)
    for model, mode in keys:
        rs = [r for r in records if r["model"] == model and r["mode"] == mode]
        ok = [r for r in rs if r["ok"]]
        warm = [r for r in ok if not r["cold"]]
        cold = next((r for r in ok if r["cold"]), None)
        out.append(
            {
                "model": model,
                "mode": mode,
                "runs_ok": len(ok),
                "runs_err": len(rs) - len(ok),
                "cold_load_s": cold["load_s"] if cold else None,
                "gen_tps_warm_mean": mean(r["gen_tps"] for r in warm),
                "gen_tps_ko": mean(r["gen_tps"] for r in warm if r["lang"] == "ko"),
                "gen_tps_en": mean(r["gen_tps"] for r in warm if r["lang"] == "en"),
                "prompt_tps_first_mean": mean(r["prompt_tps"] for r in ok if r["first_of_prompt"]),
                "ps_mem_gb": mx(r.get("ps_mem_gb") for r in ok),
                "ps_gpu_ratio": mean(r.get("ps_gpu_ratio") for r in ok),
                "processor": next((r["ps_processor"] for r in ok if r.get("ps_processor")), None),
                "runner_private_gb": mx(r.get("runner_private_gb") for r in ok),
                "runner_rss_gb": mx(r.get("runner_rss_gb") for r in ok),
                "gpu_shared_gb": mx(r.get("gpu_shared_gb") for r in ok),
                "sys_delta_gb": mx(r.get("sys_delta_gb") for r in ok),
                "min_sys_avail_gb": min((r["sys_avail_gb"] for r in ok), default=None),
                "max_pagefile_gb": mx(r.get("pagefile_used_gb") for r in ok),
                "empty_responses": sum(1 for r in ok if r.get("response_empty")),
            }
        )
    return out


def write_md(records, summary, meta, path: Path):
    by = {(s["model"], s["mode"]): s for s in summary}
    models = list(dict.fromkeys(s["model"] for s in summary))
    L = [
        "# 1단계 기준 모델 벤치마크 (CPU vs iGPU)",
        "",
        f"- 측정: {meta['started']} ~ {meta['finished']}",
        f"- 환경: {meta['platform']} / ollama {meta['ollama_version']} / 서버 OLLAMA_IGPU_ENABLE=1",
        f"- 공통 옵션: `{json.dumps(BASE_OPTIONS)}`, CPU 모드는 `num_gpu=0` 추가 / 프롬프트(ko, en)당 {REPEATS}회",
        "- 생성 속도 평균은 모델·모드별 첫 호출(로딩 포함, cold)을 제외",
        "- 메모리: `ollama ps` 값은 iGPU 공유 메모리 모드에서 실제보다 작게 나올 수 있어 러너 프로세스(llama-server) private 메모리와 GPU 공유 메모리 카운터를 함께 기록",
        "- gemma4는 기본적으로 thinking을 출력하므로 생성 토큰(eval_count)에 thinking 토큰이 포함됨",
        "",
        "## CPU vs iGPU 비교",
        "",
        "| 모델 | 생성 tok/s CPU | 생성 tok/s iGPU | 배율 | 프롬프트 tok/s CPU | 프롬프트 tok/s iGPU | 로딩(s) CPU | 로딩(s) iGPU | 러너 메모리 GB CPU | 러너 메모리 GB iGPU | GPU 공유 GB iGPU | ollama ps CPU | ollama ps iGPU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for m in models:
        c, g = by.get((m, "cpu"), {}), by.get((m, "igpu"), {})
        ratio = (
            g["gen_tps_warm_mean"] / c["gen_tps_warm_mean"]
            if g.get("gen_tps_warm_mean") and c.get("gen_tps_warm_mean") else None
        )
        L.append(
            f"| {m} | {fmt(c.get('gen_tps_warm_mean'))} | {fmt(g.get('gen_tps_warm_mean'))} | "
            f"{'-' if ratio is None else f'{ratio:.2f}x'} | {fmt(c.get('prompt_tps_first_mean'))} | "
            f"{fmt(g.get('prompt_tps_first_mean'))} | {fmt(c.get('cold_load_s'),1)} | {fmt(g.get('cold_load_s'),1)} | "
            f"{fmt(c.get('runner_private_gb'),2)} | {fmt(g.get('runner_private_gb'),2)} | {fmt(g.get('gpu_shared_gb'),2)} | "
            f"{fmt(c.get('ps_mem_gb'),2)}GB {c.get('processor') or '-'} | {fmt(g.get('ps_mem_gb'),2)}GB {g.get('processor') or '-'} |"
        )
    L += [
        "",
        "## 모드별 요약",
        "",
        "| 모델 | 모드 | 생성 tok/s | 한국어 | 영어 | 시스템 RAM 증가 GB | 최소 가용 RAM GB | 최대 페이지파일 GB | 빈 응답 | 오류 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        L.append(
            f"| {s['model']} | {s['mode']} | {fmt(s['gen_tps_warm_mean'])} | {fmt(s['gen_tps_ko'])} | "
            f"{fmt(s['gen_tps_en'])} | {fmt(s['sys_delta_gb'],2)} | {fmt(s['min_sys_avail_gb'],1)} | "
            f"{fmt(s['max_pagefile_gb'],1)} | {s['empty_responses']} | {s['runs_err']} |"
        )
    L += [
        "",
        "## 개별 실행",
        "",
        "| 모델 | 모드 | 언어 | 회차 | cold | 로딩(s) | 프롬프트 tok | 프롬프트 tok/s | 생성 tok | 생성 tok/s | ollama ps | 러너 GB | GPU 공유 GB | 가용 RAM GB | 비고 |",
        "|---|---|---|---:|:-:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in records:
        if r["ok"]:
            note = r["done_reason"] or ""
            if r.get("response_empty"):
                note += ", 응답 비어있음(thinking만)"
            L.append(
                f"| {r['model']} | {r['mode']} | {r['lang']} | {r['run']} | {'Y' if r['cold'] else ''} | "
                f"{fmt(r['load_s'],2)} | {r['prompt_eval_count']} | {fmt(r['prompt_tps'])} | "
                f"{r['eval_count']} | {fmt(r['gen_tps'])} | {r['ps_size'] or '-'} {r['ps_processor'] or ''} | "
                f"{fmt(r.get('runner_private_gb'),2)} | {fmt(r.get('gpu_shared_gb'),2)} | {fmt(r.get('sys_avail_gb'),1)} | {note} |"
            )
        else:
            L.append(
                f"| {r['model']} | {r['mode']} | {r['lang']} | {r['run']} | | | | | | | | | | | 오류: {r['error'][:80]} |"
            )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=list(MODE_OPTIONS))
    args = ap.parse_args()

    try:
        ver = requests.get(f"{HOST}/api/version", timeout=5).json().get("version")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Ollama 서버에 연결할 수 없습니다 ({HOST}): {e}")

    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "ollama_version": ver,
        "base_options": BASE_OPTIONS,
        "mode_options": MODE_OPTIONS,
        "prompts": PROMPTS,
        "repeats": REPEATS,
    }
    records = run(args.models, args.modes)
    summary = summarize(records)
    meta["finished"] = datetime.now().isoformat(timespec="seconds")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_md(records, summary, meta, RESULTS / "baseline.md")
    print(f"\n저장: {RESULTS / 'baseline.json'}, {RESULTS / 'baseline.md'}")


if __name__ == "__main__":
    main()
