"""Run an exam against an Ollama model (or the rule-based vending machine)
and save raw outputs + scores under results/.

    # cafe exam, thinking off (default)
    python scripts/run_exam.py --model gemma4:e4b --exam data/exam_cafe.jsonl
    # thinking on
    python scripts/run_exam.py --model gemma4:e4b --exam data/exam_cafe.jsonl --think
    # rule-based baseline
    python scripts/run_exam.py --model vending --exam data/exam_cafe.jsonl
    # a few cases only
    python scripts/run_exam.py --model gemma4:e4b --exam data/exam_cafe.jsonl --ids B1-1,O3-2 --out-dir results/smoke

Output: <out-dir>/<exam>__<model>__<think|nothink>__<timestamp>/
    raw.jsonl     one line per case: tool calls / text, thinking, timings
    scores.jsonl  per-case scores
    summary.json  run metadata + aggregate scores
    summary.md    short human-readable summary
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from cafe_common import ROOT, build_system_prompt, load_tools
from score import is_knowledge, knowledge_table_md, load_jsonl, score_run

OPTIONS = {"temperature": 0, "seed": 42}

KNOW_PROMPT = {
    "ko": "다음 객관식 문제의 정답을 고르세요. 설명 없이 A, B, C, D 중 알파벳 하나만 답하세요.\n\n{q}\n\n{choices}\n\n정답:",
    "en": "Choose the correct answer to the following multiple-choice question. Reply with a single letter (A, B, C, or D) and nothing else.\n\n{q}\n\n{choices}\n\nAnswer:",
}


def git_blob_sha(path: Path) -> str:
    """Same value as `git hash-object` (line endings normalized to LF like core.autocrlf)."""
    import hashlib
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# Files the vending machine depends on at runtime -> README key holding the frozen blob hash.
FROZEN_FILES = {
    "scripts/vending_machine.py": "FROZEN_VENDING_BLOB",
    "data/menu.json": "FROZEN_MENU_BLOB",
    "scripts/cafe_common.py": "FROZEN_COMMON_BLOB",
}


def frozen_blobs() -> dict[str, str | None]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    out = {}
    for path, key in FROZEN_FILES.items():
        m = re.search(rf"{key}:\s*`([0-9a-f]{{40}})`", readme)
        out[path] = m.group(1) if m else None
    return out


def check_vending_freeze() -> dict:
    """Compare current blob hashes of the vending machine and its runtime deps with README."""
    frozen = frozen_blobs()
    files = {}
    for path in FROZEN_FILES:
        cur = git_blob_sha(ROOT / path)
        ok = frozen[path] is not None and cur == frozen[path]
        files[path] = {"blob": cur, "frozen_blob": frozen[path], "matches": ok}
        if not ok:
            print(f"WARNING: {path} ({cur[:10]}) differs from the frozen version "
                  f"({(frozen[path] or 'missing')[:10]}) recorded in README.md. Vending results are not comparable.",
                  flush=True)
    return {"files": files, "all_frozen": all(f["matches"] for f in files.values())}


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).name


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def chat(host, model, messages, tools, think, num_ctx, num_predict, timeout):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "keep_alive": "10m",
        "options": {**OPTIONS, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    if tools:
        payload["tools"] = tools
    t0 = time.perf_counter()
    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    wall = time.perf_counter() - t0
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    d["_wall_s"] = wall
    return d


def tool_calls_of(msg: dict) -> list[dict]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        out.append({"name": fn.get("name"), "arguments": args or {}})
    return out


def run_case_ollama(case, a, tools):
    if is_knowledge(case):
        choices = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", case["choices"]))
        messages = [{"role": "user", "content": KNOW_PROMPT[case["lang"]].format(q=case["question"], choices=choices)}]
        num_predict = a.num_predict or (4096 if a.think else 16)
        d = chat(a.host, a.model, messages, None, a.think, a.num_ctx, num_predict, a.timeout)
    else:
        messages = [{"role": "system", "content": build_system_prompt(case.get("order_state"))}] + case["conversation"]
        num_predict = a.num_predict or (4096 if a.think else 512)
        d = chat(a.host, a.model, messages, tools, a.think, a.num_ctx, num_predict, a.timeout)
    msg = d.get("message") or {}
    return {
        "tool_calls": tool_calls_of(msg),
        "content": msg.get("content") or "",
        "thinking": msg.get("thinking") or "",
        "done_reason": d.get("done_reason"),
        "load_s": (d.get("load_duration") or 0) / 1e9,
        "prompt_eval_count": d.get("prompt_eval_count"),
        "prompt_eval_s": (d.get("prompt_eval_duration") or 0) / 1e9,
        "eval_count": d.get("eval_count"),
        "eval_s": (d.get("eval_duration") or 0) / 1e9,
        "wall_s": d["_wall_s"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help='Ollama model name, or "vending" for the rule-based baseline')
    ap.add_argument("--exam", required=True)
    ap.add_argument("--think", action="store_true", help="turn thinking on (default: off)")
    ap.add_argument("--ids", help="comma-separated case ids to run")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out-dir", default="results/runs")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--show-prompt", action="store_true", help="print the exact messages for the first selected case and exit")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    exam_path = Path(a.exam)
    exam = load_jsonl(exam_path)
    if a.ids:
        wanted = [x.strip() for x in a.ids.split(",") if x.strip()]
        by_id = {c["id"]: c for c in exam}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            sys.exit(f"unknown ids: {missing}")
        exam = [by_id[w] for w in wanted]
    if a.limit:
        exam = exam[: a.limit]
    if a.show_prompt:
        case = exam[0]
        if is_knowledge(case):
            choices = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", case["choices"]))
            print(KNOW_PROMPT[case["lang"]].format(q=case["question"], choices=choices))
        else:
            print(f"=== system ({case['id']}) ===")
            print(build_system_prompt(case.get("order_state")))
            for m in case["conversation"]:
                print(f"=== {m['role']} ===")
                print(m["content"])
            print(f"=== tools: {', '.join(t['function']['name'] for t in load_tools())} (data/tools.json) ===")
        return

    vending = a.model == "vending"
    freeze = None
    if vending:
        from vending_machine import VendingMachine
        vm = VendingMachine()
        ollama_version = None
        freeze = check_vending_freeze()
    else:
        try:
            ollama_version = requests.get(f"{a.host}/api/version", timeout=5).json().get("version")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"cannot reach Ollama at {a.host}: {e}")
    tools = load_tools()

    started = datetime.now()
    mode = "think" if a.think else "nothink"
    run_dir = ROOT / a.out_dir / f"{exam_path.stem}__{slug(a.model)}__{'rules' if vending else mode}__{started:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = []
    for k, case in enumerate(exam, 1):
        rec = {"id": case["id"]}
        try:
            if vending:
                t0 = time.perf_counter()
                rec.update({"tool_calls": vm.predict(case), "content": "", "wall_s": time.perf_counter() - t0})
            else:
                rec.update(run_case_ollama(case, a, tools))
        except Exception as e:  # noqa: BLE001
            rec.update({"tool_calls": [], "content": "", "error": repr(e)[:500]})
        raw.append(rec)
        shown = json.dumps(rec.get("tool_calls"), ensure_ascii=False) if not is_knowledge(case) else repr(rec.get("content"))
        print(f"[{k}/{len(exam)}] {case['id']}: {shown[:200]}{'  ERROR ' + rec['error'] if 'error' in rec else ''}", flush=True)

    scores, summary = score_run(load_jsonl(exam_path), raw)
    finished = datetime.now()
    meta = {
        "model": a.model,
        "exam": rel(exam_path),
        "think": None if vending else a.think,
        "options": None if vending else {**OPTIONS, "num_ctx": a.num_ctx, "num_predict": a.num_predict or "auto"},
        "ollama_version": ollama_version,
        "n_cases": len(exam),
        "ids": a.ids.split(",") if a.ids else None,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "errors": sum(1 for r in raw if "error" in r),
    }
    if vending:
        meta["vending_machine_blob"] = freeze["files"]["scripts/vending_machine.py"]["blob"]
        meta["vending_machine_frozen"] = freeze["all_frozen"]
        meta["vending_freeze_check"] = freeze["files"]
    gen = [r for r in raw if r.get("eval_count") and r.get("eval_s")]
    if gen:
        meta["mean_gen_tps"] = round(sum(r["eval_count"] for r in gen) / sum(r["eval_s"] for r in gen), 2)
        meta["mean_wall_s"] = round(sum(r["wall_s"] for r in raw) / len(raw), 3)

    def dump_jsonl(path, rows):
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    dump_jsonl(run_dir / "raw.jsonl", raw)
    dump_jsonl(run_dir / "scores.jsonl", scores)
    (run_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# {meta['exam']} / {a.model} ({'rules' if vending else mode})", "",
          f"- {meta['started']} ~ {meta['finished']}, cases: {meta['n_cases']}, errors: {meta['errors']}"]
    if "mean_gen_tps" in meta:
        md.append(f"- mean generation speed: {meta['mean_gen_tps']} tok/s, mean wall time per case: {meta['mean_wall_s']} s")
    if "always_D_baseline" in summary:
        md += ["", knowledge_table_md(summary, label=a.model)]
    md += ["", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""]
    (run_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if "always_D_baseline" in summary:
        print(knowledge_table_md(summary, label=a.model))
    print(f"saved: {rel(run_dir)}")


if __name__ == "__main__":
    main()
