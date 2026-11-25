"""Token breakdown of the fixed prompt prefix (demo config): system rules / demo rules / menu / tool definitions.

    .venv-voice\\Scripts\\python.exe v2\\voice\\app\\prefix_breakdown.py

Needs a running Ollama with the model loaded (start_demo.ps1). Text sections are counted with the
llama-server /tokenize endpoint of the loaded runner (exact model tokenizer, no generation); the tool
definitions are counted as the difference of Ollama's prompt_eval_count with and without tools (the chat
template renders them). Run between conversations: the request without tools replaces the cached prefix,
so the next turn re-evaluates it once. Writes v2/voice/results/prefix_breakdown.{md,json}.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import requests

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import engines  # noqa: E402

RES = APP.parent / "results"


def runner_port():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | "
                          "Select-Object -ExpandProperty CommandLine"], capture_output=True, text=True).stdout
    m = re.search(r"--port\s+(\d+)", out)
    return int(m[1]) if m else None


def tokenizer():
    port = runner_port()
    if port:
        try:
            requests.post(f"http://127.0.0.1:{port}/tokenize", json={"content": "a"}, timeout=5).raise_for_status()
            return lambda text: len(requests.post(f"http://127.0.0.1:{port}/tokenize",
                                                  json={"content": text}, timeout=30).json()["tokens"])
        except requests.RequestException:
            pass
    return None


def prompt_count(msgs, tools) -> int:
    payload = {"model": engines.MODEL, "messages": msgs, "stream": False, "think": False, "keep_alive": engines.KEEP_ALIVE,
               "options": engines.llm_options(num_predict=1)}
    if tools:
        payload["tools"] = engines.tools()
    r = requests.post(f"{engines.OLLAMA}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["prompt_eval_count"]


def sections(head: str) -> dict:
    """Split the head text at its [section] headings into the four groups."""
    parts = re.split(r"\n(?=\[[^\]\n]+\]\n)", head)
    groups = {"system_rules": [], "demo_rules": [], "menu": [], "menu_rules_aliases": []}
    demo_heads = {ln.strip() for ln in (APP / "demo" / "rules.txt").read_text(encoding="utf-8").splitlines()
                  if re.fullmatch(r"\[[^\]]+\]", ln.strip())}
    for p in parts:
        title = p.split("\n", 1)[0].strip()
        if title in demo_heads:
            groups["demo_rules"].append(p)
        elif title == "[메뉴판]":
            groups["menu"].append(p)
        elif title in ("[메뉴 규칙]", "[줄임말/다른 이름]"):
            groups["menu_rules_aliases"].append(p)
        else:
            groups["system_rules"].append(p)
    return {k: "\n".join(v) for k, v in groups.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    head, state_msg = engines.split_prompt([])
    user = {"role": "user", "content": "안녕하세요"}
    msgs = [head, state_msg, user]
    count = tokenizer()
    secs = sections(head["content"])
    out = {"method_sections": "llama-server /tokenize" if count else "ollama prompt_eval_count difference"}
    with_tools = prompt_count(msgs, True)          # = the warm-up prompt, cached
    without_tools = prompt_count(msgs, False)      # full evaluation (replaces the cache once)
    out["prompt_with_tools"], out["prompt_without_tools"] = with_tools, without_tools
    out["tools"] = with_tools - without_tools
    if count:
        out["sections"] = {k: count(v) for k, v in secs.items()}
        per_tool = {}
        for t in engines.tools():
            per_tool[t["function"]["name"]] = count(json.dumps(t, ensure_ascii=False))
        out["tools_json_tokens_each"] = per_tool
        lines = [ln for ln in secs["menu"].splitlines() if ln.startswith("- ")]
        out["menu_line_avg"] = round(sum(count(ln) for ln in lines) / max(1, len(lines)), 1)
        rules = {}
        for p in re.split(r"\n(?=\[[^\]\n]+\]\n)", head["content"]):
            rules[p.split("\n", 1)[0].strip()] = count(p)
        out["by_heading"] = rules
    out["state_and_user_overhead"] = with_tools - out["tools"] - sum(out.get("sections", {}).values() or [0])
    (RES / "prefix_breakdown.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# 고정 프롬프트 앞부분 토큰 (데모 설정)", "",
         f"- 세는 방법: 본문은 {out['method_sections']}, 도구 정의는 도구 있음/없음 prompt_eval_count 차이.",
         f"- 빈 주문 + '안녕하세요' 한 턴 전체: 도구 있음 {with_tools}, 도구 없음 {without_tools}.", "",
         "| 부분 | 토큰 |", "|---|---:|"]
    for k, label in (("system_rules", "기본 규칙(v1 프롬프트)"), ("demo_rules", "데모 규칙(rules.txt)"),
                     ("menu", "메뉴판"), ("menu_rules_aliases", "메뉴 규칙 + 줄임말")):
        if "sections" in out:
            L.append(f"| {label} | {out['sections'][k]} |")
    L.append(f"| 도구 정의 8개 | {out['tools']} |")
    if "by_heading" in out:
        L += ["", "| 제목별 | 토큰 |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in out["by_heading"].items()]
        L += ["", "| 도구 (JSON 기준, 참고) | 토큰 |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in out["tools_json_tokens_each"].items()]
    (RES / "prefix_breakdown.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
