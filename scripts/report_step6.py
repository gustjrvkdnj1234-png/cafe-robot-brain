"""Step 6a report: results/step6/latency.md (Korean) from results/step6/latency/*.json

    .venv\\Scripts\\python.exe scripts\\report_step6.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cafe_common import build_system_prompt  # noqa: E402
from run_exam import slug  # noqa: E402
from score import load_jsonl  # noqa: E402

D = ROOT / "results/step6/latency"
MODELS = [("gemma4:e4b", "gemma4:e4b"), ("gemma4:12b", "gemma4:12b"), ("gemma4:26b", "gemma4:26b (원본)"),
          ("song/gemma4-26b-cafe-98e", "26b 가지치기 98개"), ("song/gemma4-26b-cafe-64e", "26b 가지치기 64개"),
          ("song/gemma4-26b-cafe-48e", "26b 가지치기 48개")]
MODES = [("cpu", "CPU 전용"), ("igpu", "iGPU")]
OS_GB, LAPTOP_GB = 4.0, 16.0


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def state_changed(ids):
    """Per case: True if its system prompt (order state) differs from the previous request's (warm-up = empty order)."""
    by_id = {c["id"]: c for c in load_jsonl(ROOT / "data/exam_cafe.jsonl")}
    prev, out = build_system_prompt(None), {}
    for i in ids:
        cur = build_system_prompt(by_id[i].get("order_state"))
        out[i] = cur != prev
        prev = cur
    return out


def mem(res):
    rows = [r for r in res.get("memory_series", []) if "error" not in r]
    od = [r for r in rows if r["phase"] == "orders"]
    if not od:
        return None
    priv = [r["runner_private_gb"] for r in od]
    gpu = [r.get("gpu_shared_gb") or 0 for r in od]
    foot = [max(p, g) for p, g in zip(priv, gpu)]   # runner private and GPU shared overlap: take the larger
    return {"foot_med": statistics.median(foot), "foot_max": max(foot), "priv_med": statistics.median(priv),
            "gpu_max": max(gpu), "growth": foot[-1] - foot[0], "n": len(od)}


def fit16(m):
    need = OS_GB + m["foot_max"]
    left = LAPTOP_GB - need
    if left >= 1.5:
        return need, f"예 (여유 {left:.1f} GB)"
    if left >= 0:
        return need, f"아슬아슬 (여유 {left:.1f} GB)"
    return need, f"아니요 ({-left:.1f} GB 부족)"


def reuse_stats():
    """llama-server log: first 'cached n_tokens' per task = prompt tokens reused from the slot."""
    log = ROOT / "logs/ollama_serve_step6.log"
    if not log.exists():
        return None
    cur, out = {}, []
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"task (\d+) \| new prompt.*task\.n_tokens = (\d+)", line)
        if m:
            cur[m[1]] = {"n": int(m[2]), "cached": None}
            continue
        m = re.search(r"task (\d+) \| cached n_tokens = (\d+)", line)
        if m and m[1] in cur and cur[m[1]]["cached"] is None:
            cur[m[1]]["cached"] = int(m[2])
            out.append(cur[m[1]])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    runs = {}
    for m, _ in MODELS:
        for mode, _ in MODES:
            f = D / f"{slug(m)}__{mode}.json"
            if f.exists():
                runs[(m, mode)] = json.loads(f.read_text(encoding="utf-8"))
    any_run = next(r for r in runs.values() if r.get("case_ids"))
    changed = state_changed(any_run["case_ids"])
    n_ch = sum(changed.values())
    L = ["# 6a단계: 같은 조건에서 잰 응답 시간과 메모리", "",
         "- 카페 시험 30문항(`data/exam_cafe.jsonl`의 4번째 줄마다, 12개 유형 모두 포함)을 한 번에 하나씩 차례로 보냄. "
         "thinking 끔, temperature 0, seed 42, num_ctx 8192, num_predict 512, 3·5단계와 같은 시스템 프롬프트와 도구.",
         "- 모델마다: 모두 내림 → 시스템 프롬프트로 한 번 데우기(warm-up) → 모델을 올린 채 30문항. 실행: 2026-11-12, 같은 노트북(Core Ultra 7 255H, 32 GB, Arc 140T).",
         "- **프롬프트 캐시 문제 해결:** `ollama serve`를 이 프로세스에만 `LLAMA_ARG_CACHE_RAM=0`(3·5단계에서 최대 8 GB까지 커지던 llama-server의 RAM 프롬프트 캐시를 끔)과 "
         "`LLAMA_ARG_SWA_FULL=1`(슬라이딩 윈도 층도 전체 길이 KV를 유지해서 앞부분을 다시 쓸 수 있게 함)로 띄웠다. 슬롯이 하나라 직전 요청의 KV가 그대로 남고, 같은 앞부분은 다시 계산하지 않는다.",
         "- CPU 전용 = `num_gpu 0`(3단계와 같음). iGPU = `OLLAMA_IGPU_ENABLE=1`, `ollama ps`에서 100% GPU일 때만 잼.", ""]
    rs = reuse_stats()
    if rs:
        orders = [r for r in rs if r["n"] > 3000]
        full = [r for r in orders if r["cached"] and r["cached"] >= 3400]
        part = [r for r in orders if r["cached"] and r["cached"] < 3400]
        zero = [r for r in orders if not r["cached"]]
        L += ["## 시스템 프롬프트 재사용: 되는 경우와 안 되는 경우", "",
              f"- llama-server 로그로 확인했다. 시스템 프롬프트가 들어간 요청 {len(orders)}개(파일럿 포함) 중 {len(full)}개는 앞부분 약 3,475 토큰을 그대로 썼고, "
              f"처음부터 계산한 것은 {len(zero)}개(모델을 올린 직후의 데우기)뿐이다.",
              f"- 나머지 {len(part)}개는 앞 2,362 토큰만 다시 쓰고 약 1,100 토큰을 다시 계산했다. 이 문항들은 모두 **직전 요청과 ‘현재 주문 내역’이 다른 경우**였다"
              f"(30문항 중 {n_ch}개, 모든 모델에서 같은 문항).",
              "- 이유: 주문 내역은 시스템 프롬프트의 맨 끝에 있지만, Ollama가 도구 설명(약 1,100 토큰)을 시스템 프롬프트 **뒤에** 붙인다. "
              "그래서 주문 내역이 바뀌면 그 뒤의 도구 설명까지 다시 계산한다(2,362 토큰에서 갈라지는 것으로 보아 이렇게 추정한다).",
              "- 실제 로봇에서는 손님이 하나를 담을 때마다 주문 내역이 바뀌므로, 두 번째 주문부터는 거의 매번 이 비용을 낸다. "
              "프롬프트는 3·5단계 점수와 같은 조건을 유지하려고 바꾸지 않았다(`cafe_common.py`는 고정 파일). 아래 표는 두 경우를 나눠 적었다.", ""]
    L += ["## 결과", "",
          f"| 모델 | 모드 | 첫 토큰까지: 앞부분 재사용 (중앙값, {30 - n_ch}문항) | 첫 토큰까지: 주문 내역 바뀜 (중앙값, {n_ch}문항) | "
          "주문 1건 전체 (중앙값 / 최대, 30문항) | 생성 속도 (중앙값) | 정상 상태 메모리 (모델 + 컨텍스트) | 16 GB 노트북 (OS 4 GB + 모델) |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    notes = []
    for mode, mk in MODES:
        for m, ml in MODELS:
            r = runs.get((m, mode))
            if r is None:
                L.append(f"| {ml} | {mk} | 측정 안 함 | | | | | |")
                continue
            if not r.get("cases"):
                why = r.get("skipped") or r.get("error", "")
                L.append(f"| {ml} | {mk} | **iGPU에 안 들어감** | | | | | |")
                notes.append(f"{ml} {mk}: 데우기에서 로딩 실패 — `{why[:160]}`")
                continue
            cs = r["cases"]
            same = [c["ttft_s"] for c in cs if not changed[c["id"]]]
            ch = [c["ttft_s"] for c in cs if changed[c["id"]]]
            wt = [c["wall_s"] for c in cs]
            ts = [c["tok_s"] for c in cs if c["tok_s"]]
            mm = mem(r)
            need, fit = fit16(mm)
            mtxt = f"{mm['foot_med']:.1f} GB"
            L.append(f"| {ml} | {mk} | {statistics.median(same):.2f} s | {statistics.median(ch):.1f} s | "
                     f"{statistics.median(wt):.2f} s / {max(wt):.1f} s | {statistics.median(ts):.1f} tok/s | {mtxt} | "
                     f"{need:.1f} GB → {fit} |")
            notes.append(f"{ml} {mk}: 데우기 = 로딩 {r['warmup']['load_s']:.1f} s + 시스템 프롬프트 첫 처리 {r['warmup']['prompt_eval_s']:.1f} s. "
                         f"30문항 동안 메모리 변화 {mm['growth']:+.2f} GB"
                         + (f", 러너 private {mm['priv_med']:.1f} GB / GPU 공유 최대 {mm['gpu_max']:.1f} GB" if mode == "igpu" else "")
                         + ".")
    L += ["",
          "- 첫 토큰까지: 서버가 잰 (모델 로딩 + 프롬프트 처리 + 첫 토큰 생성 1번). 모델이 이미 올라가 있어 로딩은 0에 가깝다. "
          "Ollama는 도구 호출 하나를 다 만든 뒤에 내보내므로, 로봇이 실제로 움직이기 시작하는 시점은 ‘주문 1건 전체’에 가깝다.",
          "- 주문 1건 전체: 요청을 보내고 응답을 다 받을 때까지(클라이언트에서 잼). 생성 토큰은 대부분 20~40개다.",
          "- 정상 상태 메모리: 30문항을 처리하는 동안 2초마다 잰 값의 중앙값. CPU는 러너(llama-server) private 메모리, "
          "iGPU는 러너 private와 GPU 공유 메모리 중 큰 쪽(둘은 겹치므로 더하지 않음). 가중치 + KV 캐시(8,192 토큰, SWA 전체 길이) + 계산 버퍼 + 비전 프로젝터 + 초안(MTP) 모델이 들어 있다. "
          "프롬프트 캐시를 껐으므로 시험 중에 늘지 않았다(아래 세부의 ‘메모리 변화’).",
          "- 16 GB 노트북: OS·기본 프로그램 4 GB + 정상 상태 메모리(최댓값)로 어림했다. 여유 1.5 GB 미만이면 ‘아슬아슬’. 브라우저 등 다른 프로그램이 켜져 있으면 더 모자란다. "
          "iGPU의 GPU 공유 메모리도 같은 RAM에서 나오므로 계산은 같다. 다만 Windows는 GPU 공유 메모리를 보통 RAM의 절반까지만 주는 것으로 알려져 있어, "
          "16 GB 노트북에서는 약 8 GB를 넘는 모델이 iGPU에 다 안 올라갈 수 있다(16 GB 기계에서 직접 확인한 것은 아님).",
          ""]
    L += ["## 세부", ""] + [f"- {n}" for n in notes] + [""]
    interp = ROOT / "results/step6/latency_interpretation.md"
    if interp.exists():
        L += [interp.read_text(encoding="utf-8").strip(), ""]
    out = ROOT / "results/step6/latency.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
