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


LEAN = ROOT / "results/step6/lean"
LEAN_MODELS = [("song/gemma4-26b-cafe-64e", "26b 가지치기 64개"), ("song/gemma4-26b-cafe-48e", "26b 가지치기 48개")]
BLOB = {"song/gemma4-26b-cafe-64e": "1fa88d73", "song/gemma4-26b-cafe-48e": "9a01e444"}   # main GGUF blob (ollama show --modelfile)


def load_breakdown(log: Path, blob: str, mode: str):
    """Memory parts (MiB) of the LAST llama-server load of this blob in this mode, from its load log."""
    if not log.exists():
        return None
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    starts = [i for i, l in enumerate(lines) if "starting llama-server" in l]
    blocks = []
    for k, i in enumerate(starts):
        if blob in lines[i] and (("-ngl 0" in lines[i]) == (mode == "cpu")):
            blocks.append(lines[i:starts[k + 1] if k + 1 < len(starts) else len(lines)])
    if not blocks:
        return None
    b = blocks[-1]
    part = "main"
    d = {p: {"weights": 0.0, "kv": 0.0, "compute": 0.0} for p in ("main", "draft", "projector")}
    froze, in_compute = set(), False
    ckpt_max, ckpt_size = 0, 0.0
    for l in b:
        if "loading draft model" in l:
            part, in_compute = "draft", False
            continue
        if l.startswith("clip_model_loader: model name") and part != "projector" and d["main"]["weights"]:
            part, in_compute = "projector", False
        m = re.search(r"created context checkpoint (\d+) of \d+ .*size = ([\d.]+) MiB", l)
        if m:
            ckpt_max, ckpt_size = max(ckpt_max, int(m[1])), float(m[2])
            continue
        if part in froze:
            continue
        m = re.search(r"model buffer size =\s+([\d.]+) MiB", l)
        if m:
            d[part]["weights"] += float(m[1])
            continue
        m = re.search(r"load_hparams: model size:\s+([\d.]+) MiB", l)
        if m:
            d["projector"]["weights"] += float(m[1])
            continue
        m = re.search(r"llama_kv_cache: size =\s+([\d.]+) MiB", l)
        if m:
            d[part]["kv"] += float(m[1])
            continue
        m = re.search(r"(sched_reserve|reserve_compute_meta):.*compute buffer size =\s+([\d.]+) MiB", l)
        if m:
            d[part]["compute"] += float(m[2])
            in_compute = True
            continue
        if in_compute:   # first block of compute-buffer lines only (later lines are re-reservations at run time)
            froze.add(part)
            in_compute = False
    d["checkpoints"] = {"n_max": ckpt_max, "each_mib": ckpt_size, "total": ckpt_max * ckpt_size}
    return d


def lean_section(runs, changed):
    if not LEAN.exists():
        return []
    by_id = {c["id"]: c for c in load_jsonl(ROOT / "data/exam_cafe.jsonl")}
    from score import score_cafe_case
    L = ["# 6b단계: 카페 로봇용 텍스트 전용 가벼운 설정 (64개, 48개)", "",
         "- 6a단계 설정에서 카페 로봇에 필요 없는 것을 뺐다:",
         "  - 비전 프로젝터(이미지 입력용)와 초안(MTP) 모델: Modelfile에서 두 줄을 지운 새 모델 `song/gemma4-26b-cafe-{64,48}e-text`로 등록(가중치 블롭은 같은 파일, 템플릿·렌더러·파라미터 같음).",
         "  - `LLAMA_ARG_SWA_FULL=1`을 빼고 보통의 슬라이딩 윈도 캐시를 씀. 서버는 `OLLAMA_IGPU_ENABLE=1`, `LLAMA_ARG_CACHE_RAM=0`(이 프로세스에만).",
         "- 나머지(30문항, 순서, 데우기, thinking 끔, temperature 0, seed 42, num_ctx 8192)는 6a와 같다.", ""]
    # scores
    L += ["## 30문항 점수 확인", "", "| 모델 | 모드 | 6a 정답 | 6b 정답 | 도구 호출이 달라진 문항 |", "|---|---|---:|---:|---|"]
    lean = {}
    for m, ml in LEAN_MODELS:
        for mode, mk in MODES:
            f = LEAN / f"{slug(m + '-text')}__{mode}.json"
            a = runs.get((m, mode))
            if not f.exists() or not a or not a.get("cases"):
                L.append(f"| {ml} | {mk} | 결과 없음 | | |")
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            lean[(m, mode)] = r
            if not r.get("cases"):
                L.append(f"| {ml} | {mk} | | 실패: {(r.get('error') or r.get('skipped') or '')[:80]} | |")
                continue
            ca = {c["id"]: c for c in a["cases"]}
            cb = {c["id"]: c for c in r["cases"]}
            ea = sum(score_cafe_case(by_id[i], ca[i]["tool_calls"])["exact"] for i in ca)
            eb = sum(score_cafe_case(by_id[i], cb[i]["tool_calls"])["exact"] for i in cb)
            diff = [i for i in ca if i in cb and ca[i]["tool_calls"] != cb[i]["tool_calls"]]
            L.append(f"| {ml} | {mk} | {ea}/{len(ca)} | {eb}/{len(cb)} | {', '.join(diff) if diff else '없음'} |")
    L += ["", "- 도구 호출 비교는 도구 이름과 인자가 글자 그대로 같은지 본다. 초안 모델은 temperature 0에서 결과를 바꾸지 않는 것이 원칙이지만, 계산 순서가 달라 드물게 달라질 수 있다.", ""]
    # timing / memory
    L += ["## 응답 시간과 메모리: 6a 설정 vs 가벼운 설정", "",
          f"| 모델 | 모드 | 설정 | 첫 토큰까지: 앞부분 재사용 (중앙값) | 첫 토큰까지: 주문 내역 바뀜 (중앙값) | 주문 1건 전체 (중앙값 / 최대) | 생성 속도 | 정상 상태 메모리 | 16 GB 노트북 (OS 4 GB + 모델) |",
          "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for m, ml in LEAN_MODELS:
        for mode, mk in MODES:
            for tag, r in (("6a (전체)", runs.get((m, mode))), ("6b (가벼움)", lean.get((m, mode)))):
                if not r or not r.get("cases"):
                    continue
                cs = r["cases"]
                same = [c["ttft_s"] for c in cs if not changed[c["id"]]]
                ch = [c["ttft_s"] for c in cs if changed[c["id"]]]
                wt = [c["wall_s"] for c in cs]
                ts = [c["tok_s"] for c in cs if c["tok_s"]]
                mm = mem(r)
                need, fit = fit16(mm)
                L.append(f"| {ml} | {mk} | {tag} | {statistics.median(same):.2f} s | {statistics.median(ch):.1f} s | "
                         f"{statistics.median(wt):.2f} s / {max(wt):.1f} s | {statistics.median(ts):.1f} tok/s | "
                         f"{mm['foot_med']:.1f} GB (변화 {mm['growth']:+.2f}) | {need:.1f} GB → {fit} |")
    L.append("")
    # breakdown
    L += ["## 뺀 부분이 각각 얼마나 줄였나 (llama-server 로딩 로그의 버퍼 크기)", "",
          "| 모델 | 모드 | 본 모델 KV: SWA 전체 → 보통 (+ 체크포인트) | 초안 모델 (가중치 + KV + 계산) | 비전 프로젝터 (가중치 + 계산) | 합계 (로그) | 실제 정상 상태 메모리 차이 (측정) |",
          "|---|---|---:|---:|---:|---:|---:|"]
    gb = lambda mib: mib / 1024  # noqa: E731
    for m, ml in LEAN_MODELS:
        for mode, mk in MODES:
            full = load_breakdown(ROOT / "logs/ollama_serve_step6.log", BLOB[m], mode)
            lite = load_breakdown(ROOT / "logs/ollama_serve_step6b.log", BLOB[m], mode)
            ra, rb = runs.get((m, mode)), lean.get((m, mode))
            if not full or not lite or not ra or not rb or not rb.get("cases"):
                L.append(f"| {ml} | {mk} | 결과 없음 | | | | |")
                continue
            kv_save = full["main"]["kv"] - lite["main"]["kv"] - lite["checkpoints"]["total"]
            dr = sum(full["draft"].values())
            pj = sum(full["projector"].values())
            meas = mem(ra)["foot_med"] - mem(rb)["foot_med"]
            L.append(f"| {ml} | {mk} | {gb(kv_save):.2f} GB ({gb(full['main']['kv']):.2f} → {gb(lite['main']['kv']):.2f}, "
                     f"체크포인트 최대 {lite['checkpoints']['n_max']}개 × {lite['checkpoints']['each_mib']:.0f} MiB) | "
                     f"{gb(dr):.2f} GB | {gb(pj):.2f} GB | {gb(kv_save + dr + pj):.2f} GB | {meas:.2f} GB |")
    L += ["", "- 로그 값은 llama-server가 잡은 버퍼 크기(MiB → GB, 1 GB = 1024 MiB)이고, 측정값은 30문항 동안의 정상 상태 메모리(러너 private 또는 GPU 공유 중 큰 쪽의 중앙값) 차이다. 측정값에는 로그에 안 나오는 런타임 할당도 들어간다.",
          "- SWA 전체 → 보통: 슬라이딩 윈도 층 25개의 KV가 8,192칸에서 윈도 크기만큼으로 줄어든다. 대신 앞부분을 다시 쓰려고 llama-server가 ‘컨텍스트 체크포인트’(SWA 상태 복사본)를 만들어 둔다.", ""]
    li = ROOT / "results/step6/lean_interpretation.md"
    if li.exists():
        L += [li.read_text(encoding="utf-8").strip(), ""]
    return L


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
    L += lean_section(runs, changed)
    out = ROOT / "results/step6/latency.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
