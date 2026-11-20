"""REAP vs v1 (contribution sum + protection) across sizes 98 / 64 / 48 (light, no model).

    .venv-prune\\Scripts\\python.exe v2\\scripts\\reap_curve.py

Inputs (read only)
  original : results/step3/{exam}__gemma4-26b__cpu
  v1 sum   : results/step5/{exam}__song-gemma4-26b-cafe-{k}e__igpu, results/step5/validate_{k}e.json
  REAP     : v2/results/reap64/ (k = 64), v2/results/reap_curve/reap{k}/ (k = 98, 48)
Strict = stored v1 score (re-scored and checked), lenient = score.score_cafe_case_lenient (ask_clarification reason
ignored). McNemar = exact two-sided binomial test on discordant items (v1 report_comparison.mcnemar_exact).
Top-1 / KL: validate_pruned.py, same 8 calibration prompts (seed 7), final logits on the 482 case tokens, HF
original in float32 as the reference (so the original itself is 100% / 0 by definition).
Outputs: v2/results/reap_curve/{curve.md, curve.json, curve.png}. No question text is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from score import load_jsonl, score_cafe_case, score_cafe_case_lenient  # noqa: E402


def mcnemar_exact(b: int, c: int) -> float:
    """Same as v1 report_comparison.mcnemar_exact (not imported: it pulls in `requests`, absent in .venv-prune)."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

OUT = ROOT / "v2/results/reap_curve"
SIZES = (98, 64, 48)
EXAMS = {"cafe": ROOT / "data/exam_cafe.jsonl", "live": ROOT / "data/exam_live.jsonl"}
ALPHA = 0.05


def one(base: Path, pattern: str) -> Path:
    hits = sorted(base.glob(pattern))
    assert len(hits) == 1, (base, pattern, hits)
    return hits[0]


def run_dir(model, k, exam):
    if model == "orig":
        return ROOT / f"results/step3/{exam}__gemma4-26b__cpu"
    if model == "v1":
        return ROOT / f"results/step5/{exam}__song-gemma4-26b-cafe-{k}e__igpu"
    base = ROOT / "v2/results/reap64" if k == 64 else OUT / f"reap{k}"
    return one(base, f"{exam}__song-gemma4-26b-cafe-{k}e-reap__*")


def validate_file(model, k):
    if model == "v1":
        return ROOT / f"results/step5/validate_{k}e.json"
    return (ROOT / "v2/results/reap64/validate_reap64.json") if k == 64 else OUT / f"reap{k}/validate_reap{k}.json"


def pct(x):
    return f"{x * 100:.1f}%"


def load(model, k, cases):
    r = {}
    for e in ("cafe", "live", "knowledge"):
        d = run_dir(model, k, e)
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        scores = {x["id"]: x for x in load_jsonl(d / "scores.jsonl")}
        raw = {x["id"]: x for x in load_jsonl(d / "raw.jsonl")}
        r[e] = {"dir": d.relative_to(ROOT).as_posix(), "mode": summ["meta"]["mode"], "summary": summ["summary"],
                "strict": {i: v["exact"] for i, v in scores.items()},
                # degenerate outputs: generation hit num_predict with no tool call and no answer text
                "runaway": sum(1 for x in raw.values() if x.get("done_reason") == "length"
                               and not x.get("tool_calls") and not (x.get("content") or "").strip())}
        if e in EXAMS:
            len_ = {}
            for cid, case in cases[e].items():
                pred = raw[cid]["tool_calls"]
                if score_cafe_case(case, pred)["exact"] != scores[cid]["exact"]:
                    raise SystemExit(f"strict rescoring differs: {d} {cid}")
                len_[cid] = score_cafe_case_lenient(case, pred)["exact"]
            r[e]["lenient"] = len_
    if model == "orig":
        r["top1"], r["kl"] = 1.0, 0.0
    else:
        v = json.loads(validate_file(model, k).read_text(encoding="utf-8"))["final"]
        r["top1"], r["kl"] = v["top1_agreement_case_tokens"], v["mean_kl_orig_to_pruned"]
    return r


def mcn(a, b):
    ids = sorted(set(a) & set(b))
    ao = sum(1 for i in ids if a[i] and not b[i])
    bo = sum(1 for i in ids if b[i] and not a[i])
    return {"n": len(ids), "reap_only": ao, "v1_only": bo, "p": mcnemar_exact(ao, bo)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cases = {e: {c["id"]: c for c in load_jsonl(f)} for e, f in EXAMS.items()}
    cat_of = {cid: c["category"] for cid, c in cases["cafe"].items()}
    R = {("orig", 128): load("orig", 128, cases)}
    for k in SIZES:
        for m in ("v1", "reap"):
            R[(m, k)] = load(m, k, cases)

    rows = {}
    for (m, k), r in R.items():
        cats = {}
        for cid, v in r["cafe"]["strict"].items():
            cats.setdefault(cat_of[cid], []).append(v)
        rows[f"{m}_{k}"] = {
            "model": m, "keep": k, "mode": r["cafe"]["mode"],
            "cafe": r["cafe"]["summary"]["exact"], "cafe_lenient": sum(r["cafe"]["lenient"].values()) / 120,
            "live": r["live"]["summary"]["exact"], "live_lenient": sum(r["live"]["lenient"].values()) / 20,
            "knowledge": r["knowledge"]["summary"]["accuracy"], "top1": r["top1"], "kl": r["kl"],
            "runaway": {e: r[e]["runaway"] for e in ("cafe", "live", "knowledge")},
            "cafe_by_category": {c: sum(v) / len(v) for c, v in sorted(cats.items())},
            "dirs": {e: r[e]["dir"] for e in ("cafe", "live", "knowledge")}}
    tests = {}
    for k in SIZES:
        a, b = R[("reap", k)], R[("v1", k)]
        for e in ("cafe", "live"):
            for sc in ("strict", "lenient"):
                tests[f"{k}|{e}|{sc}"] = mcn(a[e][sc], b[e][sc])
        tests[f"{k}|knowledge|strict"] = mcn(a["knowledge"]["strict"], b["knowledge"]["strict"])
        cond = [i for i in cases["cafe"] if cat_of[i] == "conditional"]
        tests[f"{k}|cafe_conditional|strict"] = mcn({i: a["cafe"]["strict"][i] for i in cond},
                                                    {i: b["cafe"]["strict"][i] for i in cond})
    res = {"rows": rows, "mcnemar_reap_vs_v1": tests, "alpha": ALPHA}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "curve.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    chart(rows)
    write_md(res)
    print("saved", (OUT / "curve.md").relative_to(ROOT), (OUT / "curve.png").relative_to(ROOT))


def chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    xs = [128, *SIZES]
    series = {"cafe": ("카페 (120)", "#2a6fb0"), "live": ("실전 (20)", "#d0632b"), "knowledge": ("지식 (300)", "#3c9a4f")}
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=150)
    for key, (lab, col) in series.items():
        for m, ls, mk, name in (("v1", "-", "o", "v1 기여 합"), ("reap", "--", "s", "REAP")):
            ys = [rows["orig_128"][key]] + [rows[f"{m}_{k}"][key] for k in SIZES]
            ax.plot(xs, [y * 100 for y in ys], ls=ls, marker=mk, color=col, lw=2 if m == "v1" else 1.6,
                    ms=5, label=f"{lab} · {name}")
    ax.set_xticks(xs)
    ax.set_xlim(132, 44)                       # fewer experts to the right
    ax.set_xlabel("층마다 남긴 전문가 수 (128 = 원본)")
    ax.set_ylabel("정답률 (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.set_title("전문가 수에 따른 점수: v1 기여 합(실선) vs REAP(점선)")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "curve.png")
    plt.close(fig)


def write_md(res):
    R, T = res["rows"], res["mcnemar_reap_vs_v1"]
    name = lambda r: "원본 128" if r["model"] == "orig" else f"{r['keep']} · {'REAP' if r['model'] == 'reap' else 'v1 기여 합'}"  # noqa: E731
    order = ["orig_128"] + [f"{m}_{k}" for k in SIZES for m in ("v1", "reap")]
    L = ["# REAP vs v1 기여 합: 크기별 곡선", "",
         "- v1 기여 합 = v1 모델(50:50 가중 기여 합 + 바쁜 전문가 보호). REAP = 층마다 순수 REAP 상위 k개(v1 4b 계획). "
         "모두 같은 원본·같은 변환·같은 Q4_K_M 텐서 형식, imatrix 없음, 같은 시험 설정(thinking 끔, temperature 0, seed 42).",
         "- 원본은 Ollama `gemma4:26b`(CPU), 나머지는 iGPU. 엄격 = 공식 점수, 관대 = 되묻기 이유 코드만 무시.",
         "- top-1 / KL: 보정 8문항(seed 7)의 문항 토큰 482개에서 HF 원본(float32) 마지막 logits와 비교. "
         "원본은 기준이라 100% / 0. 참고로 원본을 BF16으로만 돌려도 92.5% / 0.103(`results/step5/validate_baseline_bf16.json`).", "",
         "## 점수", "", "| 모델 | 카페 (120) | 실전 엄격 (20) | 실전 관대 | 지식 (300) | top-1 일치 | KL | 빈 출력 (카페/실전/지식) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in order:
        r = R[key]
        L.append(f"| {name(r)} | {pct(r['cafe'])} | {pct(r['live'])} | {pct(r['live_lenient'])} | {pct(r['knowledge'])} | "
                 f"{pct(r['top1'])} | {r['kl']:.3f} | {r['runaway']['cafe']} / {r['runaway']['live']} / {r['runaway']['knowledge']} |")
    L += ["", "- 빈 출력 = 생성 길이 한도(num_predict)까지 가고도 도구 호출도 답도 없는 문항 수.",
          "- 카페 관대 점수는 모든 모델에서 엄격과 같거나 거의 같아 `curve.json`에만 두었다.", "",
          "![곡선](curve.png)", "",
          "## McNemar 정확 검정: REAP vs v1 기여 합 (같은 크기끼리)", "",
          "| 크기 | 시험 | 채점 | REAP만 맞힘 | v1만 맞힘 | p | p < 0.05 | 보정 후 유의 |", "|---:|---|---|---:|---:|---:|:-:|:-:|"]
    bonf = ALPHA / len(T)
    lab = {"cafe": "카페", "live": "실전", "knowledge": "지식", "cafe_conditional": "카페 conditional (10)"}
    for key, v in T.items():
        k, e, sc = key.split("|")
        L.append(f"| {k} | {lab[e]} | {'엄격' if sc == 'strict' else '관대'} | {v['reap_only']} | {v['v1_only']} | "
                 f"{v['p']:.4f} | {'예' if v['p'] < ALPHA else ''} | {'**예**' if v['p'] < bonf else ''} |")
    sig = [k for k, v in T.items() if v["p"] < ALPHA]
    strong = [k for k, v in T.items() if v["p"] < bonf]
    L += ["", f"- 보정 후 유의 = Bonferroni(검정 {len(T)}번, p < {bonf:.4f}). 같은 문항에 채점만 다른 검정도 따로 세어 보수적이다.",
          f"- p < 0.05: {', '.join(sig) if sig else '없음'}",
          f"- 보정 후에도 유의: {', '.join(strong) if strong else '없음'}", "",
          "## 카페 유형별 엄격 정답률", "",
          "| 유형 | 원본 | " + " | ".join(f"{k} v1 | {k} REAP" for k in SIZES) + " |",
          "|---|---:|" + "---:|" * (2 * len(SIZES))]
    for c in R["orig_128"]["cafe_by_category"]:
        L.append(f"| {c} | {pct(R['orig_128']['cafe_by_category'][c])} | "
                 + " | ".join(f"{pct(R[f'v1_{k}']['cafe_by_category'][c])} | {pct(R[f'reap_{k}']['cafe_by_category'][c])}"
                              for k in SIZES) + " |")
    (OUT / "curve.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
