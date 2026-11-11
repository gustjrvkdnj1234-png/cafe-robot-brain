"""Analyze REAP statistics (from reap_stats.py) and write a Korean summary + charts. Nothing is pruned.

    .venv-prune\\Scripts\\python.exe scripts\\analyze_reap.py --stats results/step4/reap_600

Outputs (next to --stats' parent, i.e. results/step4/):
    reap_summary.md            Korean report
    reap_prune_plan.json       experts to remove per layer for keep = 98 / 64 / 48 (+ per-layer numbers)
    charts/reap_cumulative.png sorted REAP score curves (cumulative share) per layer
    charts/reap_loss_by_layer.png  share of REAP score lost per layer for each keep

Ranking: per layer, experts are ranked by REAP score (combined = fixed prompt counted once + case tokens),
ties broken by router probability mass. The lowest (128 - keep) experts would be removed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKHORSE = 0.3  # expert used by >= 30% of tokens
KEEPS = (98, 64, 48)
TOPS = (32, 64, 98)


def load(stats_dir: Path):
    meta = json.loads((stats_dir / "meta.json").read_text(encoding="utf-8"))
    layers = {}
    for f in sorted(stats_dir.glob("stats_layer*.json")):
        layers[int(f.stem[-2:])] = json.loads(f.read_text(encoding="utf-8"))
    return meta, layers


def analyze_layer(s):
    c, fx, cs = s["combined"], s["fixed"], s["case"]
    E = len(c["reap"])
    reap = c["reap"]
    contrib = c["sum_gate_norm"]          # weighted sum over tokens of g * ||f||
    count = c["count"]
    order = sorted(range(E), key=lambda e: (-reap[e], -c["prob_mass"][e], e))   # best first
    tot_reap, tot_contrib, tot_count = sum(reap), sum(contrib), sum(count)
    used_f = {e for e in range(E) if fx["count"][e] > 0}
    used_c = {e for e in range(E) if cs["count"][e] > 0}
    usage = {"never": E - len(used_f | used_c), "fixed_only": len(used_f - used_c),
             "case_only": len(used_c - used_f), "both": len(used_f & used_c)}
    sorted_share = [reap[e] / tot_reap for e in order]
    cum, acc = [], 0.0
    for x in sorted_share:
        acc += x
        cum.append(acc)
    top_share = {k: cum[k - 1] for k in TOPS}
    # effective number of experts (exp of entropy of the REAP score distribution)
    p = [x for x in sorted_share if x > 0]
    eff = math.exp(-sum(x * math.log(x) for x in p))
    plans = {}
    for k in KEEPS:
        removed = sorted(order[k:])
        plans[k] = {
            "removed": removed,
            "removed_used_by_fixed": sum(1 for e in removed if e in used_f),
            "removed_fixed_only": sum(1 for e in removed if e in used_f and e not in used_c),
            "removed_never_routed": sum(1 for e in removed if e not in used_f and e not in used_c),
            "lost_reap_share": sum(reap[e] for e in removed) / tot_reap,
            "lost_contrib_share": sum(contrib[e] for e in removed) / tot_contrib,
            "rerouted_token_share": sum(count[e] for e in removed) / tot_count,
            "lost_reap": sum(reap[e] for e in removed),
            "lost_contrib": sum(contrib[e] for e in removed),
        }
    return {"order": order, "usage": usage, "cum": cum, "sorted_share": sorted_share, "top_share": top_share,
            "effective_experts": eff, "plans": plans, "tot_reap": tot_reap, "tot_contrib": tot_contrib}


def compact(nums):
    """[1,2,3,7,9,10] -> '1-3, 7, 9-10'"""
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def charts(res, out_dir: Path, n_layers_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(res)

    # 1. cumulative share curves
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    cmap = plt.get_cmap("viridis")
    for i in layers:
        ax.plot(range(1, len(res[i]["cum"]) + 1), [x * 100 for x in res[i]["cum"]], color=cmap(i / max(layers)),
                lw=1.6 if n_layers_type[i] == "full_attention" else 1.0,
                ls="-" if n_layers_type[i] == "sliding_attention" else "--", alpha=0.9)
    for k in KEEPS:
        ax.axvline(k, color="gray", lw=0.8, ls=":")
        ax.text(k + 0.8, 3, f"{k}개 남김", fontsize=9, color="dimgray", rotation=90, va="bottom")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(layers), max(layers)))
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("층 번호 (점선 = 전체 어텐션 층)")
    ax.set_xlim(1, 128)
    ax.set_ylim(0, 101)
    ax.set_xlabel("REAP 점수가 높은 순서로 남긴 전문가 수")
    ax.set_ylabel("전체 REAP 점수 중 차지하는 비율 (%)")
    ax.set_title("층별 전문가 REAP 점수 누적 곡선 (카페 문항 600개)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "reap_cumulative.png"
    fig.savefig(p1)
    plt.close(fig)

    # 2. lost share per layer
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    w = 0.27
    colors = {98: "#8fbcd4", 64: "#3b7ea1", 48: "#173f5f"}
    for j, k in enumerate(KEEPS):
        ax.bar([i + (j - 1) * w for i in layers], [res[i]["plans"][k]["lost_reap_share"] * 100 for i in layers],
               width=w, color=colors[k], label=f"{k}개 남김 ({128 - k}개 제거)", zorder=2)
    for i in layers:
        if n_layers_type[i] == "full_attention":
            ax.axvspan(i - 0.5, i + 0.5, color="#f2c94c", alpha=0.25, lw=0, zorder=0)
    ax.set_xticks(layers)
    ax.set_xlabel("층 번호 (노란 배경 = 전체 어텐션 층)")
    ax.set_ylabel("잃는 REAP 점수 비율 (%)")
    ax.set_title("전문가를 잘라낼 때 층별로 잃는 REAP 점수")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "reap_loss_by_layer.png"
    fig.savefig(p2)
    plt.close(fig)
    return p1, p2


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/step4/reap_600")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    stats_dir = ROOT / a.stats
    out = stats_dir.parent
    meta, layers = load(stats_dir)
    cfg = json.loads((ROOT / "models/gemma-4-26B-A4B-it/config.json").read_text(encoding="utf-8"))["text_config"]
    ltype = {i: cfg["layer_types"][i] for i in range(cfg["num_hidden_layers"])}
    res = {i: analyze_layer(s) for i, s in layers.items()}
    n = len(res)
    complete = n == cfg["num_hidden_layers"]

    # overall
    overall = {}
    for k in KEEPS:
        overall[k] = {
            "lost_reap_share": sum(r["plans"][k]["lost_reap"] for r in res.values()) / sum(r["tot_reap"] for r in res.values()),
            "lost_contrib_share": sum(r["plans"][k]["lost_contrib"] for r in res.values()) / sum(r["tot_contrib"] for r in res.values()),
            "removed_used_by_fixed": sum(r["plans"][k]["removed_used_by_fixed"] for r in res.values()),
            "removed_fixed_only": sum(r["plans"][k]["removed_fixed_only"] for r in res.values()),
            "removed_never_routed": sum(r["plans"][k]["removed_never_routed"] for r in res.values()),
            "removed_total": sum(len(r["plans"][k]["removed"]) for r in res.values()),
            "rerouted_token_share_mean": sum(r["plans"][k]["rerouted_token_share"] for r in res.values()) / n,
        }

    # risky layers: rank by loss at keep 64 (and effective experts)
    loss64 = sorted(res, key=lambda i: -res[i]["plans"][64]["lost_reap_share"])
    mean64 = sum(res[i]["plans"][64]["lost_reap_share"] for i in res) / n
    sd64 = (sum((res[i]["plans"][64]["lost_reap_share"] - mean64) ** 2 for i in res) / n) ** 0.5
    risky = [i for i in loss64 if res[i]["plans"][64]["lost_reap_share"] > mean64 + sd64]

    plan = {"source": a.stats, "n_cases": meta["n_cases"], "ranking": "combined REAP score, ties by router prob mass",
            "keep": {str(k): {str(i): res[i]["plans"][k]["removed"] for i in sorted(res)} for k in KEEPS},
            "per_layer": {str(i): {"usage": res[i]["usage"], "top_share": {str(t): round(v, 4) for t, v in res[i]["top_share"].items()},
                                   "effective_experts": round(res[i]["effective_experts"], 1),
                                   **{f"keep{k}": {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in res[i]["plans"][k].items()
                                                   if kk != "removed"} for k in KEEPS}} for i in sorted(res)},
            "overall": {str(k): {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in overall.items()}}
    (out / "reap_prune_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    p1, p2 = charts(res, out / "charts", ltype)

    # ---------------- markdown
    L = ["# 4b단계: 전문가 REAP 통계 요약 (가지치기 전)", ""]
    if not complete:
        L += [f"> **미완성:** {n}/{cfg['num_hidden_layers']}개 층만 계산됨", ""]
    L += [
        f"- 데이터: 카페 보정 세트 {meta['n_cases']}문항(`data/calib_cafe.jsonl`), 모델 `gemma-4-26B-A4B-it` 원본(BF16 → float32 계산), CPU에서 층 단위 실행",
        f"- 계산 토큰 {meta['tokens_computed']:,}개(접두사 트리, 따로 돌리면 {meta['tokens_if_alone']:,}개). "
        f"가중치: 고정 프롬프트(시스템 지시문 + 도구 설명, {meta['fixed_tokens_per_case']:,}토큰)는 전체 합쳐 한 번, 문항별 부분(평균 {meta['case_tokens_mean']}토큰)은 문항마다 한 번.",
        "- REAP 점수: 전문가로 보내진 토큰들의 '게이트 값 × 전문가 출력 노름' 평균. 층마다 이 점수가 낮은 전문가부터 잘라낸다고 가정한다(모든 층에서 같은 수를 남김).",
        "- 아직 아무것도 잘라내지 않았다. 아래의 '제거 대상'은 계획일 뿐이다.",
        "",
        "용어:",
        "- **REAP 점수 비율:** 층 안의 전문가별 REAP 점수 합 중 일부 전문가가 차지하는 비율.",
        "- **기여 비율:** 모든 토큰에 대해 '게이트 × 노름'을 더한 값 중 비율. 자주 쓰이는 전문가일수록 크다.",
        "- **재배정 토큰 비율:** 잘라낸 전문가로 가던 토큰 배정의 비율. 잘라낸 뒤에는 이 토큰들이 다른 전문가로 간다.",
        "",
        "## 전체 요약", "",
        "| 남기는 전문가 | 제거 (층당) | 잃는 REAP 점수 (전체) | 잃는 기여 (전체) | 평균 재배정 토큰 | 제거 대상 중 고정 프롬프트가 쓰던 전문가 | 그중 고정 프롬프트만 쓰던 전문가 | 제거 대상 중 한 번도 안 쓰인 전문가 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in KEEPS:
        o = overall[k]
        L.append(f"| {k} | {128 - k} | {pct(o['lost_reap_share'])} | {pct(o['lost_contrib_share'])} | {pct(o['rerouted_token_share_mean'])} | "
                 f"{o['removed_used_by_fixed']} / {o['removed_total']} | {o['removed_fixed_only']} | {o['removed_never_routed']} |")
    L += ["", f"![층별 누적 곡선](charts/{p1.name})", "", f"![층별 손실](charts/{p2.name})", ""]

    # 1. usage
    L += ["## 1. 층별 전문가 사용 현황", "",
          "| 층 | 종류 | 한 번도 안 쓰임 | 고정 프롬프트만 | 문항별 부분만 | 둘 다 |", "|---:|---|---:|---:|---:|---:|"]
    for i in sorted(res):
        u = res[i]["usage"]
        L.append(f"| {i} | {'전체' if ltype[i] == 'full_attention' else '슬라이딩'} | {u['never']} | {u['fixed_only']} | {u['case_only']} | {u['both']} |")
    tot = {k: sum(res[i]["usage"][k] for i in res) for k in ("never", "fixed_only", "case_only", "both")}
    L += [f"| **합계** | | **{tot['never']}** | **{tot['fixed_only']}** | **{tot['case_only']}** | **{tot['both']}** |", ""]

    # 2. distribution
    L += ["## 2. REAP 점수 분포와 집중도", "",
          "상위 N개 전문가가 차지하는 REAP 점수 비율. '유효 전문가 수'는 점수 분포의 엔트로피로 잰 값으로, 작을수록 점수가 소수에게 몰려 있다.", "",
          "| 층 | 상위 32 | 상위 64 | 상위 98 | 유효 전문가 수 | 1위 전문가 (점수 비율) |", "|---:|---:|---:|---:|---:|---|"]
    for i in sorted(res):
        r = res[i]
        e1 = r["order"][0]
        L.append(f"| {i} | {pct(r['top_share'][32])} | {pct(r['top_share'][64])} | {pct(r['top_share'][98])} | "
                 f"{r['effective_experts']:.0f} | {e1}번 ({pct(r['sorted_share'][0])}) |")
    L.append("")

    # 3. per keep
    L += ["## 3. 남기는 수별 제거 계획", ""]
    for k in KEEPS:
        L += [f"### {k}개 남김 (층당 {128 - k}개 제거)", "",
              "| 층 | 잃는 REAP 점수 | 잃는 기여 | 재배정 토큰 | 제거 대상 중 고정 프롬프트 사용 | 그중 고정만 | 한 번도 안 쓰임 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for i in sorted(res):
            pl = res[i]["plans"][k]
            L.append(f"| {i} | {pct(pl['lost_reap_share'])} | {pct(pl['lost_contrib_share'])} | {pct(pl['rerouted_token_share'])} | "
                     f"{pl['removed_used_by_fixed']} | {pl['removed_fixed_only']} | {pl['removed_never_routed']} |")
        o = overall[k]
        L += [f"| **전체** | **{pct(o['lost_reap_share'])}** | **{pct(o['lost_contrib_share'])}** | {pct(o['rerouted_token_share_mean'])} (평균) | "
              f"**{o['removed_used_by_fixed']}** | **{o['removed_fixed_only']}** | **{o['removed_never_routed']}** |", "",
              f"<details><summary>층별 제거 대상 전문가 번호 ({k}개 남김)</summary>", ""]
        for i in sorted(res):
            L.append(f"- {i}층: {compact(res[i]['plans'][k]['removed'])}")
        L += ["", "</details>", ""]

    # 4. risky
    L += ["## 4. 자르기 위험해 보이는 층", "",
          f"기준: 64개 남길 때 잃는 REAP 점수가 층 평균({pct(mean64)}) + 표준편차({pct(sd64)})보다 큰 층.", ""]
    if risky:
        L += ["| 층 | 종류 | 64개 남김 손실 | 48개 남김 손실 | 상위 64 비율 | 유효 전문가 수 | 64개 남길 때 제거되는 고정 프롬프트 전문가 |",
              "|---:|---|---:|---:|---:|---:|---:|"]
        for i in risky:
            r = res[i]
            L.append(f"| {i} | {'전체' if ltype[i] == 'full_attention' else '슬라이딩'} | {pct(r['plans'][64]['lost_reap_share'])} | "
                     f"{pct(r['plans'][48]['lost_reap_share'])} | {pct(r['top_share'][64])} | {r['effective_experts']:.0f} | "
                     f"{r['plans'][64]['removed_used_by_fixed']} |")
    else:
        L.append("- 해당하는 층 없음")
    L.append("")

    # 4b. frequently routed experts with low REAP (mean-based score hides them)
    def busy(i, thr=WORKHORSE):
        c = layers[i]["combined"]
        return {e: c["count"][e] / c["weighted_tokens"] for e in range(len(c["count"])) if c["count"][e] / c["weighted_tokens"] >= thr}
    rr = {k: [res[i]["plans"][k]["rerouted_token_share"] for i in sorted(res)] for k in KEEPS}
    m98 = sum(rr[98]) / n
    s98 = (sum((x - m98) ** 2 for x in rr[98]) / n) ** 0.5
    risky2 = sorted((i for i in res if res[i]["plans"][98]["rerouted_token_share"] > m98 + s98),
                    key=lambda i: -res[i]["plans"][98]["rerouted_token_share"])
    L += ["### 4-2. 자주 쓰이는데 REAP 점수가 낮아 잘리는 전문가가 있는 층", "",
          "REAP 점수는 '쓰일 때의 평균'이라서, 거의 모든 토큰이 거쳐 가지만 한 번에 내는 출력이 작은 전문가는 점수가 낮게 나온다. "
          f"이런 층은 손실 REAP 점수는 작아도 재배정 토큰이 크다. "
          f"기준: 98개 남길 때 재배정 토큰 비율이 층 평균({pct(m98)}) + 표준편차({pct(s98)})보다 큰 층. "
          f"'바쁜 전문가'는 토큰의 {WORKHORSE * 100:.0f}% 이상이 거쳐 가는 전문가.", ""]
    if risky2:
        L += ["| 층 | 종류 | 98개 남김: 잃는 REAP / 잃는 기여 / 재배정 | 64개 남김: 잃는 REAP / 잃는 기여 / 재배정 | 98개 남길 때 잘리는 바쁜 전문가 (토큰 비율) | 64개 남길 때 잘리는 바쁜 전문가 수 |",
              "|---:|---|---|---|---|---:|"]
        for i in risky2:
            b = busy(i)
            p98, p64 = res[i]["plans"][98], res[i]["plans"][64]
            cut98 = sorted((e for e in b if e in set(p98["removed"])), key=lambda e: -b[e])
            cut64 = [e for e in b if e in set(p64["removed"])]
            L.append(f"| {i} | {'전체' if ltype[i] == 'full_attention' else '슬라이딩'} | "
                     f"{pct(p98['lost_reap_share'])} / {pct(p98['lost_contrib_share'])} / {pct(p98['rerouted_token_share'])} | "
                     f"{pct(p64['lost_reap_share'])} / {pct(p64['lost_contrib_share'])} / {pct(p64['rerouted_token_share'])} | "
                     f"{', '.join(f'{e}번 ({b[e] * 100:.0f}%)' for e in cut98) or '-'} | {len(cut64)} / {len(b)} |")
    else:
        L.append("- 해당하는 층 없음")
    L.append("")
    L.append("<!-- INTERPRETATION -->")
    (out / "reap_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"layers analyzed: {n}; saved reap_summary.md, reap_prune_plan.json, {p1.name}, {p2.name}")
    for k in KEEPS:
        o = overall[k]
        print(f"keep {k}: lost REAP {pct(o['lost_reap_share'])}, lost contrib {pct(o['lost_contrib_share'])}, "
              f"rerouted {pct(o['rerouted_token_share_mean'])}, removed used-by-fixed {o['removed_used_by_fixed']}/{o['removed_total']} "
              f"(fixed-only {o['removed_fixed_only']}, never {o['removed_never_routed']})")
    print("risky (keep 64):", [(i, pct(res[i]['plans'][64]['lost_reap_share'])) for i in risky])


if __name__ == "__main__":
    main()
