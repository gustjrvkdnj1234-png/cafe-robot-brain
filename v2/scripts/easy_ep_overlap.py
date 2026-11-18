"""EASY-EP keep-sets from v2/results/easy_ep/stats and comparison with the Phase A criteria (light, no model).

    .venv\\Scripts\\python.exe v2\\scripts\\easy_ep_overlap.py

- Sanity check: count and sum g*||f|| of the new run vs v1 (results/step4/reap_600), per layer / expert / group.
- EASY-EP ranking per layer: sum over tokens of g*||e||*s_t, with the same 50:50 reweighting as the other
  criteria (fixed-prompt part scaled so its token weight equals the case tokens' weight). Ties: reweighted router
  probability mass (v1 stats), then index. Variant A is the default; B and C for reference.
- Compared with REAP / frequency / sum / sum + protection (v1) from v2/scripts/criteria_overlap.py:
  Jaccard of keep-sets (mean over 30 layers, min layer) and reassignment rate (v1 metric) on plain and 50:50 stats.
Outputs: v2/results/easy_ep/{overlap.json, summary.md}
"""
from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "v2/scripts"))
from criteria_overlap import KEEPS, LABEL, jaccard, lost_contrib, orders, reassign  # noqa: E402

V1 = ROOT / "results/step4/reap_600"
NEW = ROOT / "v2/results/easy_ep/stats"
OUT = ROOT / "v2/results/easy_ep"
VARIANTS = ("A", "B", "C")
LABEL = {**LABEL, "easy_A": "EASY-EP A (기본)", "easy_B": "EASY-EP B", "easy_C": "EASY-EP C"}
BASE = ("reap", "freq", "sum", "sum_prot")


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    v1 = {i: json.loads((V1 / f"stats_layer{i:02d}.json").read_text(encoding="utf-8")) for i in range(30)}
    ez = {i: json.loads((NEW / f"easy_layer{i:02d}.json").read_text(encoding="utf-8")) for i in range(30)}
    meta = json.loads((NEW / "meta.json").read_text(encoding="utf-8"))

    # ---- sanity: same counts and g*||f|| as v1
    worst = {"count": (0.0, None), "sum_gate_norm": (0.0, None), "weighted_tokens": (0.0, None)}
    for i in range(30):
        for g in ("fixed", "case", "combined"):
            for f in ("count", "sum_gate_norm"):
                for e, (x, y) in enumerate(zip(ez[i][g][f], v1[i][g][f])):
                    r = abs(x - y) / max(abs(y), 1e-9) if (x or y) else 0.0
                    if r > worst[f][0]:
                        worst[f] = (r, f"layer {i} {g} expert {e}: {x} vs {y}")
            r = abs(ez[i][g]["weighted_tokens"] - v1[i][g]["weighted_tokens"]) / v1[i][g]["weighted_tokens"]
            if r > worst["weighted_tokens"][0]:
                worst["weighted_tokens"] = (r, f"layer {i} {g}")

    # ---- orders
    per = {}
    for i in range(30):
        rw, busy, od = orders(v1[i])
        f, c = ez[i]["fixed"], ez[i]["case"]
        w = c["weighted_tokens"] / f["weighted_tokens"]
        for v in VARIANTS:
            imp = [c[f"easy_{v}"][e] + w * f[f"easy_{v}"][e] for e in range(128)]
            od[f"easy_{v}"] = sorted(range(128), key=lambda e: (-imp[e], -rw["prob_mass"][e], e))
        per[i] = {"rw": rw, "plain": v1[i]["combined"], "orders": od,
                  "s_mean": ez[i]["combined"]["s_mean"]}
    crits = (*(f"easy_{v}" for v in VARIANTS), *BASE)

    res = {"sanity_max_rel_diff": {k: {"value": v[0], "where": v[1]} for k, v in worst.items()},
           "runtime_s": meta.get("total_s"), "peak_rss_gb": meta.get("peak_rss_gb"), "threads": meta.get("threads"),
           "s_mean_by_layer": {str(i): per[i]["s_mean"] for i in per}, "keeps": {}}
    for k in KEEPS:
        kept = {c: {i: set(per[i]["orders"][c][:k]) for i in per} for c in crits}
        rem = {c: {i: per[i]["orders"][c][k:] for i in per} for c in crits}
        pairs = {}
        for a, b in [(a, b) for a in crits[:3] for b in crits if crits.index(b) > crits.index(a)]:
            js = {i: jaccard(kept[a][i], kept[b][i]) for i in per}
            lo = min(js, key=js.get)
            pairs[f"{a}|{b}"] = {"mean": statistics.mean(js.values()), "min": js[lo], "min_layer": lo,
                                 "per_layer": {str(i): round(x, 4) for i, x in js.items()}}
        cr = {}
        for c in crits:
            rp = {i: reassign(per[i]["plain"], rem[c][i]) for i in per}
            mx = max(rp, key=rp.get)
            cr[c] = {"reassign_plain": statistics.mean(rp.values()),
                     "reassign_5050": statistics.mean(reassign(per[i]["rw"], rem[c][i]) for i in per),
                     "lost_contrib_plain": sum(sum(per[i]["plain"]["sum_gate_norm"][e] for e in rem[c][i]) for i in per)
                     / sum(sum(per[i]["plain"]["sum_gate_norm"]) for i in per),
                     "max_layer": mx, "max_layer_reassign_plain": rp[mx],
                     "kept": {str(i): sorted(kept[c][i]) for i in per} if c.startswith("easy_") else None}
        res["keeps"][str(k)] = {"pairs": pairs, "criteria": cr}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res)
    print(json.dumps(res["sanity_max_rel_diff"], ensure_ascii=False))
    print("saved", (OUT / "summary.md").relative_to(ROOT))


def write_md(res):
    s = res["sanity_max_rel_diff"]
    L = ["# EASY-EP 선택과 다른 기준 비교", "",
         f"- 통계: `v2/results/easy_ep/stats/easy_layer*.json` (v1과 같은 600문항, 같은 층별 CPU 방식). "
         f"실행 시간 {res['runtime_s'] / 3600:.2f}시간({res['runtime_s']:,.0f} s), 최대 RSS {res['peak_rss_gb']} GB, 스레드 {res['threads']}.",
         "- 순위: 토큰마다 g·‖e‖·s_t를 더한 값. 다른 기준과 같은 50:50 가중(고정 프롬프트 전체 가중치 = 문항별 토큰 전체 가중치), "
         "동점은 라우터 확률 합, 그다음 번호. 변형 정의는 `README.md`.",
         "- 재배정 비율은 v1과 같은 지표(층의 top-8 배정 중 잘린 전문가로 가던 비율, 30개 층 평균). 층 번호는 0부터 센다.", "",
         "## 확인: v1 통계와 같은가", "",
         f"- 배정 수(count) 최대 상대 차이 {s['count']['value']:.2e}, g·‖f‖ 합 {s['sum_gate_norm']['value']:.2e}, "
         f"토큰 가중치 {s['weighted_tokens']['value']:.2e} (모든 층·전문가·그룹)."]
    for k, r in res["keeps"].items():
        L += ["", f"## {k}개 남김", "", "| 기준 | 재배정 (4b 기준) | 재배정 (50:50 기준) | 가장 큰 층 | 잃는 기여 (4b 기준) |",
              "|---|---:|---:|---:|---:|"]
        for c, v in r["criteria"].items():
            L.append(f"| {LABEL[c]} | {pct(v['reassign_plain'])} | {pct(v['reassign_5050'])} | "
                     f"{v['max_layer']}층 {pct(v['max_layer_reassign_plain'])} | {pct(v['lost_contrib_plain'])} |")
        L += ["", "| 비교 | Jaccard 평균 (30개 층) | 최소 (층) |", "|---|---:|---:|"]
        for p, v in r["pairs"].items():
            a, b = p.split("|")
            L.append(f"| {LABEL[a]} vs {LABEL[b]} | {v['mean']:.3f} | {v['min']:.3f} ({v['min_layer']}층) |")
        js = r["pairs"]["easy_A|sum"]["per_layer"]
        lo = sorted(js.items(), key=lambda x: x[1])[:5]
        L += ["", "- EASY-EP A와 기여 합(v1)이 가장 많이 다른 층: " + ", ".join(f"{i}층 {x:.3f}" for i, x in lo)]
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
