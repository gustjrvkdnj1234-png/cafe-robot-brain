"""Phase A: compare expert-selection criteria using the EXISTING v1 statistics (no forward passes).

    .venv\\Scripts\\python.exe v2\\scripts\\criteria_overlap.py

Input : results/step4/reap_600/stats_layer*.json (v1, read only)
Checks: v1 plans results/step5/prune_plan_{98,64,48}.json and the expert lists used to build the models
        (models/gemma-4-26B-A4B-it-cafe-{98,64,48}e/prune_info.json, if present)
Output: v2/results/criteria_overlap/{overlap.json, summary.md}

Criteria, per layer, keep k of 128 experts. All four rank on the same statistics: v1's 50:50 reweighting
(fixed prompt scaled so its token weight equals the case tokens' weight), so only the criterion differs.
  reap      mean g*||f(x)|| over tokens routed to the expert      (sum_gate_norm / count)
  freq      routed count                                          (count)
  sum       sum of g*||f(x)|| over all tokens, no protection      (sum_gate_norm)
  sum_prot  v1 rule: experts with >=30% of the layer's tokens first, then sum (must equal the v1 plans)
Ties: reweighted router probability mass, then expert index (as in v1).
Reassignment rate (v1 metric): share of the layer's top-8 assignments that went to removed experts, i.e. the
assignments that must move to another expert after pruning; averaged over the 30 layers. Reported on the
plain v1 statistics (fixed prompt counted once, "4b basis") and on the 50:50 statistics ("new basis").
"""
from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from plan_prune import BUSY, plain_rank, reweight  # noqa: E402  (v1 code, imported read-only)

STATS = ROOT / "results/step4/reap_600"
OUT = ROOT / "v2/results/criteria_overlap"
KEEPS = (98, 64, 48)
CRITERIA = ("reap", "freq", "sum", "sum_prot")
LABEL = {"reap": "REAP (평균)", "freq": "빈도", "sum": "기여 합", "sum_prot": "기여 합 + 보호 (v1)"}


def orders(s):
    rw, _ = reweight(s)
    E = len(rw["count"])
    busy = {e for e in range(E) if rw["frequency"][e] >= BUSY}
    tie = lambda e: (-rw["prob_mass"][e], e)  # noqa: E731
    return rw, busy, {
        "reap": sorted(range(E), key=lambda e: (-rw["reap"][e], *tie(e))),
        "freq": sorted(range(E), key=lambda e: (-rw["count"][e], *tie(e))),
        "sum": sorted(range(E), key=lambda e: (-rw["sum_gate_norm"][e], *tie(e))),
        "sum_prot": sorted(range(E), key=lambda e: (e not in busy, -rw["sum_gate_norm"][e], *tie(e))),
    }


def reassign(stats_c, removed):
    return sum(stats_c["count"][e] for e in removed) / sum(stats_c["count"])


def lost_contrib(stats_c, removed):
    return sum(stats_c["sum_gate_norm"][e] for e in removed) / sum(stats_c["sum_gate_norm"])


def jaccard(a, b):
    return len(a & b) / len(a | b)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    layers = {int(p.stem[-2:]): json.loads(p.read_text(encoding="utf-8")) for p in sorted(STATS.glob("stats_layer*.json"))}
    assert sorted(layers) == list(range(30)), sorted(layers)
    per = {}
    for i, s in layers.items():
        rw, busy, od = orders(s)
        per[i] = {"rw": rw, "plain": s["combined"], "busy": sorted(busy), "orders": od,
                  "plain_reap_order": plain_rank(s["combined"])}

    res = {"source": str(STATS.relative_to(ROOT)).replace("\\", "/"), "busy_threshold": BUSY, "keeps": {}}
    checks = []
    for k in KEEPS:
        # --- verification against v1
        v1 = json.loads((ROOT / f"results/step5/prune_plan_{k}.json").read_text(encoding="utf-8"))["layers"]
        mism_plan = [i for i in per if sorted(per[i]["orders"]["sum_prot"][:k]) != v1[str(i)]["kept"]]
        # v1 pure-REAP (Step 4) removed set = v1 removed - newly removed + saved
        mism_plain = []
        for i in per:
            L = v1[str(i)]
            plain_rm = (set(L["removed"]) - set(L["newly_removed_vs_plain"])) | set(L["saved_vs_plain"])
            if plain_rm != set(per[i]["plain_reap_order"][k:]):
                mism_plain.append(i)
        info = ROOT / f"models/gemma-4-26B-A4B-it-cafe-{k}e/prune_info.json"
        mism_build = None
        if info.exists():
            pi = json.loads(info.read_text(encoding="utf-8"))
            kept_by_layer = pi["kept_experts"]          # {"layer": [expert ids]}
            mism_build = [i for i in per if sorted(kept_by_layer[str(i)]) != sorted(per[i]["orders"]["sum_prot"][:k])]
        checks.append({"keep": k, "sum_prot_vs_v1_plan_mismatch_layers": mism_plan,
                       "plain_reap_vs_v1_step4_plan_mismatch_layers": mism_plain,
                       "sum_prot_vs_built_model_mismatch_layers": mism_build})

        # --- comparison
        kept = {c: {i: set(per[i]["orders"][c][:k]) for i in per} for c in CRITERIA}
        rem = {c: {i: per[i]["orders"][c][k:] for i in per} for c in CRITERIA}
        pairs = {}
        for a, b in itertools.combinations(CRITERIA, 2):
            js = {i: jaccard(kept[a][i], kept[b][i]) for i in per}
            lo = min(js, key=js.get)
            pairs[f"{a}|{b}"] = {"mean": statistics.mean(js.values()), "min": js[lo], "min_layer": lo,
                                 "per_layer": {str(i): round(v, 4) for i, v in js.items()}}
        crit = {}
        for c in CRITERIA:
            crit[c] = {
                "reassign_plain": statistics.mean(reassign(per[i]["plain"], rem[c][i]) for i in per),
                "reassign_5050": statistics.mean(reassign(per[i]["rw"], rem[c][i]) for i in per),
                "lost_contrib_plain": sum(sum(per[i]["plain"]["sum_gate_norm"][e] for e in rem[c][i]) for i in per)
                / sum(sum(per[i]["plain"]["sum_gate_norm"]) for i in per),
                "max_layer_reassign_plain": max(((reassign(per[i]["plain"], rem[c][i]), i) for i in per)),
                "per_layer_reassign_plain": {str(i): round(reassign(per[i]["plain"], rem[c][i]), 4) for i in per},
            }
        layer_mean_j = {i: statistics.mean(pairs[p]["per_layer"][str(i)] for p in pairs) for i in per}
        res["keeps"][str(k)] = {"pairs": pairs, "criteria": crit,
                                "layer_mean_jaccard": {str(i): round(v, 4) for i, v in layer_mean_j.items()},
                                "busy_per_layer": {str(i): len(per[i]["busy"]) for i in per},
                                "sum_vs_sumprot_changed_layers": [i for i in per if kept["sum"][i] != kept["sum_prot"][i]]}
    res["checks"] = checks
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res)
    print(json.dumps(checks))
    print("saved", (OUT / "summary.md").relative_to(ROOT))


def pct(x):
    return f"{x * 100:.1f}%"


def write_md(res):
    L = ["# 전문가 선택 기준 비교 (v1 통계 재사용, 새 계산 없음)", "",
         f"- 입력: `{res['source']}/stats_layer*.json` (v1, 카페 보정 600문항). 새 순전파는 하지 않았다.",
         "- 네 기준 모두 v1의 50:50 가중 통계(고정 프롬프트 전체 가중치 = 문항별 토큰 전체 가중치)로 순위를 매겨, 기준만 다르게 했다. "
         "동점은 라우터 확률 합, 그다음 번호 순(v1과 같음).",
         "- REAP = 배정된 토큰에서의 g·‖f(x)‖ 평균, 빈도 = 배정 수, 기여 합 = 모든 토큰의 g·‖f(x)‖ 합, "
         "기여 합 + 보호 = v1 규칙(층 토큰의 30% 이상을 받는 전문가는 항상 남김).",
         "- 재배정 비율(v1과 같은 지표): 층의 top-8 배정 중 잘린 전문가로 가던 배정의 비율 = 가지치기 뒤 다른 전문가로 바뀌어야 하는 배정. "
         "30개 층 평균. ‘4b 기준’은 고정 프롬프트를 한 번만 센 원래 통계, ‘50:50 기준’은 가중 통계.",
         "- 층 번호는 0부터 센다(0~29).", "", "## 검증", ""]
    for c in res["checks"]:
        b = c["sum_prot_vs_built_model_mismatch_layers"]
        L.append(f"- {c['keep']}개: 기여 합 + 보호 = v1 계획(`results/step5/prune_plan_{c['keep']}.json`) "
                 f"{'**30개 층 모두 일치**' if not c['sum_prot_vs_v1_plan_mismatch_layers'] else '불일치 층 ' + str(c['sum_prot_vs_v1_plan_mismatch_layers'])}; "
                 f"모델을 만들 때 쓴 목록(`prune_info.json`) "
                 f"{'확인 불가' if b is None else ('일치' if not b else '불일치 층 ' + str(b))}; "
                 f"4b 통계의 순수 REAP = v1 4b 계획 "
                 f"{'일치' if not c['plain_reap_vs_v1_step4_plan_mismatch_layers'] else '불일치 층 ' + str(c['plain_reap_vs_v1_step4_plan_mismatch_layers'])}")
    for k, r in res["keeps"].items():
        L += ["", f"## {k}개 남김", "", "| 기준 | 재배정 (4b 기준) | 재배정 (50:50 기준) | 가장 큰 층 (4b 기준) | 잃는 기여 (4b 기준, 전체) |",
              "|---|---:|---:|---:|---:|"]
        for c, v in r["criteria"].items():
            mx, ml = v["max_layer_reassign_plain"]
            L.append(f"| {LABEL[c]} | {pct(v['reassign_plain'])} | {pct(v['reassign_5050'])} | {ml}층 {pct(mx)} | {pct(v['lost_contrib_plain'])} |")
        L += ["", "| 비교 | Jaccard 평균 (30개 층) | 최소 (층) |", "|---|---:|---:|"]
        for p, v in r["pairs"].items():
            a, b = p.split("|")
            L.append(f"| {LABEL[a]} vs {LABEL[b]} | {v['mean']:.3f} | {v['min']:.3f} ({v['min_layer']}층) |")
        lm = sorted(r["layer_mean_jaccard"].items(), key=lambda x: x[1])[:5]
        L += ["", "- 기준끼리 가장 많이 다른 층(6개 쌍의 Jaccard 평균이 낮은 순): "
              + ", ".join(f"{i}층 {v:.3f}" for i, v in lm)]
        ch = r["sum_vs_sumprot_changed_layers"]
        L.append(f"- 보호 규칙이 기여 합 순위를 실제로 바꾼 층: {', '.join(map(str, ch)) if ch else '없음'} "
                 f"(바쁜 전문가가 있는 층: {', '.join(i for i, n in r['busy_per_layer'].items() if n) or '없음'})")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
