"""Step 5a.1: expert prune plans with two rules on top of REAP (nothing is pruned here).

    .venv-prune\\Scripts\\python.exe scripts\\plan_prune.py --stats results/step4/reap_600 --out results/step5

Rules (decided by the project owner):
  A. Busy experts are always kept: an expert that at least 30% of a layer's tokens are routed to.
  B. Fixed-prompt weight: the fixed-prompt statistics (system prompt + tool declarations, counted once in Step 4)
     are scaled so their total token weight equals the total weight of all case-specific tokens (50:50).
Ranking per layer (chosen, "option 3"): busy experts first, then by the reweighted contribution sum
  sum over tokens of g * ||f|| (so frequency counts), ties by reweighted router prob mass, then index.
Why not the REAP mean with B: REAP = sum / count, so scaling the fixed part leaves experts used only by the
  fixed prompt unchanged (rule B saved none of them). The report compares the three options at each keep.

Outputs:
  prune_plan_{98,64,48}.json   kept / removed experts per layer + numbers
  prune_plans.md               Korean comparison with the plain REAP plan (results/step4/reap_prune_plan.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUSY = 0.30
KEEPS = (98, 64, 48)
ADD = ("count", "sum_gate", "sum_norm", "sum_gate_norm", "prob_mass")


def reweight(s):
    """Combined stats with fixed part scaled to the case part's total weight."""
    f, c = s["fixed"], s["case"]
    w = c["weighted_tokens"] / f["weighted_tokens"]
    out = {k: [c[k][e] + w * f[k][e] for e in range(len(c[k]))] for k in ADD}
    out["weighted_tokens"] = c["weighted_tokens"] + w * f["weighted_tokens"]
    out["reap"] = [out["sum_gate_norm"][e] / out["count"][e] if out["count"][e] else 0.0 for e in range(len(c["count"]))]
    out["frequency"] = [x / out["weighted_tokens"] for x in out["count"]]
    return out, w


def plain_rank(c):
    E = len(c["reap"])
    return sorted(range(E), key=lambda e: (-c["reap"][e], -c["prob_mass"][e], e))


def metrics(c, removed):
    """Losses measured on stats c (dict with reap, sum_gate_norm, count)."""
    return {"lost_reap_share": sum(c["reap"][e] for e in removed) / sum(c["reap"]),
            "lost_contrib_share": sum(c["sum_gate_norm"][e] for e in removed) / sum(c["sum_gate_norm"]),
            "rerouted_share": sum(c["count"][e] for e in removed) / sum(c["count"]),
            "lost_reap": sum(c["reap"][e] for e in removed), "tot_reap": sum(c["reap"]),
            "lost_contrib": sum(c["sum_gate_norm"][e] for e in removed), "tot_contrib": sum(c["sum_gate_norm"])}


def pct(x):
    return f"{x * 100:.1f}%"


def compact(nums):
    out, i = [], 0
    nums = sorted(nums)
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out) or "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/step4/reap_600")
    ap.add_argument("--out", default="results/step5")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sd, out = ROOT / a.stats, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    layers = {int(p.stem[-2:]): json.loads(p.read_text(encoding="utf-8")) for p in sorted(sd.glob("stats_layer*.json"))}
    assert len(layers) == 30, len(layers)

    per = {}
    for i, s in layers.items():
        rw, w = reweight(s)
        c0 = s["combined"]
        E = len(c0["count"])
        busy = sorted(e for e in range(E) if rw["frequency"][e] >= BUSY)
        busy_plain = sorted(e for e in range(E) if c0["frequency"][e] >= BUSY)
        new_order = sorted(range(E), key=lambda e: (e not in busy, -rw["sum_gate_norm"][e], -rw["prob_mass"][e], e))
        fo = {e for e in range(E) if s["fixed"]["count"][e] > 0 and s["case"]["count"][e] == 0}
        opt = {"opt1": sorted(range(E), key=lambda e: (e not in busy and e not in fo, -rw["reap"][e], -rw["prob_mass"][e], e)),
               "opt2": sorted(range(E), key=lambda e: (e not in busy, -rw["reap"][e], -rw["prob_mass"][e], e)),
               "opt3": new_order}
        old_order = plain_rank(c0)
        used_f = {e for e in range(E) if s["fixed"]["count"][e] > 0}
        used_c = {e for e in range(E) if s["case"]["count"][e] > 0}
        fixed_only = used_f - used_c
        per[i] = {"w": w, "busy": busy, "busy_plain": busy_plain, "fixed_only": sorted(fixed_only), "plans": {},
                  "options": {k: {o: {"plain": metrics(c0, ordr[k:]), "new": metrics(rw, ordr[k:]),
                                      "fixed_only_cut": len(set(ordr[k:]) & fo)} for o, ordr in opt.items()} for k in KEEPS}}
        for k in KEEPS:
            assert len(busy) <= k
            new_rm, old_rm = sorted(new_order[k:]), sorted(old_order[k:])
            per[i]["plans"][k] = {
                "kept": sorted(new_order[:k]), "removed": new_rm,
                "plain_removed": old_rm,
                "saved": sorted(set(old_rm) - set(new_rm)),       # kept now, removed by plain plan
                "newly_removed": sorted(set(new_rm) - set(old_rm)),
                "saved_fixed_only": sorted((set(old_rm) - set(new_rm)) & fixed_only),
                "saved_busy": sorted((set(old_rm) - set(new_rm)) & set(busy)),
                "removed_fixed_only": sorted(set(new_rm) & fixed_only),
                "new_on_plain_stats": metrics(c0, new_rm), "plain_on_plain_stats": metrics(c0, old_rm),
                "new_on_new_stats": metrics(rw, new_rm), "plain_on_new_stats": metrics(rw, old_rm),
            }

    def overall(k, key):
        lr = sum(per[i]["plans"][k][key]["lost_reap"] for i in per) / sum(per[i]["plans"][k][key]["tot_reap"] for i in per)
        lc = sum(per[i]["plans"][k][key]["lost_contrib"] for i in per) / sum(per[i]["plans"][k][key]["tot_contrib"] for i in per)
        rr = sum(per[i]["plans"][k][key]["rerouted_share"] for i in per) / len(per)
        return lr, lc, rr

    r4 = lambda x: round(x, 4) if isinstance(x, float) else x
    for k in KEEPS:
        plan = {"keep": k, "num_experts_original": 128, "top_k": 8, "source_stats": a.stats,
                "rules": {"A_busy_threshold": BUSY, "B_fixed_weight": "fixed total weight = case total weight",
                          "fixed_scale_factor": round(per[0]["w"], 4),
                          "ranking": "busy first, then reweighted contribution sum (g*||f|| over tokens), ties by reweighted prob mass, then index"},
                "layers": {str(i): {"kept": per[i]["plans"][k]["kept"], "removed": per[i]["plans"][k]["removed"],
                                    "busy": per[i]["busy"],
                                    "saved_vs_plain": per[i]["plans"][k]["saved"],
                                    "newly_removed_vs_plain": per[i]["plans"][k]["newly_removed"],
                                    "removed_fixed_only": per[i]["plans"][k]["removed_fixed_only"],
                                    **{key: {kk: r4(vv) for kk, vv in per[i]["plans"][k][key].items() if not kk.startswith(("lost_reap", "tot_", "lost_contrib")) or kk.endswith("share")}
                                       for key in ("new_on_plain_stats", "plain_on_plain_stats", "new_on_new_stats", "plain_on_new_stats")}}
                           for i in sorted(per)},
                "overall": {key: dict(zip(("lost_reap_share", "lost_contrib_share", "rerouted_share_mean"), map(r4, overall(k, key))))
                            for key in ("new_on_plain_stats", "plain_on_plain_stats", "new_on_new_stats", "plain_on_new_stats")}}
        (out / f"prune_plan_{k}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------------- Korean report
    fo_total = sum(len(per[i]["fixed_only"]) for i in per)
    L = ["# 5a단계: 규칙 A·B를 넣은 가지치기 계획", "",
         "4b단계의 순수 REAP 계획과 비교한다. 아직 아무것도 잘라내지 않았다.", "",
         f"- **규칙 A (바쁜 전문가 보호):** 층의 토큰 중 {BUSY * 100:.0f}% 이상이 거쳐 가는 전문가는 항상 남긴다. "
         "판정은 규칙 B로 가중치를 바꾼 통계로 했다.",
         f"- **규칙 B (고정 프롬프트 가중치):** 고정 프롬프트 통계에 {per[0]['w']:.2f}배를 곱해, 고정 프롬프트 전체 가중치"
         f"({layers[0]['fixed']['weighted_tokens']:,.0f}토큰)를 문항별 토큰 전체 가중치({layers[0]['case']['weighted_tokens']:,.0f}토큰)와 같게(50:50) 만든 뒤 "
         "통계(배정 수, 게이트 × 노름의 합 등)를 다시 계산했다.",
         "- **순위 (선택한 방식, 아래 선택지 3):** 바쁜 전문가 먼저, 그다음 50:50 가중치로 잰 **기여 합**(모든 토큰에 대한 '게이트 × 노름'의 합, 자주 쓰일수록 커짐) 순. 같으면 라우터 확률 합, 그다음 번호 순.",
         "- **왜 REAP 점수를 그대로 쓰지 않았나:** REAP 점수는 '합 ÷ 배정 수'인 평균이라, 고정 프롬프트 통계에 몇 배를 곱해도 고정 프롬프트만 쓰는 전문가의 점수는 그대로다. "
         "실제로 규칙 B를 REAP 점수에 적용했을 때 살아난 '고정 프롬프트만' 전문가는 0개였다.",
         "- 손실은 두 가지 기준으로 잰다. **4b 기준**은 고정 프롬프트를 한 번만 센 원래 통계, **새 기준**은 규칙 B의 50:50 통계다. "
         "각 계획은 자기 기준에서 유리하므로 두 기준을 함께 본다.", "",
         "## 세 가지 선택지 비교", "",
         "모두 규칙 A(바쁜 전문가 보호)와 규칙 B(50:50 가중치)를 쓴다. 차이는 순위를 매기는 기준이다.", "",
         "1. **REAP 점수 + '고정 프롬프트만' 전문가도 보호:** 규칙 A처럼 고정 프롬프트만 쓰는 전문가를 항상 남긴다.",
         "2. **REAP 점수 그대로:** 처음 정한 규칙 A+B 그대로.",
         "3. **기여 합 (선택):** 쓰일 때의 평균이 아니라 전체 합으로 순위를 매긴다. 자주 쓰이지 않는 전문가가 먼저 잘린다.", "",
         "손실은 4b 기준(고정 프롬프트 한 번) 통계로 쟀다. 괄호 안은 새 기준(50:50).", ""]
    for k in KEEPS:
        L += [f"**{k}개 남김**", "",
              "| 선택지 | 잃는 REAP 점수 | 잃는 기여 | 재배정 (층 평균) | 잘리는 '고정 프롬프트만' 전문가 (328개 중) |", "|---|---:|---:|---:|---:|"]
        for o, name in (("opt1", "1. REAP + 고정만 보호"), ("opt2", "2. REAP 그대로"), ("opt3", "3. 기여 합 (선택)")):
            vals = {}
            for b in ("plain", "new"):
                X = [per[i]["options"][k][o][b] for i in per]
                vals[b] = (sum(x["lost_reap"] for x in X) / sum(x["tot_reap"] for x in X),
                           sum(x["lost_contrib"] for x in X) / sum(x["tot_contrib"] for x in X),
                           sum(x["rerouted_share"] for x in X) / len(X))
            cut = sum(per[i]["options"][k][o]["fixed_only_cut"] for i in per)
            L.append(f"| {name} | {pct(vals['plain'][0])} ({pct(vals['new'][0])}) | {pct(vals['plain'][1])} ({pct(vals['new'][1])}) | "
                     f"{pct(vals['plain'][2])} ({pct(vals['new'][2])}) | {cut} |")
        L.append("")
    L += ["## 순수 REAP 계획(4b)과 비교한 전체 요약", "",
         "| 남김 | 계획 | 잃는 REAP (4b 기준) | 잃는 기여 (4b 기준) | 재배정 (4b 기준, 층 평균) | 잃는 REAP (새 기준) | 잃는 기여 (새 기준) | 재배정 (새 기준) |",
         "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for k in KEEPS:
        for name, a_key, b_key in (("순수 REAP (4b)", "plain_on_plain_stats", "plain_on_new_stats"),
                                   ("규칙 A+B (기여 합)", "new_on_plain_stats", "new_on_new_stats")):
            x, y = overall(k, a_key), overall(k, b_key)
            L.append(f"| {k} | {name} | {pct(x[0])} | {pct(x[1])} | {pct(x[2])} | {pct(y[0])} | {pct(y[1])} | {pct(y[2])} |")
    L.append("")
    L += ["| 남김 | 계획이 바뀐 전문가 (층 합계, 새로 살린 수 = 새로 자른 수) | 살린 바쁜 전문가 | 살린 '고정 프롬프트만' 전문가 | 그래도 잘리는 '고정 프롬프트만' 전문가 |",
          "|---:|---:|---:|---:|---:|"]
    for k in KEEPS:
        sv = sum(len(per[i]["plans"][k]["saved"]) for i in per)
        sb = sum(len(per[i]["plans"][k]["saved_busy"]) for i in per)
        sf = sum(len(per[i]["plans"][k]["saved_fixed_only"]) for i in per)
        rf = sum(len(per[i]["plans"][k]["removed_fixed_only"]) for i in per)
        L.append(f"| {k} | {sv} | {sb} | {sf} | {rf} / {fo_total} |")
    L.append("")
    nb = sum(len(per[i]["busy"]) for i in per)
    L += [f"바쁜 전문가는 모두 {nb}개다(층별 목록은 아래 표). "
          + ("4b 원래 통계로 판정해도 같은 집합이다." if all(per[i]["busy"] == per[i]["busy_plain"] for i in per)
             else "4b 원래 통계로 판정하면 일부 층에서 집합이 다르다: "
             + "; ".join(f"{i}층 새 {per[i]['busy']} / 원래 {per[i]['busy_plain']}" for i in per if per[i]["busy"] != per[i]["busy_plain"])), ""]

    for k in KEEPS:
        L += [f"## {k}개 남김: 층별 변화", "",
              "| 층 | 바쁜 전문가 | 새로 살린 전문가 | 새로 자른 전문가 | 살린 '고정만' | 잃는 REAP 4b기준 (순수 → A+B) | 잃는 기여 4b기준 (순수 → A+B) | 재배정 4b기준 (순수 → A+B) |",
              "|---:|---|---|---|---:|---|---|---|"]
        for i in sorted(per):
            p = per[i]["plans"][k]
            o, n = p["plain_on_plain_stats"], p["new_on_plain_stats"]
            L.append(f"| {i} | {compact(per[i]['busy'])} | {compact(p['saved'])} | {compact(p['newly_removed'])} | {len(p['saved_fixed_only'])} | "
                     f"{pct(o['lost_reap_share'])} → {pct(n['lost_reap_share'])} | {pct(o['lost_contrib_share'])} → {pct(n['lost_contrib_share'])} | "
                     f"{pct(o['rerouted_share'])} → {pct(n['rerouted_share'])} |")
        L.append("")
    (out / "prune_plans.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for k in KEEPS:
        print(k, "plain", [pct(x) for x in overall(k, "plain_on_plain_stats")], "new", [pct(x) for x in overall(k, "new_on_plain_stats")],
              "| new stats: plain", [pct(x) for x in overall(k, "plain_on_new_stats")], "new", [pct(x) for x in overall(k, "new_on_new_stats")])


if __name__ == "__main__":
    main()
