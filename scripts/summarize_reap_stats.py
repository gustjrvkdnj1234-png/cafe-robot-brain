"""Print a readable summary of reap_stats.py output.

    python scripts/summarize_reap_stats.py results/step4/pilot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = Path(sys.argv[1])
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    print(f"cases {meta['n_cases']}, tokens computed {meta['tokens_computed']:,} (alone {meta['tokens_if_alone']:,}), "
          f"fixed tokens/case {meta['fixed_tokens_per_case']}, case tokens/case {meta['case_tokens_mean']}")
    for f in sorted(d.glob("stats_layer*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        E = len(s["combined"]["count"])
        print(f"\n== {f.stem}: {E} experts")
        for g in ("fixed", "case", "combined"):
            x = s[g]
            used = sum(1 for c in x["count"] if c > 0)
            routed = sum(x["raw_routed"])
            topk = routed / max(1, sum(1 for _ in x["count"]))
            order = sorted(range(E), key=lambda e: -x["reap"][e])
            print(f"  {g:8} weighted tokens {x['weighted_tokens']:9.1f} | experts used {used:3d}/{E} | "
                  f"routed (raw) {int(routed):6d} | top REAP {[(e, round(x['reap'][e], 3)) for e in order[:5]]}")
            print(f"  {'':8} lowest REAP among used {[(e, round(x['reap'][e], 4)) for e in order if x['count'][e] > 0][-5:]}")
        f_used = {e for e in range(E) if s["fixed"]["count"][e] > 0}
        c_used = {e for e in range(E) if s["case"]["count"][e] > 0}
        print(f"  experts used only by fixed prompt {len(f_used - c_used)}, only by case part {len(c_used - f_used)}, "
              f"by both {len(f_used & c_used)}, by neither {E - len(f_used | c_used)}")
        sc = s["per_expert_scale"]
        print(f"  per_expert_scale min {min(sc):.3f} max {max(sc):.3f}")


if __name__ == "__main__":
    main()
