"""Scoring for the cafe exam (tool calls) and the knowledge exam (letters).

Cafe scoring, per case:
  * calls are normalized (defaults filled for add_item, menu aliases mapped to
    canonical names, ask_clarification compared on `reason` only; the question
    wording is ignored)
  * exact   = 1 if the multiset of (tool name, key arguments) equals the
              expected one (order-insensitive, no missing or extra calls)
  * partial = sum over matched pairs of [0.5 for the tool name +
              0.5 * fraction of key arguments that match], divided by
              max(len(expected), len(predicted)) -> extra/missing calls cost
  * if a case lists `expected_alternatives`, the best-scoring one is used

Usage:
    python scripts/score.py --exam data/exam_cafe.jsonl --pred results/runs/<run>/raw.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from cafe_common import normalize_call

# ---------------------------------------------------------------- cafe


def _arg_fraction(exp: dict, pred: dict) -> float:
    if exp["name"] != pred["name"]:
        return 0.0
    ea, pa = exp["args"], pred["args"]
    keys = sorted(set(ea) | set(pa))
    if not keys:
        return 1.0
    if exp["name"] == "modify_item":
        idx_ok = ea.get("item_index") == pa.get("item_index")
        ec, pc = ea.get("changes") or {}, pa.get("changes") or {}
        ck = set(ec) | set(pc)
        ch = sum(ec.get(k) == pc.get(k) for k in ck) / len(ck) if ck else 1.0
        return 0.5 * idx_ok + 0.5 * ch
    return sum(ea.get(k) == pa.get(k) for k in keys) / len(keys)


def _pair_score(exp: dict, pred: dict) -> float:
    if exp["name"] != pred["name"]:
        return 0.0
    return 0.5 + 0.5 * _arg_fraction(exp, pred)


def _merge_adds(calls: list[dict]) -> list[dict]:
    """Two add_item calls that differ only in quantity count as one line."""
    out, seen = [], {}
    for c in calls:
        if c["name"] == "add_item" and isinstance(c["args"].get("quantity"), int):
            key = json.dumps({k: v for k, v in c["args"].items() if k != "quantity"}, ensure_ascii=False, sort_keys=True)
            if key in seen:
                seen[key]["args"]["quantity"] += c["args"]["quantity"]
                continue
            c = {"name": c["name"], "args": dict(c["args"])}
            seen[key] = c
        out.append(c)
    return out


def _score_against(expected: list, predicted: list) -> dict:
    E = _merge_adds([normalize_call(c) for c in expected])
    P = _merge_adds([normalize_call(c) for c in predicted])
    n = max(len(E), len(P))
    if n == 0:
        return {"exact": 1, "partial": 1.0}
    # best one-to-one assignment (lists are tiny, brute force is fine)
    best = 0.0
    if E and P:
        small, large, flip = (E, P, False) if len(E) <= len(P) else (P, E, True)
        for perm in itertools.permutations(range(len(large)), len(small)):
            s = sum(
                _pair_score(small[i], large[j]) if not flip else _pair_score(large[j], small[i])
                for i, j in enumerate(perm)
            )
            best = max(best, s)
    exact = int(
        len(E) == len(P)
        and sorted(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in E)
        == sorted(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in P)
    )
    return {"exact": exact, "partial": round(best / n, 4)}


def score_cafe_case(case: dict, predicted: list) -> dict:
    options = [case["expected"]] + list(case.get("expected_alternatives") or [])
    results = [_score_against(opt, predicted) for opt in options]
    best = max(results, key=lambda r: (r["exact"], r["partial"]))
    pred_names = [c.get("name") for c in predicted]
    exp_names = [c["name"] for c in case["expected"]]
    return {
        "id": case["id"],
        "category": case["category"],
        "paraphrase_group": case["paraphrase_group"],
        "exact": best["exact"],
        "partial": best["partial"],
        "tool_names_match": int(sorted(pred_names) == sorted(exp_names)),
    }


def _any_reason(calls: list) -> list:
    """ask_clarification with its reason code replaced by a wildcard (same on both sides -> always equal)."""
    out = []
    for c in calls or []:
        if c.get("name") == "ask_clarification":
            args = c.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            c = {**c, "arguments": {**args, "reason": "*"}}
        out.append(c)
    return out


def score_cafe_case_lenient(case: dict, predicted: list) -> dict:
    """Supplementary score that ignores the ask_clarification reason code (the official score stays strict).
    A case with strict exact 0 and lenient exact 1 is wrong only because of the reason code."""
    case = {**case, "expected": _any_reason(case["expected"]),
            "expected_alternatives": [_any_reason(a) for a in case.get("expected_alternatives") or []]}
    return score_cafe_case(case, _any_reason(predicted))


def summarize_cafe(scores: list[dict]) -> dict:
    by_cat = defaultdict(list)
    by_group = defaultdict(list)
    for s in scores:
        by_cat[s["category"]].append(s)
        by_group[s["paraphrase_group"]].append(s)
    return {
        "n": len(scores),
        "exact": round(statistics.mean(s["exact"] for s in scores), 4) if scores else None,
        "partial": round(statistics.mean(s["partial"] for s in scores), 4) if scores else None,
        "tool_names_match": round(statistics.mean(s["tool_names_match"] for s in scores), 4) if scores else None,
        # all paraphrases of one intent answered exactly right
        "group_consistency": round(
            statistics.mean(int(all(x["exact"] for x in g)) for g in by_group.values()), 4
        ) if by_group else None,
        "by_category": {
            c: {
                "n": len(v),
                "exact": round(statistics.mean(x["exact"] for x in v), 4),
                "partial": round(statistics.mean(x["partial"] for x in v), 4),
            }
            for c, v in sorted(by_cat.items())
        },
    }


# ---------------------------------------------------------------- knowledge

LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def extract_letter(text: str) -> str | None:
    if not text:
        return None
    t = text.strip()
    m = re.search(r"(?:정답|answer)\s*(?:은|는|:|is)?\s*\(?([ABCD])\)?", t, re.I)
    if m:
        return m.group(1).upper()
    m = LETTER_RE.search(t)
    return m.group(1) if m else None


def score_knowledge_case(case: dict, response_text: str) -> dict:
    letter = extract_letter(response_text)
    return {
        "id": case["id"],
        "source": case["source"],
        "subject": case["subject"],
        "pred": letter,
        "answer": case["answer"],
        "exact": int(letter == case["answer"]),
    }


def _always_d(rows: list[dict]) -> float | None:
    """Score of a 'model' that always answers D (KMMLU's answer key is skewed)."""
    return round(statistics.mean(int(s["answer"] == "D") for s in rows), 4) if rows else None


def summarize_knowledge(scores: list[dict]) -> dict:
    by_src = defaultdict(list)
    for s in scores:
        by_src[s["source"]].append(s)
    return {
        "n": len(scores),
        "accuracy": round(statistics.mean(s["exact"] for s in scores), 4) if scores else None,
        "always_D_baseline": _always_d(scores),
        "unparsed": sum(1 for s in scores if s["pred"] is None),
        "by_source": {
            k: {
                "n": len(v),
                "accuracy": round(statistics.mean(x["exact"] for x in v), 4),
                "always_D_baseline": _always_d(v),
            }
            for k, v in sorted(by_src.items())
        },
    }


def knowledge_table_md(summary: dict, label: str = "model") -> str:
    """Markdown table: model accuracy next to the always-D baseline."""
    rows = [f"| source | n | {label} | always D |", "|---|---:|---:|---:|"]
    for k, v in summary["by_source"].items():
        rows.append(f"| {k} | {v['n']} | {v['accuracy']:.1%} | {v['always_D_baseline']:.1%} |")
    rows.append(f"| **all** | {summary['n']} | {summary['accuracy']:.1%} | {summary['always_D_baseline']:.1%} |")
    return "\n".join(rows)


# ---------------------------------------------------------------- io


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def is_knowledge(case: dict) -> bool:
    return "choices" in case and "answer" in case


def score_run(exam: list[dict], raw: list[dict]) -> tuple[list[dict], dict]:
    by_id = {c["id"]: c for c in exam}
    scores = []
    for r in raw:
        case = by_id[r["id"]]
        if is_knowledge(case):
            scores.append(score_knowledge_case(case, r.get("content") or ""))
        else:
            scores.append(score_cafe_case(case, r.get("tool_calls") or []))
    if scores and "source" in scores[0]:
        return scores, summarize_knowledge(scores)
    return scores, summarize_cafe(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", required=True)
    ap.add_argument("--pred", help="raw.jsonl written by run_exam.py")
    ap.add_argument("--self-check", action="store_true", help="score the expected answers against themselves")
    a = ap.parse_args()
    exam = load_jsonl(a.exam)
    sys.stdout.reconfigure(encoding="utf-8")
    if a.self_check:
        raw = [{"id": c["id"], "tool_calls": c.get("expected"), "content": c.get("answer")} for c in exam]
    else:
        raw = load_jsonl(a.pred)
    scores, summary = score_run(exam, raw)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if "always_D_baseline" in summary:
        print()
        print(knowledge_table_md(summary))


if __name__ == "__main__":
    main()
