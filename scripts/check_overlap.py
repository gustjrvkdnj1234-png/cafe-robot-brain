"""Verify that the calibration set shares nothing with the exam.

Checks (exit code 1 if any check finds an overlap):
  1. exact sentence overlap        - any utterance (user or assistant turn),
                                     compared after removing spaces/punctuation
  2. near-duplicate sentences      - char-trigram Jaccard >= 0.6, or
                                     difflib ratio >= 0.85 (strings >= 4 chars)
  3. paraphrase_group id overlap
  4. intent overlap                - same order state + same expected calls,
                                     or the same clarification intent label

    python scripts/check_overlap.py
    python scripts/check_overlap.py --exam data/exam_cafe.jsonl --calib data/calib_cafe.jsonl
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

from cafe_common import normalize_call, normalize_add

JACCARD_T = 0.6
RATIO_T = 0.85


def norm_text(s: str) -> str:
    return re.sub(r"[\s\.,!?~·…\-+/]+", "", s).lower()


def trigrams(s: str) -> set:
    return {s[i: i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def utterances(case: dict) -> list[str]:
    return [m["content"] for m in case["conversation"]]


def intent_key(case: dict) -> str:
    """Explicit `intent` label if present; otherwise order state + normalized
    expected calls, prefixed by an optional `intent_tag` (e.g. takeout, conditional)."""
    if case.get("intent"):
        return case["intent"]
    state = [normalize_add(s) for s in case.get("order_state") or []]
    calls = sorted(
        json.dumps(normalize_call(c), ensure_ascii=False, sort_keys=True) for c in case["expected"]
    )
    sig = json.dumps({"state": state, "calls": calls}, ensure_ascii=False, sort_keys=True)
    return f"{case['intent_tag']}|{sig}" if case.get("intent_tag") else sig


class ExamIndex:
    """Pre-computed exam sentences for fast near-duplicate lookups."""

    def __init__(self, exam: list[dict]):
        self.sentences = {}
        for c in exam:
            for u in utterances(c):
                n = norm_text(u)
                self.sentences.setdefault(n, (c["id"], u))
        self.tri = {n: trigrams(n) for n in self.sentences}
        self.groups = {c["paraphrase_group"] for c in exam}
        self.intents = {intent_key(c): c["id"] for c in exam}

    def near_duplicate(self, text: str):
        n = norm_text(text)
        if n in self.sentences:
            return ("exact", self.sentences[n], 1.0)
        t = trigrams(n)
        for en, src in self.sentences.items():
            if len(n) >= 4 and len(en) >= 4:
                j = len(t & self.tri[en]) / len(t | self.tri[en])
                if j >= JACCARD_T:
                    return ("near", src, round(j, 3))
                sm = difflib.SequenceMatcher(None, n, en)
                if sm.real_quick_ratio() >= RATIO_T and sm.quick_ratio() >= RATIO_T and sm.ratio() >= RATIO_T:
                    return ("near", src, round(sm.ratio(), 3))
        return None


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default=str(root / "data" / "exam_cafe.jsonl"))
    ap.add_argument("--calib", default=str(root / "data" / "calib_cafe.jsonl"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    exam, calib = load_jsonl(a.exam), load_jsonl(a.calib)
    idx = ExamIndex(exam)
    exact, near, groups, intents = [], [], [], []
    for c in calib:
        for u in utterances(c):
            hit = idx.near_duplicate(u)
            if hit:
                (exact if hit[0] == "exact" else near).append((c["id"], u, hit[1][0], hit[1][1], hit[2]))
        if c["paraphrase_group"] in idx.groups:
            groups.append(c["id"])
        k = intent_key(c)
        if k in idx.intents:
            intents.append((c["id"], idx.intents[k]))

    print(f"exam cases: {len(exam)}  calibration samples: {len(calib)}")
    print(f"1. exact sentence overlap   : {len(exact)}")
    print(f"2. near-duplicate sentences : {len(near)}  (trigram Jaccard >= {JACCARD_T} or ratio >= {RATIO_T})")
    print(f"3. paraphrase group overlap : {len(groups)}")
    print(f"4. intent overlap           : {len(intents)}")
    for row in (exact + near)[:20]:
        print("   ", row)
    for row in intents[:20]:
        print("   intent", row)
    bad = len(exact) + len(near) + len(groups) + len(intents)
    print("RESULT:", "PASS (zero overlap)" if bad == 0 else f"FAIL ({bad} overlaps)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
