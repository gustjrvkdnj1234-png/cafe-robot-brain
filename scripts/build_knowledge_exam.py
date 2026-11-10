"""Build data/exam_knowledge.jsonl: 200 KMMLU (Korean) + 100 MMLU (English)
multiple-choice questions, sampled with a fixed seed over all subjects.

Rows are fetched one by one from the Hugging Face datasets-server API, so the
full datasets are never downloaded. Neither dataset requires a login.

  KMMLU: HAERAE-HUB/KMMLU, test split of all 45 subject configs (CC BY-ND 4.0)
  MMLU : cais/mmlu, config "all", test split (MIT)

    python scripts/build_knowledge_exam.py
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "exam_knowledge.jsonl"
API = "https://datasets-server.huggingface.co"
SEED = 42
N_KMMLU, N_MMLU = 200, 100
LETTERS = "ABCD"


PAGE = 100          # rows per request; sampled rows on the same page share one request
THROTTLE_S = 1.0    # pause between requests (the API rate-limits bursts)
_page_cache: dict = {}


def get(path: str, **params) -> dict:
    for attempt in range(8):
        time.sleep(THROTTLE_S)
        r = requests.get(f"{API}/{path}", params=params, timeout=60)
        if r.status_code in (401, 403):
            raise SystemExit(f"{params.get('dataset')}: access denied ({r.status_code}). Login or license acceptance needed.")
        if r.status_code == 200:
            return r.json()
        wait = int(r.headers.get("Retry-After", 0) or 0) or min(120, 10 * 2 ** attempt)
        print(f"\n  HTTP {r.status_code}, retrying in {wait}s", flush=True)
        time.sleep(wait)
    r.raise_for_status()


def test_splits(dataset: str) -> list[tuple[str, int]]:
    size = get("size", dataset=dataset)
    return [(s["config"], s["num_rows"]) for s in size["size"]["splits"] if s["split"] == "test"]


def fetch_row(dataset: str, config: str, idx: int) -> dict:
    page = idx // PAGE
    key = (dataset, config, page)
    if key not in _page_cache:
        data = get("rows", dataset=dataset, config=config, split="test", offset=page * PAGE, length=PAGE)
        _page_cache[key] = {r["row_idx"]: r["row"] for r in data["rows"]}
    return _page_cache[key][idx]


def main():
    rng = random.Random(SEED)
    cases = []

    # ---- KMMLU: sample over the union of all subject test splits
    splits = sorted(test_splits("HAERAE-HUB/KMMLU"))
    population = [(cfg, i) for cfg, n in splits for i in range(n)]
    picks = sorted(rng.sample(population, N_KMMLU))
    for k, (cfg, i) in enumerate(picks, 1):
        row = fetch_row("HAERAE-HUB/KMMLU", cfg, i)
        ans = int(row["answer"])
        if ans not in (1, 2, 3, 4):
            raise ValueError(f"unexpected KMMLU answer {ans} at {cfg}/{i}")
        cases.append({
            "id": f"kmmlu-{k:03d}", "source": "kmmlu", "lang": "ko", "subject": cfg,
            "question": row["question"].strip(),
            "choices": [str(row[L]).strip() for L in LETTERS],
            "answer": LETTERS[ans - 1],
            "origin": {"dataset": "HAERAE-HUB/KMMLU", "config": cfg, "split": "test", "row_idx": i},
        })
        print(f"\rKMMLU {k}/{N_KMMLU}", end="", flush=True)
    print()

    # ---- MMLU: config "all" already mixes the 57 subjects
    n_all = dict(test_splits("cais/mmlu"))["all"]
    for k, i in enumerate(sorted(rng.sample(range(n_all), N_MMLU)), 1):
        row = fetch_row("cais/mmlu", "all", i)
        ans = int(row["answer"])
        if ans not in (0, 1, 2, 3) or len(row["choices"]) != 4:
            raise ValueError(f"unexpected MMLU row at {i}")
        cases.append({
            "id": f"mmlu-{k:03d}", "source": "mmlu", "lang": "en", "subject": row["subject"],
            "question": row["question"].strip(),
            "choices": [str(c).strip() for c in row["choices"]],
            "answer": LETTERS[ans],
            "origin": {"dataset": "cais/mmlu", "config": "all", "split": "test", "row_idx": i},
        })
        print(f"\rMMLU {k}/{N_MMLU}", end="", flush=True)
    print()

    OUT.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    subj = lambda src: len({c["subject"] for c in cases if c["source"] == src})
    print(f"{len(cases)} questions -> data/exam_knowledge.jsonl "
          f"(KMMLU subjects: {subj('kmmlu')}, MMLU subjects: {subj('mmlu')})")
    for src in ("kmmlu", "mmlu"):
        dist = {L: sum(1 for c in cases if c["source"] == src and c["answer"] == L) for L in LETTERS}
        print(f"  {src} answer distribution: {dist}")


if __name__ == "__main__":
    main()
