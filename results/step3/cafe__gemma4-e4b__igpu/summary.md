# data/exam_cafe.jsonl / gemma4:e4b (nothink, igpu)

- 2026-11-10T13:36:39 ~ 2026-11-10T13:39:45, cases: 120, errors: 0
- mean generation speed: 41.38 tok/s, mean wall time per case: 1.548 s

```json
{
  "n": 120,
  "exact": 0.7417,
  "partial": 0.8408,
  "tool_names_match": 0.825,
  "group_consistency": 0.5946,
  "by_category": {
    "ambiguous": {
      "n": 10,
      "exact": 0.3,
      "partial": 0.3
    },
    "basic": {
      "n": 10,
      "exact": 1,
      "partial": 1.0
    },
    "call_staff": {
      "n": 10,
      "exact": 0.9,
      "partial": 0.9
    },
    "conditional": {
      "n": 10,
      "exact": 0.2,
      "partial": 0.25
    },
    "impossible": {
      "n": 10,
      "exact": 1,
      "partial": 1.0
    },
    "modify_cancel": {
      "n": 9,
      "exact": 0.8889,
      "partial": 0.8889
    },
    "multi_item": {
      "n": 9,
      "exact": 0.6667,
      "partial": 0.9881
    },
    "off_menu": {
      "n": 10,
      "exact": 1,
      "partial": 1.0
    },
    "options": {
      "n": 10,
      "exact": 0.7,
      "partial": 0.9786
    },
    "slang": {
      "n": 10,
      "exact": 0.6,
      "partial": 0.9714
    },
    "takeout": {
      "n": 10,
      "exact": 0.6,
      "partial": 0.8
    },
    "unavailable": {
      "n": 12,
      "exact": 1,
      "partial": 1.0
    }
  }
}
```
