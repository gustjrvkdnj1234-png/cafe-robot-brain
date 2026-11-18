# data/exam_cafe.jsonl / song/gemma4-26b-cafe-64e-reap (nothink, igpu)

- 2026-11-18T14:59:27 ~ 2026-11-18T15:13:27, cases: 120, errors: 0
- mean generation speed: 20.17 tok/s, mean wall time per case: 6.996 s

```json
{
  "n": 120,
  "exact": 0.8917,
  "partial": 0.9238,
  "tool_names_match": 0.9083,
  "group_consistency": 0.7838,
  "by_category": {
    "ambiguous": {
      "n": 10,
      "exact": 0.7,
      "partial": 0.7
    },
    "basic": {
      "n": 10,
      "exact": 1,
      "partial": 1.0
    },
    "call_staff": {
      "n": 10,
      "exact": 0.9,
      "partial": 0.95
    },
    "conditional": {
      "n": 10,
      "exact": 0.5,
      "partial": 0.5929
    },
    "impossible": {
      "n": 10,
      "exact": 1,
      "partial": 1.0
    },
    "modify_cancel": {
      "n": 9,
      "exact": 1,
      "partial": 1.0
    },
    "multi_item": {
      "n": 9,
      "exact": 1,
      "partial": 1.0
    },
    "off_menu": {
      "n": 10,
      "exact": 1,
      "partial": 1.0
    },
    "options": {
      "n": 10,
      "exact": 0.9,
      "partial": 0.9929
    },
    "slang": {
      "n": 10,
      "exact": 1,
      "partial": 1.0
    },
    "takeout": {
      "n": 10,
      "exact": 0.8,
      "partial": 0.9
    },
    "unavailable": {
      "n": 12,
      "exact": 0.9167,
      "partial": 0.9583
    }
  }
}
```
