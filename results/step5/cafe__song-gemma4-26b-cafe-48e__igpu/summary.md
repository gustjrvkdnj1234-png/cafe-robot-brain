# data/exam_cafe.jsonl / song/gemma4-26b-cafe-48e (nothink, igpu)

- 2026-11-11T20:15:16 ~ 2026-11-11T20:29:14, cases: 120, errors: 0
- mean generation speed: 20.57 tok/s, mean wall time per case: 6.975 s

```json
{
  "n": 120,
  "exact": 0.9333,
  "partial": 0.9577,
  "tool_names_match": 0.9417,
  "group_consistency": 0.9189,
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
      "exact": 1,
      "partial": 1.0
    },
    "conditional": {
      "n": 10,
      "exact": 0.9,
      "partial": 0.9929
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
      "exact": 1,
      "partial": 1.0
    },
    "slang": {
      "n": 10,
      "exact": 1,
      "partial": 1.0
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
