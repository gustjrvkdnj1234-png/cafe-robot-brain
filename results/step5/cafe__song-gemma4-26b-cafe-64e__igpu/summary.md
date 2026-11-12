# data/exam_cafe.jsonl / song/gemma4-26b-cafe-64e (nothink, igpu)

- 2026-11-11T19:28:27 ~ 2026-11-11T19:42:33, cases: 120, errors: 0
- mean generation speed: 20.83 tok/s, mean wall time per case: 7.045 s

```json
{
  "n": 120,
  "exact": 0.9333,
  "partial": 0.9613,
  "tool_names_match": 0.95,
  "group_consistency": 0.8919,
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
      "exact": 0.7,
      "partial": 0.85
    },
    "unavailable": {
      "n": 12,
      "exact": 1,
      "partial": 1.0
    }
  }
}
```
