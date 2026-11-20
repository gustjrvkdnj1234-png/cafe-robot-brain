# data/exam_cafe.jsonl / song/gemma4-26b-cafe-98e-reap (nothink, igpu)

- 2026-11-20T14:37:53 ~ 2026-11-20T14:56:01, cases: 120, errors: 0
- mean generation speed: 12.77 tok/s, mean wall time per case: 9.061 s

```json
{
  "n": 120,
  "exact": 0.9417,
  "partial": 0.9655,
  "tool_names_match": 0.9583,
  "group_consistency": 0.8919,
  "by_category": {
    "ambiguous": {
      "n": 10,
      "exact": 0.7,
      "partial": 0.7
    },
    "basic": {
      "n": 10,
      "exact": 0.9,
      "partial": 0.9929
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
      "exact": 0.8,
      "partial": 0.9
    },
    "unavailable": {
      "n": 12,
      "exact": 1,
      "partial": 1.0
    }
  }
}
```
