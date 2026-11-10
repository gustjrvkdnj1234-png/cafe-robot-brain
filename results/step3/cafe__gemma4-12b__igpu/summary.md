# data/exam_cafe.jsonl / gemma4:12b (nothink, igpu)

- 2026-11-10T13:42:39 ~ 2026-11-10T14:04:27, cases: 120, errors: 0
- mean generation speed: 14.41 tok/s, mean wall time per case: 10.891 s

```json
{
  "n": 120,
  "exact": 0.9083,
  "partial": 0.9202,
  "tool_names_match": 0.9167,
  "group_consistency": 0.8378,
  "by_category": {
    "ambiguous": {
      "n": 10,
      "exact": 0.5,
      "partial": 0.5
    },
    "basic": {
      "n": 10,
      "exact": 0.9,
      "partial": 0.9929
    },
    "call_staff": {
      "n": 10,
      "exact": 0.9,
      "partial": 0.95
    },
    "conditional": {
      "n": 10,
      "exact": 0.6,
      "partial": 0.6
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
      "exact": 1,
      "partial": 1.0
    },
    "unavailable": {
      "n": 12,
      "exact": 1,
      "partial": 1.0
    }
  }
}
```
