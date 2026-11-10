# data/exam_live.jsonl / gemma4:12b (nothink, igpu)

- 2026-11-10T14:04:28 ~ 2026-11-10T14:08:14, cases: 20, errors: 0
- mean generation speed: 12.91 tok/s, mean wall time per case: 11.258 s

```json
{
  "n": 20,
  "exact": 0.45,
  "partial": 0.5964,
  "tool_names_match": 0.6,
  "group_consistency": 0.45,
  "by_category": {
    "ambiguous": {
      "n": 1,
      "exact": 0,
      "partial": 0.5
    },
    "call_staff": {
      "n": 3,
      "exact": 0.3333,
      "partial": 0.5
    },
    "conditional": {
      "n": 3,
      "exact": 0.3333,
      "partial": 0.3333
    },
    "impossible": {
      "n": 1,
      "exact": 0,
      "partial": 0.0
    },
    "multi_item": {
      "n": 1,
      "exact": 0,
      "partial": 0.5
    },
    "off_menu": {
      "n": 6,
      "exact": 0.6667,
      "partial": 0.75
    },
    "options": {
      "n": 4,
      "exact": 0.5,
      "partial": 0.7321
    },
    "slang": {
      "n": 1,
      "exact": 1,
      "partial": 1.0
    }
  }
}
```
