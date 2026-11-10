# data/exam_knowledge.jsonl / gemma4:e4b (nothink, igpu)

- 2026-11-10T13:40:24 ~ 2026-11-10T13:42:25, cases: 300, errors: 0
- mean generation speed: 27.4 tok/s, mean wall time per case: 0.409 s

| source | n | gemma4:e4b | always D |
|---|---:|---:|---:|
| kmmlu | 200 | 45.5% | 33.5% |
| mmlu | 100 | 71.0% | 23.0% |
| **all** | 300 | 54.0% | 30.0% |

```json
{
  "n": 300,
  "accuracy": 0.54,
  "always_D_baseline": 0.3,
  "unparsed": 0,
  "by_source": {
    "kmmlu": {
      "n": 200,
      "accuracy": 0.455,
      "always_D_baseline": 0.335
    },
    "mmlu": {
      "n": 100,
      "accuracy": 0.71,
      "always_D_baseline": 0.23
    }
  }
}
```
