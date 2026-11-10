# data/exam_knowledge.jsonl / gemma4:12b (nothink, igpu)

- 2026-11-10T14:08:15 ~ 2026-11-10T14:18:15, cases: 300, errors: 0
- mean generation speed: 11.39 tok/s, mean wall time per case: 1.998 s

| source | n | gemma4:12b | always D |
|---|---:|---:|---:|
| kmmlu | 200 | 54.5% | 33.5% |
| mmlu | 100 | 84.0% | 23.0% |
| **all** | 300 | 64.3% | 30.0% |

```json
{
  "n": 300,
  "accuracy": 0.6433,
  "always_D_baseline": 0.3,
  "unparsed": 4,
  "by_source": {
    "kmmlu": {
      "n": 200,
      "accuracy": 0.545,
      "always_D_baseline": 0.335
    },
    "mmlu": {
      "n": 100,
      "accuracy": 0.84,
      "always_D_baseline": 0.23
    }
  }
}
```
