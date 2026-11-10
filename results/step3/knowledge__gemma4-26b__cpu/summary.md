# data/exam_knowledge.jsonl / gemma4:26b (nothink, cpu)

- 2026-11-10T14:36:10 ~ 2026-11-10T14:51:33, cases: 300, errors: 0
- mean generation speed: 12.19 tok/s, mean wall time per case: 3.074 s

| source | n | gemma4:26b | always D |
|---|---:|---:|---:|
| kmmlu | 200 | 67.5% | 33.5% |
| mmlu | 100 | 83.0% | 23.0% |
| **all** | 300 | 72.7% | 30.0% |

```json
{
  "n": 300,
  "accuracy": 0.7267,
  "always_D_baseline": 0.3,
  "unparsed": 0,
  "by_source": {
    "kmmlu": {
      "n": 200,
      "accuracy": 0.675,
      "always_D_baseline": 0.335
    },
    "mmlu": {
      "n": 100,
      "accuracy": 0.83,
      "always_D_baseline": 0.23
    }
  }
}
```
