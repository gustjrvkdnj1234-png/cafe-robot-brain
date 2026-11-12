# data/exam_knowledge.jsonl / song/gemma4-26b-cafe-64e (nothink, igpu)

- 2026-11-11T19:44:59 ~ 2026-11-11T19:50:28, cases: 300, errors: 0
- mean generation speed: 14.54 tok/s, mean wall time per case: 1.099 s

| source | n | song/gemma4-26b-cafe-64e | always D |
|---|---:|---:|---:|
| kmmlu | 200 | 43.5% | 33.5% |
| mmlu | 100 | 55.0% | 23.0% |
| **all** | 300 | 47.3% | 30.0% |

```json
{
  "n": 300,
  "accuracy": 0.4733,
  "always_D_baseline": 0.3,
  "unparsed": 1,
  "by_source": {
    "kmmlu": {
      "n": 200,
      "accuracy": 0.435,
      "always_D_baseline": 0.335
    },
    "mmlu": {
      "n": 100,
      "accuracy": 0.55,
      "always_D_baseline": 0.23
    }
  }
}
```
