# REAP vs v1 기여 합: 98 · 64 · 48 곡선

REAP-64(`v2/results/reap64/`)에 이어 REAP-98, REAP-48을 같은 방법으로 만들고 시험해, 크기별로 v1 모델(기여 합 + 보호)과 비교했다. 표와 검정은 `curve.md`, 그림은 `curve.png`.

- 코드: `v2/scripts/reap64_pipeline.py --keep 98|48`(REAP-64와 같은 파이프라인, 출력만 `reap<k>/`), `v2/scripts/reap_curve.py`(표·검정·그림)
- 남김 목록: `reap<k>/prune_plan_reap<k>.json`. v1 4단계 순수 REAP 계획(`results/step4/reap_prune_plan.json`)과 30개 층 모두 같음을 확인. v1 모델과 다른 전문가는 층당 평균 98: 13.3개, 64: 21.5개, 48: 20.2개.
- 빌드: 잘라 내기 → 은닉 상태 검증 → BF16 GGUF → 변환 검증 → Q4_K_M(v1 텐서 형식 파일, 임베딩 q8_0, imatrix 없음) → Q4 검증 → 등록. 검증 뒤 BF16 GGUF·잘라 낸 safetensors·로컬 Q4를 지우고 설정·`prune_info.json`만 남겼다. Modelfile은 같은 크기 v1 모델과 같다(FROM 줄만 다름).
- 시험: v1 `scripts/run_step5.py` 그대로(thinking 끔, temperature 0, seed 42, num_ctx 8192, Ollama 서버는 `OLLAMA_IGPU_ENABLE=1`만). 둘 다 iGPU에 다 들어가 iGPU 모드.

| 모델 | Ollama 이름 | Q4 크기 | 빌드 | 시험 | 단계 최대 private 메모리 | Ollama 러너 최대 (카페 / 실전 / 지식) |
|---|---|---:|---:|---:|---:|---:|
| REAP-98 | `song/gemma4-26b-cafe-98e-reap` | 13.67 GB | 38분 | 29분 | 9.5 GB | 17.8 / 17.6 / 24.9 GB |
| REAP-48 | `song/gemma4-26b-cafe-48e-reap` | 8.00 GB | 21분 | 42분 | 8.4 GB | 12.9 / 12.7 / 20.0 GB |

- Ollama 러너 메모리는 v1과 같은 설정이라 지식 시험에서 20 GB를 넘을 수 있다(v1 64e도 21.6 GB). 설정은 바꾸지 않았다.
- 디스크 여유는 계속 100 GB 이상이었다.

## REAP-48 스모크 테스트

v1 파이프라인은 5문항 스모크 테스트에서 도구 호출이 3건 미만이면 전체 시험을 건너뛴다. REAP-48은 2/5건이었다. 빌드 검증(텐서 해시, GGUF 메타데이터, llama.cpp 생성)은 모두 통과했으므로 고장 난 빌드가 아니라 모델 자체의 결과로 보고, 이번 과제가 두 모델의 평가를 요구해 `--continue-after-smoke-fail`로 전체 시험을 쳤다(`reap48/pipeline_state.json`의 `smoke.gate_passed = false`).

REAP-48의 오답 대부분은 **빈 출력**이다: 생성 한도 512 토큰을 다 쓰고도 도구 호출도 글도 나오지 않은 문항이 카페 68 / 실전 11 / 지식 1개. thinking도 비어 있어서, 도구 호출을 열고 닫지 못해 파서가 내용을 내보내지 않은 것으로 추정한다(원문 토큰은 기록되지 않아 확인하지 못함).
