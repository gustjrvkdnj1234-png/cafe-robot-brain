# REAP-64: 순수 REAP로 고른 64개 전문가 모델

v1 64e(50:50 가중 기여 합 + 바쁜 전문가 보호)와 맞대 보려고, 층마다 **순수 REAP**(배정된 토큰의 g·‖f(x)‖ 평균, v1 통계 그대로) 상위 64개 전문가만 남긴 모델을 만들고 v1과 같은 시험을 쳤다. 결과 표는 `comparison.md`.

- 코드: `v2/scripts/reap64_pipeline.py`(목록·빌드·등록·시험), `v2/scripts/reap64_compare.py`(비교)
- 남김 목록: `prune_plan_reap64.json`. v1 4단계 계획(`results/step4/reap_prune_plan.json`, 64개)과 30개 층 모두 같은지 실행 때 확인했다. v1 64e와 층당 평균 21.5개(전체 644개) 전문가가 다르다.
- Ollama 이름: `song/gemma4-26b-cafe-64e-reap` (blob sha256 `b92b9962…dcd275`, Q4_K_M 9,812,825,280 B). Modelfile은 v1 64e와 같고(`ollama show --modelfile` 비교, FROM 줄만 다름) `Modelfile.gemma4-26b-cafe-64e-reap`에 복사해 두었다.

## 만든 방법 (v1 `scripts/overnight_step5.py`와 같음)

| 단계 | 도구 | 시간 | 최대 메모리 |
|---|---|---:|---:|
| 전문가·라우터 행·`per_expert_scale` 잘라 내기 | `scripts/build_pruned.py` | 150 s | 0.2 GB |
| 은닉 상태 검증 (8문항, 참고용) | `scripts/validate_pruned.py` | 662 s | 7.6 GB |
| BF16 GGUF 변환 | `convert_hf_to_gguf.py` (llama.cpp b12283) | 254 s | 6.0 GB |
| 변환 검증 | `scripts/verify_pruned_files.py` | 126 s | 3.8 GB (RSS 23.5 GB) |
| Q4_K_M 양자화 (v1 텐서 형식 파일, 임베딩 q8_0, imatrix 없음) | `llama-quantize` | 113 s | 3.9 GB (RSS 24.1 GB) |
| Q4 검증 | `scripts/verify_q4.py` | 41 s | 7.4 GB |
| Ollama 등록·blob 해시 확인 | `ollama create` | 약 15 s | – |
| **합계 (빌드)** | | **약 23분** | |

- 메모리는 각 단계 프로세스 트리의 private 메모리(자기 몫으로 잡은 메모리)다. 괄호 안 RSS에는 28 GB GGUF를 메모리 매핑(np.memmap / mmap)한 파일 페이지가 들어가 있어 OS가 언제든 내려놓을 수 있다. 처음에는 RSS로 20 GB 제한을 걸었는데, 변환 검증이 RSS 20.1 GB에서 멈춰서 private 기준으로 바꾸고 그 단계부터 다시 돌렸다.
- GGUF가 검증된 뒤 BF16 GGUF(27.6 GB)와 잘라 낸 safetensors(28.7 GB)를 지웠다. `models/gemma-4-26B-A4B-it-cafe-64e-reap/`에는 설정·토크나이저·`prune_info.json`만 남았다. 로컬 Q4 복사본은 Ollama blob과 해시가 같은 것을 확인한 뒤 지웠다.
- 은닉 상태 검증(마지막 logits, 문항 토큰 482개): REAP-64는 원본과 top-1 일치 73.2%, KL 1.77. v1 64e는 80.3%, KL 0.91 (`validate_reap64.json`, `results/step5/validate_64e.json`).

## 시험

v1 `scripts/run_step5.py`를 그대로 쓰고 출력 폴더만 여기로 바꿨다. thinking 끔, temperature 0, seed 42, num_ctx 8192, 같은 시스템 프롬프트와 도구, Ollama 서버는 `OLLAMA_IGPU_ENABLE=1`만(서버 프로세스에만). iGPU에 다 들어가서(`load_igpu.json`) v1 64e와 같은 iGPU 모드로 쳤다. 시작 전 5문항 스모크 테스트는 5/5 도구 호출.

| 시험 | 시간 | 문항당 평균 | Ollama 러너 최대 메모리 |
|---|---:|---:|---:|
| 카페 120 | 14.0분 | 7.0 s | 14.6 GB |
| 실전 20 | 2.4분 | 7.2 s | 14.4 GB |
| 지식 300 | 6.5분 | 1.3 s | 21.7 GB |

- 지식 시험에서 러너 메모리가 20 GB를 넘었다. v1 64e도 같은 설정에서 21.6 GB였다(`results/step5/knowledge__song-gemma4-26b-cafe-64e__igpu/summary.json`). iGPU 모드에서는 GPU 메모리도 시스템 RAM에서 잡기 때문이고, v1과 같은 조건을 지키려고 설정을 바꾸지 않았다.
