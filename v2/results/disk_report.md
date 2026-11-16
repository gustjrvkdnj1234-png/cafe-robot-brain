# 디스크 점검 (2026-11-16, 아무것도 지우지 않음)

- C: 드라이브 476 GB 중 340 GB 사용, **여유 136 GB**.
- ‘필요’ = v1을 다시 만들거나 다시 가지치기하는 데 필요. ‘재생성 가능’ = 원본과 코드로 다시 만들 수 있음.

## 저장소 안 (`tech_blog/`)

| 경로 | 크기 | 내용 | 판정 |
|---|---:|---|---|
| `models/gemma-4-26B-A4B-it/` | 49 GB | HF 원본 BF16 safetensors(47 GB + 1.6 GB) + 설정·토크나이저 | **필요** (다시 가지치기, 새 통계 수집. 다시 받는 데 v1에서 약 16시간 걸림) |
| `models/gemma-4-26B-A4B-it-cafe-{98,64,48}e/` | 31 MB × 3 | 가지치기 체크포인트의 설정·토크나이저·`prune_info.json`만 남음(가중치는 v1에서 이미 지움) | 작아서 보관 권장. 가중치는 재생성 가능 |
| `models/gguf/` | 113 KB | Modelfile만(로컬 GGUF 복사본은 v1에서 지움, Ollama blob과 같았음) | 보관 |
| `results/step4/reap_600/states_after_layer.pt` | 1.4 GB | REAP 수집을 이어서 돌리기 위한 마지막 층 출력 | 재생성 가능(분석에는 안 씀) |
| `results/step4/timing_600/states_after_layer.pt` | 1.4 GB | 시간 측정용 시범 실행의 같은 파일 | 재생성 가능 |
| `results/step4/pilot/states_after_layer.pt` | 54 MB | 첫 시범 실행 | 재생성 가능 |
| `tools/llama.cpp-b12283/` | 313 MB | GGUF 변환·양자화 도구(시험용 vocab GGUF 포함) | **필요** (다시 가지치기) |
| `.venv-prune/` | 994 MB | torch, transformers | **필요** (다시 가지치기, 통계 수집) |

## Ollama (`~/.ollama/models/blobs`, 실제 디스크 81.9 GB)

`ollama list` 크기는 비전 프로젝터·초안 모델을 포함하며, 같은 blob을 여러 모델이 나눠 쓴다.

| 모델 | `ollama list` | 혼자 쓰는 blob | 판정 |
|---|---:|---:|---|
| `gemma4:e4b` | 6.6 GB | 6.6 GB | v1 비교용. 레지스트리에서 다시 받을 수 있음 |
| `gemma4:12b` | 8.0 GB | 8.0 GB | v1 비교용. 다시 받을 수 있음 |
| `gemma4:26b` | 18 GB | 18.7 GB (언어 17.07 + 프로젝터 1.19 + 초안 0.46) | v1 기준. 다시 받을 수 있음. 프로젝터·초안 blob은 song 모델들도 씀 |
| `song/gemma4-26b-128e-noimat` | 18 GB | 17.07 GB | 대조군 최종 GGUF. 재생성 가능(원본 + llama.cpp) |
| `song/gemma4-26b-cafe-98e` | 15 GB | 13.67 GB | 최종 GGUF. 재생성 가능 |
| `song/gemma4-26b-cafe-64e` | 11 GB | 9.81 GB | **최종 GGUF (HF에도 있음)** |
| `song/gemma4-26b-cafe-48e` | 9.7 GB | 8.00 GB | 최종 GGUF. 재생성 가능 |
| `song/gemma4-26b-cafe-64e-text` | 9.8 GB | 0 (64e와 같은 blob) | v1 6b·6c |
| `song/gemma4-26b-cafe-48e-text` | 8.0 GB | 0 (48e와 같은 blob) | v1 6b·6c |
| `hf.co/SONGSSONG123/gemma4-26b-cafe-64e-text-GGUF:Q4_K_M` | 9.8 GB | 0 (64e와 같은 blob) | HF 검증용 |

- 혼자 쓰는 blob 합계 81.9 GB = blob 폴더 크기와 같다(남는 고아 blob 없음).
- 최종 GGUF 네 개(98·64·48·대조군 48.6 GB)는 원본 + 계획 + llama.cpp로 다시 만들 수 있지만, 만들 때마다 변환·양자화·검증 시간이 든다. v1 숫자를 다시 재려면 이 blob들이 있어야 한다.

## 저장소 밖

| 경로 | 크기 | 내용 | 판정 |
|---|---:|---|---|
| `~/.cache/huggingface/hub/models--Bllossom--llama-3.2-Korean-Bllossom-3B` | 6.1 GB | 이 프로젝트에서 쓰지 않은 모델 | 다른 작업의 것으로 보임 — 확인 필요 |
| `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct` | 2.9 GB | 이 프로젝트에서 쓰지 않은 모델 | 다른 작업의 것으로 보임 — 확인 필요 |
| `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2` | 88 MB | 이 프로젝트에서 쓰지 않은 모델 | 확인 필요 |
| `~/.cache/huggingface/xet/` | 209 MB | HF 업로드 때 생긴 전송 캐시 | 재생성 가능 |
| `~/.cache/huggingface/hub/models--SONGSSONG123--…` | 수십 KB | HF 검증 때 받은 README·Modelfile·LICENSE | 재생성 가능 |
