# 1단계 기준 모델 벤치마크 (CPU vs iGPU)

- 측정: 2026-11-06T17:37:00 ~ 2026-11-06T17:57:08
- 환경: Windows-11-10.0.26200-SP0 / ollama 0.40.15 / 서버 OLLAMA_IGPU_ENABLE=1
- 공통 옵션: `{"temperature": 0, "num_predict": 256, "num_ctx": 8192, "seed": 42}`, CPU 모드는 `num_gpu=0` 추가 / 프롬프트(ko, en)당 3회
- 생성 속도 평균은 모델·모드별 첫 호출(로딩 포함, cold)을 제외
- 메모리: `ollama ps` 값은 iGPU 공유 메모리 모드에서 실제보다 작게 나올 수 있어 러너 프로세스(llama-server) private 메모리와 GPU 공유 메모리 카운터를 함께 기록
- gemma4는 기본적으로 thinking을 출력하므로 생성 토큰(eval_count)에 thinking 토큰이 포함됨

## CPU vs iGPU 비교

| 모델 | 생성 tok/s CPU | 생성 tok/s iGPU | 배율 | 프롬프트 tok/s CPU | 프롬프트 tok/s iGPU | 로딩(s) CPU | 로딩(s) iGPU | 러너 메모리 GB CPU | 러너 메모리 GB iGPU | GPU 공유 GB iGPU | ollama ps CPU | ollama ps iGPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| gemma4:e4b | 21.2 | 33.2 | 1.57x | 123.7 | 143.0 | 5.2 | 5.8 | 7.52 | 8.46 | 7.26 | 1.94GB 100% CPU | 0.49GB 100% GPU |
| gemma4:12b | 7.8 | 16.7 | 2.14x | 32.4 | 53.0 | 7.7 | 10.9 | 9.88 | 11.26 | 11.37 | 6.26GB 100% CPU | 1.35GB 100% GPU |
| gemma4:26b | 14.4 | - | - | 44.7 | - | 25.6 | - | 19.28 | - | - | 8.62GB 100% CPU | -GB - |

## 모드별 요약

| 모델 | 모드 | 생성 tok/s | 한국어 | 영어 | 시스템 RAM 증가 GB | 최소 가용 RAM GB | 최대 페이지파일 GB | 빈 응답 | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4:e4b | igpu | 33.2 | 34.0 | 32.7 | 7.72 | 9.5 | 0.1 | 6 | 0 |
| gemma4:e4b | cpu | 21.2 | 23.3 | 19.8 | 6.56 | 11.2 | 0.1 | 6 | 0 |
| gemma4:12b | igpu | 16.7 | 17.0 | 16.5 | 10.76 | 8.0 | 0.1 | 6 | 0 |
| gemma4:12b | cpu | 7.8 | 5.8 | 9.1 | 8.85 | 10.0 | 0.1 | 6 | 0 |
| gemma4:26b | igpu | - | - | - | - | - | - | 0 | 6 |
| gemma4:26b | cpu | 14.4 | 13.8 | 14.8 | 19.42 | 7.1 | 3.6 | 6 | 0 |

## 개별 실행

| 모델 | 모드 | 언어 | 회차 | cold | 로딩(s) | 프롬프트 tok | 프롬프트 tok/s | 생성 tok | 생성 tok/s | ollama ps | 러너 GB | GPU 공유 GB | 가용 RAM GB | 비고 |
|---|---|---|---:|:-:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| gemma4:e4b | igpu | ko | 1 | Y | 5.77 | 40 | 134.9 | 256 | 28.9 | 529 MB 100% GPU | 8.40 | 7.26 | 9.5 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | igpu | ko | 2 |  | 0.01 | 40 | 312.7 | 256 | 34.5 | 529 MB 100% GPU | 8.42 | 7.14 | 9.5 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | igpu | ko | 3 |  | 0.01 | 40 | 300.4 | 256 | 33.6 | 529 MB 100% GPU | 8.42 | 7.19 | 9.6 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | igpu | en | 1 |  | 0.02 | 35 | 151.2 | 256 | 32.8 | 529 MB 100% GPU | 8.43 | 7.17 | 9.7 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | igpu | en | 2 |  | 0.01 | 35 | 256.3 | 256 | 32.8 | 529 MB 100% GPU | 8.44 | 7.17 | 9.8 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | igpu | en | 3 |  | 0.01 | 35 | 253.0 | 256 | 32.5 | 529 MB 100% GPU | 8.46 | 7.16 | 9.8 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | cpu | ko | 1 | Y | 5.21 | 40 | 122.0 | 256 | 22.7 | 2.1 GB 100% CPU | 7.45 | 1.71 | 11.2 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | cpu | ko | 2 |  | 0.01 | 40 | 343.4 | 256 | 23.6 | 2.1 GB 100% CPU | 7.47 | 1.34 | 11.3 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | cpu | ko | 3 |  | 0.01 | 40 | 343.6 | 256 | 22.9 | 2.1 GB 100% CPU | 7.49 | 1.34 | 12.0 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | cpu | en | 1 |  | 0.01 | 35 | 125.4 | 256 | 21.4 | 2.1 GB 100% CPU | 7.49 | 1.31 | 12.2 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | cpu | en | 2 |  | 0.02 | 35 | 304.7 | 256 | 19.1 | 2.1 GB 100% CPU | 7.50 | 1.29 | 12.2 | length, 응답 비어있음(thinking만) |
| gemma4:e4b | cpu | en | 3 |  | 0.01 | 35 | 284.0 | 256 | 19.0 | 2.1 GB 100% CPU | 7.52 | 1.27 | 12.2 | length, 응답 비어있음(thinking만) |
| gemma4:12b | igpu | ko | 1 | Y | 10.91 | 40 | 55.4 | 256 | 16.6 | 1.5 GB 100% GPU | 10.81 | 11.36 | 8.6 | length, 응답 비어있음(thinking만) |
| gemma4:12b | igpu | ko | 2 |  | 0.00 | 40 | 112.1 | 256 | 17.0 | 1.5 GB 100% GPU | 10.92 | 11.20 | 8.5 | length, 응답 비어있음(thinking만) |
| gemma4:12b | igpu | ko | 3 |  | 0.01 | 40 | 114.5 | 256 | 16.9 | 1.5 GB 100% GPU | 11.04 | 11.21 | 8.4 | length, 응답 비어있음(thinking만) |
| gemma4:12b | igpu | en | 1 |  | 0.01 | 35 | 50.6 | 256 | 16.5 | 1.5 GB 100% GPU | 11.03 | 11.20 | 8.4 | length, 응답 비어있음(thinking만) |
| gemma4:12b | igpu | en | 2 |  | 0.02 | 35 | 80.3 | 256 | 16.3 | 1.5 GB 100% GPU | 11.15 | 11.23 | 8.2 | length, 응답 비어있음(thinking만) |
| gemma4:12b | igpu | en | 3 |  | 0.01 | 35 | 61.7 | 256 | 16.6 | 1.5 GB 100% GPU | 11.26 | 11.37 | 8.0 | length, 응답 비어있음(thinking만) |
| gemma4:12b | cpu | ko | 1 | Y | 7.74 | 40 | 30.6 | 256 | 9.2 | 6.7 GB 100% CPU | 9.44 | 1.25 | 10.2 | length, 응답 비어있음(thinking만) |
| gemma4:12b | cpu | ko | 2 |  | 0.01 | 40 | 115.7 | 256 | 2.0 | 6.7 GB 100% CPU | 9.55 | 1.24 | 10.2 | length, 응답 비어있음(thinking만) |
| gemma4:12b | cpu | ko | 3 |  | 0.00 | 40 | 120.8 | 256 | 9.6 | 6.7 GB 100% CPU | 9.67 | 1.20 | 10.1 | length, 응답 비어있음(thinking만) |
| gemma4:12b | cpu | en | 1 |  | 0.00 | 35 | 34.2 | 256 | 9.2 | 6.7 GB 100% CPU | 9.66 | 1.22 | 10.1 | length, 응답 비어있음(thinking만) |
| gemma4:12b | cpu | en | 2 |  | 0.03 | 35 | 107.2 | 256 | 9.2 | 6.7 GB 100% CPU | 9.77 | 1.21 | 10.1 | length, 응답 비어있음(thinking만) |
| gemma4:12b | cpu | en | 3 |  | 0.01 | 35 | 102.1 | 256 | 9.1 | 6.7 GB 100% CPU | 9.88 | 1.22 | 10.0 | length, 응답 비어있음(thinking만) |
| gemma4:26b | igpu | ko | 1 | | | | | | | | | | | 오류: HTTPError('500 Server Error: Internal Server Error for url: http://127.0.0.1:114 |
| gemma4:26b | igpu | ko | 2 | | | | | | | | | | | 오류: HTTPError('500 Server Error: Internal Server Error for url: http://127.0.0.1:114 |
| gemma4:26b | igpu | ko | 3 | | | | | | | | | | | 오류: HTTPError('500 Server Error: Internal Server Error for url: http://127.0.0.1:114 |
| gemma4:26b | igpu | en | 1 | | | | | | | | | | | 오류: HTTPError('500 Server Error: Internal Server Error for url: http://127.0.0.1:114 |
| gemma4:26b | igpu | en | 2 | | | | | | | | | | | 오류: HTTPError('500 Server Error: Internal Server Error for url: http://127.0.0.1:114 |
| gemma4:26b | igpu | en | 3 | | | | | | | | | | | 오류: HTTPError('500 Server Error: Internal Server Error for url: http://127.0.0.1:114 |
| gemma4:26b | cpu | ko | 1 | Y | 25.63 | 40 | 43.3 | 256 | 16.2 | 9.3 GB 100% CPU | 18.99 | 0.73 | 7.6 | length, 응답 비어있음(thinking만) |
| gemma4:26b | cpu | ko | 2 |  | 0.01 | 40 | 206.2 | 256 | 14.0 | 9.3 GB 100% CPU | 19.06 | 0.74 | 7.5 | length, 응답 비어있음(thinking만) |
| gemma4:26b | cpu | ko | 3 |  | 0.01 | 40 | 189.9 | 256 | 13.7 | 9.3 GB 100% CPU | 19.14 | 0.72 | 7.3 | length, 응답 비어있음(thinking만) |
| gemma4:26b | cpu | en | 1 |  | 0.01 | 35 | 46.0 | 256 | 14.8 | 9.3 GB 100% CPU | 19.13 | 0.72 | 7.3 | length, 응답 비어있음(thinking만) |
| gemma4:26b | cpu | en | 2 |  | 0.01 | 35 | 173.4 | 256 | 14.7 | 9.3 GB 100% CPU | 19.20 | 0.73 | 7.3 | length, 응답 비어있음(thinking만) |
| gemma4:26b | cpu | en | 3 |  | 0.01 | 35 | 191.3 | 256 | 14.9 | 9.3 GB 100% CPU | 19.28 | 0.72 | 7.1 | length, 응답 비어있음(thinking만) |
