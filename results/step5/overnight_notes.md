# 밤사이 실행 기록 (5단계: 대조군 · 64개 · 48개)

사용자가 자리를 비운 동안 내린 결정과 이유, 실패를 적는다. (커밋하지 않음)

## 결정

- 1단계(디스크 정리): 98개 모델의 BF16 safetensors와 BF16 GGUF는 5a단계에서 이미 지웠다. 추가로 `models/gguf/gemma4-26b-cafe-98e-Q4_K_M.gguf`(13.7 GB)를 지웠다. Ollama 블롭(sha256-061cc70f…)과 SHA-256이 같은 복사본이라 Ollama 등록에는 영향이 없다.
- 11-11 18:07 스모크 테스트 통과 기준: 5문항 중 3문항 이상에서 도구 호출이 나오면 전체 시험을 돌린다(지시: '도구 호출이 되면'). 모드는 iGPU 로딩 확인에서 `ollama ps` 100% GPU면 iGPU, 아니면 CPU.
- 11-11 18:24 `song/gemma4-26b-128e-noimat`: Ollama 블롭과 SHA-256이 같은 것을 확인한 뒤 로컬 Q4 복사본(gemma4-26b-128e-noimat-Q4_K_M.gguf)을 지웠다(디스크 절약, 등록은 그대로).
- 11-11 19:26 `song/gemma4-26b-cafe-64e`: Ollama 블롭과 SHA-256이 같은 것을 확인한 뒤 로컬 Q4 복사본(gemma4-26b-cafe-64e-Q4_K_M.gguf)을 지웠다(디스크 절약, 등록은 그대로).
- 11-11 20:13 `song/gemma4-26b-cafe-48e`: Ollama 블롭과 SHA-256이 같은 것을 확인한 뒤 로컬 Q4 복사본(gemma4-26b-cafe-48e-Q4_K_M.gguf)을 지웠다(디스크 절약, 등록은 그대로).
