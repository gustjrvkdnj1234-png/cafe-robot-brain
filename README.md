# 카페 알바 로봇 언어 두뇌

카페 아르바이트 로봇의 "언어 두뇌"로 쓸 로컬 LLM을 노트북(Intel Core Ultra 7 255H, RAM 32GB, Arc 140T iGPU)에서 고르고 다듬는 실험 프로젝트.

## 이 저장소에 대해

한국어 기술 블로그 글에 쓰인 실험 코드와 측정 결과를 담은 저장소입니다.
모델 가중치는 포함하지 않습니다. 아래 `ollama pull` 명령으로 직접 받아야 합니다.

## 1단계: 기준 모델 속도·메모리 측정

대상: `gemma4:e4b`, `gemma4:12b`, `gemma4:26b`

```powershell
# 1) 환경 (Python 3.10+, Ollama 최신)
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install requests

# 2) 모델 받기
ollama pull gemma4:e4b
ollama pull gemma4:12b
ollama pull gemma4:26b
ollama list > results\models.txt

# 3) Ollama 서버 실행 (Ollama 0.40.15 이상, 트레이 앱은 끈 상태)
#    내장 GPU(Arc 140T)는 기본으로 제외되므로 이 프로세스에만 OLLAMA_IGPU_ENABLE=1 을 준다.
$env:OLLAMA_IGPU_ENABLE = "1"; ollama serve *> logs\ollama_serve.log
#    다른 창에서: ollama ps  /  Select-String -Path logs\ollama_serve.log -Pattern "inference compute"

# 4) 벤치마크 (iGPU 모드 + CPU 전용 모드, psutil 필요)
.\.venv\Scripts\python.exe -m pip install psutil
.\.venv\Scripts\python.exe scripts\bench_baseline.py
#    일부만: --models gemma4:e4b gemma4:12b --modes igpu cpu
```

결과물: `results/baseline.json`(원자료), `results/baseline.md`(CPU vs iGPU 비교표).

측정 조건: `/api/generate`, stream=false, temperature 0, num_predict 256, num_ctx 8192, seed 42.
CPU 모드는 같은 서버에 `num_gpu=0` 옵션을 줘서 강제한다. 한국어/영어 프롬프트를 각각 3번씩 돌리고, 모델·모드별 첫 호출(로딩 포함)은 따로 표시한다. 모델·모드 사이에는 keep_alive 0으로 메모리에서 내린다.

주의: iGPU 모드에서는 `ollama ps` 의 메모리가 실제보다 훨씬 작게 표시된다. 그래서 러너(`llama-server.exe`) 메모리와 GPU 공유 메모리 카운터도 함께 기록한다.
