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

## 2단계: 시험 문제 만들기

| 파일 | 내용 |
|---|---|
| `data/menu.json` | 메뉴 15종, 옵션, 제약 규칙, 줄임말 |
| `data/tools.json` | Ollama tools 형식의 도구 5개 (`add_item`, `modify_item`, `remove_item`, `ask_clarification`, `confirm_order`) |
| `data/exam_cafe.jsonl` | 카페 주문 시험 120문항 (8개 유형 × 15, 같은 뜻을 3~4가지로 바꿔 쓴 32개 묶음) |
| `data/exam_cafe_review.md` | 사람이 직접 검토할 20문항 샘플 |
| `data/calib_cafe.jsonl` | 나중에 가지치기(pruning)에 쓸 보정용 600문항. 시험과 문장·의도가 겹치지 않음 |
| `data/exam_knowledge.jsonl` | 일반 지식 시험: KMMLU 200 + MMLU 100 (seed 42) |

```powershell
.\.venv\Scripts\python.exe scripts\build_exam_cafe.py        # 카페 시험 + 검토용 샘플
.\.venv\Scripts\python.exe scripts\build_calib_cafe.py       # 보정용 세트
.\.venv\Scripts\python.exe scripts\check_overlap.py          # 시험과 보정 세트가 겹치지 않는지 확인
.\.venv\Scripts\python.exe scripts\build_knowledge_exam.py   # Hugging Face에서 KMMLU/MMLU 샘플링

# 채점 실행 (서버는 1단계와 같이 OLLAMA_IGPU_ENABLE=1 로 실행)
.\.venv\Scripts\python.exe scripts\run_exam.py --model gemma4:e4b --exam data\exam_cafe.jsonl          # thinking 끔(기본)
.\.venv\Scripts\python.exe scripts\run_exam.py --model gemma4:e4b --exam data\exam_cafe.jsonl --think  # thinking 켬
.\.venv\Scripts\python.exe scripts\run_exam.py --model vending --exam data\exam_cafe.jsonl             # 규칙 기반 자판기
```

채점 방식(`scripts/score.py`): 도구 이름과 핵심 인자가 모두 맞으면 정답(exact), 일부만 맞으면 부분 점수(partial)를 준다. 말하지 않은 옵션은 기본값과 같다고 보고, 되묻기는 질문 문장과 관계없이 이유(ambiguous / impossible / off_menu)만 본다. 지식 시험은 정답 알파벳이 일치하는지만 본다.

비교 기준인 "자판기"(`scripts/vending_machine.py`)는 키워드, 동의어 사전, 정규식으로 만든 규칙 기반 파서다. 시험 문제가 아니라 메뉴와 규칙만 보고 작성했다.
