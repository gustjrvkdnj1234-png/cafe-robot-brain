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
| `data/menu.json` | 메뉴 15종, 옵션, 기본값, 제약 규칙, 제공하지 않는 옵션, 직원 호출 대상, 줄임말 |
| `data/tools.json` | Ollama tools 형식의 도구 6개 (`add_item`, `modify_item`, `remove_item`, `ask_clarification`, `call_staff`, `confirm_order`) |
| `data/exam_cafe.jsonl` | 카페 주문 시험 120문항 (12개 유형, 같은 뜻을 3~4가지로 바꿔 쓴 37개 묶음) |
| `data/exam_cafe_review.md` | 사람이 직접 검토할 20문항 샘플 |
| `data/calib_cafe.jsonl` | 나중에 가지치기(pruning)에 쓸 보정용 600문항 (12개 유형 × 50). 시험과 문장·의도가 겹치지 않음 |
| `data/exam_knowledge.jsonl` | 일반 지식 시험: KMMLU 200 + MMLU 100 (seed 42). KMMLU 라이선스(CC BY-ND 4.0) 때문에 저장소에는 없음 → `build_knowledge_exam.py`로 생성 |
| `data/exam_live.jsonl` | 사람이 직접 쓰는 문항 자리 (아래 참고) |

### 과제 규칙 (2b에서 추가)

| | 규칙 |
|---|---|
| A | 로봇이 정하면 안 되는 일(이미 들어간 주문의 준비 상태, 시간 약속, 단체 주문 시간, 캐리어·봉투, 얼음컵, 그 밖의 서비스 요청)은 `call_staff`로 직원에게 넘긴다. |
| B | 샷은 0.5(반 샷) 단위로 더하거나 뺄 수 있다. |
| C | `confirm_order(dine_option)`: 대화 중에 포장이라고 했으면 `takeout`, 아니면 `dine_in`(기본값). 주문 중 포장 언급은 되묻지 않고 받아들인다. |
| D | "X 있으면 X, 없으면 Y": X가 안 되면 되묻지 않고 Y(또는 그 종류의 기본 메뉴)를 담는다. |
| E | 커피(에스프레소·아메리카노·콜드브루)는 당도·시럽 옵션이 없다. 시럽 요청은 `ask_clarification(impossible)`로 셀프바를 안내한다. |
| F | 디카페인, 원두 선택, 휘핑크림, 락토프리 우유, 메뉴에 없는 음료는 제공하지 않는다 → `ask_clarification(off_menu)`로 정중히 거절. |

기본값: 따뜻하게(콜드브루·에이드는 아이스), 레귤러, 샷 그대로, 보통 당도, 일반 우유, 수량 1, 매장. 손님이 말하지 않은 옵션은 묻지 않고 기본값을 쓴다.
모델에 실제로 들어가는 시스템 프롬프트는 `run_exam.py --show-prompt`로 확인할 수 있다.

```powershell
.\.venv\Scripts\python.exe scripts\build_exam_cafe.py        # 카페 시험 + 검토용 샘플
.\.venv\Scripts\python.exe scripts\build_calib_cafe.py       # 보정용 세트
.\.venv\Scripts\python.exe scripts\check_overlap.py          # 시험과 보정 세트가 겹치지 않는지 확인
.\.venv\Scripts\python.exe scripts\build_knowledge_exam.py   # Hugging Face에서 KMMLU/MMLU 샘플링

# 채점 실행 (서버는 1단계와 같이 OLLAMA_IGPU_ENABLE=1 로 실행)
.\.venv\Scripts\python.exe scripts\run_exam.py --model gemma4:e4b --exam data\exam_cafe.jsonl          # thinking 끔(기본)
.\.venv\Scripts\python.exe scripts\run_exam.py --model gemma4:e4b --exam data\exam_cafe.jsonl --think  # thinking 켬
.\.venv\Scripts\python.exe scripts\run_exam.py --model vending --exam data\exam_cafe.jsonl             # 규칙 기반 자판기
.\.venv\Scripts\python.exe scripts\run_exam.py --model gemma4:e4b --exam data\exam_cafe.jsonl --ids T2-1 --show-prompt  # 프롬프트 확인
```

채점 방식(`scripts/score.py`): 도구 이름과 핵심 인자가 모두 맞으면 정답(exact), 일부만 맞으면 부분 점수(partial)를 준다. 말하지 않은 옵션은 기본값과 같다고 본다. 되묻기는 질문 문장과 관계없이 이유(ambiguous / impossible / off_menu)만 보고, 직원 호출은 호출 사유 문장과 관계없이 호출했는지만 본다.
지식 시험은 정답 알파벳이 일치하는지만 보며, 결과 표에는 항상 "무조건 D라고 답하기" 기준 점수를 함께 적는다(KMMLU 정답이 D에 쏠려 있어서 무작위 25%보다 높다).

### 자판기(규칙 기반 비교 기준) 고정

`scripts/vending_machine.py`는 키워드, 동의어 사전, 정규식으로 만든 규칙 기반 파서다. 시험 문제가 아니라 메뉴와 규칙만 보고 작성했다(다만 시험 문제와 같은 사람이 작성했다는 한계가 있다).
자판기는 실행 중에 `data/menu.json`과 `scripts/cafe_common.py`도 읽으므로 이 세 파일을 함께 아래 버전으로 **고정**하며, 이후 절대 수정하지 않는다.
`run_exam.py --model vending`은 실행할 때마다 세 파일의 해시를 확인해 하나라도 다르면 경고하고, 결과(`summary.json`)에 해시와 일치 여부를 기록한다.

| 파일 | 고정 해시 (`git hash-object <파일>`로 확인) |
|---|---|
| `scripts/vending_machine.py` | FROZEN_VENDING_BLOB: `a6d693cf211540a499c2b83c7261106952e018f0` |
| `data/menu.json` | FROZEN_MENU_BLOB: `a898ac58530839da92bab1d27353c03ddce44dc0` |
| `scripts/cafe_common.py` | FROZEN_COMMON_BLOB: `043454654f0323850bb57d1e4df6ce21318de675` |

- 고정 커밋: `9118e0a1182570e4cdc4d8f317c5dd33121e0933` (이 커밋에서 세 파일이 위 해시 그대로 들어갔다)
- 메뉴나 공통 코드를 꼭 바꿔야 한다면, 그 뒤의 자판기 결과는 이 고정 버전과 비교할 수 없다.

### 사람이 쓰는 문항: `data/exam_live.jsonl`

이 파일의 문항은 사람이 직접 쓴다. AI(코딩 에이전트 포함)는 이 파일을 생성하거나, 고치거나, 무언가를 튜닝할 목적으로 읽어서는 안 된다. 모델·자판기·프롬프트를 모두 확정한 뒤 최종 평가에만 쓴다.

## 7단계: 로봇 비교 화면 (블로그용)

같은 손님 말에 다섯 로봇(자판기, E4B, 12B, 26B 원본, 64e)이 어떻게 답했는지 만화처럼 보여 주는 로컬 웹 페이지다.

```powershell
.\.venv\Scripts\python.exe scripts\robot_ui.py      # http://127.0.0.1:8765/ (이 컴퓨터에서만 열림)
```

- **다시 보기(기본):** 3·5단계 시험에서 기록한 답을 그대로 재생한다. 도구 호출은 쉬운 말("아이스 아메리카노 1잔 담았어요", "직원 불러드릴게요!")로 바꿔 보여 주고, 채점 결과(○ 정답 / ✗ 오답), 실제 응답 시간, 그 시험의 최대 메모리를 붙인다. 생각하는 시간은 로그 눈금으로 줄여서 보여 준다. 문항 목록 맨 위에 블로그용 추천 6개(L09 반샷만, L04 따뜻한 아이스, D1-2 녹차라떼 없으면 라떼, L18 캐리어, L13 아샷추, L14 시럽 2펌프)가 있다.
- **직접 주문:** 로봇 하나를 골라 입력한 말을 Ollama에 보낸다. 6c의 배포용 프롬프트 배치(주문 내역을 맨 끝에)를 쓰고, 옵션은 시험과 같다. 정답이 없으므로 채점하지 않는다. "대화 이어가기"를 켜 두면 담은 메뉴가 다음 말의 주문 내역이 된다.
- **내보내기:** 지금 장면을 GIF, PNG 프레임 묶음, 마지막 장면 PNG로 `ui/exports/<문항>/`에 저장한다(git에는 올리지 않음). 주소로도 할 수 있다: `/?case=L09&export=gif`, `/?case=L09&t=end`(마지막 장면 고정).
- 실전 문항(`data/exam_live.jsonl`)은 손님 말과 모범 답안을 **화면에 보여 주기 위해서만** 읽는다. 튜닝에는 쓰지 않는다.
- 글꼴(Jua)은 Google Fonts에서 받는다. 인터넷이 없으면 맑은 고딕으로 그린다.
