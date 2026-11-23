# 음성 데모 1단계: 오프라인 한국어 STT·TTS 고르기

이 노트북(Core Ultra 7 255H, RAM 32 GB, Arc iGPU, CUDA 없음)에서 **설치 후 인터넷 없이 CPU로** 돌아가는 무료 한국어 STT와 TTS를 비교했다.

- 모델은 만들지 않았고, Ollama는 계속 꺼 두었다.
- 클라우드 음성 API는 쓰지 않았다.
- 시험하는 동안 각 후보 프로세스의 바깥 네트워크 연결을 막았다. 로컬(127.0.0.1)만 허용했다.

## 추천

| 역할 | 추천 | 이유 |
|---|---|---|
| STT | **faster-whisper medium (int8)** | CER 7.1%로 가장 정확하다. 한 문장 3.1 s, RAM 1.8 GB, MIT. |
| TTS | **Supertonic 3 int8, 여성 sid 0, 4단계** | 한 문장 1.0 s, RAM 0.32 GB. 되알아듣기에서 표기 차이를 빼면 오류 0이다. 44 kHz 신경망 음성이다. 모델은 OpenRAIL-M이라 상업 이용은 되지만 용도 제한 조항이 있다. |

- **STT:**
  - **medium을 고른 이유:** large-v3-turbo보다 오류가 절반이고(17자 대 33자) 더 빠르다.
  - **SenseVoice는 빠르지만 오류가 위험하다.** 0.07 s로 45배 빠르지만 "반샷 → 반찬", "녹차라떼 → 녹찰 할 때"처럼 메뉴 이름을 틀린다. 이런 오류는 주문 두뇌(64e)가 바로 잘못 알아듣는다.
  - **whisper.cpp는 이 기계에서 느렸다.** CPU 빌드가 9.7 s였다.
- **TTS:**
  - **4단계와 8단계는 발음 차이가 없었다.** 8단계 1.39 s 대비 25% 빠르다.
  - **Windows Heami는 예비로 둔다.** 설치가 필요 없고 0.01 s로 즉시 나오지만, 예전 방식 음성이다.
  - **음질은 직접 들어 봐야 한다.** 숫자로 판단할 수 없으므로 `results/tts/`의 WAV를 들어 보고 정한다.

### 예상 한 턴 시간 (STT + 64e 2.55 s + TTS)

| 조합 | STT | 64e (v1, iGPU) | TTS | 합계 |
|---|---:|---:|---:|---:|
| **추천: medium + Supertonic 3 (4단계)** | 3.14 s | 2.55 s | 1.04 s | **6.7 s** |
| medium + Windows Heami | 3.14 s | 2.55 s | 0.01 s | 5.7 s |
| SenseVoice + Supertonic 3 (4단계) | 0.07 s | 2.55 s | 1.04 s | 3.7 s |
| large-v3-turbo + Supertonic 3 (4단계) | 4.97 s | 2.55 s | 1.04 s | 8.6 s |

- **이 합계는 각각 따로 잰 값을 더한 추정이다.** 녹음이 끝난 뒤부터 소리가 나기 시작할 때까지를 뜻한다.
  - TTS는 문장 전체를 다 만든 뒤 재생하는 기준이다. 응답이 두 문장이면 첫 문장부터 재생해 줄일 수 있다.
  - Whisper는 입력을 늘 30초로 채워 처리하므로, 말이 짧아도 STT 시간은 거의 같다.
- **메모리:** 64e 러너는 v1 카페 시험에서 14.6 GB였다. 여기에 STT 1.8 GB와 TTS 0.3 GB를 더하면 약 17 GB로, 32 GB 안에 들어간다.
- **시험 규모:** 화자 한 명이 말한 20문장(240자)뿐이다. CER 차이가 몇 %p 정도면 문장이 바뀔 때 뒤집힐 수 있다. medium과 나머지의 차이(17자 대 33자 이상)만 비교적 뚜렷하다.

## STT 결과 (`results/stt.md`)

녹음 20개(내장 마이크 10, Buds4 Pro 10)로 쟀다. 모든 후보에 같은 설정을 썼다: 한국어 고정, greedy, temperature 0, 스레드 8. 예열 1회 뒤에 쟀다.

| 후보 | CER 내장 | CER 블루투스 | CER 전체 | 라테/라떼 같게 | 평균 지연 | RTF | 최대 RAM | 다운로드 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| faster-whisper small int8 | 17.5% | 20.8% | 19.2% | 19.2% | 1.14 s | 0.22 | 0.73 GB | 0.49 GB |
| **faster-whisper medium int8** | 6.7% | 7.5% | **7.1%** | 7.1% | 3.14 s | 0.61 | 1.83 GB | 1.53 GB |
| faster-whisper large-v3-turbo int8 | 14.2% | 13.3% | 13.8% | 11.3% | 4.97 s | 0.97 | 1.97 GB | 1.62 GB |
| whisper.cpp large-v3-turbo q5_0 | 16.7% | 13.3% | 15.0% | 12.1% | 9.70 s | 1.89 | 0.83 GB | 0.58 GB |
| SenseVoice-Small int8 (sherpa-onnx) | 15.0% | 12.5% | 13.8% | 13.8% | **0.07 s** | 0.01 | 0.37 GB | 0.24 GB |

- **CER 계산:** 띄어쓰기와 문장부호를 빼고 글자 단위로 쟀다. '라테/라떼 같게' 열은 표준 표기 '라테'를 오류로 세지 않은 값이다.
- **장치 차이는 작다.** 내장 마이크와 블루투스(32 kHz LE Audio)의 차이는 모델마다 방향이 달랐다.
- **모든 모델이 틀린 단어:** "아바라", "캐리어"(→ 헤리어·테리어), "아아"는 거의 모든 모델이 틀렸다. 메뉴 단어 목록을 힌트로 주면 나아질 수 있지만, 이번 문장으로 조정하면 과적합이라 하지 않았다.
- **원본 인식 결과:** 문장별 결과는 `results/stt/<후보>.json`과 `stt.md`에 있다.

## TTS 결과 (`results/tts.md`)

응답 5문장을 예열 1회 뒤에 쟀다(스레드 8). 지연은 문장 전체 파형이 나올 때까지의 시간이다.

**되알아듣기**는 합성음을 faster-whisper medium이 받아 적어 정답과 비교한 CER이다. 발음이 또렷한지만 보며, 자연스러움 점수는 아니다.

| 후보 | 출력 | 평균 지연 | RTF | 최대 RAM | 되알아듣기 CER | 표기 차이 제외 | 다운로드 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows SAPI Heami | 22 kHz | 0.01 s | 0.003 | 0.12 GB (PowerShell 0.08 포함) | 2.3% | 0.0% | 0 |
| Windows OneCore Heami | 16 kHz | 0.01 s | 0.002 | 0.12 GB (PowerShell 0.08 포함) | 3.4% | 0.0% | 0 |
| Supertonic 3 int8 여성 sid 0, 8단계 | 44 kHz | 1.39 s | 0.36 | 0.32 GB | 2.3% | 0.0% | 0.13 GB |
| **Supertonic 3 int8 여성 sid 0, 4단계** | 44 kHz | **1.04 s** | 0.27 | 0.32 GB | 2.3% | **0.0%** | 0.13 GB |
| Supertonic 3 int8 남성 sid 5, 8단계 | 44 kHz | 1.36 s | 0.35 | 0.32 GB | 3.4% | 2.3% (시럽 → 실업) | 0.13 GB |
| Supertonic 2 int8 여성 sid 0, 8단계 | 44 kHz | 0.68 s | 0.18 | 0.25 GB | 4.5% | 2.3% (시럽 → 실업) | 0.08 GB |

- **표기 차이 제외:** STT가 "두 잔"을 "2잔"으로, "게요"를 "께요"로 적은 것은 발음 오류가 아니므로 이 열에서는 오류로 세지 않았다.
- **Supertonic 3 화자:** 10명이며, 1번 문장 샘플이 `results/tts/supertonic3_voices/sidN.wav`에 있다. 음높이로 보면 sid 0–4가 여성, 5–9가 남성이다.
- **Mimic3 KSS는 받았지만 제외했다.** sherpa-onnx 2.0.2에 발음 변환기(espeak-ng)가 없어 합성할 수 없다. 학습 데이터도 비상업용이다.
- **MeloTTS Korean은 시도하지 않았다.** ONNX 한국어판이 없고, PyTorch로 돌리려면 예전 transformers와 Python이 필요하다.

## 라이선스

| 구성 요소 | 라이선스 | 상업 이용 |
|---|---|---|
| faster-whisper, CTranslate2 | MIT | 가능 |
| Whisper 가중치 (small / medium / large-v3-turbo, CT2·ggml 변환본 포함) | MIT (OpenAI) | 가능 |
| whisper.cpp | MIT | 가능 |
| sherpa-onnx | Apache-2.0 | 가능 |
| SenseVoice-Small 가중치 | FunASR Model Open Source License v1.1 | **불명확.** 사용·수정·공유를 허용하고 출처·모델명 표기를 요구하지만, 상업 이용을 따로 언급하지 않고 "참고·학습용" 문구가 있다. |
| Supertonic 2 / 3 가중치 | BigScience OpenRAIL-M (코드는 MIT) | **가능, 용도 제한 있음.** 부속서 A 제한(동의 없는 사람 흉내, 기계 생성물임을 밝히지 않는 사용 등 금지)을 지키고, 재배포할 때 같은 제한을 넘겨야 한다. |
| Windows Heami (SAPI / OneCore) | Windows 라이선스에 포함 | 추가 비용은 없다. 합성 음성을 상업적으로 배포하는 조건은 확인하지 않았다. |
| Mimic3 KSS 음성 (제외) | 모델은 CC0, 학습 데이터 KSS는 CC BY-NC-SA 4.0 | **위험(비상업 데이터)** |
| sounddevice, soundfile, numpy, psutil | MIT / BSD | 가능 |

- **Supertone 회사 상황(확인 안 됨):** 한 HF 미러에 Supertone이 2026-07 해산을 결의했고 저장소를 보관 처리한다는 메모가 있다. 공식 출처로 확인하지는 않았다. 가중치는 이미 로컬에 있고 라이선스는 그대로 유지된다. 다만 앞으로 업데이트나 지원은 기대하기 어렵다.

## 녹음 장치 (`results/devices.json`, `results/levels.json`)

| 장치 | WASAPI 기본 샘플레이트 | 채널 | 녹음 최대 레벨 |
|---|---:|---:|---:|
| 내장 마이크 배열 (인텔 SST) | 48 kHz | 2 | -22 ~ -11 dBFS |
| Buds4 Pro 마이크 (BT LE Audio) | 32 kHz | 1 | -9 ~ -3.5 dBFS |

- **Buds4 Pro:** 클래식 블루투스의 통화 모드(HFP, 8–16 kHz)가 아니라 LE Audio 32 kHz다. 마이크를 연 동안에도 Windows가 알려 주는 출력 형식은 48 kHz 그대로였다. 실제 무선 구간 음질은 이 방법으로는 확인할 수 없다.
- **내장 마이크:** 소음 억제 때문에 쉬는 구간이 완전한 0으로 기록된다.
- **녹음 상태:** 20개 모두 무음이나 클리핑이 없다.

## 실행 방법

전용 환경은 `.venv-voice`(Python 3.13)다. 설치한 패키지는 sounddevice, soundfile, numpy, faster-whisper, sherpa-onnx, psutil이다.

```
.venv-voice\Scripts\python.exe v2\voice\devices.py              # 장치 목록
.venv-voice\Scripts\python.exe v2\voice\record.py builtin       # 녹음 (bluetooth도 같음)
.venv-voice\Scripts\python.exe v2\voice\check_levels.py         # 녹음 레벨 검사
.venv-voice\Scripts\python.exe v2\voice\download_models.py      # 모델 받기 (v2/voice/models/)
.venv-voice\Scripts\python.exe v2\voice\bench_stt.py            # STT 비교
.venv-voice\Scripts\python.exe v2\voice\bench_tts.py voices     # Supertonic 화자 훑기
.venv-voice\Scripts\python.exe v2\voice\bench_tts.py            # TTS 비교 (tts_windows.ps1 포함)
```

녹음(`recordings/`), 합성 음성(`*.wav`), 모델(`models/`, 총 4.98 GB 받음)은 커밋하지 않는다(`.gitignore`).

### 진행 중 고친 것

- **SenseVoice:** 처음 받은 `sense-voice-...-int8-2025-09-09`가 광둥어 미세조정판(WSYue)이라 한국어 음성에 중국어를 출력했다. 원래 다국어판(2024-07-17 int8)으로 바꿨다.
- **whisper.cpp:** 한글이 든 경로의 모델 파일을 열지 못해 서버가 바로 꺼졌다. 모델 폴더에서 실행하고 파일 이름만 넘기도록 바꿨다.
- **`record.py` 들어 보기:** 블루투스 연결 중에 출력 장치를 열지 못하면 멈추던 문제를 고쳐, 이제는 경고만 내고 녹음·저장은 계속한다.
