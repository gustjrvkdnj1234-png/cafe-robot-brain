# Ollama 모델 등록 점검 (2026-11-23)

## 결론

**매니페스트는 사라지지 않았다.** 이 Ollama(0.40.15)는 직접 만든 모델의 매니페스트를 `~/.ollama/models/manifests-v2/ollama.com/` 아래에 쓴다. 처음 점검할 때 예전 폴더인 `manifests/`만 봐서, 모델이 없어졌다고 잘못 판단했다. `song/` 모델 9개는 모두 등록되어 있었다. 따라서 다시 등록하지 않았다(`ollama create`를 실행하지 않음).

## 근거

| 확인 | 결과 |
|---|---|
| Ollama 버전과 설치 시각 | 0.40.15. `unins000.exe`가 2026-11-06 16:54, `ollama.exe`가 11-06 01:20이고, 그 뒤 갱신 없음. |
| `manifests/` 폴더 | `library` 3개(11-06 pull)와 `hf.co` 64e-text 1개(11-13 pull)만 있다. `registry.ollama.ai` 폴더의 수정 시각은 11-06 17:07:40이다. 같은 디스크에서 하위 폴더를 지우면 상위 폴더 수정 시각이 바뀌는 것을 실험으로 확인했으므로, 이곳에 `song` 폴더가 생겼다 지워진 적은 없다. |
| `manifests-v2/ollama.com/song/` | 9개 모델이 각각 등록 시각에 만들어져 있다. 예: 98e 11-11 17:32:35, 64e 11-11 19:26:39, 48e-reap 11-20 15:26:33. 서버 로그의 `POST /api/create` 시각과 맞는다(예: `ollama_serve_step5.log` 17:32:35). |
| 서버 로그 | 모든 `ollama serve` 로그가 같은 `OLLAMA_MODELS`(`%USERPROFILE%\.ollama\models`)를 쓴다. `/api/delete` 호출은 없다. 11-23 시작 때 `total unused blobs removed: 0`. |
| 우리 스크립트와 명령 | 스크립트에 `ollama rm`, `/api/delete`, 매니페스트 삭제 코드가 없다. PowerShell 기록의 `ollama rm`은 예전 `haccp-*` 모델뿐이다. 휴지통에 Ollama 관련 항목은 `OllamaSetup.exe`뿐이다. |
| 블롭 | 52개, 113,372,362,330 B로, 점검 전후가 같다. 9개 모델의 주 GGUF 블롭은 모두 있고, 다시 해시를 계산해 파일 이름과 내용이 같은 것을 확인했다. |

## 모델 확인 (`verify.md`, `verify.json`)

| 모델 | 결과 | 주 GGUF 블롭 | 기대값 출처 |
|---|---|---|---|
| song/gemma4-26b-cafe-98e | 등록 정상 | `061cc70f4814` | `results/step5/overnight_notes.md` |
| song/gemma4-26b-cafe-64e | 등록 정상 | `1fa88d73cd3f` | `scripts/report_step6.py` |
| song/gemma4-26b-cafe-48e | 등록 정상 | `9a01e444a066` | `scripts/report_step6.py` |
| song/gemma4-26b-cafe-64e-text | 등록 정상 | `1fa88d73cd3f` | `models/gguf/lean/Modelfile.64e-text` |
| song/gemma4-26b-cafe-48e-text | 등록 정상 | `9a01e444a066` | `models/gguf/lean/Modelfile.48e-text` |
| song/gemma4-26b-128e-noimat | 등록 정상 (CPU 전용) | `caf0cbfc0553` | 해시 기록 없음. 등록 시각과 크기(17.07 GB)가 v1 로그와 맞는다. |
| song/gemma4-26b-cafe-98e-reap | 등록 정상 | `0da089756251` | `v2/results/reap_curve/reap98/pipeline_state.json` |
| song/gemma4-26b-cafe-64e-reap | 등록 정상 | `b92b9962bec4` | `v2/results/reap64/pipeline_state.json` |
| song/gemma4-26b-cafe-48e-reap | 등록 정상 | `a8832fd38f80` | `v2/results/reap_curve/reap48/pipeline_state.json` |

- 확인 방법: 각 모델을 서버에 띄워 짧은 응답을 받은 뒤 바로 내렸다. 서버는 `OLLAMA_NOPRUNE=1`로 띄웠다.
- 프로젝터와 DRAFT: 전체 빌드 7개는 v1처럼 둘 다 들어 있다. text 2개는 둘 다 없다.
- 128e-noimat: iGPU로는 Vulkan 버퍼를 할당하지 못해 뜨지 않는다. v1과 같이 CPU 전용(`num_gpu 0`)으로 확인했다.

## 고아 블롭

**없다.** 처음에 "참조 없음"으로 보였던 1–2 KB 파일 9개는 각 `song/` 모델의 매니페스트 자체다. `ollama list`의 ID와 같다(예: 64e `69587d1545c6`). 지운 것은 없다.

## 자동 시작 경로

- **시작프로그램:** 시작프로그램 폴더의 `Ollama.lnk`(트레이 앱)는 2026-06-05부터 비활성 상태였다(StartupApproved 03). 이번에 `%USERPROFILE%\ollama_autostart_backup\`으로 옮겼으며, 되돌리려면 다시 옮기면 된다.
- **나머지 경로:** 작업 스케줄러, 서비스, Run 키에는 Ollama 항목이 없다.
- **CLI가 앱을 띄움:** 서버가 꺼진 상태에서 `ollama` 명령(예: `ollama list`)을 실행하면 데스크톱 앱이 뜬다. 그 앱의 서버는 **NOPRUNE 없이** 시작된다. 11-23 10:19에 실제로 한 번 이렇게 떴다가 바로 종료했다. 지워진 블롭은 0개였다. 이때 업데이트 확인이 0.40.17을 찾았고, `updates_v2`에 빈 폴더만 남았다(아래에서 삭제).
- **앱 설정:** `auto_update_enabled`가 1이었고, 아래에서 0으로 바꿨다.

## 사용자 결정에 따른 설정 변경 (2026-11-23)

| 변경 | 방법 | 되돌리기 |
|---|---|---|
| 사용자 환경 변수 `OLLAMA_NOPRUNE=1` (영구) | `[Environment]::SetEnvironmentVariable('OLLAMA_NOPRUNE','1','User')`. 데스크톱 앱이나 CLI가 띄운 서버도 이 값을 물려받는다(새로 시작한 프로세스부터). | 같은 명령에 값 `$null` |
| Ollama 앱 자동 업데이트 끔, 0.40.15 유지 | 앱이 꺼진 상태에서 `%LOCALAPPDATA%\Ollama\db.sqlite`의 `settings.auto_update_enabled`를 1에서 0으로 바꿨다. 바꾸기 전 `db.sqlite*`를 `%USERPROFILE%\ollama_autostart_backup\db_backup_20261123\`에 복사해 두었다. | 백업 복사본을 되돌리거나 값을 1로 |
| 남은 업데이트 흔적 삭제 | 앞에서 "1 바이트 파일"이라고 적은 것은 실제로는 **빈 폴더**였다(폴더 크기를 잘못 읽음). `updates_v2\b38d147d…`와 그 부모 `updates_v2`는 둘 다 11-23 10:19:11.444에 만들어졌다. 이는 `app.log`의 "New update available … v0.40.17/OllamaSetup.exe"(10:19:11.342) 직후이고, 안은 비어 있어 중단된 0.40.17 다운로드 준비 폴더로 판단했다. 이 두 빈 폴더만 지웠다. | 필요 없음 |

## NOPRUNE 적용

- `v2/scripts/start_ollama.ps1`: 새로 만들었다.
  - 환경 변수는 서버 프로세스에만 적용되고, `OLLAMA_NOPRUNE=1`은 항상 들어간다.
  - 이미 서버가 떠 있으면 시작하지 않는다.
  - 로그에서 `OLLAMA_NOPRUNE:true`를 확인하지 못하면 서버를 끈다.
- 서버를 직접 띄우는 기존 스크립트 두 개에 `OLLAMA_NOPRUNE="1"`을 추가했다:
  - `scripts/overnight_step5.py` `ensure_ollama()`
  - `v2/scripts/reap64_pipeline.py` `ensure_ollama()`
- 음성 데모 시작 스크립트는 `start_ollama.ps1`을 쓴다.
