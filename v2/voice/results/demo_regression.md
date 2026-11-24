# 데모 설정 회귀 확인 (64e, iGPU, 카페 120 + 실전 20)

- 같은 모델·옵션(thinking 끔, temperature 0, seed 42, num_ctx 8192, num_predict 512), 앱과 같은 배포용 배치.
- v1 설정 = 실험 프롬프트·도구 그대로, 데모 설정 = 데모 규칙 + inform 도구. 둘 다 이번에 같은 서버에서 실행.
- 채점 = v1 엄격/관대 채점. inform 호출은 빼고 채점하되, 순수 메뉴 질문(기대 답이 되묻기뿐이고, 마지막 말이 질문이며 주문 동사가 없는 문항, 자동 판정)은 inform만 불러도 정답.
- 참고: 5단계 저장 결과(시험용 배치)도 같은 규칙으로 다시 채점.

## 카페 120

| 실행 | 엄격 정답 | 관대 정답 | inform 호출 문항 | 순수 메뉴 질문 (그중 inform) |
|---|---:|---:|---:|---:|
| 5단계 저장 결과 (시험용 배치) | 112/120 (93.3%) | 112/120 | 0 | 2 (0) |
| v1 설정 (이번) | 112/120 (93.3%) | 112/120 | 0 | 2 (0) |
| 데모 설정 (이번) | 112/120 (93.3%) | 112/120 | 0 | 2 (0) |
| 데모 설정, CPU 전용 (이번) | 112/120 (93.3%) | 112/120 | 0 | 2 (0) |

| 유형 | 문항 | v1 설정 엄격 | 데모 설정 엄격 | 데모 CPU 엄격 | 5단계 저장 엄격 |
|---|---:|---:|---:|---:|---:|
| ambiguous | 10 | 7 | 7 | 7 | 7 |
| basic | 10 | 9 | 10 | 10 | 10 |
| call_staff | 10 | 9 | 9 | 9 | 10 |
| conditional | 10 | 9 | 9 | 9 | 9 |
| impossible | 10 | 10 | 10 | 10 | 10 |
| modify_cancel | 9 | 9 | 9 | 9 | 9 |
| multi_item | 9 | 9 | 9 | 9 | 9 |
| off_menu | 10 | 10 | 10 | 10 | 10 |
| options | 10 | 10 | 9 | 9 | 9 |
| slang | 10 | 10 | 10 | 10 | 10 |
| takeout | 10 | 8 | 8 | 8 | 7 |
| unavailable | 12 | 12 | 12 | 12 | 12 |

결과가 바뀐 문항 (v1 설정 → 데모 설정, 엄격): 2개

| id | 유형 | v1 → 데모 | v1 설정 도구 호출 | 데모 설정 도구 호출 |
|---|---|---|---|---|
| B1-3 | basic | X → O | `add_item({"menu": "아메리카노", "temperature": "iced"})` | `add_item({"menu": "아메리카노"})` |
| O2-2 | options | O → X | `add_item({"menu": "바닐라라떼", "shots_delta": 1, "sweetness": "less"})` | `add_item({"menu": "바닐라라떼", "shots_delta": 0.5, "sweetness": "less"})` |

## 실전 20

| 실행 | 엄격 정답 | 관대 정답 | inform 호출 문항 | 순수 메뉴 질문 (그중 inform) |
|---|---:|---:|---:|---:|
| 5단계 저장 결과 (시험용 배치) | 14/20 (70.0%) | 14/20 | 0 | 1 (0) |
| v1 설정 (이번) | 12/20 (60.0%) | 13/20 | 0 | 1 (0) |
| 데모 설정 (이번) | 11/20 (55.0%) | 12/20 | 1 | 1 (0) |
| 데모 설정, CPU 전용 (이번) | 12/20 (60.0%) | 13/20 | 1 | 1 (0) |

| 유형 | 문항 | v1 설정 엄격 | 데모 설정 엄격 | 데모 CPU 엄격 | 5단계 저장 엄격 |
|---|---:|---:|---:|---:|---:|
| ambiguous | 1 | 1 | 1 | 1 | 1 |
| call_staff | 3 | 1 | 1 | 1 | 1 |
| conditional | 3 | 0 | 0 | 0 | 1 |
| impossible | 1 | 0 | 0 | 0 | 0 |
| multi_item | 1 | 1 | 1 | 1 | 1 |
| off_menu | 6 | 5 | 5 | 5 | 6 |
| options | 4 | 3 | 3 | 3 | 3 |
| slang | 1 | 1 | 0 | 1 | 1 |

결과가 바뀐 문항 (v1 설정 → 데모 설정, 엄격): 3개

| id | 유형 | v1 → 데모 | v1 설정 도구 호출 | 데모 설정 도구 호출 |
|---|---|---|---|---|
| L06 | off_menu | X → O | `ask_clarification({"question": "…", "reason": "ambiguous"})` | `ask_clarification({"question": "…", "reason": "off_menu"})` |
| L16 | slang | O → X | `add_item({"menu": "바닐라라떼", "temperature": "iced"})` | `add_item({"menu": "바닐라라떼", "temperature": "iced"}); confirm_order({"dine_option": "takeout"})` |
| L19 | off_menu | O → X | `ask_clarification({"question": "…", "reason": "off_menu"})` | `ask_clarification({"question": "…", "reason": "ambiguous"})` |

