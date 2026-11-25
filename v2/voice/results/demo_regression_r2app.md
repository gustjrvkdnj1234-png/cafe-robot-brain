# 데모 설정 회귀 확인 2 (2026-11-25, 64e, CPU 전용): 앱 안전장치만 (A3 = app)

- 비교: 11-24 데모 설정(`demo__*__cpu.jsonl`) → 11-25 데모 설정(`demo_r2app__*__cpu.jsonl`). 같은 모델·옵션·배포용 배치.
- 11-25 변경: A3 온도 묻기 규칙(따뜻한/아이스 둘 다 되는 메뉴에 온도를 말하지 않으면 묻기), read_order 도구, 온도 기본값 문구. 시험 문항에는 주문 단계(확인 대기 / 확정) 줄이 붙지 않는다.
- 채점: v1 엄격 채점, inform·read_order 호출은 빼고 센다(순수 메뉴 질문 예외는 그대로).
- **의도된 변화(A3)**: 기대 답에 따뜻한/아이스 둘 다 되는 메뉴의 add_item이 있고, 손님이 어느 턴에서도 온도를 말하지 않았으며(자동 판정), 틀린 이유가 온도 질문인 문항. prompt: 모델이 온도를 되물음(카페는 질문에 온도 말이 있을 때만, 실전은 질문 문장을 저장하지 않아 ask_clarification이면). app: 모델 호출은 채점 그대로이고, 앱 안전장치가 붙잡아 온도를 물었을 add_item이 있는 문항을 따로 센다(이 경우 모델 답은 v1 기본값대로라 대부분 정답).

## 카페 120

| 실행 | 엄격 정답 |
|---|---:|
| 11-24 데모 설정, CPU | 112/120 |
| 11-25 데모 설정, CPU | 110/120 |
| 11-25, 의도된 변화(A3)를 정답으로 셀 때 | 110/120 |

- 의도된 변화(A3): 21개 (그중 11-24에 정답이던 것 20개)
- 그 밖에 결과가 바뀐 문항: 3개
- read_order를 부른 문항: 0개

| 유형 | 문항 | 11-24 엄격 | 11-25 엄격 | 의도된 변화(A3) |
|---|---:|---:|---:|---:|
| ambiguous | 10 | 7 | 7 | 0 |
| basic | 10 | 10 | 9 | 6 |
| call_staff | 10 | 9 | 9 | 0 |
| conditional | 10 | 9 | 8 | 6 |
| impossible | 10 | 10 | 10 | 0 |
| modify_cancel | 9 | 9 | 9 | 0 |
| multi_item | 9 | 9 | 9 | 3 |
| off_menu | 10 | 10 | 10 | 0 |
| options | 10 | 9 | 10 | 3 |
| slang | 10 | 10 | 10 | 3 |
| takeout | 10 | 8 | 7 | 0 |
| unavailable | 12 | 12 | 12 | 0 |

의도된 변화(A3) 목록 (카페 120)

| id | 유형 | 11-24 | 11-25 도구 호출 |
|---|---|---|---|
| B1-1 | basic | O | `add_item({"menu": "아메리카노", "quantity": 1})` |
| B1-2 | basic | O | `add_item({"menu": "아메리카노", "quantity": 1})` |
| B1-4 | basic | O | `add_item({"menu": "아메리카노", "quantity": 1})` |
| B3-1 | basic | O | `add_item({"menu": "유자차", "quantity": 1})` |
| B3-2 | basic | O | `add_item({"menu": "유자차", "quantity": 1})` |
| B3-3 | basic | O | `add_item({"menu": "유자차", "quantity": 1})` |
| O2-1 | options | O | `add_item({"menu": "바닐라라떼", "shots_delta": 1, "sweetness": "less"})` |
| O2-2 | options | X | `add_item({"menu": "바닐라라떼", "shots_delta": 1, "sweetness": "less"})` |
| O2-3 | options | O | `add_item({"menu": "바닐라라떼", "shots_delta": 1, "sweetness": "less"})` |
| S3-1 | slang | O | `add_item({"menu": "카라멜마키아또", "quantity": 1})` |
| S3-2 | slang | O | `add_item({"menu": "카라멜마키아또", "quantity": 1})` |
| S3-3 | slang | O | `add_item({"menu": "카라멜마키아또", "quantity": 1})` |
| M2-1 | multi_item | O | `add_item({"menu": "카페라떼", "quantity": 2}); add_item({"menu": "레몬에이드", "quantity": 1})` |
| M2-2 | multi_item | O | `add_item({"menu": "카페라떼", "quantity": 2}); add_item({"menu": "레몬에이드", "quantity": 1})` |
| M2-3 | multi_item | O | `add_item({"menu": "카페라떼", "quantity": 2}); add_item({"menu": "레몬에이드", "quantity": 1})` |
| D1-1 | conditional | O | `add_item({"menu": "카페라떼"})` |
| D1-3 | conditional | O | `add_item({"menu": "카페라떼"})` |
| D1-4 | conditional | O | `add_item({"menu": "카페라떼"})` |
| D3-1 | conditional | O | `add_item({"menu": "아메리카노"})` |
| D3-2 | conditional | O | `add_item({"menu": "아메리카노"})` |
| D3-3 | conditional | O | `add_item({"menu": "아메리카노"})` |

그 밖에 바뀐 문항 (카페 120)

| id | 유형 | 11-24 → 11-25 | 11-24 도구 호출 | 11-25 도구 호출 |
|---|---|---|---|---|
| B1-3 | basic | O → X | `add_item({"menu": "아메리카노"})` | `add_item({"menu": "아메리카노", "temperature": "iced"})` |
| T1-1 | takeout | O → X | `add_item({"menu": "아메리카노", "temperature": "iced"})` | `add_item({"menu": "아메리카노", "temperature": "iced"}); confirm_order({"dine_option": "takeout"})` |
| D1-2 | conditional | O → X | `add_item({"menu": "카페라떼"})` | `ask_clarification({"question": "죄송합니다. 녹차라떼는 메뉴에 없어서, 카페라떼로 준비해 드릴까요?", "reason": "off_menu"})` |

## 실전 20

| 실행 | 엄격 정답 |
|---|---:|
| 11-24 데모 설정, CPU | 12/20 |
| 11-25 데모 설정, CPU | 10/20 |
| 11-25, 의도된 변화(A3)를 정답으로 셀 때 | 12/20 |

- 의도된 변화(A3): 3개 (그중 11-24에 정답이던 것 1개)
- 그 밖에 결과가 바뀐 문항: 4개
- read_order를 부른 문항: 0개

| 유형 | 문항 | 11-24 엄격 | 11-25 엄격 | 의도된 변화(A3) |
|---|---:|---:|---:|---:|
| ambiguous | 1 | 1 | 1 | 0 |
| call_staff | 3 | 1 | 1 | 0 |
| conditional | 3 | 0 | 1 | 1 |
| impossible | 1 | 0 | 0 | 0 |
| multi_item | 1 | 1 | 0 | 0 |
| off_menu | 6 | 5 | 4 | 0 |
| options | 4 | 3 | 3 | 2 |
| slang | 1 | 1 | 0 | 0 |

의도된 변화(A3) 목록 (실전 20)

| id | 유형 | 11-24 | 11-25 도구 호출 |
|---|---|---|---|
| L01 | conditional | X | `ask_clarification` |
| L09 | options | O | `add_item` |
| L14 | options | X | `ask_clarification` |

그 밖에 바뀐 문항 (실전 20)

| id | 유형 | 11-24 → 11-25 | 11-24 도구 호출 | 11-25 도구 호출 |
|---|---|---|---|---|
| L12 | multi_item | O → X | `add_item; ask_clarification` | `add_item; ask_clarification` |
| L13 | off_menu | O → X | `ask_clarification` | `ask_clarification` |
| L16 | slang | O → X | `add_item` | `add_item; confirm_order` |
| L20 | conditional | X → O | `ask_clarification; call_staff` | `add_item; call_staff` |

