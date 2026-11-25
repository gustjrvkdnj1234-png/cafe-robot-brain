# 데모 설정 회귀 확인 2 (2026-11-25, 64e, CPU 전용): 프롬프트 규칙 + 앱 안전장치 (A3 = prompt)

- 비교: 11-24 데모 설정(`demo__*__cpu.jsonl`) → 11-25 데모 설정(`demo_r2__*__cpu.jsonl`). 같은 모델·옵션·배포용 배치.
- 11-25 변경: A3 온도 묻기 규칙(따뜻한/아이스 둘 다 되는 메뉴에 온도를 말하지 않으면 묻기), read_order 도구, 온도 기본값 문구. 시험 문항에는 주문 단계(확인 대기 / 확정) 줄이 붙지 않는다.
- 채점: v1 엄격 채점, inform·read_order 호출은 빼고 센다(순수 메뉴 질문 예외는 그대로).
- **의도된 변화(A3)**: 기대 답에 따뜻한/아이스 둘 다 되는 메뉴의 add_item이 있고, 손님이 어느 턴에서도 온도를 말하지 않았으며(자동 판정), 틀린 이유가 온도 질문인 문항. prompt: 모델이 온도를 되물음(카페는 질문에 온도 말이 있을 때만, 실전은 질문 문장을 저장하지 않아 ask_clarification이면). app: 모델 호출은 채점 그대로이고, 앱 안전장치가 붙잡아 온도를 물었을 add_item이 있는 문항을 따로 센다(이 경우 모델 답은 v1 기본값대로라 대부분 정답).

## 카페 120

| 실행 | 엄격 정답 |
|---|---:|
| 11-24 데모 설정, CPU | 112/120 |
| 11-25 데모 설정, CPU | 88/120 |
| 11-25, 의도된 변화(A3)를 정답으로 셀 때 | 105/120 |

- 의도된 변화(A3): 17개 (그중 11-24에 정답이던 것 16개)
- 그 밖에 결과가 바뀐 문항: 8개
- read_order를 부른 문항: 0개

| 유형 | 문항 | 11-24 엄격 | 11-25 엄격 | 의도된 변화(A3) |
|---|---:|---:|---:|---:|
| ambiguous | 10 | 7 | 7 | 0 |
| basic | 10 | 10 | 3 | 7 |
| call_staff | 10 | 9 | 9 | 0 |
| conditional | 10 | 9 | 7 | 1 |
| impossible | 10 | 10 | 9 | 0 |
| modify_cancel | 9 | 9 | 9 | 0 |
| multi_item | 9 | 9 | 6 | 3 |
| off_menu | 10 | 10 | 10 | 0 |
| options | 10 | 9 | 6 | 3 |
| slang | 10 | 10 | 6 | 3 |
| takeout | 10 | 8 | 6 | 0 |
| unavailable | 12 | 12 | 10 | 0 |

의도된 변화(A3) 목록 (카페 120)

| id | 유형 | 11-24 | 11-25 도구 호출 |
|---|---|---|---|
| B1-1 | basic | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| B1-2 | basic | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| B1-3 | basic | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| B1-4 | basic | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| B3-1 | basic | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| B3-2 | basic | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| B3-3 | basic | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| O2-1 | options | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| O2-2 | options | X | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| O2-3 | options | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| S3-1 | slang | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| S3-2 | slang | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| S3-3 | slang | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| M2-1 | multi_item | O | `add_item({"menu": "카페라떼", "quantity": 2}); add_item({"menu": "레몬에이드"}); ask_clarification({"question": "카페라떼는 따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| M2-2 | multi_item | O | `ask_clarification({"question": "라떼는 따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| M2-3 | multi_item | O | `add_item({"menu": "카페라떼", "quantity": 2}); add_item({"menu": "레몬에이드", "temperature": "iced"}); ask_clarification({"question": "라떼는 따뜻한 거요, 아이스요?", "re` |
| D3-1 | conditional | O | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |

그 밖에 바뀐 문항 (카페 120)

| id | 유형 | 11-24 → 11-25 | 11-24 도구 호출 | 11-25 도구 호출 |
|---|---|---|---|---|
| O3-3 | options | O → X | `add_item({"menu": "카페라떼", "shots_delta": 0.5, "temperature": "iced"})` | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| S2-2 | slang | O → X | `add_item({"menu": "바닐라라떼", "temperature": "iced"})` | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| I3-3 | impossible | O → X | `ask_clarification({"question": "아메리카노는 시럽을 따로 제공해 드리고 있어요. 셀프바에서 직접 넣어주시겠어요?", "reason": "impossible"})` | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| T1-1 | takeout | O → X | `add_item({"menu": "아메리카노", "temperature": "iced"})` | `add_item({"menu": "아메리카노", "temperature": "iced"}); confirm_order({"dine_option": "takeout"})` |
| T1-4 | takeout | O → X | `add_item({"menu": "아메리카노", "temperature": "iced"})` | `add_item({"menu": "아메리카노", "temperature": "iced"}); confirm_order({"dine_option": "takeout"})` |
| D1-2 | conditional | O → X | `add_item({"menu": "카페라떼"})` | `ask_clarification({"question": "녹차라떼는 메뉴에 없어서, 카페라떼로 준비해 드릴까요?", "reason": "off_menu"})` |
| U2-2 | unavailable | O → X | `ask_clarification({"question": "죄송합니다, 휘핑크림은 제공하지 않습니다.", "reason": "off_menu"})` | `ask_clarification({"question": "따뜻한 거요, 아이스요?", "reason": "ambiguous"})` |
| U2-3 | unavailable | O → X | `ask_clarification({"question": "죄송합니다, 휘핑크림은 제공하지 않습니다.", "reason": "off_menu"})` | `ask_clarification({"question": "카페모카는 따뜻한 거요, 아이스요? 그리고 저희 카페는 휘핑크림을 제공하지 않아요.", "reason": "impossible"})` |

## 실전 20

| 실행 | 엄격 정답 |
|---|---:|
| 11-24 데모 설정, CPU | 12/20 |
| 11-25 데모 설정, CPU | 11/20 |
| 11-25, 의도된 변화(A3)를 정답으로 셀 때 | 14/20 |

- 의도된 변화(A3): 3개 (그중 11-24에 정답이던 것 1개)
- 그 밖에 결과가 바뀐 문항: 2개
- read_order를 부른 문항: 0개

| 유형 | 문항 | 11-24 엄격 | 11-25 엄격 | 의도된 변화(A3) |
|---|---:|---:|---:|---:|
| ambiguous | 1 | 1 | 1 | 0 |
| call_staff | 3 | 1 | 1 | 0 |
| conditional | 3 | 0 | 1 | 1 |
| impossible | 1 | 0 | 0 | 0 |
| multi_item | 1 | 1 | 1 | 0 |
| off_menu | 6 | 5 | 4 | 0 |
| options | 4 | 3 | 2 | 2 |
| slang | 1 | 1 | 1 | 0 |

의도된 변화(A3) 목록 (실전 20)

| id | 유형 | 11-24 | 11-25 도구 호출 |
|---|---|---|---|
| L01 | conditional | X | `ask_clarification` |
| L09 | options | O | `ask_clarification` |
| L14 | options | X | `ask_clarification` |

그 밖에 바뀐 문항 (실전 20)

| id | 유형 | 11-24 → 11-25 | 11-24 도구 호출 | 11-25 도구 호출 |
|---|---|---|---|---|
| L13 | off_menu | O → X | `ask_clarification` | `ask_clarification` |
| L20 | conditional | X → O | `ask_clarification; call_staff` | `add_item; call_staff` |

