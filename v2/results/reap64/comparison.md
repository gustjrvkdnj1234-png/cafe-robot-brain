# REAP-64 vs v1 64e vs 26B 원본

- REAP-64: 층마다 순수 REAP(배정된 토큰의 g·‖f(x)‖ 평균, v1 통계, 보호·가중 없음) 상위 64개 전문가. v1 64e: 50:50 가중 기여 합 + 바쁜 전문가 보호. 둘 다 같은 원본·같은 변환·같은 Q4_K_M 텐서 형식, imatrix 없음.
- 시험 설정은 v1과 같다(thinking 끔, temperature 0, seed 42, num_ctx 8192, 같은 시스템 프롬프트·도구, Ollama 서버는 `OLLAMA_IGPU_ENABLE=1`만). 엄격 = 공식 점수, 관대 = 되묻기 이유 코드만 무시.
- 실행 모드: REAP-64 igpu, v1 64e igpu, 26B 원본 cpu(26B 원본은 iGPU에 들어가지 않아 v1에서 CPU로 잼).

## 점수

| 모델 | 카페 엄격 (120) | 카페 관대 | 실전 엄격 (20) | 실전 관대 | 지식 (300) | KMMLU (200) | MMLU (100) |
|---|---:|---:|---:|---:|---:|---:|---:|
| REAP-64 | 89.2% | 89.2% | 55.0% | 70.0% | 46.0% | 38.0% | 62.0% |
| v1 64e (기여 합 + 보호) | 93.3% | 93.3% | 70.0% | 70.0% | 47.3% | 43.5% | 55.0% |
| 26B 원본 | 94.2% | 94.2% | 75.0% | 75.0% | 72.7% | 67.5% | 83.0% |

### 카페 유형별 엄격 정답률

| 유형 | REAP-64 | v1 64e (기여 합 + 보호) | 26B 원본 |
|---|---:|---:|---:|
| ambiguous | 70.0% | 70.0% | 70.0% |
| basic | 100.0% | 100.0% | 100.0% |
| call_staff | 90.0% | 100.0% | 100.0% |
| conditional | 50.0% | 90.0% | 100.0% |
| impossible | 100.0% | 100.0% | 100.0% |
| modify_cancel | 100.0% | 100.0% | 100.0% |
| multi_item | 100.0% | 100.0% | 100.0% |
| off_menu | 100.0% | 100.0% | 100.0% |
| options | 90.0% | 90.0% | 100.0% |
| slang | 100.0% | 100.0% | 100.0% |
| takeout | 80.0% | 70.0% | 60.0% |
| unavailable | 91.7% | 100.0% | 100.0% |

## McNemar 정확 검정 (양쪽, 한쪽만 맞힌 문항 수)

| 비교 (A vs B) | 시험 | 채점 | 둘 다 | A만 | B만 | 둘 다 틀림 | p |
|---|---|---|---:|---:|---:|---:|---:|
| REAP-64 vs v1 64e (기여 합 + 보호) | cafe | 엄격 | 106 | 1 | 6 | 7 | 0.125 |
| REAP-64 vs v1 64e (기여 합 + 보호) | cafe | 관대 | 106 | 1 | 6 | 7 | 0.125 |
| REAP-64 vs v1 64e (기여 합 + 보호) | live | 엄격 | 9 | 2 | 5 | 4 | 0.453 |
| REAP-64 vs v1 64e (기여 합 + 보호) | live | 관대 | 12 | 2 | 2 | 4 | 1.000 |
| REAP-64 vs v1 64e (기여 합 + 보호) | knowledge | 엄격 | 97 | 41 | 45 | 117 | 0.747 |
| REAP-64 vs 26B 원본 | cafe | 엄격 | 105 | 2 | 8 | 5 | 0.109 |
| REAP-64 vs 26B 원본 | cafe | 관대 | 105 | 2 | 8 | 5 | 0.109 |
| REAP-64 vs 26B 원본 | live | 엄격 | 10 | 1 | 5 | 4 | 0.219 |
| REAP-64 vs 26B 원본 | live | 관대 | 13 | 1 | 2 | 4 | 1.000 |
| REAP-64 vs 26B 원본 | knowledge | 엄격 | 115 | 23 | 103 | 59 | 0.000 |
| v1 64e (기여 합 + 보호) vs 26B 원본 | cafe | 엄격 | 111 | 1 | 2 | 6 | 1.000 |
| v1 64e (기여 합 + 보호) vs 26B 원본 | cafe | 관대 | 111 | 1 | 2 | 6 | 1.000 |
| v1 64e (기여 합 + 보호) vs 26B 원본 | live | 엄격 | 14 | 0 | 1 | 5 | 1.000 |
| v1 64e (기여 합 + 보호) vs 26B 원본 | live | 관대 | 14 | 0 | 1 | 5 | 1.000 |
| v1 64e (기여 합 + 보호) vs 26B 원본 | knowledge | 엄격 | 120 | 22 | 98 | 60 | 0.000 |

## 두 64e 모델의 채점 결과가 다른 문항: 카페 (7개)

- ○ = 엄격 정답, △ = 관대로만 정답, ✗ = 오답.

| 문항 | 유형 | REAP-64 | v1 64e | 26B 원본 | REAP-64 호출 | v1 64e 호출 | 정답 |
|---|---|:-:|:-:|:-:|---|---|---|
| C3-2 | call_staff | ✗ | ○ | ○ | `add_item(menu=아메리카노, quantity=30, temperature=iced); call_staff(reason=내일 아침 아메리카노 30잔 10시까지 가능한지 문의)` | `call_staff(reason=내일 아침 대량 주문(아메리카노 30잔) 10시까지 가능 여부 문의)` | `call_staff(reason=단체 주문 시간 문의)` |
| T1-2 | takeout | ○ | ✗ | ✗ | `add_item(menu=아메리카노, size=regular, temperature=iced)` | `add_item(menu=아메리카노, temperature=iced); confirm_order(dine_option=takeout)` | `add_item(menu=아메리카노, temperature=iced)` |
| D1-2 | conditional | ✗ | ○ | ○ | `ask_clarification(question=죄송합니다. 저희 메뉴에는 녹차라떼가 없습니다. 대신 카페라떼로 준비해 드릴까요?, reason=off_menu)` | `add_item(menu=카페라떼)` | `add_item(menu=카페라떼)` |
| D1-3 | conditional | ✗ | ○ | ○ | `ask_clarification(question=죄송합니다. 저희 메뉴에는 녹차라떼가 없습니다. 대신 카페라떼로 준비해 드릴까요?, reason=off_menu)` | `add_item(menu=카페라떼)` | `add_item(menu=카페라떼)` |
| D1-4 | conditional | ✗ | ○ | ○ | `ask_clarification(question=죄송합니다. 저희 메뉴에는 녹차라떼가 준비되어 있지 않습니다. 대신 카페라떼로 준비해 드릴까요?, reason=off_menu)` | `add_item(menu=카페라떼)` | `add_item(menu=카페라떼)` |
| D3-3 | conditional | ✗ | ○ | ○ | `ask_clarification(question=죄송합니다. 저희 카페는 디카페인 옵션을 제공하지 않습니다. 대신 일반 아메리카노로 준비해 드릴까요?, reason=off_menu)` | `add_item(menu=아메리카노)` | `add_item(menu=아메리카노)` |
| U4-2 | unavailable | ✗ | ○ | ○ | `add_item(menu=아메리카노, temperature=iced); ask_clarification(question=죄송합니다. 저희 카페에서는 원두 선택이나 디카페인 옵션은 제공하지 않습니다., reason=off_menu)` | `ask_clarification(question=죄송합니다. 저희 카페에서는 원두를 선택하실 수 없습니다., reason=off_menu)` | `ask_clarification(question=죄송하지만 원두는 고르실 수 없어요. 기본 원두로 드릴까요?, reason=off_menu)` |

- REAP-64만 틀린 문항의 유형: conditional 4, call_staff 1, unavailable 1
- v1 64e만 틀린 문항의 유형: takeout 1
- 채점 결과는 같지만 도구 호출이 다른 문항 63개(문구·추가 인자 차이, 호출 전체는 `comparison.json`): ambiguous A1-1, A1-3, A1-4, A2-2, A2-3, A3-2, A3-3; basic B3-1; call_staff C1-2, C1-3, C1-4, C2-3, C3-1, C3-3; conditional D2-1, D2-2, D2-3; impossible I1-1, I1-2, I1-3, I1-4, I2-1, I2-2, I2-3, I3-1, I3-2, I3-3; modify_cancel X2-1, X2-2; multi_item M1-1, M1-2, M1-3, M2-1, M2-2, M2-3; off_menu F1-1, F1-2, F1-3, F1-4, F2-1, F2-2, F2-3, F3-1, F3-2, F3-3; options O2-2; slang S2-1, S2-2, S2-3; takeout T1-1, T1-3, T1-4; unavailable U1-1, U1-2, U1-3, U2-1, U2-2, U2-3, U3-1, U3-2, U3-3, U4-1, U4-3

## 두 64e 모델의 채점 결과가 다른 문항: 실전 (7개)

- ○ = 엄격 정답, △ = 관대로만 정답, ✗ = 오답. 실전 문항은 문장과 정답을 적지 않는다(사람이 쓴 시험).

| 문항 | 유형 | REAP-64 | v1 64e | 26B 원본 | REAP-64 호출 | v1 64e 호출 |
|---|---|:-:|:-:|:-:|---|---|
| L06 | off_menu | △ | ○ | ○ | `ask_clarification(question=죄송합니다, 초콜릿케이크는 당도 조절이 되지 않습니다. 다른 메뉴를 도와드릴까요?, reason=impossible)` | `ask_clarification(question=죄송합니다, 저희 카페에는 초코라떼 메뉴가 없습니다. 혹시 카페모카를 말씀하시는 걸까요?, reason=off_menu)` |
| L07 | impossible | ○ | ✗ | ✗ | `ask_clarification(question=아이스 아메리카노 한 잔 준비해 드릴게요. 다만, 커피 종류에는 시럽을 넣어드리지 않고 있어요. 시럽은 셀프바에 준비되어 있으니 직접 넣어 드실 수 있습니다., reason=impossible)` | `add_item(menu=아메리카노, temperature=iced); ask_clarification(question=아이스 아메리카노 한 잔 준비해 드릴게요. 커피에 시럽을 넣으시려면 셀프바에서 직접 넣어 드실 수 있습니다!, reason=impossible)` |
| L12 | multi_item | △ | ○ | ○ | `add_item(menu=레몬에이드, size=regular, temperature=iced); ask_clarification(question=죄송합니다, 라떼 종류 중 어떤 메뉴로 드릴까요? 카페라떼, 바닐라라떼, 카라멜마키아또, 카페모카 중에서 선택해 주세요. 그리고 저희 카페는 원두 선택이 불가능합니다., reason=ambiguous)` | `add_item(menu=레몬에이드, quantity=1); ask_clarification(question=죄송합니다, 저희 카페에서는 원두를 선택하실 수 없습니다. 라떼 2잔은 기본 원두로 준비해 드릴까요?, reason=off_menu)` |
| L13 | off_menu | ✗ | ○ | ○ | `add_item(menu=아메리카노, shots_delta=0.5, size=regular, temperature=iced)` | `ask_clarification(question=죄송합니다, 저희 카페에는 아이스티 메뉴가 없습니다. 혹시 다른 음료를 주문하시겠어요?, reason=off_menu)` |
| L14 | options | ○ | ✗ | ○ | `add_item(menu=바닐라라떼, sweetness=less)` | `ask_clarification(question=커피 메뉴에는 시럽을 직접 넣어드릴 수 없어요. 시럽은 셀프바에서 직접 넣어 드실 수 있습니다!, reason=impossible)` |
| L19 | off_menu | △ | ○ | ○ | `ask_clarification(question=죄송합니다. 유자차에 에이드 탄산 추가는 불가능합니다. 유자차는 차 종류라 에이드처럼 탄산이 들어가지 않아요. 대신 시원한 유자차나 상큼한 자몽에이드 같은 에이드 메뉴는 어떠신가요?, reason=impossible)` | `ask_clarification(question=죄송합니다. 저희 카페에는 유자 에이드는 메뉴에 없어서, 유자차로 준비해 드릴까요?, reason=off_menu)` |
| L20 | conditional | ✗ | ○ | ○ | `ask_clarification(question=죄송합니다. 에스프레소는 아이스로 제공되지 않습니다. 얼음 컵을 따로 드릴까요, 아니면 그냥 에스프레소를 주문하시겠어요?, reason=impossible); call_staff(reason=얼음컵 요청)` | `add_item(menu=에스프레소); call_staff(reason=얼음컵 요청)` |

- REAP-64만 틀린 문항의 유형: off_menu 3, multi_item 1, conditional 1
- v1 64e만 틀린 문항의 유형: impossible 1, options 1
- 채점 결과는 같지만 도구 호출이 다른 문항 9개(문구·추가 인자 차이, 호출 전체는 `comparison.json`): ambiguous L04; call_staff L11, L15; conditional L01, L03; off_menu L02, L05, L08; slang L16

## 지식 시험

- 두 64e 모델의 답이 다른 문항 139개, REAP-64만 맞힘 41개, v1 64e만 맞힘 45개 (문항 번호는 `comparison.json`).
