# EASY-EP 통계 (v2)

EASY-EP(Dong et al., arXiv:2504.06792) 식 (8)의 토큰 기여 s_t = 1 − cos(h_t, h̃_t)를 Gemma 4 26B-A4B에서 모으고, 전문가 중요도 I_i = Σ_t g_t,i · ‖e_i(x_t)‖ · s_t로 98·64·48개 남김 목록을 만들어 v1 기준들과 비교했다. 모델은 만들지 않았다.

- 코드: `v2/scripts/easy_ep_stats.py`(수집), `v2/scripts/easy_ep_overlap.py`(남김 목록·비교)
- 결과: `stats/easy_layer{00..29}.json`, `stats/meta.json`, `overlap.json`, `summary.md`
- 수집 방식: v1(`scripts/reap_stats.py`)과 같다. 같은 600문항(seed 42, `case_ids`가 v1 `results/step4/reap_600/meta.json`과 같은지 실행 때 확인), 접두사 트리, 한 번에 한 층만 RAM에, float32, CPU. 계측 외의 코드는 v1 것을 읽기 전용으로 가져다 썼다.

## Gemma 4 층 구조와 세 변형

Gemma 4의 FFN 블록은 밀집 MLP와 라우팅 전문가가 같은 FFN 앞 잔차에서 나란히 돈다(`transformers` `modeling_gemma4.py` `Gemma4TextDecoderLayer.forward`).

```
residual = (어텐션 뒤 은닉 상태)
mlp_out  = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(residual)))
router(residual)                                   # 라우터 입력 = residual
x        = pre_feedforward_layernorm_2(residual)   # 라우팅 전문가 입력
moe_raw  = Σ_k g_k · e_k(x)                        # 라우팅 전문가 출력(게이트 합)
moe_out  = post_feedforward_layernorm_2(moe_raw)
out      = (residual + post_feedforward_layernorm(mlp_out + moe_out)) · layer_scalar
```

| 변형 | h_t | h̃_t | 코드에서 잡는 텐서 |
|---|---|---|---|
| **A (기본)** | 라우팅 전문가 입력 `x = pre_feedforward_layernorm_2(residual)` | `x + moe_raw` (정규화 전 라우팅 전문가 출력 + 그 입력) | `experts.forward`의 `hidden_states`와 반환값 `final` |
| B | FFN 앞 잔차 `residual` | `residual + moe_out` (`post_feedforward_layernorm_2`를 거친 라우팅 전문가 출력) | `router.forward` 입력, `post_feedforward_layernorm_2` 출력 훅 |
| C | FFN 앞 잔차 `residual` | 층 출력 `out` (밀집 MLP 포함) | `router.forward` 입력, 층 `forward` 반환값. `layer_scalar > 0`을 실행 때 확인해 코사인에 영향 없음 |

- 셋 모두 g·‖e‖는 같은 값(정규화 전 전문가 출력 e_i(x)의 노름에 게이트 g를 곱한 것, v1 `sum_gate_norm`과 같음)이고 s_t만 다르다.
- 모델에 층별 입력(`hidden_size_per_layer_input`)이 없음을 실행 때 확인했다.

### 왜 A가 기본인가

논문은 h_t를 **라우팅 전문가 모듈의 입력**, h̃_t를 **그 입력 + 라우팅 전문가 출력(게이트 합)** 으로 두고, 공유 전문가(여기서는 밀집 MLP에 해당)는 뺀다. A는 이 두 텐서를 그대로 쓴다. B는 h를 라우터 입력(정규화 전 잔차)으로 바꾸고 출력에 정규화를 한 번 더 거친 값이라 논문 정의에서 한 걸음 벗어나고, C는 밀집 MLP까지 섞여 논문이 제외한 부분이 들어간다. 그래서 A를 기본, B·C를 참고로 둔다.

## 실행

| 항목 | 값 |
|---|---:|
| 문항 / 계산한 토큰 / 구간 | 600 / 129,734 / 868 (v1과 같음) |
| 실행 시간 | 4,362 s (1.21시간, v1 REAP 수집 6,429 s) |
| 최대 RSS | 7.62 GB (제한 20 GB) |
| 스레드 | 16 |

층별 불러오기·순전파 시간과 RSS는 `stats/meta.json`의 `layers`에 있다. 이어서 돌리기용 `stats/states_after_layer.pt`(1.4 GB)는 재생성할 수 있어 git에 넣지 않았다.

## 확인: v1 통계와 같은가

같은 실행에서 전문가별 배정 수(`count`)와 g·‖f(x)‖ 합(`sum_gate_norm`)을 다시 기록해 v1(`results/step4/reap_600/stats_layer*.json`)과 비교했다. 30개 층 × 128개 전문가 × fixed/case/combined 전부에서 **최대 상대 차이 0**(토큰 가중치도 0)으로 완전히 같다. 따라서 EASY-EP 값은 v1과 같은 토큰·라우팅 위에서 계산된 것이다.

## s_t 크기 (30개 층의 combined 평균)

| 변형 | 평균 | 최소 (층) | 최대 (층) |
|---|---:|---:|---:|
| A | 0.041 | 0.004 (29층) | 0.228 (0층) |
| B | 0.575 | 0.270 (29층) | 0.834 (4층) |
| C | 0.180 | 0.039 (13층) | 0.862 (29층) |

## 남김 목록 만드는 법

층마다 I_i를 v1과 같은 50:50 가중(고정 프롬프트 전체 가중치 = 문항별 토큰 전체 가중치)으로 합산해 큰 순서로 k개를 남긴다. 동점은 v1과 같이 가중 라우터 확률 합, 그다음 전문가 번호. 보호 규칙은 넣지 않았다. 변형별 남김 목록은 `overlap.json`의 `keeps.<k>.criteria.easy_<A|B|C>.kept`에 있다.

비교 결과(Jaccard, 재배정 비율, 잃는 기여)는 `summary.md`를 본다.
