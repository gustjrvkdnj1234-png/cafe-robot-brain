# EASY-EP 점검: 지금 통계로 쓸 수 있는가

EASY-EP(Dong et al., arXiv:2504.06792) 식 (8): 토큰 기여 s_t = 1 − cos(h_t, h̃_t). h_t는 라우팅 전문가 모듈의 입력, h̃_t = h_t + Σ g·e(h_t)(라우팅 전문가 출력의 게이트 합 + 잔차). 공유 전문가는 빼고 본다. 전문가 중요도 I(E_i) = Σ_t g·‖e‖·s_t.

## 결론: v1 통계로는 계산할 수 없다

- v1 통계(`results/step4/reap_600/stats_layer*.json`)에는 전문가별 **합계만** 있다(배정 수, g 합, ‖f‖ 합, g·‖f‖ 합, 최댓값, 라우터 확률 합). s_t는 토큰마다 다른 가중치라 g·‖f‖와 토큰 단위로 곱한 뒤 더해야 하는데, 합계에서는 이것을 되돌릴 수 없다.
- 토큰별 은닉 상태도 남아 있지 않다. `states_after_layer.pt`는 이어서 돌리기 위한 **마지막으로 끝낸 층의 출력** 하나뿐이라, 층마다 필요한 ‘전문가 전/후’ 표현이 없다.
- 그러므로 근사하지 않고, 새로 모아야 한다.

## Gemma 4에 옮길 때 정해야 할 것

Gemma 4 층은 밀집 MLP와 라우팅 전문가가 같은 입력(FFN 앞 잔차)에서 **나란히** 돌고, 각각 RMSNorm을 거쳐 더한 뒤 다시 RMSNorm하고 잔차에 더한다(`transformers` `modeling_gemma4.py` `Gemma4TextDecoderLayer.forward`). 논문의 h, h̃에 정확히 대응하는 텐서가 없어서 다음 중 하나를 골라야 한다(한 번의 계산에서 셋 다 모을 수 있다).

| 안 | h_t | h̃_t | 특징 |
|---|---|---|---|
| A | 라우팅 전문가 입력 `pre_feedforward_layernorm_2(residual)` | h_t + Σ g·f(h_t) (정규화 전 전문가 출력) | 논문 식 (1)과 가장 비슷하다 |
| B | FFN 앞 잔차 `residual` | residual + `post_feedforward_layernorm_2(Σ g·f)` | 실제로 잔차 흐름에 더해지는 크기에 가깝다 |
| C | FFN 앞 잔차 `residual` | 층 FFN 블록의 최종 출력(밀집 MLP 포함) | 밀집 MLP까지 섞인다. 논문 정의(공유 전문가 제외)와 다르다 |

## 새로 모으는 비용 (v1과 같은 층별 CPU 방식)

- 계산: v1 계측 코드(`scripts/reap_stats.py`의 `experts_forward`)는 토큰별 g와 ‖f(x)‖, 전문가 출력 합(`final`)을 이미 만든다. 추가되는 것은 토큰당 2,816차원 코사인 1번과, 토큰의 8개 배정에 s_t를 곱해 더하는 일뿐이다. 전문가 행렬곱에 비하면 무시할 만하다.
- 시간: v1 전체 통계 수집과 같다. 600문항, 30개 층, 6,429 s(약 1.8시간, `results/step4/reap_600/meta.json`의 `total_s`).
- 메모리: v1 최대 RSS 7.84 GB. 추가는 구간별 토큰 × 8개의 g·‖f‖ 임시 값 정도(구간당 수 MB)라 8 GB 안쪽, 20 GB 제한에 넉넉하다.
- 필요한 것: 원본 safetensors(`models/gemma-4-26B-A4B-it`, 49 GB, 디스크에 있음), `.venv-prune`(torch, transformers).
- 같은 계산에서 A·B·C 세 가지 s_t와 v1 통계를 함께 기록하면, 한 번 돌려서 모든 기준을 비교할 수 있다.
