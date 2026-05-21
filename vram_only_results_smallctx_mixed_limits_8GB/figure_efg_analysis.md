# Figure E/F/G Analysis: TBT, TTFT Components, and Batch Context

생성일: 2026-05-21  
대상 결과: `vram_only_results_smallctx_mixed_limits_8GB`  
대상 그래프: Figure E, Figure F, Figure G

## 1. 분석 목적

이 문서는 multi-tenant / multi-turn vLLM 실험에서 TBT가 Group1/Group2의 input/output token 차이에 비해 유사하게 관찰되는 이유를 해석하기 위한 보조 분석이다.

검증하려는 가설은 다음과 같다.

- TBT는 개별 request의 input/output token 수보다 같은 decode iteration 또는 같은 batch wave에서 공유하는 GPU execution cadence의 영향을 크게 받는다.
- tenant 수가 낮은 8/16에서는 batch context 증가와 TBT 증가가 거의 같이 움직인다.
- tenant 수가 32로 증가하면 KV pressure와 scheduler blocking이 지배적이 되어, turn 후반부에서는 input context가 계속 증가해도 TBT는 plateau 또는 감소할 수 있다.
- 따라서 turn 8-10에서 TBT가 낮아지는 것은 성능이 좋아진 것이 아니라, 병목이 decode TBT에서 blocking/prefill/TTFT로 이동한 결과일 수 있다.

## 2. 사용한 그래프와 데이터

### Figure E

파일:

```text
graphs/generate_selected_paper_graphs/paper_figure_e_total_tbt_sum_by_turn.png
graphs/generate_selected_paper_graphs/paper_figure_e_total_tbt_sum_by_turn_summary.csv
```

Figure E는 tenant count별로 각 turn에서 모든 tenant request의 `p95_tbt_ms`를 합산한 값이다.

정의:

```text
sum_p95_TBT(turn) = sum(p95_tbt_ms of all successful tenant requests in that turn)
```

bar는 exp1/exp2/exp3 세 번의 반복 실험에서 계산한 `sum_p95_TBT`의 평균이다. error bar는 request-level min/max가 아니라, exp1/exp2/exp3 세 값의 min-max 범위이다.

### Figure F

파일:

```text
graphs/generate_selected_paper_graphs/paper_figure_f_tbt_vs_ttft_components_32tenants.png
graphs/generate_selected_paper_graphs/paper_figure_f_tbt_vs_ttft_components_32tenants.csv
```

Figure F는 32 tenants만 대상으로, TBT와 TTFT component를 같은 turn 축에서 비교한다.

왼쪽:

```text
sum of tenant p95 TBT
```

오른쪽:

```text
P99 blocking time
P99 prefill time
P99 TTFT
```

이 그래프는 TBT가 turn 후반부에 감소해도 실제 latency 병목이 줄어든 것인지 확인하기 위한 것이다.

### Figure G

파일:

```text
graphs/generate_selected_paper_graphs/paper_figure_g_context_tbt_by_turn.png
graphs/generate_selected_paper_graphs/paper_figure_g_context_tbt_by_turn_summary.csv
```

Figure G는 tenant count별로 turn 단위 batch context와 TBT를 함께 표시한다.

표시값:

```text
total input context = sum(input_tokens)
uncached context proxy = sum(input_tokens - prefix_hit_tokens)
sum p95 TBT = sum(p95_tbt_ms)
```

주의: Figure G의 context는 실제 vLLM scheduler iteration 단위 active context가 아니라, turn 단위 request 결과에서 계산한 proxy이다.

## 3. Figure E 결과: 전체 Tenant TBT 합

Figure E의 핵심은 tenant 수가 증가할수록 전체 TBT 합이 커지지만, 32 tenants에서는 turn 7 이후 증가하지 않고 plateau 또는 감소한다는 점이다.

### Turn 1-10 비교

| tenants | turn | sum p95 TBT sec | total input K tokens | total output tokens |
|---:|---:|---:|---:|---:|
| 8 | 1 | 0.079 | 3.0 | 10,247.7 |
| 8 | 2 | 0.111 | 14.4 | 10,246.0 |
| 8 | 3 | 0.149 | 26.1 | 10,248.0 |
| 8 | 4 | 0.185 | 37.5 | 10,248.0 |
| 8 | 5 | 0.220 | 48.2 | 10,248.0 |
| 8 | 6 | 0.257 | 59.7 | 10,248.0 |
| 8 | 7 | 0.293 | 71.2 | 10,248.0 |
| 8 | 8 | 0.333 | 83.2 | 10,248.0 |
| 8 | 9 | 0.368 | 94.3 | 10,248.0 |
| 8 | 10 | 0.402 | 105.0 | 10,248.0 |
| 16 | 1 | 0.212 | 3.4 | 20,493.7 |
| 16 | 2 | 0.340 | 26.1 | 20,496.0 |
| 16 | 3 | 0.483 | 50.0 | 20,496.0 |
| 16 | 4 | 0.632 | 73.5 | 20,496.0 |
| 16 | 5 | 0.769 | 95.3 | 20,496.0 |
| 16 | 6 | 0.913 | 118.2 | 20,496.0 |
| 16 | 7 | 1.055 | 140.7 | 20,496.0 |
| 16 | 8 | 1.206 | 164.7 | 20,496.0 |
| 16 | 9 | 1.348 | 187.4 | 20,496.0 |
| 16 | 10 | 1.486 | 209.2 | 20,496.0 |
| 32 | 1 | 0.661 | 5.5 | 40,989.0 |
| 32 | 2 | 1.178 | 51.2 | 40,991.7 |
| 32 | 3 | 1.780 | 98.0 | 40,992.0 |
| 32 | 4 | 2.381 | 144.7 | 40,991.7 |
| 32 | 5 | 2.955 | 189.2 | 40,992.0 |
| 32 | 6 | 3.546 | 234.9 | 40,992.0 |
| 32 | 7 | 3.726 | 279.9 | 40,992.0 |
| 32 | 8 | 3.674 | 325.9 | 40,992.0 |
| 32 | 9 | 3.637 | 372.7 | 40,992.0 |
| 32 | 10 | 3.512 | 417.2 | 40,991.3 |

관찰:

- 8 tenants에서는 turn 1부터 turn 10까지 total input context와 TBT가 함께 증가한다.
- 16 tenants에서도 같은 패턴이 나타난다.
- 32 tenants에서는 turn 1부터 turn 7까지 TBT가 증가하지만, turn 8-10에서는 total input context가 계속 증가하는데도 TBT는 감소한다.

특히 32 tenants에서 다음 현상이 중요하다.

```text
turn7  input 279.9K, TBT 3.726s
turn8  input 325.9K, TBT 3.674s
turn9  input 372.7K, TBT 3.637s
turn10 input 417.2K, TBT 3.512s
```

즉, turn 후반부의 TBT 감소는 input context가 줄었기 때문이 아니다.

## 4. Figure F 결과: TBT는 감소하지만 Blocking/TTFT는 증가

Figure F는 32 tenants에서 TBT와 TTFT component를 비교한다.

### 32 tenants turn 1-10

| turn | sum p95 TBT sec | P99 blocking sec | P99 prefill sec | P99 TTFT sec |
|---:|---:|---:|---:|---:|
| 1 | 0.661 | 0.0 | 0.5 | 0.6 |
| 2 | 1.178 | 7.1 | 11.3 | 13.1 |
| 3 | 1.780 | 10.4 | 27.8 | 28.4 |
| 4 | 2.381 | 17.7 | 34.4 | 42.7 |
| 5 | 2.955 | 20.9 | 51.0 | 54.5 |
| 6 | 3.546 | 27.2 | 67.8 | 73.2 |
| 7 | 3.726 | 234.8 | 69.3 | 275.1 |
| 8 | 3.674 | 706.9 | 171.3 | 723.3 |
| 9 | 3.637 | 832.4 | 103.7 | 919.2 |
| 10 | 3.512 | 923.8 | 432.2 | 987.7 |

관찰:

- turn 7에서 turn 10으로 갈수록 `sum p95 TBT`는 3.726초에서 3.512초로 감소한다.
- 반면 P99 blocking time은 234.8초에서 923.8초로 크게 증가한다.
- P99 TTFT도 275.1초에서 987.7초로 증가한다.
- P99 prefill은 turn별 변동이 있지만, turn 10에서 432.2초로 매우 크다.

해석:

- turn 8-10에서 TBT가 감소한다고 해서 end-to-end latency가 좋아진 것이 아니다.
- 실제 latency 병목은 decode token gap이 아니라 scheduler queue blocking과 prefill 쪽으로 이동했다.
- request가 decode에 들어간 이후의 per-token cadence는 plateau 또는 감소하지만, decode에 들어가기 전 대기 시간이 급격히 증가한다.

따라서 Figure F는 다음 결론을 뒷받침한다.

```text
32 tenants 후반부 성능 악화의 핵심은 TBT 증가가 아니라 blocking/prefill/TTFT 증가이다.
```

## 5. Figure G 결과: Batch Context와 TBT 관계

Figure G는 turn 단위 batch context proxy와 TBT를 함께 표시한다.

### Correlation

| tenants | corr(total input context, sum p95 TBT) | corr(uncached context, sum p95 TBT) |
|---:|---:|---:|
| 8 | 0.9999 | 0.4420 |
| 16 | 0.9999 | 0.4705 |
| 32 | 0.9156 | 0.6535 |

관찰:

- 8 tenants와 16 tenants에서는 total input context와 TBT의 상관이 거의 1.0이다.
- 32 tenants에서도 전체 상관은 0.9156으로 여전히 높지만, turn 8-10에서 관계가 깨진다.
- uncached context와 TBT의 상관은 total input context보다 낮다. 이는 prefix cache hit, scheduler behavior, queueing 효과가 TBT에 함께 영향을 준다는 뜻이다.

### 8 tenants

| turn | total input K | uncached K | sum p95 TBT sec | mean p95 TBT ms |
|---:|---:|---:|---:|---:|
| 1 | 3.0 | 1.9 | 0.079 | 9.84 |
| 2 | 14.4 | 11.5 | 0.111 | 13.94 |
| 3 | 26.1 | 11.7 | 0.149 | 18.62 |
| 4 | 37.5 | 11.4 | 0.185 | 23.14 |
| 5 | 48.2 | 10.8 | 0.220 | 27.46 |
| 6 | 59.7 | 11.6 | 0.257 | 32.08 |
| 7 | 71.2 | 11.5 | 0.293 | 36.67 |
| 8 | 83.2 | 12.1 | 0.333 | 41.57 |
| 9 | 94.3 | 11.1 | 0.368 | 45.99 |
| 10 | 105.0 | 10.1 | 0.402 | 50.29 |

8 tenants에서는 turn이 진행될수록 total input context와 TBT가 안정적으로 증가한다. 이 구간에서는 scheduler pressure가 작고, decode cadence가 context 증가를 비교적 직접적으로 반영한다.

### 16 tenants

| turn | total input K | uncached K | sum p95 TBT sec | mean p95 TBT ms |
|---:|---:|---:|---:|---:|
| 1 | 3.4 | 1.9 | 0.212 | 13.26 |
| 2 | 26.1 | 22.8 | 0.340 | 21.25 |
| 3 | 50.0 | 24.0 | 0.483 | 30.22 |
| 4 | 73.5 | 23.6 | 0.632 | 39.53 |
| 5 | 95.3 | 21.9 | 0.769 | 48.07 |
| 6 | 118.2 | 23.1 | 0.913 | 57.08 |
| 7 | 140.7 | 22.5 | 1.055 | 65.95 |
| 8 | 164.7 | 24.2 | 1.206 | 75.38 |
| 9 | 187.4 | 22.5 | 1.348 | 84.22 |
| 10 | 209.2 | 21.6 | 1.486 | 92.90 |

16 tenants에서도 8 tenants와 유사하게 total input context와 TBT가 함께 증가한다. 이 구간까지는 batch context 증가가 TBT 증가를 잘 설명한다.

### 32 tenants

| turn | total input K | uncached K | sum p95 TBT sec | mean p95 TBT ms |
|---:|---:|---:|---:|---:|
| 1 | 5.5 | 3.6 | 0.661 | 20.66 |
| 2 | 51.2 | 45.7 | 1.178 | 36.81 |
| 3 | 98.0 | 47.0 | 1.780 | 55.63 |
| 4 | 144.7 | 46.8 | 2.381 | 74.42 |
| 5 | 189.2 | 44.7 | 2.955 | 92.36 |
| 6 | 234.9 | 51.2 | 3.546 | 110.82 |
| 7 | 279.9 | 122.6 | 3.726 | 116.43 |
| 8 | 325.9 | 202.7 | 3.674 | 114.82 |
| 9 | 372.7 | 274.1 | 3.637 | 113.66 |
| 10 | 417.2 | 335.1 | 3.512 | 109.74 |

32 tenants에서는 turn 7까지 TBT가 빠르게 증가하지만, turn 8 이후에는 total input context와 uncached context가 모두 증가하는데 TBT는 오히려 감소한다.

이 결과는 다음을 의미한다.

- TBT 감소는 batch context가 줄어서 발생한 것이 아니다.
- turn 8-10에서는 decode iteration 자체가 더 느려지는 것이 아니라, scheduler blocking 또는 prefill 단계에서 request가 오래 머무는 구조가 된다.
- 즉, 실제 병목은 decode token gap보다 queueing과 prefill 쪽이다.

## 6. 핵심 해석

### 6.1 낮은 tenant count에서는 context가 TBT를 잘 설명한다

8 tenants와 16 tenants에서는 total input context와 TBT의 correlation이 거의 1.0이다.

```text
8 tenants  corr = 0.9999
16 tenants corr = 0.9999
```

이 구간에서는 request들이 비교적 안정적으로 batch에 올라가고, scheduler queue pressure가 작다. 따라서 turn이 진행되며 history가 길어질수록 decode step의 attention context도 증가하고, 그 결과 TBT가 함께 증가한다.

### 6.2 32 tenants에서는 turn 8 이후 관계가 깨진다

32 tenants에서는 turn 7까지는 TBT가 증가한다.

```text
turn1 -> turn7: TBT 0.661s -> 3.726s
```

하지만 turn 8-10에서는 context가 계속 증가하는데 TBT는 감소한다.

```text
turn7 -> turn10:
input context 279.9K -> 417.2K
uncached context 122.6K -> 335.1K
TBT 3.726s -> 3.512s
```

이 구간은 decode TBT가 더 이상 전체 병목을 설명하지 못하는 saturation 이후 구간으로 볼 수 있다.

### 6.3 TBT 감소는 성능 개선이 아니다

Figure F에서 보듯 turn 7-10 사이에 P99 TTFT와 blocking은 크게 증가한다.

```text
P99 blocking: 234.8s -> 923.8s
P99 TTFT:    275.1s -> 987.7s
```

따라서 TBT가 낮아진 것은 성능 개선이 아니라, 더 많은 시간이 decode 이전의 waiting/blocking/prefill 단계에서 소비되기 때문이다.

### 6.4 Group1/Group2 TBT가 비슷한 이유

Group1과 Group2는 input/output token 수가 다르지만, 같은 vLLM instance에서 같은 turn wave에 들어간다. decode 중에는 GPU kernel completion cadence를 공유하기 때문에, 개별 request의 token 길이 차이가 TBT에 직접적으로 크게 반영되지 않을 수 있다.

즉, request-local 특성보다 다음 요소가 TBT를 더 강하게 지배할 수 있다.

- 같은 iteration에 들어간 active request 수
- decode iteration의 aggregate context
- scheduler가 실제로 token을 배정한 request 수
- KV pressure로 인한 queueing/preemption/eviction/recomputation
- prefix cache hit 상태

## 7. 현재 분석의 한계

Figure E/F/G는 현재 저장된 request-level 결과를 기반으로 한다. 따라서 다음 정보는 직접 포함하지 않는다.

- vLLM scheduler iteration ID
- iteration별 running queue request 수
- iteration별 실제 scheduled request 수
- iteration별 decode request 수
- iteration별 active decode context length
- token이 어느 iteration에서 생성되었는지
- iteration elapsed time

따라서 Figure G의 `total input context`는 실제 iteration context가 아니라 turn-level proxy이다.

정확한 검증을 위해서는 vLLM에 iteration-level metrics writer를 추가해야 한다.

필요한 JSONL 예시는 다음과 같다.

```json
{
  "iteration_id": 12345,
  "timestamp_start": 0.0,
  "timestamp_end": 0.0,
  "iteration_elapsed_ms": 0.0,
  "num_running_reqs": 32,
  "num_waiting_reqs": 0,
  "num_scheduled_reqs": 32,
  "num_context_requests": 0,
  "num_context_tokens": 0,
  "num_generation_requests": 32,
  "num_generation_tokens": 32,
  "sum_decode_context_tokens": 0,
  "mean_decode_context_tokens": 0,
  "max_decode_context_tokens": 0
}
```

이 metric이 있으면 다음 그래프를 직접 그릴 수 있다.

- `sum_decode_context_tokens` vs `iteration_elapsed_ms`
- turn별 `p50/p95 iteration context` vs `p95 TBT`
- per-token TBT vs previous iteration context
- turn 7/8/9/10의 iteration context distribution 비교

## 8. 결론

Figure E/F/G를 종합하면 다음과 같다.

1. 8 tenants와 16 tenants에서는 total input context 증가가 TBT 증가를 잘 설명한다.
2. 32 tenants에서는 turn 7까지 TBT가 증가하지만, turn 8-10에서는 input context와 uncached context가 계속 증가해도 TBT는 감소한다.
3. 이 감소는 성능 개선이 아니라, 병목이 decode TBT에서 scheduler blocking/prefill/TTFT로 이동한 결과이다.
4. 32 tenants 후반부에서는 TBT만 보면 실제 system pressure를 과소평가할 수 있다.
5. Group1/Group2의 TBT가 유사한 것은 request-local token 길이보다 shared decode iteration cadence가 더 강하게 작용했기 때문일 가능성이 있다.
6. 다만 이를 확정하려면 vLLM iteration-level metrics가 필요하다.
