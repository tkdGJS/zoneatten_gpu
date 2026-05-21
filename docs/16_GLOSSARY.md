# 16. Glossary

## TTFT

Time To First Token. 요청이 들어온 시점부터 첫 output token이 생성/전달될 때까지의 시간.

Approximation:

\[
TTFT \approx Blocking + Prefill
\]

## TBT

Time Between Tokens. 연속 output token 사이의 시간 간격.

\[
TBT_k = t_{k+1} - t_k
\]

## TTLT

Time To Last Token. 요청이 들어온 시점부터 마지막 output token까지의 시간.

## Prefill

Input prompt/context를 transformer에 통과시켜 KV cache를 생성하는 단계.

## Decode

이미 계산된 KV cache를 사용해 output token을 하나씩 생성하는 단계.

## KV Cache

Transformer attention에서 이전 token들의 key/value를 저장한 cache. Multi-turn에서는 prefix/history reuse를 통해 prefill recomputation을 줄일 수 있다.

## KV Retention

Request/turn이 끝난 뒤에도 KV cache를 보존하여 future turn에서 재사용하려는 정책.

## Protected KV

Tenant guarantee 또는 policy에 의해 다른 tenant 때문에 eviction되지 않도록 보호되는 KV.

## Elastic KV

필요 시 reclaim 가능한 KV. Keep score가 낮으면 victim이 된다.

## Blocking Time

요청이 실제 prefill/decode를 시작하기 전 queue/admission에서 기다리는 시간.

## Continuous Batching

여러 request를 동적으로 batch에 넣고 빼면서 GPU를 지속적으로 활용하는 serving 방식.

## Decode Externality

한 tenant/request의 long context 또는 KV state가 batch step latency를 증가시켜 co-batched 다른 tenant의 TBT를 악화시키는 효과.

## TBT Debt

Tenant가 받아야 할 token service보다 실제 받은 token 수가 부족한 정도.

\[
Debt = ExpectedTokens - ServedTokens
\]

## Service Curve

특정 시간 구간 동안 최소 token emission service를 보장하는 모델.

\[
S(t_2)-S(t_1) \ge r(t_2-t_1)-\sigma
\]

## Admission Control

새 request를 바로 batch/scheduler에 넣을지, 기다리게 할지, 제한할지 결정하는 정책.

## Temporal Isolation Lane

특정 tenant/class에 decode time window 또는 quantum을 예약하여 token service opportunity를 보장하는 방식.

## Prefix Hit

이미 저장된 KV/prefix를 재사용해 recomputation을 피하는 것.

## Recompute Tokens

KV/prefix가 없어 다시 prefill해야 하는 token 수.

## Queue Pressure

Global blocking 또는 waiting queue 상태가 budget을 초과하는 정도.

## Prefill Pressure

Tenant의 prefill latency가 prefill budget을 초과하는 정도.

## SLO

Service Level Objective. 예: P95 TBT <= 40ms, P99 TTFT <= 800ms.
