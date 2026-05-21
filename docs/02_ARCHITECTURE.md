# 02. Proposed Architecture

## 1. High-Level Components

```text
OpenAI/API Request
      |
      v
Request Metadata / Tenant Resolver
      |
      v
Scheduler
  |       |        |          |
  |       |        |          +--> TBT-aware Admission Controller
  |       |        +-------------> Decode Externality Meter
  |       +----------------------> TBT Debt Manager
  +------------------------------> Dynamic KV Retention Controller
                                      |
                                      v
                              Elastic KV Reclaimer
                                      |
                                      v
                             KV Cache Manager
                                      |
                                      v
                            Worker / Model Runner
                                      |
                                      v
                             Output Processor / Metrics
```

## 2. Request Metadata / Tenant Resolver

역할:

- request에서 tenant_id를 추출한다.
- tenant_id가 없으면 `"default"`를 사용한다.
- tenant별 SLO config를 조회한다.
- request_id와 tenant_id mapping을 scheduler와 metrics에 전달한다.

초기 구현에서는 request metadata, header, extra body field 중 현재 vLLM에서 가장 덜 침습적인 경로를 사용한다.

## 3. TBT Debt Manager

역할:

- tenant별 expected token service 계산
- served output tokens 누적
- debt 계산
- debt가 큰 tenant에 priority credit 제공

상태 예시:

```python
tenant_state = {
    tenant_id: {
        "rate": tokens_per_sec,
        "burst": sigma,
        "served_tokens": int,
        "expected_tokens": float,
        "debt": float,
        "last_update_time": float,
    }
}
```

주의:

- wall-clock time과 scheduler iteration time을 혼동하지 않는다.
- request completion만 보지 말고 streaming output token 단위로 served count를 갱신해야 한다.
- default off 상태에서는 scheduling behavior가 변하지 않아야 한다.

## 4. Decode Externality Meter

역할:

- request 또는 tenant가 batch step latency에 기여한 정도를 추정한다.
- 정확한 Shapley value 계산은 비싸므로 초기에는 approximation을 사용한다.
- candidate admission 시 \(\hat{T}_{batch}(B \cup r)-\hat{T}_{batch}(B)\)를 예측한다.

가능한 feature:

- active batch size
- request context length
- total KV tokens
- max KV tokens
- scheduled decode tokens
- prefill tokens inserted
- attention backend
- model type
- GPU type

초기 approximation:

```text
externality_score =
    alpha * request_context_tokens
  + beta  * request_kv_blocks
  + gamma * batch_max_context_increase
  + delta * prefill_interference_score
```

## 5. TBT-Aware Admission Controller

역할:

- 새 request를 admit하면 기존 SLO tenant의 TBT/TBT debt가 악화되는지 예측한다.
- 조건을 만족하면 admit, 아니면 wait/throttle/scale signal을 보낸다.

Admission condition:

\[
Admit(r) \iff \hat{TBT}_j(B \cup r) \le \tau_j, \quad \forall j \in SLO\ tenants
\]

또는 debt 기준:

\[
Debt_j(t+\Delta) \le Debt_j^{max}
\]

초기 구현은 hard reject보다 queue delay를 선택한다.

## 6. Dynamic KV Retention Controller

역할:

- tenant별 protected KV budget \(G_i\)를 조절한다.
- prefill pressure, queue pressure, TBT debt, externality budget을 함께 본다.

Increase condition:

\[
PrefillPressure_i > 1+h
\]

and:

\[
QueuePressure < 1
\]

Decrease condition:

\[
QueuePressure > 1+h
\]

or:

\[
TBTDebt_{victim} > threshold
\]

AIMD-style:

```text
increase slowly
decrease quickly
```

## 7. Elastic KV Reclaimer

역할:

- protected KV는 기본적으로 보호한다.
- elastic KV 중 keep score가 낮은 block/session을 회수한다.

Score:

\[
KeepScore(b) =
ReuseBenefit(b) - ConcurrencyCost(b) - DecodeExternalityCost(b) + FairnessCredit_i
\]

victim:

\[
victim = \arg\min_{b \in ElasticKV} KeepScore(b)
\]

## 8. Temporal Isolation Lane

강한 TBT guarantee가 필요하고 mixed batching만으로 부족하면 사용한다.

예:

```text
Premium tenant decode windows: reserved quantum
Best-effort tenant decode windows: shared quantum
```

장점:

- token service opportunity를 시간적으로 보장한다.

단점:

- batching opportunity 감소
- TPS 감소 가능
- implementation complexity 증가

## 9. Recommended Implementation Order

1. Observability only
2. TBT decomposition
3. TBT Debt Manager skeleton
4. Externality Meter skeleton
5. Admission hook behind flag
6. KV Reclaimer hook behind flag
7. Dynamic controller
8. Temporal lane if needed
