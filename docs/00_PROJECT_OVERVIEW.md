# 00. Project Overview

## 목적

이 프로젝트는 vLLM source code를 수정하여 **multi-turn multi-tenant LLM serving에서 TTFT/TBT isolation을 보장하거나, 최소한 보장 가능한 조건과 trade-off를 실험적으로 밝히는 prototype**을 만든다.

## 핵심 문제

기존 관점에서는 KV cache retention을 주로 prefix recomputation 감소, 즉 TTFT 개선 수단으로 본다.

하지만 multi-tenant continuous batching 환경에서는 KV retention이 다음 비용도 만든다.

1. GPU KV capacity 점유
2. Admission capacity 감소
3. Blocking time 증가
4. Long-context decode에 의한 batch-level TBT externality
5. Short-context tenant의 TBT 악화

따라서 KV retention은 cache policy가 아니라 **shared scheduling resource control problem**으로 다뤄야 한다.

## One-Sentence Framing

> TBT isolation requires more than length-aware batching. We enforce tenant-level token service guarantees by metering decode externality and controlling KV residency, so that one tenant’s long KV state cannot persistently degrade another tenant’s token emission rate.

## 연구 질문

### RQ1. KV retention은 언제 TTFT를 줄이고 언제 blocking을 증가시키는가?

\[
TTFT_i \approx Blocking_i + Prefill_i
\]

KV retention은 prefill을 줄이지만, concurrency budget을 소모하여 blocking을 증가시킬 수 있다.

### RQ2. Short-context tenant도 long-context tenant와 co-batch되면 TBT 손해를 보는가?

Continuous batching에서 같은 decode iteration에 들어간 request들은 batch step latency를 공유한다.

\[
TBT_s \approx T_{batch}
\]

따라서 short tenant 자체 context가 짧아도, long tenant와 같은 batch에 있으면 batch-level decode externality를 함께 부담할 수 있다.

### RQ3. TBT isolation을 scheduling heuristic이 아니라 guarantee로 정의할 수 있는가?

Tenant별 TBT SLO 또는 token service curve를 둔다.

\[
P95(TBT_i) \le \tau_i
\]

또는:

\[
S_i(t_2) - S_i(t_1) \ge r_i(t_2-t_1) - \sigma_i
\]

## 최종 목표

\[
\max TPS
\]

subject to:

\[
P95(TBT_i) \le \tau_i, \quad \forall i \in SLO\ tenants
\]

\[
P99(TTFT_i) \le \theta_i, \quad optional
\]

\[
NoStarvation_i
\]

TPS는 primary objective가 아니라, SLO를 만족하는 범위에서 최대화하는 secondary objective다.

## 구현 범위

### 포함

- tenant id propagation
- TTFT/TBT metrics
- observed TBT decomposition
- KV usage / prefix hit / recompute tokens metrics
- TBT debt manager
- decode externality meter
- TBT-aware admission hook
- elastic KV reclaimer hook
- baseline comparison scripts and reports

### 제외 또는 후순위

- CUDA kernel 자체 최적화
- model architecture 변경
- distributed serving full redesign
- production-grade autoscaler
- 완전한 hard real-time guarantee

## Feasibility 조건

강한 guarantee는 overload 상황에서는 불가능하다. 따라서 다음을 명시한다.

- feasible capacity 영역에서 service curve를 enforce한다.
- overload에서는 queueing, throttling, admission reject, scale-out trigger가 필요하다.
- 실험에서는 offered load를 명시하고, capacity boundary를 함께 보고한다.
