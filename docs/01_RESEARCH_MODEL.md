# 01. Research Model

## 1. Basic Variables

| Symbol | Meaning |
|---|---|
| \(i\) | tenant id |
| \(r\) | request |
| \(B\) | current batch |
| \(G_i\) | tenant \(i\)의 KV guarantee budget |
| \(P_i\) | protected KV of tenant \(i\) |
| \(E_i\) | elastic KV of tenant \(i\) |
| \(C\) | total KV capacity |
| \(S_i(t)\) | tenant \(i\)가 시간 \(t\)까지 받은 output token 수 |
| \(r_i\) | tenant \(i\)의 minimum token emission rate |
| \(\tau_i\) | tenant \(i\)의 TBT SLO |
| \(\theta_i\) | tenant \(i\)의 TTFT SLO |

## 2. TTFT Model

\[
TTFT_i \approx Blocking_i + Prefill_i
\]

KV retention의 이득:

\[
KV\ retention \Rightarrow prefix\ hit \uparrow \Rightarrow Prefill \downarrow
\]

KV retention의 비용:

\[
KV\ retention \Rightarrow free/reclaimable\ KV \downarrow
\Rightarrow admission\ capacity \downarrow
\Rightarrow Blocking \uparrow
\]

따라서 KV를 보호할 조건은 다음이다.

\[
\Delta PrefillSaving > \Delta BlockingCost
\]

## 3. KV Guarantee Invariant

Tenant \(i\)의 KV를 나눈다.

\[
K_i = P_i \cup E_i
\]

Protected KV bound:

\[
|P_i| \le \min(G_i, |K_i|)
\]

No cross-tenant eviction:

\[
b \in P_i \Rightarrow b \text{ cannot be evicted for tenant } j \ne i
\]

Admission reserve:

\[
\sum_i G_i \le C - C^{min}_{admission}
\]

중요한 점은 \(G_i\)가 빈 공간 예약이 아니라, 실제 reusable KV 중 protected로 취급할 상한이라는 것이다.

## 4. TBT Model

Request-local model:

\[
TBT_i \approx f(context_i)
\]

하지만 continuous batching에서는 다음 모델이 더 적절하다.

\[
TBT_i \approx T^{shared}_{batch-step} + T^{local}_i
\]

Batch step latency는 batch 전체 context/KV traffic에 의존할 수 있다.

\[
T_{batch} \approx f\left(\sum_{r \in B} context_r, \max_{r \in B} context_r, |B|\right)
\]

Short-context tenant도 long-context tenant와 co-batch되면:

\[
\sum_{r \in B} context_r \uparrow
\Rightarrow T_{batch} \uparrow
\Rightarrow TBT_s \uparrow
\]

## 5. Observed TBT Decomposition

Observed TBT는 순수 decode kernel time만이 아니다.

\[
ObservedTBT = DecodeKernelTime + SchedulerGap + PrefillInterference + KVAllocationDelay + PreemptionDelay
\]

따라서 실험에서 반드시 분리해야 한다.

- Decode kernel duration
- Scheduler gap
- Prefill chunks inserted between decode steps
- KV allocation stall
- Preemption/recompute delay
- Token timestamp gap

## 6. Token Service Curve

Tenant \(i\)에 대한 token service guarantee:

\[
S_i(t_2) - S_i(t_1) \ge r_i \cdot (t_2 - t_1) - \sigma_i
\]

- \(S_i(t)\): 누적 served output tokens
- \(r_i\): 최소 token emission rate
- \(\sigma_i\): burst tolerance

## 7. TBT Debt

Expected token service:

\[
Expected_i(t) = r_i t
\]

Actual service:

\[
Served_i(t)
\]

Debt:

\[
Debt_i(t) = Expected_i(t) - Served_i(t)
\]

정책:

\[
Debt_i \uparrow \Rightarrow Priority_i \uparrow
\]

이 방식은 context length similarity가 아니라 실제 token service deficit을 기준으로 tenant를 보호한다.

## 8. Decode Externality

Tenant \(i\)의 request가 batch에 들어오며 증가시키는 batch latency:

\[
Externality_i = \hat{T}_{batch}(B \cup r_i) - \hat{T}_{batch}(B)
\]

누적 externality:

\[
X_i(t) = \sum Externality_i
\]

Budget 초과 조건:

\[
X_i(t) > X_i^{budget}
\]

조치:

- long-context decode admission 제한
- elastic KV retention 감소
- lower priority
- temporal lane 사용
- overload signal 발생

## 9. KV Retention Condition Under TBT Guarantee

\[
Keep(k_i) \quad \text{only if} \quad
ReuseBenefit_i > ConcurrencyCost_i + DecodeExternalityCost_i
\]

그리고:

\[
TBTDebt_{victim\ tenants} \le threshold
\]

즉, future prefill saving이 있어도 다른 tenant의 TBT guarantee를 깨는 KV는 보호하지 않는다.
