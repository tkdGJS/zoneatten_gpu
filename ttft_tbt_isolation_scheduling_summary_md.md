# TTFT부터 TBT까지: KV Retention, Isolation, Scheduling 논의 정리

## 0. 한 문장 요약

Multi-turn multi-tenant LLM serving에서 KV cache는 단순히 재사용 가능한 상태가 아니라 **prefill recomputation을 줄이는 동시에 serving concurrency와 decode cadence를 소모하는 shared scheduling resource**이다. 따라서 좋은 시스템은 KV를 많이 보존하는 것이 아니라, **KV retention의 recomputation benefit, blocking cost, decode externality, tenant-level TBT guarantee**를 함께 고려해야 한다.

---

## 1. 출발점: tenant별 KV minimum guarantee

초기 아이디어는 tenant마다 일정량의 KV cache를 보호해서 multi-turn workload에서 prefix recomputation을 줄이는 것이었다.

Tenant \(i\)의 KV guarantee를 \(G_i\)라고 하면:

\[
G_i \uparrow
\Rightarrow
prefix\ hit_i \uparrow
\Rightarrow
recompute\ tokens_i \downarrow
\Rightarrow
prefill_i \downarrow
\Rightarrow
TTFT_i \downarrow
\]

즉, tenant별 KV를 보호하면 다음 turn에서 동일 prefix/history를 다시 prefill하지 않아도 되므로 TTFT가 줄어들 수 있다.

그러나 논의가 진행되면서 이 관점은 불충분하다는 결론이 나왔다. KV를 많이 보호하면 prefill은 줄어들지만, 동시에 GPU KV capacity를 점유해서 새 요청을 admit할 수 있는 여유가 줄어들고, 그 결과 blocking time이 증가할 수 있다.

---

## 2. TTFT 관점의 핵심 trade-off

LLM serving에서 TTFT는 대략 다음처럼 볼 수 있다.

\[
TTFT_i \approx Blocking_i + Prefill_i
\]

KV retention은 두 방향으로 작용한다.

### 2.1 KV retention의 이득

\[
KV\ retention \Rightarrow prefix\ hit \uparrow \Rightarrow Prefill \downarrow
\]

### 2.2 KV retention의 비용

\[
KV\ retention \Rightarrow free/reclaimable\ KV \downarrow
\Rightarrow admission\ capacity \downarrow
\Rightarrow active\ batch\ size \downarrow
\Rightarrow Blocking \uparrow
\]

따라서 KV guarantee를 늘리는 것이 항상 TTFT를 줄이지 않는다.

중요한 판단 기준은 다음이다.

\[
\Delta PrefillSaving > \Delta BlockingCost
\]

즉, **KV를 보호해서 줄어드는 prefill time이, 그 KV가 capacity를 점유해서 증가시키는 blocking time보다 클 때만 보호해야 한다.**

---

## 3. Minimum guarantee sweep의 역할

Tenant별 \(G_i\)를 조금씩 늘려가며 성능을 측정하는 실험은 필요하다. 다만 목적은 “KV를 늘리면 좋아진다”를 보이는 것이 아니라, 다음을 찾는 것이다.

1. \(G_i\) 증가에 따른 prefill 감소량
2. \(\sum_i G_i\) 증가에 따른 blocking 증가량
3. TTFT/TTLT가 최소가 되는 knee point
4. online controller의 초기값과 증감폭
5. tenant/workload별 marginal benefit curve

예상되는 형태는 다음과 같다.

| KV guarantee | Prefill | Blocking | TTFT/TTLT |
|---:|---:|---:|---:|
| 너무 작음 | 큼 | 작음 | 큼 |
| 적당함 | 작음 | 적당함 | 최소 |
| 너무 큼 | 작음 | 큼 | 다시 증가 |

초기값은 다음처럼 정할 수 있다.

\[
G_i^{init} = \min G_i \quad s.t. \quad P99(Prefill_i) \le PrefillBudget_i
\]

또는 marginal prefill saving이 급격히 작아지는 knee point를 사용할 수 있다.

---

## 4. KV guarantee invariant

Tenant \(i\)의 KV cache를 다음처럼 나눈다.

\[
K_i = P_i \cup E_i
\]

- \(P_i\): protected KV
- \(E_i\): elastic KV

### 4.1 Protected KV bound

\[
|P_i| \le \min(G_i, |K_i|)
\]

중요한 점은 \(G_i\)가 “빈 공간 예약”이 아니라는 것이다. Tenant가 실제로 reusable KV를 가지고 있을 때, 그중 최대 \(G_i\)까지만 protected로 취급한다.

### 4.2 No cross-tenant eviction of protected KV

\[
b \in P_i \Rightarrow b \text{ cannot be evicted for tenant } j \ne i
\]

### 4.3 Elastic-only reclamation

새 요청을 admit하기 위해 KV가 부족할 때 victim은 elastic KV에서만 고른다.

\[
b \in \bigcup_i E_i
\]

### 4.4 Admission reserve

Protected KV가 전체 capacity를 다 먹지 않도록 다음 제약이 필요하다.

\[
\sum_i G_i \le C - C^{min}_{admission}
\]

이 제약이 없으면 KV isolation이 prefill hit rate만 높이고, 실제 serving concurrency를 무너뜨릴 수 있다.

---

## 5. Dynamic KV guarantee controller

정적 guarantee는 workload 변화에 취약하다. 따라서 tenant별 prefill pressure와 global queue pressure를 보고 \(G_i\)를 동적으로 조절해야 한다.

### 5.1 관찰 metric

Tenant별 metric:

\[
PrefillPressure_i = \frac{P99(Prefill_i)}{PrefillBudget_i}
\]

Global metric:

\[
QueuePressure = \frac{P95(Blocking)}{BlockingBudget}
\]

추가로 다음도 봐야 한다.

- prefix hit rate
- recompute tokens
- active batch size
- output TPS
- free/reclaimable KV
- observed TBT
- preemption/stall events

### 5.2 증가 조건

\[
PrefillPressure_i > 1+h
\]

그리고:

\[
QueuePressure < 1
\]

이면:

\[
G_i \leftarrow G_i + \Delta_{up}
\]

즉, tenant의 prefill은 나쁘지만 global blocking pressure가 낮을 때만 guarantee를 늘린다.

### 5.3 감소 조건

\[
QueuePressure > 1+h
\]

이면 guarantee를 줄인다. 단, 아무 tenant나 줄이지 않고 marginal benefit이 낮은 tenant부터 줄인다.

\[
victim = \arg\min_i MarginalBenefit_i(G_i)
\]

\[
G_{victim} \leftarrow G_{victim} - \Delta_{down}
\]

여기서:

\[
MarginalBenefit_i(G_i) = \frac{\Delta PrefillSaving_i}{\Delta G_i}
\]

정책적으로는 증가보다 감소를 빠르게 하는 AIMD-style이 적절하다.

\[
\Delta_{down} > \Delta_{up}
\]

---

## 6. Elastic KV reclaim

Protected KV는 건드리지 않고, elastic KV 중 가치가 낮은 것을 회수한다.

기본 keep score는 다음과 같다.

\[
KeepScore(b) =
\frac{
P_{reuse}(b) \cdot RecomputeCost(b) \cdot SLOUrgency_i
}{
Size(b)
}
\]

KV가 부족하면:

\[
victim = \arg\min_{b \in ElasticKV} KeepScore(b)
\]

그러나 이후 논의에서 단순 reuse benefit만으로는 부족하다는 결론이 나왔다. 최종적으로는 다음 비용을 함께 고려해야 한다.

\[
KeepScore(b) = ReuseBenefit(b) - ConcurrencyCost(b) - DecodeExternalityCost(b) + FairnessCredit_i
\]

즉, 어떤 KV가 재사용될 가능성이 높더라도, 그 KV가 긴 context decode를 유발해 다른 tenant의 TBT를 악화시키면 보호 가치가 낮아진다.

---

## 7. Blocking time과 scheduling

큐잉된 요청의 처리 순서는 단순 실행 순서가 아니라 **blocking time을 어떤 request/tenant에게 부과할지 결정하는 정책**이다.

\[
TTFT_i = Blocking_i + Prefill_i
\]

Scheduler가 누구를 먼저 admit하느냐에 따라 \(Blocking_i\)의 분포가 달라진다.

### 7.1 단순 정책의 한계

- FCFS: 구현은 쉽지만 convoy effect 가능
- Short-first: 평균 latency는 줄지만 long request starvation 가능
- Long-first: short request가 unnecessary blocking을 먹음
- Random: variance가 크고 tail이 불안정
- EDF/slack: SLO-aware이지만 KV/externality를 직접 반영하지 않음

### 7.2 KV-aware queue ordering

요청 \(r\)의 score는 다음 요소를 포함해야 한다.

\[
Score(r) = Urgency(r) + Aging(r) + ReuseBenefit(r)
- \lambda KVNeed(r)
- \mu PrefillCost(r)
- \gamma ExternalityCost(r)
\]

- \(Urgency\): SLO까지 얼마나 급한가
- \(Aging\): 오래 기다린 요청 구제
- \(ReuseBenefit\): protected KV를 활용해 prefill을 얼마나 줄이는가
- \(KVNeed\): 이 요청이 추가로 필요한 KV footprint
- \(PrefillCost\): decode cadence를 방해할 가능성
- \(ExternalityCost\): 다른 request의 TBT를 얼마나 늘리는가

중요한 점은 protected KV가 있는 요청을 무조건 먼저 처리하면 안 된다는 것이다. 그 요청이 long-context이고 batch-level externality가 크면, prefill saving보다 blocking/TBT cost가 클 수 있다.

---

## 8. Admission capacity에서 concurrency budget으로 추상화

논의 중 vLLM의 `max-num-batched-tokens`, `max-num-seqs`, KV capacity 등이 admission을 제한한다는 점을 다뤘다. 그러나 이를 논문 contribution으로 직접 밀면 너무 엔지니어링스럽다.

연구 abstraction은 다음이 더 적절하다.

> KV retention consumes serving concurrency budget.

즉, 특정 vLLM option이 핵심이 아니라, KV를 보호하는 정책이 future batch를 구성할 수 있는 동시성 여유를 줄인다는 구조적 문제가 핵심이다.

구현 섹션에서는 다음처럼 말할 수 있다.

> In our vLLM prototype, concurrency pressure manifests through concrete limits such as KV blocks, sequence slots, and per-iteration token budget. We use these as measurement signals to estimate the abstract concurrency cost.

본문에서는 다음 구조로 추상화한다.

\[
KV\ retention \Rightarrow ConcurrencyBudget\ consumption \Rightarrow Blocking
\]

---

## 9. TBT 관찰: 같은 output token 수인데 turn이 진행되면 TBT가 증가

초기 가설은 다음이었다.

> Turn이 진행될수록 context/KV length가 증가하므로, 같은 output token 수라도 token 하나를 생성하는 비용이 증가한다.

즉:

\[
TBT_i \approx f(context_i)
\]

하지만 group1과 group2의 input token 수가 크게 다른데도 TBT가 유사하다는 관찰이 나왔다. 이 경우 request-local context length만으로는 설명이 부족하다.

더 적절한 모델은 다음이다.

\[
TBT_i \approx T^{shared}_{batch-step} + T^{local}_i
\]

Continuous batching에서는 같은 decode iteration에 들어간 request들이 하나의 batch step latency를 공유한다. 따라서 짧은 context tenant도 긴 context tenant와 같은 batch에 들어가면 비슷한 TBT를 겪을 수 있다.

---

## 10. Cross-tenant TBT interference

TBT 간섭은 다음과 같이 정의할 수 있다.

> Under continuous batching, a tenant’s long KV/context increases the decode-step latency experienced by co-batched tenants, even when those tenants have short contexts.

이를 **batch-level decode externality** 또는 **cross-tenant TBT interference**라고 부를 수 있다.

### 10.1 Mechanism

Decode step latency는 batch 전체의 active context/KV traffic에 의존한다.

\[
T_{batch} \approx f\left(\sum_{r \in B} context_r, \max_{r \in B} context_r, |B|\right)
\]

따라서 short tenant \(s\)의 own context가 작아도:

\[
TBT_s \approx T_{batch}
\]

이고, 같은 batch에 long-context tenant가 있으면:

\[
\sum_{r \in B} context_r \uparrow
\Rightarrow T_{batch} \uparrow
\Rightarrow TBT_s \uparrow
\]

### 10.2 Group2가 손해 보는 이유

Group2가 input token 수가 작더라도, saturation 이후에는 shared batch-step latency가 local context 차이를 덮어버릴 수 있다.

\[
T^{shared}_{batch-step} \gg T^{local}_{group2}
\]

그러면:

\[
TBT_{group2} \approx TBT_{group1}
\]

이것은 group2가 자기보다 긴 context를 가진 group1의 decode externality를 함께 부담한다는 뜻이다.

---

## 11. TBT saturation과 observed TBT 분해

만약 blocking이 시작된 이후 active batch의 total KV/input tokens가 거의 일정하다면, 순수 decode kernel 관점의 TBT는 어느 정도 saturation되는 것이 자연스럽다.

\[
ActiveKV_{batch} \approx constant
\Rightarrow PureDecodeTBT \approx plateau
\]

그런데 observed TBT가 계속 증가한다면, 그것은 순수 decode kernel time만이 아닐 수 있다.

Observed TBT는 다음을 포함할 수 있다.

\[
ObservedTBT = DecodeKernelTime + SchedulerGap + PrefillInterference + KVAllocationDelay + PreemptionDelay
\]

따라서 TBT 분석에서는 반드시 다음을 분리해야 한다.

- 순수 decode kernel duration
- token timestamp gap
- prefill chunks inserted between decode steps
- scheduler gap
- KV allocation stall
- preemption/recompute delay

만약 median TBT는 saturation되고 p95/p99만 계속 증가한다면, 원인은 decode cost 증가가 아니라 tail stall일 가능성이 크다.

---

## 12. CascadeInfer류 접근과의 관계

CascadeInfer류 접근은 context length heterogeneity를 줄여 batch inefficiency를 완화한다. 즉, 길이가 비슷한 요청끼리 묶으면 batch step latency가 안정화된다.

하지만 우리가 원하는 방향이 단순히 “긴 것끼리, 짧은 것끼리 묶자”로 끝나면 novelty가 약하다.

따라서 차별화해야 할 점은 다음이다.

- CascadeInfer류: sequence/context length heterogeneity를 줄여 efficiency 개선
- 본 논의의 방향: tenant-level TBT service guarantee와 KV retention/externality control

즉, length grouping은 하나의 구현 수단일 수 있지만, 논문 핵심은 **tenant별 TBT guarantee enforcement**여야 한다.

---

## 13. TBT isolation을 guarantee 관점으로 재정의

TBT isolation을 단순 scheduling 문제가 아니라 **token service guarantee** 문제로 볼 수 있다.

Tenant \(i\)에 대해 TBT SLO를 둔다.

\[
P95(TBT_i) \le \tau_i
\]

또는 token emission rate로 표현하면:

\[
Rate_i \ge \frac{1}{\tau_i}
\]

Cumulative token service로 정의하면:

\[
S_i(t_2) - S_i(t_1) \ge r_i \cdot (t_2 - t_1) - \sigma_i
\]

- \(S_i(t)\): 시간 \(t\)까지 tenant \(i\)가 받은 output tokens
- \(r_i\): 최소 token emission rate
- \(\sigma_i\): burst tolerance

이렇게 정의하면 연구 질문은 다음으로 바뀐다.

> tenant \(i\)가 약속된 token service curve보다 뒤처지지 않도록 KV retention과 decode admission을 어떻게 제어할 것인가?

---

## 14. TBT debt 기반 isolation

Tenant \(i\)가 받아야 할 token 수:

\[
Expected_i(t) = r_i t
\]

실제로 받은 token 수:

\[
Served_i(t)
\]

TBT service debt:

\[
Debt_i(t) = Expected_i(t) - Served_i(t)
\]

- \(Debt_i > 0\): tenant가 약속보다 덜 받음
- \(Debt_i < 0\): tenant가 약속보다 더 받음

정책:

\[
Debt_i \uparrow \Rightarrow Priority_i \uparrow
\]

이 방식은 context length similarity가 아니라 실제 token service deficit을 기준으로 tenant를 보호한다.

---

## 15. Decode externality budget

TBT guarantee를 위해서는 피해 tenant뿐 아니라 가해 tenant도 제어해야 한다.

Tenant \(i\)의 요청이 batch에 들어가며 증가시키는 batch step latency를 다음처럼 둔다.

\[
Externality_i = \hat{T}_{batch}(B \cup r_i) - \hat{T}_{batch}(B)
\]

Tenant \(i\)의 누적 externality:

\[
X_i(t) = \sum Externality_i
\]

Tenant \(i\)가 허용된 externality budget을 초과하면:

\[
X_i(t) > X_i^{budget}
\]

그 tenant의 long-context decode admission을 제한하거나 elastic KV retention을 줄인다.

핵심은 다음이다.

> We do not merely group similar lengths. We meter and limit each tenant’s contribution to shared decode-step latency.

---

## 16. KV retention을 TBT guarantee에 종속시키기

기존 KV retention은 TTFT/prefill 개선을 위해 사용되었다. 그러나 TBT isolation 관점에서는 KV retention이 다른 tenant의 TBT guarantee를 깨지 않는 범위에서만 허용되어야 한다.

KV \(k_i\)를 유지할 조건:

\[
Keep(k_i) \quad \text{only if} \quad
ReuseBenefit_i > ConcurrencyCost_i + DecodeExternalityCost_i
\]

그리고:

\[
TBTDebt_{victim\ tenants} \le threshold
\]

즉, 어떤 tenant의 KV를 유지하면 그 tenant의 future prefill은 줄지만, 그 KV가 long-context decode externality를 만들어 다른 tenant의 TBT SLO를 깨면 유지하면 안 된다.

이렇게 하면 KV budget policy와 TBT isolation이 연결된다.

---

## 17. TBT-aware admission control

강한 TBT guarantee를 주장하려면 overload 상황을 인정해야 한다. 모든 tenant가 동시에 높은 token rate를 요구하면 어떤 정책도 guarantee를 만족시킬 수 없다.

따라서 admission control이 필요하다.

새 request \(r\)를 admit할 때:

\[
Admit(r) \iff \hat{TBT}_j(B \cup r) \le \tau_j, \quad \forall j \in SLO\ tenants
\]

또는 debt 기준으로:

\[
Debt_j(t+\Delta) \le Debt_j^{max}
\]

조건을 만족하지 못하면 request를 바로 admit하지 않고 queueing, throttling, scale-out 등의 조치를 취해야 한다.

---

## 18. Temporal isolation lane

정말 강한 TBT guarantee가 필요한 tenant에는 mixed batching만으로는 부족할 수 있다. 이 경우 tenant class별 temporal decode lane을 둘 수 있다.

예:

- Premium TBT-SLO tenant: dedicated decode windows
- Best-effort tenant: shared mixed windows

시간을 작은 quantum으로 나누고 tenant별 decode quantum을 보장한다.

\[
Q_i = \text{tenant } i \text{의 decode quantum}
\]

이 방식은 context length heterogeneity reduction과 다르다. 길이가 비슷한 요청끼리 묶는 것이 아니라, tenant별 token service 기회를 시간적으로 보장하는 것이다.

단점은 batching opportunity가 줄어 TPS가 감소할 수 있다는 점이다.

---

## 19. 최종 system objective

최종 objective는 두 가지 버전으로 잡을 수 있다.

### 19.1 Throughput-oriented version

\[
\max TPS
\]

subject to weak fairness / no starvation.

이 경우 연구는 isolation보다 global efficiency에 가깝다.

### 19.2 Isolation-oriented version

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

이 경우 TPS는 TBT/TTFT guarantee를 만족하는 범위에서 최대화하는 secondary objective가 된다.

현재 논문을 isolation 관점으로 가져가려면 19.2가 더 적절하다.

---

## 20. 최종 시스템 구조

### 20.1 Dynamic KV Retention Controller

- tenant별 KV protection/residency budget 조절
- prefill pressure, queue pressure, TBT debt, externality budget을 함께 봄
- KV를 보호할지 말지는 recomputation benefit과 concurrency/TBT cost를 비교해 결정

### 20.2 Elastic KV Reclaimer

- protected KV는 유지
- elastic KV 중 score가 낮은 것을 회수
- score는 reuse benefit뿐 아니라 concurrency cost와 decode externality cost 포함

### 20.3 TBT Debt Manager

- tenant별 token service curve 추적
- 실제 served tokens와 expected served tokens의 차이를 debt로 관리
- debt가 큰 tenant의 decode service priority 증가

### 20.4 Decode Externality Meter

- tenant별로 batch step latency에 기여한 정도를 추정
- externality budget 초과 tenant의 long-context admission 또는 KV retention 제한

### 20.5 TBT-aware Admission Controller

- 새 요청을 admit하면 기존 tenant의 TBT guarantee가 깨지는지 예측
- 깨질 경우 queueing/throttling/scale-out 또는 temporal lane 사용

---

## 21. 필요한 실험

### 21.1 Guarantee sweep

Tenant별 KV budget을 sweep해서 prefill 감소와 blocking 증가의 trade-off를 보인다.

필요 metric:

- p99 TTFT
- p99 blocking
- p99 prefill
- prefix hit rate
- recompute tokens
- active batch size
- output TPS

### 21.2 Group1-only / Group2-only / Mixed

Short-context tenant가 long-context tenant와 섞일 때 TBT 손해를 보는지 확인한다.

핵심 결과:

\[
TBT_{G2|mixed} > TBT_{G2|only}
\]

그리고:

\[
TBT_{G2|mixed} \approx TBT_{G1|mixed}
\]

이면 batch-level decode externality를 강하게 주장할 수 있다.

### 21.3 TBT decomposition

Observed TBT를 분해한다.

\[
ObservedTBT = DecodeKernelTime + SchedulerGap + PrefillInterference + KVAllocationDelay + PreemptionDelay
\]

로깅해야 할 것:

- per-iteration decode step time
- active batch size
- batch total context/KV tokens
- batch max context/KV tokens
- prefill tokens scheduled between decode steps
- KV blocks used/free
- preemption count
- token timestamp gap

### 21.4 TBT service guarantee evaluation

Baseline:

1. vLLM default
2. length-aware batching
3. static tenant partition
4. KV retention only
5. dynamic KV retention only
6. TBT debt only
7. proposed: KV retention + externality budget + TBT-aware admission

Metric:

- P95/P99 TBT per tenant
- TBT SLO violation rate
- token service debt over time
- output TPS
- TTFT/TTLT
- blocking
- prefix hit/recompute tokens
- short tenant degradation under mixed workload

---

## 22. 논문 contribution 후보

### Contribution 1: Characterization

Multi-turn multi-tenant serving에서 KV retention은 prefill recomputation을 줄이지만, blocking과 batch-level TBT externality를 유발함을 보인다.

### Contribution 2: Abstraction

KV cache를 단순 reusable memory가 아니라 **concurrency-consuming scheduling resource**로 정의한다.

### Contribution 3: TBT isolation metric

Tenant별 TBT를 token service curve 또는 token service debt로 정의하고, TBT guarantee를 enforce해야 함을 제안한다.

### Contribution 4: Control policy

KV retention, elastic reclaim, decode externality metering, TBT-aware admission을 결합해 tenant-level TBT guarantee를 만족하면서 TPS 손실을 최소화한다.

---

## 23. Reviewer 관점의 위험 요소

### 위험 1: 그냥 length-aware batching 아닌가?

방어:

- 우리는 context length heterogeneity reduction이 아니라 tenant-level token service guarantee를 목표로 한다.
- 길이 grouping은 하나의 heuristic일 뿐이며, 핵심은 TBT debt와 externality budget이다.

### 위험 2: KV guarantee가 isolation이 아니라 throughput optimization 아닌가?

방어:

- Throughput-oriented version이라면 “isolation” 용어를 줄여야 한다.
- Isolation version으로 가려면 TBT/TTFT SLO constraint를 명확히 둬야 한다.

### 위험 3: Strong guarantee가 feasible한가?

방어:

- overload 상황에서는 guarantee가 불가능하므로 admission control 또는 scale-out trigger가 필요하다.
- 우리는 feasible capacity 영역에서 token service curve를 enforce한다고 명시해야 한다.

### 위험 4: TBT 증가가 순수 decode 때문인가, scheduler gap 때문인가?

방어:

- observed TBT를 decode kernel time, scheduler gap, prefill interference, KV stall, preemption delay로 분해하는 실험을 해야 한다.

---

## 24. 최종 framing 후보

### Throughput-oriented framing

> KV cache retention is a throughput-control problem: retaining more KV reduces recomputation but consumes concurrency budget. We dynamically retain only KV states whose recomputation savings outweigh their concurrency and decode-externality costs.

### Isolation-oriented framing

> TBT isolation requires more than length-aware batching. We enforce tenant-level token service guarantees by metering decode externality and controlling KV residency, so that one tenant’s long KV state cannot persistently degrade another tenant’s token emission rate.

현재 사용자의 문제의식이 “tenant 간 TBT 간섭을 isolation 관점에서 어떻게 보장할 수 있나”로 이동했으므로, 최종적으로는 isolation-oriented framing이 더 적절하다.

---

## 25. 가장 중요한 결론

1. TTFT 관점에서 KV retention은 prefill saving과 blocking cost의 trade-off다.
2. TBT 관점에서 KV retention은 batch-level decode externality를 만들 수 있다.
3. Short-context tenant도 long-context tenant와 co-batch되면 TBT 손해를 볼 수 있다.
4. 단순 length-aware batching은 이 문제를 완전히 설명하거나 해결하지 못한다.
5. TBT isolation을 주장하려면 tenant별 token service curve 또는 TBT debt를 정의해야 한다.
6. KV retention은 TBT guarantee를 깨지 않는 범위에서만 허용되어야 한다.
7. 최종 시스템은 KV budget controller, elastic reclaimer, TBT debt manager, decode externality meter, TBT-aware admission controller를 결합해야 한다.

한 문장으로 정리하면:

> KV cache는 future prefill을 줄이는 cache인 동시에 current/future decode service를 오염시키는 shared state이다. 따라서 multi-tenant LLM serving에서 진짜 isolation은 KV를 얼마나 보존할지가 아니라, tenant별 token emission service를 보장하면서 KV retention과 decode externality를 어떻게 제어할지의 문제다.

