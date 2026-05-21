# 05. Implementation Plan

## Phase 0: Baseline and Repository Mapping

목표: 코드 수정 없이 현재 vLLM checkout을 이해한다.

Tasks:

- vLLM version 확인
- Scheduler/KV/Request/Metrics path 확인
- 최소 smoke test 실행
- `docs/03_VLLM_CODE_MAP.md` 업데이트

Exit criteria:

- 현재 버전의 주요 파일 경로가 문서화됨
- 기본 import/pytest smoke test 통과

## Phase 1: Observability Patch

목표: scheduling behavior를 바꾸지 않고 필요한 데이터를 모은다.

Add:

- tenant_id propagation
- request arrival/first token/output token timestamp
- scheduler iteration id
- active batch size
- scheduled prefill/decode tokens
- batch total/max context/KV tokens
- KV blocks used/free
- preemption event count
- per-iteration scheduling and execution time

Rules:

- default behavior unchanged
- metrics can be disabled
- no scheduling policy change

Exit criteria:

- default off 테스트 통과
- enabled 상태에서 metric/log 확인 가능

## Phase 2: TBT Decomposition

목표: observed TBT 증가 원인을 분해한다.

Compute:

\[
ObservedTBT = DecodeKernelTime + SchedulerGap + PrefillInterference + KVAllocationDelay + PreemptionDelay
\]

Tasks:

- token timestamp gap 계산
- scheduler loop gap 계산
- model execution duration 기록
- prefill chunk insertion 여부 기록
- KV allocation delay 기록
- preemption/recompute delay 기록

Exit criteria:

- request/tenant별 TBT breakdown csv/json 생성 가능

## Phase 3: Baseline Policies

목표: 제안 정책 전 비교군을 만든다.

Baselines:

1. vLLM default
2. length-aware batching
3. static tenant partition
4. KV retention only
5. TBT debt only

주의:

- baseline도 config flag로 구분
- 실험 script에서 seed, workload, model, GPU 정보를 기록

## Phase 4: TBT Debt Manager

목표: tenant별 token service debt를 계산하고 priority signal을 만든다.

Minimal class:

```python
class TBTDebtManager:
    def register_tenant(self, tenant_id, rate, burst): ...
    def update_time(self, now): ...
    def on_tokens_served(self, tenant_id, n_tokens, now): ...
    def get_debt(self, tenant_id, now): ...
    def get_priority_credit(self, tenant_id, now): ...
```

초기에는 scheduling에 연결하지 않고 unit test부터 작성한다.

## Phase 5: Decode Externality Meter

목표: tenant/request가 batch step latency에 주는 externality를 추정한다.

Initial estimator:

```text
externality_score =
    alpha * context_tokens
  + beta * kv_blocks
  + gamma * max_context_increase
  + delta * prefill_interference
```

Later:

- actual measured batch step time으로 online regression
- per-model/per-GPU calibration

## Phase 6: TBT-Aware Admission

목표: candidate request가 existing SLO tenant를 망가뜨리는지 판단한다.

Initial policy:

```text
if predicted_tbt_violation or debt_violation:
    keep in waiting queue
else:
    admit
```

주의:

- starvation 방지를 위해 aging 필요
- overload에서 무한 대기하지 않도록 policy action을 명확히 둠

## Phase 7: Elastic KV Reclaimer

목표: KV retention을 TBT guarantee에 종속시킨다.

Keep condition:

\[
ReuseBenefit > ConcurrencyCost + DecodeExternalityCost
\]

Tasks:

- protected vs elastic 구분
- elastic victim score 계산
- protected budget bound 유지
- admission reserve 유지

## Phase 8: Full Controller

목표: Dynamic KV Retention + TBT Debt + Externality + Admission 결합.

Controller signals:

- PrefillPressure
- QueuePressure
- TBTDebt
- ExternalityBudget
- KVFree/Reclaimable
- Output TPS

## Phase 9: Experiment and Report

목표: 논문/보고서에 들어갈 데이터 생성.

Experiments:

- guarantee sweep
- group-only vs mixed
- TBT decomposition
- service guarantee evaluation
- ablation study

Outputs:

- tables
- figures
- raw logs
- experiment report
- patch notes
