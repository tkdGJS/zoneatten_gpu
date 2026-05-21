# 09. Experiment Design

## 1. Experiment Principles

- 정책 구현보다 먼저 baseline과 metric을 안정화한다.
- 평균보다 P95/P99를 중시한다.
- TTFT와 TBT를 따로 보고, TBT는 반드시 decomposition한다.
- Throughput만 좋아지는 정책을 isolation으로 포장하지 않는다.
- Overload와 feasible region을 분리한다.

## 2. Workload Dimensions

| Dimension | Values |
|---|---|
| Tenants | 1, 2, 4, 8 |
| Context length | short, medium, long, mixed |
| Output tokens | fixed, variable |
| Turn count | 1, 2, 4, 8, 16 |
| Arrival pattern | closed-loop, Poisson, burst |
| Tenant class | SLO, best-effort |
| Prefix reuse | none, low, medium, high |
| Load | underload, near capacity, overload |

## 3. Experiment A: KV Guarantee Sweep

목적:

Tenant별 KV budget \(G_i\)를 sweep해서 prefill 감소와 blocking 증가 trade-off를 보인다.

### A.1 Confirmed Scope

이번 실험의 minimum guarantee는 **KV blocks** 단위로 정의한다.

Sweep values:

```text
kv_min_budget_blocks = 0, 8, 16, 32, 64, 128, 256
```

Primary metric source:

```text
patched vLLM request metrics JSONL
```

Primary outcome:

```text
prefill_time_s
```

`prefill_time_s`는 patched vLLM이 request-level JSONL에 직접 기록한 값을 사용한다. 분석 스크립트에서 `scheduled_ts`, `first_token_ts` 등을 다시 조합해 임의로 prefill을 재계산하지 않는다. 단, sanity check 목적으로 `first_token_ts - scheduled_ts`와의 차이는 별도 컬럼으로 비교할 수 있다.

### A.2 Workload Source

Workload source는 기존 실험 스크립트를 그대로 재사용한다.

Canonical runner:

```bash
./run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh
```

이 실험의 비교 방식은 "새 workload를 만들지 않고, 같은 runner/workload를 수정된 vLLM 코드 위에서 다시 실행했을 때 request metrics JSONL의 `prefill_time_s`가 어떻게 바뀌는가"다.

현재 runner에서 확인된 주요 설정:

| Script variable | Current value | Meaning |
|---|---:|---|
| `BLOCK_VALUE` | `16384` | `VLLM_NUM_GPU_BLOCKS_OVERRIDE`로 전달되는 전체 GPU KV block override |
| `TENANT_VALUES` | `32 16 8` | tenant count sweep |
| `TENANT_KV_MIN_BLOCK_VALUES` | `0 8 16 32 64 128 256` | `VLLM_TENANT_KV_MIN_BLOCKS`로 전달되는 per-tenant minimum KV block guarantee sweep |
| `RUNS_PER_SETTING` | `1` | setting당 반복 횟수 |
| `TURNS_PER_TENANT` | `10` | tenant별 turn 수 |
| `METRICS_DIR` | `vram_only_artifacts_smallctx_mixed_limits_8GB/request_metrics` | request metrics JSONL output directory |

Runner는 다음 env를 vLLM 서버 시작 시 넘긴다.

```text
VLLM_REQUEST_METRICS_JSONL
VLLM_NUM_GPU_BLOCKS_OVERRIDE
VLLM_TENANT_KV_MIN_BLOCKS
VLLM_MAX_NUM_SEQS
VLLM_DISABLE_PREFIX_CACHING=0
VLLM_MAX_MODEL_LEN
VLLM_MAX_NUM_BATCHED_TOKENS
```

따라서 sweep 해석은 다음 조건을 전제로 한다.

- runner script는 기존 workload behavior를 유지하되 `TENANT_KV_MIN_BLOCK_VALUES` sweep 축을 추가한다.
- workload generation/reuse 경로는 기존 script semantics를 따른다.
- patched vLLM이 request metrics JSONL에 `prefill_time_s`를 기록한다.
- patched vLLM이 tenant KV guarantee 정책을 적용하므로 같은 runner에서도 결과가 달라진다.

주의: runner의 `TENANT_VALUES=(32 16 8)`은 tenant 수 sweep이고, `TENANT_KV_MIN_BLOCK_VALUES="0 8 16 32 64 128 256"`가 per-tenant KV guarantee sweep이다. 각 vLLM server run은 `VLLM_TENANT_KV_MIN_BLOCKS=<value>`를 환경 변수로 받는다.

### A.3 Tenant Setup

Tenant 수:

```text
num_tenants = 8, 16, 32
```

현재 runner는 `TENANT_VALUES=(32 16 8)` 순서로 실행한다. 분석에서는 이를 `8`, `16`, `32` tenant settings로 정규화해 비교한다.

기본 guarantee sweep은 모든 tenant에 같은 per-tenant minimum KV block guarantee를 적용한다. 즉 `VLLM_TENANT_KV_MIN_BLOCKS=8`이면 각 tenant가 동일하게 8 KV blocks minimum guarantee를 받는 정책으로 해석한다.

All-tenant sweep:

| Run | all tenants `kv_min_budget_blocks` |
|---|---:|
| all_g0 | 0 |
| all_g8 | 8 |
| all_g16 | 16 |
| all_g32 | 32 |
| all_g64 | 64 |
| all_g128 | 128 |
| all_g256 | 256 |

All-tenant sweep은 "모두에게 guarantee를 주면 capacity pressure 속에서도 prefill benefit이 유지되는가"를 본다. 특정 tenant만 보호하는 single-target sweep은 별도 후속 실험으로 둔다.

Feasibility note:

```text
theoretical_protected_blocks = tenant_count * kv_min_budget_blocks
```

예를 들어 `tenant_count=32`, `kv_min_budget_blocks=256`이면 이론상 최대 `8192` reusable cached blocks가 protected가 될 수 있다. 현재 runner의 `BLOCK_VALUE=16384`보다 작지만, active KV, admission 여유, long-context footprint를 제외한 실제 evictable capacity는 더 작다. 따라서 high guarantee point에서는 prefill 감소보다 queueing/preemption 증가가 더 커질 수 있다.

### A.4 Controlled Variables

Sweep 간 다음 값은 고정한다.

- model
- GPU
- vLLM version and git commit
- block size
- max model len
- `max_num_batched_tokens`
- `max_num_seqs`
- scheduler policy
- prefix caching setting
- random seed
- tenant request mix
- prompt length distribution
- output length distribution
- turn count distribution
- request arrival timestamps or closed-loop concurrency

### A.5 Run Matrix

Minimum matrix:

```text
tenant_count in [8, 16, 32] x kv_min_budget_blocks in [0, 8, 16, 32, 64, 128, 256] x N seeds
```

Runner matrix:

```text
tenant_count in [32, 16, 8] x kv_min_budget_blocks in [0, 8, 16, 32, 64, 128, 256] x RUNS_PER_SETTING=1
```

With the current defaults:

```text
3 tenant-count settings x 7 KV guarantee settings x 1 run = 21 runs
```

Metrics:

- per-tenant p50/p95/p99 `prefill_time_s`
- per-tenant p50/p95/p99 TTFT
- per-tenant p50/p95/p99 blocking / queueing time
- prefix hit rate
- recompute tokens
- active batch size
- output TPS
- free/reclaimable KV
- preemption count
- protected KV blocks
- elastic KV blocks
- evicted KV blocks
- request count per tenant

Primary plots:

- x-axis: `kv_min_budget_blocks`
- y-axis: target tenant p50/p95/p99 `prefill_time_s`
- secondary y-axis or separate plot: other tenants p95 `prefill_time_s`
- queueing/TTFT plot to show whether prefill gains were offset by blocking
- KV pressure plot: free KV blocks and eviction count

Acceptance evidence:

- For a target tenant, increasing `kv_min_budget_blocks` from 0 to 8/16/32/64/128/256 should reduce p50 or p95 `prefill_time_s` if reusable prefix exists.
- If `prefill_time_s` does not decrease, report whether prefix reuse was absent, guarantee was too small, KV pressure caused eviction anyway, or the metric does not isolate model prefill execution.
- If `prefill_time_s` decreases but TTFT does not, inspect queueing/blocking time.
- If target tenant improves while other tenants degrade, quantify the cost as isolation trade-off rather than claiming pure improvement.

Expected shape:

| KV guarantee | Prefill | Blocking | TTFT |
|---:|---:|---:|---:|
| too low | high | low | high |
| proper | low | moderate | minimum |
| too high | low | high | high |

## 4. Experiment B: Group1-only / Group2-only / Mixed

목적:

Short-context tenant가 long-context tenant와 섞일 때 TBT 손해를 보는지 확인한다.

Groups:

- Group1: long context tenant
- Group2: short context tenant

Runs:

1. Group1-only
2. Group2-only
3. Group1 + Group2 mixed

Key evidence:

\[
TBT_{G2|mixed} > TBT_{G2|only}
\]

and:

\[
TBT_{G2|mixed} \approx TBT_{G1|mixed}
\]

이면 batch-level decode externality를 주장할 수 있다.

Metrics:

- per-tenant P50/P95/P99 TBT
- batch step latency
- batch total context/KV tokens
- batch max context/KV tokens
- active batch size
- token timestamp gap

## 5. Experiment C: TBT Decomposition

목적:

Observed TBT 증가가 순수 decode 때문인지 scheduler/prefill/KV stall 때문인지 분해한다.

Model:

\[
ObservedTBT = DecodeKernelTime + SchedulerGap + PrefillInterference + KVAllocationDelay + PreemptionDelay
\]

Required logs:

- per-iteration decode/model execution time
- scheduler loop time
- prefill tokens scheduled between decode steps
- KV blocks used/free
- KV allocation delay
- preemption count
- output token timestamp gap

## 6. Experiment D: TBT Service Guarantee Evaluation

Baselines:

1. vLLM default
2. length-aware batching
3. static tenant partition
4. KV retention only
5. dynamic KV retention only
6. TBT debt only
7. proposed: KV retention + externality budget + TBT-aware admission

Metrics:

- P95/P99 TBT per tenant
- TBT SLO violation rate
- token service debt over time
- output TPS
- TTFT/TTLT
- blocking
- prefix hit/recompute tokens
- short tenant degradation under mixed workload

## 7. Experiment E: Ablation Study

Ablations:

| Policy | TBT Debt | Externality Meter | KV Reclaim | Admission |
|---|---:|---:|---:|---:|
| default | off | off | off | off |
| debt only | on | off | off | off |
| externality only | off | on | off | off |
| KV only | off | off | on | off |
| admission only | off | on | off | on |
| full | on | on | on | on |

## 8. Reporting Template

Every experiment report must include:

- git commit hash
- vLLM version
- model
- GPU
- CLI flags
- workload config
- random seed
- policy flags
- raw metrics path
- summary table
- plots
- conclusion
- limitations
