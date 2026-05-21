# 10. Metrics and Logging Design

## 1. Goals

Metrics must answer:

1. Did KV retention reduce prefill?
2. Did KV retention increase blocking?
3. Did long-context tenant increase other tenant's TBT?
4. Is observed TBT caused by decode kernel, scheduler gap, prefill interference, KV stall, or preemption?
5. Did a policy satisfy tenant-level TBT SLO?
6. What throughput cost was paid?

## 2. Request-Level Fields

| Field | Type | Description |
|---|---|---|
| request_id | str | internal request id |
| tenant_id | str | tenant id, fallback `default` |
| tenant_kv_min_blocks | int | per-tenant KV minimum block guarantee from `VLLM_TENANT_KV_MIN_BLOCKS` |
| arrival_ts | float | request arrival time |
| first_scheduled_ts | float | first time scheduler selected request |
| first_token_ts | float | first output token time |
| finish_ts | float | request completion time |
| prompt_tokens | int | input tokens |
| output_tokens | int | generated output tokens |
| num_preemptions | int | preemption count |
| status | str | request status |
| ttft | float | first_token_ts - arrival_ts |
| ttl | float | finish_ts - arrival_ts |

## 3. Token-Level Fields

| Field | Type | Description |
|---|---|---|
| request_id | str | request id |
| tenant_id | str | tenant id |
| token_index | int | output token index |
| token_ts | float | token emitted timestamp |
| tbt | float | token_ts[k] - token_ts[k-1] |
| scheduler_iteration_id | int | iteration that produced token |
| stream_send_ts | float | optional client-visible send time |

## 4. Scheduler Iteration Fields

| Field | Type | Description |
|---|---|---|
| iteration_id | int | monotonic scheduler step id |
| scheduler_start_ts | float | scheduler start |
| scheduler_end_ts | float | scheduler end |
| model_start_ts | float | worker/model runner start |
| model_end_ts | float | worker/model runner end |
| active_batch_size | int | running/scheduled seq count |
| scheduled_prefill_tokens | int | scheduled prefill tokens |
| scheduled_decode_tokens | int | scheduled decode tokens |
| scheduled_total_tokens | int | total scheduled tokens |
| batch_total_context_tokens | int | sum context tokens |
| batch_max_context_tokens | int | max context tokens |
| batch_total_kv_blocks | int | sum KV blocks |
| batch_free_kv_blocks | int | free KV blocks |
| preemptions_this_iter | int | preemption count |
| kv_alloc_delay_ms | float | KV allocation time |
| queue_len_waiting | int | waiting queue length |
| queue_len_running | int | running queue length |

## 5. Tenant-Level Fields

| Field | Type | Description |
|---|---|---|
| tenant_id | str | tenant |
| served_tokens | int | cumulative served tokens |
| expected_tokens | float | service curve expectation |
| tbt_debt | float | expected - served |
| tbt_slo_ms | float | target TBT |
| tbt_violation_count | int | SLO violations |
| externality_budget | float | budget |
| externality_used | float | cumulative |
| kv_protected_blocks | int | protected KV |
| kv_elastic_blocks | int | elastic KV |
| prefix_hit_tokens | int | saved tokens |
| recompute_tokens | int | recomputed tokens |

## 6. TBT Decomposition Fields

| Component | How to Approximate Initially |
|---|---|
| DecodeKernelTime | model runner execution duration for decode step |
| SchedulerGap | time between previous model end and next model start excluding known waits |
| PrefillInterference | prefill tokens scheduled between consecutive decode outputs |
| KVAllocationDelay | KV allocation/reclaim time |
| PreemptionDelay | recompute or wait after preemption |
| StreamingOverhead | stream_send_ts - internal token_ts |

## 7. Log Format

Prefer JSON Lines for raw event logs.

```json
{"event":"scheduler_iter","iteration_id":1,"active_batch_size":8}
{"event":"token","request_id":"r1","tenant_id":"a","token_index":3,"tbt_ms":21.4}
{"event":"tenant_debt","tenant_id":"a","debt":5.2}
```

## 8. Prometheus vs File Logs

Use both if feasible.

- Prometheus: aggregate online metrics
- JSONL/CSV: experiment analysis and reproducibility

## 9. Default Behavior

Metrics must be gated by config/env flag if overhead is non-trivial.

Candidate flags:

```text
--enable-tenant-isolation-metrics
--tenant-isolation-log-path
--enable-tbt-decomposition
```
