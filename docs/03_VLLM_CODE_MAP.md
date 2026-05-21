# 03. vLLM Code Map

> 주의: vLLM 파일 구조는 버전마다 변한다. 이 문서는 시작점이다. 실제 구현 전 반드시 현재 checkout에서 `rg`, `git grep`, `find`로 확인한다.

## 0. Verified Checkout Snapshot

- Verified date: 2026-05-21
- Source root: `vendor/vllm`
- vLLM version: `0.14.0`
- vLLM commit id: `gb17039bcc`
- Current checkout note: top-level `tests/` is not present. Test paths below are target locations to create or restore if this vendor snapshot remains source-only.

## 1. Search Commands

```bash
rg "class Scheduler|def schedule|SchedulerOutput|SchedulerConfig" vendor/vllm
rg "KVCacheManager|KVCacheBlocks|BlockPool|free\(|allocate" vendor/vllm
rg "RequestStatus|class Request|request_id|arrival_time" vendor/vllm/v1 vendor/vllm
rg "max_num_batched_tokens|max_num_seqs|enable_chunked_prefill|max_num_partial_prefills" vendor/vllm
rg "Prometheus|metrics|Stats|record_event|EngineCoreEventType" vendor/vllm
rg "OpenAI|chat/completions|responses|metadata|extra_body|vllm_xargs" vendor/vllm/entrypoints/openai
find vendor/vllm -maxdepth 4 -type f | rg "scheduler|request_queue|output|kv_cache_manager|request\.py|metrics|openai"
```

## 2. Likely Important Files

| Area | Candidate Path | Role |
|---|---|---|
| Scheduler config | `vendor/vllm/config/scheduler.py` | `SchedulerConfig`, `max_num_batched_tokens`, `max_num_seqs`, chunked prefill, async scheduling 등 scheduler options |
| Scheduler main | `vendor/vllm/v1/core/sched/scheduler.py` | `Scheduler`, request queue 관리, running/waiting scheduling, preemption, KV manager 호출 |
| Scheduler interface | `vendor/vllm/v1/core/sched/interface.py` | `SchedulerInterface`, stats API |
| Scheduler output | `vendor/vllm/v1/core/sched/output.py` | `SchedulerOutput`, worker/model runner로 전달되는 scheduled request 정보 |
| Request queue | `vendor/vllm/v1/core/sched/request_queue.py` | `FCFSRequestQueue`, `PriorityRequestQueue`, waiting queue ordering/policy 후보 |
| KV cache manager | `vendor/vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`, `KVCacheBlocks`, Scheduler와 KV manager 사이 allocation result interface |
| Single type KV manager | `vendor/vllm/v1/core/single_type_kv_cache_manager.py` | `SingleTypeKVCacheManager`, `FullAttentionManager`, `SlidingWindowManager` 등 실제 block allocation/free/reuse 구현 |
| KV cache utilities | `vendor/vllm/v1/core/kv_cache_utils.py` | KV block 계산/utility |
| KV cache metrics | `vendor/vllm/v1/core/kv_cache_metrics.py` | block residency lifecycle metrics |
| Request object | `vendor/vllm/v1/request.py` | `Request`, `RequestStatus`, request_id, status, token counts, timestamps 후보 |
| Engine request structs | `vendor/vllm/v1/engine/__init__.py` | `EngineCoreRequest`, `EngineCoreOutput`, `EngineCoreOutputs` |
| Engine core | `vendor/vllm/v1/engine/core.py` | scheduler와 model execution loop 연결 |
| Async LLM | `vendor/vllm/v1/engine/async_llm.py` | API layer와 engine request submission 연결 |
| Output processor | `vendor/vllm/v1/engine/output_processor.py` | request metrics jsonl, first/last token timestamp aggregation 후보 |
| Worker/model runner | `vendor/vllm/v1/worker/gpu_model_runner.py` 또는 `vendor/vllm/v1/worker/gpu/model_runner.py` | GPU execution, per-step timing 후보 |
| Metrics | `vendor/vllm/v1/metrics/*` | `SchedulerStats`, `IterationStats`, Prometheus/stat logger 추가 후보 |
| OpenAI entrypoints | `vendor/vllm/entrypoints/openai/*` | tenant_id를 request metadata/header에서 받는 후보 |
| Tests | currently absent | `vendor/vllm` snapshot에는 test tree가 없음. 새 단위 테스트 위치를 별도로 결정해야 함 |

## 3. Scheduler Fields to Inspect

현재 Scheduler에 다음이 있는지 확인한다.

- `self.waiting`
- `self.running`
- `self.kv_cache_manager`
- `self.encoder_cache_manager`
- `self.log_stats`
- preemption handler
- `schedule()` method
- `update_from_output()` 또는 비슷한 output update method
- `finish_requests()`
- `get_request_counts()`

## 4. Request Fields to Inspect

확인할 후보:

- `request_id`
- `prompt_token_ids`
- `output_token_ids`
- `num_computed_tokens`
- `num_tokens`
- `status`
- `arrival_time`
- `first_token_time`
- `num_preemptions`
- `metadata`
- `sampling_params`
- `client_index`

현재 `vendor/vllm/v1/request.py`에서 확인된 field:

- `request_id`
- `client_index`
- `sampling_params`
- `arrival_time`
- `status`
- `prompt_token_ids`
- `num_prompt_tokens`
- `num_computed_tokens`
- `output_token_ids`
- `num_preemptions`
- `kv_transfer_params` from `sampling_params.extra_args`
- `record_event()`

현재 Request에는 `tenant_id`, generic `metadata`, `first_token_time` field가 없다. tenant propagation 후보는 다음 중 하나를 별도 patch에서 결정한다.

- OpenAI request `metadata`
- OpenAI `vllm_xargs` -> `SamplingParams.extra_args`
- HTTP header, 예: `X-Tenant-ID`
- 새 internal `Request.tenant_id` field

필요한 추가 후보:

- `tenant_id`
- `tenant_slo_class`
- `tbt_last_token_ts`
- `tbt_observed_gaps`
- `served_output_tokens`
- `externality_score_accum`

## 5. SchedulerConfig Fields to Inspect

공식 문서상 확인해야 할 field:

- `max_num_batched_tokens`: single iteration에서 처리 가능한 최대 token 수
- `max_num_seqs`: single iteration에서 처리 가능한 최대 sequence 수
- `enable_chunked_prefill`
- `max_num_partial_prefills`
- `max_long_partial_prefills`
- `long_prefill_token_threshold`
- `disable_hybrid_kv_cache_manager`
- `policy`
- `scheduler_cls`
- `async_scheduling`
- `stream_interval`

추가 실험 flag 후보:

- `enable_tenant_isolation`
- `enable_tbt_metrics`
- `enable_tbt_debt_priority`
- `enable_decode_externality_meter`
- `enable_tbt_aware_admission`
- `enable_elastic_kv_reclaim`
- `tenant_slo_config_path`

## 6. Modification Strategy

### Safe First Edits

- metrics/logging only
- config flag default off
- new helper class with isolated unit tests
- test-only hooks

### Risky Edits

- changing queue ordering by default
- changing KV eviction/free semantics
- changing scheduler output format without worker update
- modifying CUDA kernels
- changing public API request schema without compatibility path

## 7. Common Pitfall

`max_num_batched_tokens`는 연구 contribution 자체가 아니라 구현 signal이다. 논문/연구 framing에서는 다음 abstraction을 사용한다.

> KV retention consumes serving concurrency budget.

vLLM 구현에서는 이 abstraction이 KV blocks, sequence slots, per-iteration token budget 등으로 나타난다.
