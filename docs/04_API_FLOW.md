# 04. API and Scheduler Flow

> 이 문서는 개념적 흐름이다. 실제 symbol 이름은 현재 vLLM checkout에서 확인한다.

## 1. Online Serving Request Flow

```text
Client
  |
  | HTTP OpenAI-compatible request
  v
OpenAI API entrypoint
  |
  | parse request body / headers / metadata
  v
AsyncLLM / Engine client
  |
  | create internal Request
  v
EngineCore
  |
  | add request to Scheduler
  v
Scheduler waiting queue
  |
  | schedule iteration
  v
SchedulerOutput
  |
  | worker/model runner execution
  v
Model output tokens
  |
  | output processor / detokenizer
  v
Streaming or final response
```

## 2. Tenant ID Flow

목표 흐름:

```text
HTTP request metadata/header
  -> internal request metadata
  -> Request.tenant_id
  -> Scheduler state
  -> SchedulerOutput / metrics
  -> token timestamp logs
```

Fallback:

```text
tenant_id = "default"
```

가능한 입력 방식:

- request body의 `metadata.tenant_id`
- OpenAI-compatible extra field
- HTTP header, 예: `X-Tenant-ID`
- benchmark script에서 request마다 부여

주의:

- vLLM의 OpenAI compatibility를 깨지 않도록 unknown field handling을 확인한다.
- public API를 바꾸기보다 experimental metadata path를 우선 사용한다.

## 3. Scheduler Iteration Flow

개념 흐름:

```text
while has_requests:
    collect waiting/running requests
    compute per-iteration token budget
    decide scheduled tokens per request
    allocate KV blocks
    possibly preempt/requeue
    emit SchedulerOutput
    worker executes
    update request states from outputs
    emit metrics
```

여기서 intervention 후보는 다음이다.

| Hook | Purpose |
|---|---|
| before scheduling | queue ordering, TBT debt priority |
| during candidate admission | TBT-aware admission, externality prediction |
| KV allocation/reclaim | elastic KV reclaim, protected KV policy |
| after schedule output | scheduled token metrics |
| after model execution | observed TBT, batch step latency |
| after token output | served token count, TBT debt update |

## 4. TTFT Measurement Flow

```text
request_arrival_time
  -> request scheduled
  -> prefill executed
  -> first output token emitted
  -> TTFT = first_token_time - arrival_time
```

TTFT decomposition:

```text
TTFT = Blocking + Prefill + OtherOverheads
```

필요 timestamp:

- request arrival
- admitted/scheduled time
- prefill start/end
- first token emitted

## 5. TBT Measurement Flow

```text
output token k emitted at t_k
output token k+1 emitted at t_{k+1}
TBT_k = t_{k+1} - t_k
```

Streaming에서 실제 client-visible TBT와 internal token generation time은 다를 수 있다.

따라서 두 종류를 분리한다.

| Metric | Meaning |
|---|---|
| internal TBT | model runner/scheduler 관점 token emission gap |
| client-visible TBT | streaming response가 client에 도착하는 gap |

초기 연구에서는 internal TBT를 우선 측정하고, 이후 streaming overhead를 별도 측정한다.

## 6. TBT Decomposition Flow

```text
observed token gap
  = model execution duration
  + scheduler gap
  + prefill interference
  + KV allocation/reclaim stall
  + preemption/recompute delay
  + output processing/streaming overhead
```

로깅 위치 후보:

- scheduler loop start/end
- model runner start/end
- KV allocation start/end
- preemption event
- output processor token timestamp
- API streaming send timestamp

## 7. KV Retention Flow

```text
request has reusable KV/prefix
  -> KV manager checks cached blocks
  -> scheduler decides tokens to compute
  -> KV blocks are allocated or reused
  -> request runs
  -> completion / preemption / eviction
```

실험 intervention:

- protected KV cannot be evicted by other tenants
- elastic KV can be reclaimed based on keep score
- KV keep decision considers reuse benefit, concurrency cost, decode externality cost

## 8. Policy Flow

```text
candidate request r
  |
  v
estimate prefill saving
estimate KV footprint
estimate decode externality
check tenant debt / victim tenant debt
check queue pressure
  |
  +--> admit and schedule
  |
  +--> delay / throttle / reclaim elastic KV / scale-out signal
```
