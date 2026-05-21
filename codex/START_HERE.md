# Codex START_HERE

너는 vLLM fork에서 TTFT/TBT isolation prototype을 개발하는 coding agent다.

## 1. Project in One Sentence

Multi-tenant LLM serving에서 KV cache retention, admission, scheduling, decode externality를 제어하여 tenant별 TTFT/TBT service guarantee를 실험한다.

## 2. Current Research Framing

단순히 length-aware batching을 구현하는 것이 아니다.

우리가 구현하려는 것은 다음이다.

- Tenant별 token service curve 또는 TBT debt를 추적한다.
- Long-context tenant가 co-batched short-context tenant의 TBT를 악화시키는 decode externality를 측정한다.
- KV retention은 prefix recomputation saving이 있어도 다른 tenant의 TBT SLO를 깨면 줄이거나 회수한다.
- Admission은 batch에 새 request를 넣었을 때 기존 tenant의 TBT guarantee가 깨지는지 판단한다.

## 3. Start With Observability

정책 구현 전 먼저 아래 메트릭이 필요하다.

- request_id
- tenant_id
- arrival time
- first token time
- per-output-token timestamp
- observed TBT
- scheduler iteration id
- scheduled prefill/decode token count
- active batch size
- total context/KV tokens in batch
- max context/KV tokens in batch
- KV blocks used/free
- preemption count
- per-iteration scheduler duration
- per-iteration model execution duration

## 4. Initial Code Search

작업 전 현재 vLLM checkout에서 아래를 실행하라.

```bash
rg "class Scheduler|def schedule|SchedulerOutput|KVCacheManager|RequestStatus|Request" vllm/v1 vllm tests
rg "max_num_batched_tokens|max_num_seqs|max_num_partial_prefills|enable_chunked_prefill" vllm
rg "metrics|prometheus|Stats|record_event|EngineCoreEventType" vllm
```

## 5. First Patch Recommendation

가장 안전한 첫 patch는 policy가 아니라 logging이다.

추천 첫 patch:

> Request에 tenant_id metadata를 전달할 수 있는 경로를 확인하고, scheduler iteration output에 tenant별 scheduled token count와 batch context summary를 default-disabled debug metric으로 남긴다.

## 6. Required Update After Patch

수정 후 반드시 `docs/06_PATCH_NOTES.md`에 다음을 기록하라.

- version/date
- modified files
- purpose
- behavioral change
- test command
- known limitation
