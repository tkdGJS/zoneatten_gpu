# 07. Troubleshooting Wiki

이 문서는 프로젝트 진행 중 반복되는 문제와 해결법을 기록한다.

## Entry Format

```markdown
## Problem: short title

### Symptoms

### Context

### Likely Cause

### Diagnosis Commands

### Fix

### Prevention

### Related Patch
```

---

## Problem: Current vLLM path differs from documentation

### Symptoms

- `vllm/v1/core/sched/scheduler.py`가 없거나 class 이름이 다름
- Codex가 문서에 있는 경로를 수정하려다 실패함

### Likely Cause

vLLM upstream 구조 변경 또는 checkout version 차이.

### Diagnosis Commands

```bash
git rev-parse HEAD
python -c "import vllm; print(vllm.__version__)"
find vllm -iname '*scheduler*' -o -iname '*kv*'
rg "class Scheduler|def schedule|KVCacheManager|SchedulerConfig" vllm tests
```

### Fix

- 실제 path를 `docs/03_VLLM_CODE_MAP.md`에 반영한다.
- line number가 아니라 symbol 기반으로 수정한다.
- 불확실한 파일은 먼저 read-only mapping patch로 처리한다.

### Prevention

Codex 작업 전 `codex/START_HERE.md`의 search commands를 실행한다.

---

## Problem: Added metrics changed scheduler behavior

### Symptoms

- default test failure
- output order changed
- throughput/latency suddenly changed with metrics flag off

### Likely Cause

로깅 추가 과정에서 queue iteration, object mutation, timestamp update가 scheduling path에 영향을 줌.

### Diagnosis Commands

```bash
git diff
pytest -q tests/v1/core/sched -k "scheduler or schedule"
rg "enable_tbt_metrics|tenant_id|record" vllm
```

### Fix

- metrics data structure를 read-only snapshot으로 만든다.
- default-disabled flag가 실제로 모든 path를 보호하는지 확인한다.
- timestamp 기록이 ordering key를 바꾸지 않도록 분리한다.

### Prevention

default behavior unchanged test를 추가한다.

---

## Problem: TBT measurement is noisy

### Symptoms

- median TBT는 안정적인데 p95/p99가 크게 흔들림
- 같은 workload에서도 TBT가 재현되지 않음

### Likely Cause

Observed TBT에 scheduler gap, prefill interference, streaming overhead, host overhead가 섞여 있음.

### Diagnosis Commands

```bash
nvidia-smi dmon
rg "scheduler.*time|model.*time|token.*timestamp" logs results
```

### Fix

- internal TBT와 client-visible TBT를 분리한다.
- scheduler loop time과 model execution time을 별도 기록한다.
- prefill chunk가 decode step 사이에 들어갔는지 기록한다.

### Prevention

`docs/10_METRICS_LOGGING.md`의 TBT decomposition fields를 모두 기록한다.

---

## Problem: Strong TBT guarantee cannot be met

### Symptoms

- 모든 tenant debt가 계속 증가
- admission controller가 계속 block함
- output TPS가 capacity를 초과한 offered load를 따라가지 못함

### Likely Cause

Overload. Feasible capacity 영역 밖에서는 어떤 scheduler도 guarantee를 만족시킬 수 없음.

### Diagnosis Commands

```bash
# offered load, output TPS, queue length, debt trend 확인
rg "offered_load|output_tps|queue|debt|admission" results logs
```

### Fix

- overload mode를 명시한다.
- queueing/throttling/reject/scale-out signal 중 하나를 정책 action으로 둔다.
- experiment report에 feasible/infeasible region을 구분한다.

### Prevention

실험 설계에서 capacity sweep을 먼저 수행한다.

---

## Problem: KV retention improves prefill but worsens TTFT

### Symptoms

- prefix hit rate 증가
- recompute tokens 감소
- p99 TTFT 또는 blocking 증가

### Likely Cause

KV retention이 concurrency budget을 소모하여 admission capacity를 줄임.

### Fix

- `PrefillSaving > BlockingCost` 조건을 적용한다.
- admission reserve를 둔다.
- marginal benefit이 낮은 tenant의 guarantee부터 줄인다.

---

## Problem: Short-context tenant TBT worsens in mixed workload

### Symptoms

- Group2-only에서는 TBT가 낮음
- Group1+Group2 mixed에서는 Group2 TBT가 Group1과 비슷해짐

### Likely Cause

Batch-level decode externality. Short tenant가 long-context tenant의 shared batch-step latency를 같이 부담함.

### Fix

- externality meter를 활성화한다.
- long-context tenant의 externality budget을 제한한다.
- debt가 큰 victim tenant의 priority를 높인다.
- 필요시 temporal lane을 사용한다.
