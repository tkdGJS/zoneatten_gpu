# 15. Architecture Decision Records

이 파일은 중요한 설계 결정을 기록한다.

## ADR Format

```markdown
## ADR-000X: Title

Date:

Status: proposed / accepted / rejected / superseded

### Context

### Decision

### Alternatives Considered

### Consequences

### Related Patch
```

---

## ADR-0001: Use tenant-level token service debt as isolation signal

Date: 2026-05-21

Status: proposed

### Context

단순 length-aware batching은 context length heterogeneity를 줄일 수 있지만, tenant별 TBT guarantee를 직접 표현하지 못한다.

### Decision

Tenant별 expected token service와 actual served tokens를 비교하여 TBT debt를 계산한다.

\[
Debt_i(t) = Expected_i(t) - Served_i(t)
\]

Debt가 큰 tenant는 scheduler priority credit을 받는다.

### Alternatives Considered

- length-aware grouping only
- static partition
- FCFS with aging
- pure KV guarantee

### Consequences

- tenant-level guarantee를 직접 표현할 수 있다.
- overload 상황에서는 debt가 무한 증가할 수 있으므로 admission/throttling/scale-out 정책이 필요하다.

### Related Patch

N/A

---

## ADR-0002: Treat KV retention as concurrency-consuming scheduling resource

Date: 2026-05-21

Status: proposed

### Context

KV retention은 prefix recomputation을 줄이지만 GPU KV capacity와 batch admission 여유를 소모한다.

### Decision

KV keep/reclaim 판단에 reuse benefit뿐 아니라 concurrency cost와 decode externality cost를 포함한다.

\[
KeepScore = ReuseBenefit - ConcurrencyCost - DecodeExternalityCost + FairnessCredit
\]

### Alternatives Considered

- LRU eviction
- prefix hit probability only
- static per-tenant KV partition

### Consequences

- TTFT와 TBT trade-off를 함께 볼 수 있다.
- KV manager와 scheduler 사이의 coupling이 증가하므로 flag와 tests가 필요하다.

### Related Patch

N/A

---

## ADR-0003: Pass tenant KV minimum guarantee sweep via environment variable

Date: 2026-05-21

Status: accepted

### Context

Experiment A reruns the existing `run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh` workload runner on patched vLLM. The runner already sweeps tenant count with `TENANT_VALUES=(32 16 8)` and fixes total GPU KV blocks with `BLOCK_VALUE=16384`. It needs a separate sweep axis for per-tenant minimum KV block guarantee without changing the workload generation semantics.

### Decision

The runner sweeps:

```text
TENANT_KV_MIN_BLOCK_VALUES="0 8 16 32 64 128 256"
```

For each vLLM server run, it exports:

```text
VLLM_TENANT_KV_MIN_BLOCKS=<value>
```

Patched vLLM will read `VLLM_TENANT_KV_MIN_BLOCKS` as the all-tenant per-tenant minimum KV block guarantee. `0` is the baseline.

### Alternatives Considered

- Add a public vLLM CLI flag immediately.
- Add a tenant SLO config file to the runner.
- Encode the guarantee in request metadata.
- Hardcode the sweep value in patched vLLM.

### Consequences

- The existing workload runner keeps the same workload generation/reuse behavior.
- Raw and summary CSVs must include `tenant_kv_min_blocks` so sweep results remain attributable.
- This is suitable for the first experiment but should become a formal config/CLI flag before any production-facing patch.

### Related Patch

`v0.0.4`, `v0.1.0`
