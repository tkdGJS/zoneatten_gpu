# 14. Test Plan

## 1. Test Categories

| Category | Purpose |
|---|---|
| Unit tests | helper class correctness |
| Scheduler tests | deterministic ordering/admission behavior |
| KV tests | protected/elastic invariant |
| Metrics tests | default-off and enabled output |
| API tests | tenant_id propagation |
| Integration tests | small model serving |
| Performance tests | overhead and SLO evaluation |

## 2. Unit Tests

### TBTDebtManager

Cases:

- new tenant has zero debt initially
- served tokens reduce debt
- time passing increases expected service
- burst tolerance caps urgency
- unknown tenant fallback behavior
- disabled manager returns neutral priority

### DecodeExternalityMeter

Cases:

- zero context produces low score
- long context produces higher score
- max context increase reflected
- budget window accumulates and decays
- disabled meter returns zero score

### KV Keep Score

Cases:

- high reuse benefit keeps KV
- high concurrency cost lowers score
- high externality cost lowers score
- fairness credit increases score
- protected KV is not selected as elastic victim

## 3. Scheduler Tests

Cases:

- default flag off preserves existing ordering
- debt priority changes ordering only when enabled
- aging prevents starvation
- admission controller delays violating request
- overload mode is explicit
- preempted request retains tenant metadata

## 4. API Tests

Cases:

- no tenant_id -> `default`
- metadata tenant_id parsed
- invalid tenant_id handled safely
- tenant_id included in metrics output

## 5. Metrics Tests

Cases:

- metrics disabled: no behavior change, no heavy log
- metrics enabled: JSONL contains required fields
- TBT calculated from token timestamps
- scheduler iteration id is monotonic
- no sensitive prompt content is logged unless explicitly enabled

## 6. Integration Tests

Small smoke:

```bash
vllm serve <small-model> --enable-tenant-isolation-metrics --tenant-isolation-log-path /tmp/iso.jsonl
```

Send two tenants:

- short prompt tenant
- long prompt tenant

Check:

- responses complete
- log file exists
- tenant ids appear
- token timestamps appear
- scheduler iteration records appear

## 7. Performance Checks

Measure overhead:

- default vLLM
- metrics disabled
- metrics enabled
- full policy enabled

Report:

- output TPS
- p95 TTFT
- p95 TBT
- CPU overhead
- memory overhead

## 8. Regression Rule

Every behavior-changing patch requires a test showing:

1. default behavior unchanged
2. enabled behavior intentionally changed
