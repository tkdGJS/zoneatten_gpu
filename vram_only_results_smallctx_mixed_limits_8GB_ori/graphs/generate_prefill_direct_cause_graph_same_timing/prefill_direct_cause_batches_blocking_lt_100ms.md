# Prefill Direct Cause Graph

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32`, `blocking_time_ms < 100`
- Batch samples: `21`

## Why These Axes

- One point is one filtered `(exp, turn)` batch, not one request.
- This is closer to direct cause because the x-axis is a batch-level load variable and the y-axis is a batch-level prefill outcome.
- It avoids mixing request-level outputs with batch-level cause in the same point.

## Correlations

- `batch_total_computation_tokens` vs `prefill time average (ms)`: `0.734`
- `batch_compute_ratio_8192` vs `batch_p95_prefill_ms`: `0.746`
- `batch_total_resident_kv_mib` vs `prefill time average (ms)`: `0.995`

