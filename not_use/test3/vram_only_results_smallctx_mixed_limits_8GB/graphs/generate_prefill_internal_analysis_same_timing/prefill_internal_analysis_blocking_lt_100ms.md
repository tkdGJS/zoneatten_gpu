# Prefill Internal Analysis With Blocking < 100 ms

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Filter: `blocking_time_ms < 100.0`
- Samples after filtering: `819`

## Why This Filter

- The goal is to remove most queueing effect and focus on internal prefill execution differences.
- In this slice, prefill variation should be interpreted mainly as execution/context difference inside the engine, not API waiting.

## Axis Ranking

- `Delay from first request in batch (ms)`: Pearson `0.999`, Spearman `0.994`
- `Batch first-token spread (ms)`: Pearson `0.848`, Spearman `0.859`
- `Computation tokens`: Pearson `0.360`, Spearman `0.416`
- `Batch total compute / 8192`: Pearson `0.429`, Spearman `0.293`
- `Batch total computation tokens`: Pearson `0.429`, Spearman `0.293`

## Interpretation

- If `time_from_batch_first_token_ms` or `batch_first_token_spread_ms` still tracks prefill after queueing is filtered out, that is evidence for internal stagger/chunk/interleaving effects.
- If `batch_total_computation_tokens` remains the strongest axis, then batch-wide prefill load is still the most direct driver even without explicit blocking.

