# Prefill Internal Analysis With Blocking < 100 ms

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Filter: `blocking_time_ms < 100.0`
- Samples after filtering: `609`

## Why This Filter

- The goal is to remove most queueing effect and focus on internal prefill execution differences.
- In this slice, prefill variation should be interpreted mainly as execution/context difference inside the engine, not API waiting.

## Axis Ranking

- `Delay from first request in batch (ms)`: Pearson `0.999`, Spearman `0.983`
- `Batch first-token spread (ms)`: Pearson `0.807`, Spearman `0.795`
- `Batch total compute / 8192`: Pearson `0.603`, Spearman `0.675`
- `Batch total computation tokens`: Pearson `0.603`, Spearman `0.675`
- `Computation tokens`: Pearson `0.408`, Spearman `0.472`

## Interpretation

- If `time_from_batch_first_token_ms` or `batch_first_token_spread_ms` still tracks prefill after queueing is filtered out, that is evidence for internal stagger/chunk/interleaving effects.
- If `batch_total_computation_tokens` remains the strongest axis, then batch-wide prefill load is still the most direct driver even without explicit blocking.

