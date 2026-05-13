# Prefill vs Compute Ratio By Blocking Slice

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32`, one point = one filtered `(exp, turn)` batch
- Comparison:
  - left panel: requests with `blocking < 100 ms`
  - right panel: requests with `blocking > 100 ms`

## Correlation

- `blocking < 100 ms`: Pearson `0.609`, batches `30`
- `blocking > 100 ms`: Pearson `0.953`, batches `12`

## Interpretation

- If the `blocking > 100 ms` panel still shows a clear positive trend, then prefill remains strongly tied to execution load even after queueing is already present.
- If the trend weakens substantially, that means queueing/memory pressure dominates and compute ratio becomes a less direct driver in that slice.

