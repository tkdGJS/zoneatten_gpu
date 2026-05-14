# Prefill Batch Rank Analysis

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32`, `blocking_time_ms < 100`
- Request samples: `723`
- Filtered batches: `30`

## Rank Interpretation

- `rank_in_batch=1` means the earliest first-token completion inside the filtered batch.
- Larger rank means the request completed its first-token later inside the same filtered batch.

## Key Correlations

- `rank_in_batch` vs `prefill_ms`: Pearson `-0.061`, Spearman `-0.063`
- `rank_frac` vs `prefill_ms`: Pearson `0.073`, Spearman `0.072`

## Fastest vs Slowest Mean Comparison

- Prefill: fastest `13132.4 ms`, slowest `15524.5 ms`
- Computation tokens: fastest `916.1`, slowest `1070.1`
- KV history tokens: fastest `3642.5`, slowest `3845.2`
- Group1 ratio: fastest `0.41`, slowest `0.51`

## Interpretation

- If slowest requests also have higher computation or longer KV history, request shape contributes to late completion.
- If wave plots show stair-step clusters, that is evidence for internal chunk/interleaving waves inside the filtered batch.
- If fast/slow gaps remain large even when request-shape gaps are small, scheduler/interleaving effects are likely dominant.

