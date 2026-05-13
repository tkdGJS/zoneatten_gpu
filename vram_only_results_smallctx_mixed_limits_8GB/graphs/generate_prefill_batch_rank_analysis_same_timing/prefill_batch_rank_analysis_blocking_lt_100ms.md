# Prefill Batch Rank Analysis

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32`, `blocking_time_ms < 100`
- Request samples: `819`
- Filtered batches: `30`

## Rank Interpretation

- `rank_in_batch=1` means the earliest first-token completion inside the filtered batch.
- Larger rank means the request completed its first-token later inside the same filtered batch.

## Key Correlations

- `rank_in_batch` vs `prefill_ms`: Pearson `0.092`, Spearman `0.160`
- `rank_frac` vs `prefill_ms`: Pearson `0.300`, Spearman `0.342`

## Fastest vs Slowest Mean Comparison

- Prefill: fastest `4882.1 ms`, slowest `14395.8 ms`
- Computation tokens: fastest `696.7`, slowest `858.0`
- KV history tokens: fastest `3312.7`, slowest `4058.7`
- Group1 ratio: fastest `0.50`, slowest `0.65`

## Interpretation

- If slowest requests also have higher computation or longer KV history, request shape contributes to late completion.
- If wave plots show stair-step clusters, that is evidence for internal chunk/interleaving waves inside the filtered batch.
- If fast/slow gaps remain large even when request-shape gaps are small, scheduler/interleaving effects are likely dominant.

