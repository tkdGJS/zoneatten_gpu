# Prefill Batch Rank Analysis

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32`, `blocking_time_ms < 100`
- Request samples: `609`
- Filtered batches: `21`

## Rank Interpretation

- `rank_in_batch=1` means the earliest first-token completion inside the filtered batch.
- Larger rank means the request completed its first-token later inside the same filtered batch.

## Key Correlations

- `rank_in_batch` vs `prefill_ms`: Pearson `0.098`, Spearman `0.087`
- `rank_frac` vs `prefill_ms`: Pearson `0.134`, Spearman `0.148`

## Fastest vs Slowest Mean Comparison

- Prefill: fastest `6466.2 ms`, slowest `8755.2 ms`
- Computation tokens: fastest `530.5`, slowest `414.8`
- KV history tokens: fastest `2016.6`, slowest `1530.7`
- Group1 ratio: fastest `0.68`, slowest `0.18`

## Interpretation

- If slowest requests also have higher computation or longer KV history, request shape contributes to late completion.
- If wave plots show stair-step clusters, that is evidence for internal chunk/interleaving waves inside the filtered batch.
- If fast/slow gaps remain large even when request-shape gaps are small, scheduler/interleaving effects are likely dominant.

