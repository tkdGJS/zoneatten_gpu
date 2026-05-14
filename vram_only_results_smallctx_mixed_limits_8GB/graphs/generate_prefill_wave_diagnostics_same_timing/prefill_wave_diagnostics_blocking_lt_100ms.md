# Prefill Wave Diagnostics

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32`, `blocking_time_ms < 100`
- Request samples: `723`
- Filtered batches: `30`
- Plots focus on the top-6 batches with the widest relative first-token spread

## Terms

- `filtered batch`: same `(exp, turn)` after applying `blocking < 100 ms`
- `completion rank`: order of `first_token_ts` inside the filtered batch
- `relative first-token delay`: `first_token_ts - min(first_token_ts in same filtered batch)`
- `delta gap`: difference in relative first-token delay between adjacent completion ranks

## What To Look For

- Wave plot: stair-step patterns indicate multiple completion waves
- Timeline strip: clustered points indicate groups of requests completing near the same time
- Delta-gap plot: large bars indicate likely wave boundaries

## Batch Summary

- Mean delay spread across filtered batches: `0.0 ms`
- Mean max delta gap across filtered batches: `0.0 ms`

