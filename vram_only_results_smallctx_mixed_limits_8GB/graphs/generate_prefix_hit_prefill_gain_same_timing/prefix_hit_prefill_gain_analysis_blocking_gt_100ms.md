# Prefix Hit vs Prefill Gain

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32`, `blocking time > 100 ms`
- One point = one filtered `(exp, turn, group)` batch

## Why This Analysis

- `Prefill vs total compute / 8192` explains execution pressure, but it does not directly explain why prefix hit itself matters.
- These graphs add the missing counterfactual view: how much extra prefill would have been needed if the same input had not benefited from prefix reuse.

## Fitted Baseline

- A batch-level linear fit was estimated from `batch_total_computation_tokens -> batch_mean_prefill_ms`: `prefill ≈ 3.992563 * compute + 10085.140`
- This fit is then used to estimate a no-hit counterfactual by replacing `compute` with `input`.

## Estimated Gain Definition

- `fit_prefill_no_hit_ms = slope * batch_total_input_tokens + intercept`
- `estimated_prefix_hit_gain_ms = fit_prefill_no_hit_ms - actual_batch_mean_prefill_ms`
- This is an estimate, not a directly observed vLLM counterfactual run.

## Key Readout

- `batch_prefix_hit_rate` vs `prefill time average / input token`: r = -0.692
- `batch_prefix_hit_rate` vs `estimated_prefix_hit_gain_ms`: r = 0.649
- `batch_total_prefix_hit_tokens` vs `estimated_prefix_hit_gain_ms`: r = 0.988

## Interpretation

- If higher prefix-hit rate lowers prefill cost per input token, then prefix reuse is not just reducing token count on paper; it is translating into lower execution cost.
- If saved tokens track estimated gain, then prefix hit can be framed as a concrete prefill-latency optimization rather than only a cache statistic.

