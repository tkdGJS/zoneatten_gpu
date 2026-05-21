# Prefill vs Computation Analysis

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Samples: 960 successful requests
- Main focus for visual inspection: turns 5, 6, 7

## What This Checks

- Whether `prefill time` is explained only by `computation tokens = input tokens - prefix hit`.
- Whether the remaining unexplained part grows with `resident KV usage`.

## Key Findings

- In turns 5/6/7, correlation between computation tokens and prefill is `0.354`.
- In the same turns, correlation between resident KV per tenant and prefill is `0.173`.
- After subtracting the baseline `prefill = computation_tokens / 6`, correlation between resident KV and prefill residual is `0.164`.
- Correlation between computation tokens and that residual is `0.331`.

## Interpretation

- If computation tokens alone explained prefill, the scatter would collapse toward a single narrow curve. It does not.
- Higher resident KV bins tend to sit higher for the same computation-token range, which means the same compute load costs more under higher KV pressure.
- The residual plot shows the same point more directly: even after subtracting a simple token-count baseline, higher resident KV still tends to push requests upward.
- This supports the stricter claim: `computation tokens create the base prefill load, but KV pressure / batching / scheduling state add extra spread and extra delay`.

## KV Bins

- Low KV: `resident_kv_mib <= 84.9`
- Mid KV: `84.9 < resident_kv_mib <= 206.3`
- High KV: `resident_kv_mib > 206.3`

