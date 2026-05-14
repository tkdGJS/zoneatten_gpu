# Turn 5/6/7 Root Cause Analysis

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Samples: 3 experiments x 32 tenants x 10 turns = 960 successful requests
- Group1: `history_limit_tokens=8192`
- Group2: `history_limit_tokens=2048`

## What Happened At Turn 5, Turn 6, And Turn 7

- `turn 5` itself is not the real queueing onset. Combined mean blocking at turn 5 is `150.0 ms`, still effectively zero.
- The first structural queueing starts at `turn 6`: mean blocking rises to `427.4 ms` while mean resident KV per experiment reaches `4.516 GiB`.
- By `turn 7`, mean blocking reaches `396.5 ms`, mean prefill reaches `24391.3 ms`, and mean TTFT reaches `24787.8 ms`.
- Prefix reuse collapses across the same transition: global prefix-hit rate drops from `73.8%` at turn 5 to `76.7%` at turn 6 and `81.7%` at turn 7.
- Tail prefill also starts to break at `turn 6`: p99 prefill is `20465.3 ms` at turn 5, `29477.1 ms` at turn 6, and `34923.5 ms` at turn 7.

## vLLM Capacity Evidence

- From the local model config, KV cache cost is `2 x layers x KV heads x head_dim x bytes = 32,768 bytes/token`, about `32.0 KiB/token`.
- With `block_size=16`, one GPU block is `512.0 KiB` and `2048` blocks correspond to about `1.000 GiB` of active KV capacity.

## Capacity Crossing

- Mean total resident KV before prefill is `3.475 GiB` at turn 5, `4.516 GiB` at turn 6, and `5.575 GiB` at turn 7.
- This means the system is still below the effective `~1.0 GiB` block budget at turn 5, sits right on the boundary at turn 6, and exceeds it clearly at turn 7.
- Mean total prompt KV after prefill is even larger: `3.581 GiB` at turn 5 and `5.696 GiB` at turn 7.

## Group-Level Difference At Turn 7

- Group2: mean input `5799.4`, mean prefix-hit `4687.0`, mean blocking `244.1 ms`, mean prefill `21508.3 ms`, p99 prefill `34923.5 ms`, mean resident KV per experiment `2.771 GiB`.
- Group1: mean input `5865.6`, mean prefix-hit `4845.0`, mean blocking `548.9 ms`, mean prefill `27274.3 ms`, p99 prefill `33946.1 ms`, mean resident KV per experiment `2.804 GiB`.
- Group2 carries less resident history per tenant, but it also loses more reusable prefix. That is why its prompt footprint can be smaller while its prefill pressure still remains high.

## TTFT Relationship

- Correlation, estimated resident KV per tenant vs TTFT: `0.525`
- Correlation, estimated resident KV per tenant vs prefill: `0.778`
- Correlation, input tokens vs TTFT: `0.524`
- Correlation, output tokens vs TTFT: `0.178`
- Output tokens have weak correlation because TTFT is dominated by queueing plus prefill, not by decode length.

## Interpretation

- The turn-7 prefill jump is not best explained by VRAM shortage alone. It is better explained as `effective KV-capacity saturation + lower prefix reuse + 32-way synchronized arrival`, which forces more real prompt computation into the same prefill window.
- Once the resident KV sum approaches the 2048-block budget, the scheduler cannot admit all 32 requests into active execution immediately. That shows up first as blocking at turn 6.
- After that, prefix-hit quality drops sharply, so each admitted request also carries more computation tokens. That pushes prefill time up at turn 7.
- The combined effect is `blocking increase first, then prefill increase on top`, which matches the measured TTFT breakdown.

## Measurement Limits

- This repository does not contain per-request `nvidia-smi` samples or allocator traces. Per-tenant VRAM numbers in the CSV are therefore estimates derived from KV token footprint, not direct GPU telemetry.
- The local model snapshot size on disk is about `2.310 GiB`, but runtime VRAM for weights/activations can differ from on-disk size.

