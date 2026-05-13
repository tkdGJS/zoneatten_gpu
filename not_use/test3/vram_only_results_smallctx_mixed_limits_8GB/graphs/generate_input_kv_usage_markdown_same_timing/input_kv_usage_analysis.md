# Input KV Usage By Turn Analysis

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Outputs:
  - `exp1/exp2/exp3_input_kv_usage_by_turn_32tenants.png`
  - `exp123_mean_input_kv_usage_by_turn_32tenants.png`
  - `exp1/exp2/exp3_total_input_kv_usage_by_turn_32tenants.png`
  - `exp123_mean_total_input_kv_usage_by_turn_32tenants.png`

## What These Graphs Measure

- The graphs convert `input_tokens` into a KV-style footprint in MiB using the same per-token KV size used elsewhere in the analysis.
- This is not `resident KV` already stored from history. It is the footprint of the prompt tokens presented in that turn.
- Group1 corresponds to `history_limit_tokens=8192` and Group2 corresponds to `history_limit_tokens=2048`.
- The dashed `1024 MiB` line is the reference KV capacity used in the rest of the study.

## How To Read The Graphs

- The stacked graphs show how much of the turn-level input footprint comes from Group1 versus Group2.
- The total-only graphs collapse Group1+Group2 into a single bar so the turn-level aggregate is easier to compare against the `1024 MiB` reference line.
- Because this is based on `input_tokens`, it is a prompt-size view, not an active-cache view.

## Mean Pattern Across exp1+exp2+exp3

- Turn 1: total `109.1 MiB`, Group1 `105.2 MiB`, Group2 `3.8 MiB`
- Turn 5: total `2748.6 MiB`, Group1 `1641.6 MiB`, Group2 `1106.9 MiB`
- Turn 6: total `3536.8 MiB`, Group1 `2173.7 MiB`, Group2 `1363.1 MiB`
- Turn 7: total `4368.9 MiB`, Group1 `2672.3 MiB`, Group2 `1696.6 MiB`
- Turn 8: total `5252.3 MiB`, Group1 `3218.9 MiB`, Group2 `2033.4 MiB`
- Turn 9: total `6224.3 MiB`, Group1 `3785.6 MiB`, Group2 `2438.8 MiB`
- Turn 10: total `7167.6 MiB`, Group1 `4308.0 MiB`, Group2 `2859.6 MiB`

## Main Observations

- Mean total input-KV footprint is already high by turn 5 (`2748.6 MiB`) and exceeds the `1024 MiB` reference by turn 6 (`3536.8 MiB`).
- By turn 7 the mean total input-KV footprint grows further to `4368.9 MiB`, and by turn 10 it reaches `7167.6 MiB`.
- Group1 contributes the larger share of input-KV footprint in every later turn, which is expected because the longer-history group builds larger prompts.
- Group2 still contributes a meaningful fraction, so later-turn batch pressure is not only a Group1 phenomenon; it is a mixed-batch load problem.

## Comparison To Resident KV Graphs

- `Resident KV` graphs answer: how much history is already active in cache before the new turn starts.
- `Input KV` graphs answer: how large the current prompt payload is when translated into KV-sized token footprint.
- In practice, `resident KV` is more relevant to admission/blocking pressure, while `input KV` is more relevant to prefill execution load.
- This is why a turn can show strong prefill growth even before the resident-KV sum alone obviously exceeds the capacity line.

## Interpretation

- These graphs support the interpretation that later turns increase prefill pressure mainly because prompt payload grows substantially across tenants.
- The input-KV view therefore complements the resident-KV view: one captures prompt compute load, the other captures active-cache pressure.
- Together they explain why prefill can begin degrading even before blocking becomes dominant.

