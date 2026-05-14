# Prefill Cost Driver Analysis

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Main dataset: all turns 1~10
- Additional focus view: turns 5, 6, 7
- Residual definition: `prefill residual = prefill - computation_tokens / 6`

## Question 1: Why is the cost of one miss token not constant?

- Correlation between per-tenant resident KV and prefill cost per token is `-0.362` in turns 5/6/7.
- This means the same number of computation tokens can still cost different time depending on how long the already-resident history is.
- The practical interpretation is that new miss tokens are not appended into identical execution context. They are appended on top of different history length and different active-cache state.

## Full-Range View

- Across all turns, correlation between computation tokens and prefill is `0.453`.
- Across all turns, correlation between computation tokens and prefill residual is `0.435`.
- Across all turns, correlation between batch total resident KV and prefill residual is `0.816`.
- Across all turns, correlation between batch total resident KV and prefill cost per token is `-0.130`.
- The new all-range scatter plots are meant to show the global shape directly instead of splitting the x-axis into computation bins.

## Question 2: Does cost per token change near capacity?

- Correlation between batch total resident KV and prefill cost per token is `0.098`.
- Correlation between batch total computation tokens and prefill cost per token is `0.004`.
- Correlation between batch total resident KV and residual is `0.447`.
- Correlation between batch total computation tokens and residual is `0.333`.
- This supports a batch-level explanation: once the synchronized batch itself becomes heavier, token cost and unexplained residual both move.

## Estimated Chunk/Batch Pressure

- Batch total computation tokens are compared against `max_num_batched_tokens=8192` to estimate how many prefill rounds would be needed if all computation tokens had to be served through the chunked-prefill budget.
- This is an estimate, not a direct vLLM internal trace. The repository does not contain per-iteration scheduler dumps.

## Batch Pressure Bins

- Low: `0 <= batch_total_resident_kv_mib <= 512.0`
- High: `512.0 < batch_total_resident_kv_mib <= 1024.0`
- Extra: `batch_total_resident_kv_mib > 1024.0`

## Computation Bins

- Low compute: `computation_tokens <= 1114.7`
- Mid compute: `1114.7 < computation_tokens <= 1180.0`
- High compute: `computation_tokens > 1180.0`

## Interpretation

- The data does not support a single fixed ms-per-token model. Token cost depends on both local context length and the global synchronized batch state.
- In other words, `computation tokens` are the base load, but `existing KV length` and `batch pressure near capacity` act like extra modifiers on that load.

