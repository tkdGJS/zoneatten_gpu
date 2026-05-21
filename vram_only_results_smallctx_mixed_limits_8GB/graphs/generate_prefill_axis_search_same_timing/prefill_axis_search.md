# Prefill Axis Search

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Samples: 960 successful requests
- Goal: find an x-axis that organizes `prefill time` more consistently than plain computation tokens

## Ranking By Spearman Correlation

- `Computation tokens x batch pressure ratio`: Pearson `0.602`, Spearman `0.756`
- `Batch total resident KV (MiB)`: Pearson `0.567`, Spearman `0.748`
- `Batch resident KV / KV capacity`: Pearson `0.567`, Spearman `0.748`
- `KV history tokens + computation tokens`: Pearson `0.624`, Spearman `0.722`
- `Resident KV per tenant (MiB)`: Pearson `0.564`, Spearman `0.713`
- `Computation tokens x estimated prefill rounds`: Pearson `0.582`, Spearman `0.680`

## Interpretation

- If a better axis exists, it should produce both stronger rank correlation and visibly tighter alignment in scatter form.
- Plain `computation tokens` is included as a baseline so it can be compared directly against batch-level and joint axes.
- The most promising candidates are the ones that combine `my compute load` with `global batch pressure`, because prefill is executed inside that shared context.

## Group Split

- Group1, `Batch total computation tokens`: Pearson `0.564`, Spearman `0.746`
- Group1, `Batch total resident KV (MiB)`: Pearson `0.588`, Spearman `0.830`
- Group2, `Batch total computation tokens`: Pearson `0.501`, Spearman `0.584`
- Group2, `Batch total resident KV (MiB)`: Pearson `0.667`, Spearman `0.679`

