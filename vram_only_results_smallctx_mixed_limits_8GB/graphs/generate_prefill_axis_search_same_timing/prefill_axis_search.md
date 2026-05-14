# Prefill Axis Search

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Samples: 960 successful requests
- Goal: find an x-axis that organizes `prefill time` more consistently than plain computation tokens

## Ranking By Spearman Correlation

- `Batch total resident KV (MiB)`: Pearson `0.551`, Spearman `0.762`
- `Batch resident KV / KV capacity`: Pearson `0.551`, Spearman `0.762`
- `Computation tokens x batch pressure ratio`: Pearson `0.600`, Spearman `0.745`
- `KV history tokens + computation tokens`: Pearson `0.615`, Spearman `0.715`
- `Resident KV per tenant (MiB)`: Pearson `0.551`, Spearman `0.714`
- `Batch total computation tokens`: Pearson `0.508`, Spearman `0.701`

## Interpretation

- If a better axis exists, it should produce both stronger rank correlation and visibly tighter alignment in scatter form.
- Plain `computation tokens` is included as a baseline so it can be compared directly against batch-level and joint axes.
- The most promising candidates are the ones that combine `my compute load` with `global batch pressure`, because prefill is executed inside that shared context.

## Group Split

- Group1, `Batch total computation tokens`: Pearson `0.571`, Spearman `0.763`
- Group1, `Batch total resident KV (MiB)`: Pearson `0.569`, Spearman `0.841`
- Group2, `Batch total computation tokens`: Pearson `0.518`, Spearman `0.651`
- Group2, `Batch total resident KV (MiB)`: Pearson `0.682`, Spearman `0.699`

