# Prefill Axis Search

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Samples: 960 successful requests
- Goal: find an x-axis that organizes `prefill time` more consistently than plain computation tokens

## Ranking By Spearman Correlation

- `Computation tokens`: Pearson `0.000`, Spearman `0.000`
- `Resident KV per tenant (MiB)`: Pearson `0.000`, Spearman `0.000`
- `Batch total computation tokens`: Pearson `0.000`, Spearman `0.000`
- `Batch total resident KV (MiB)`: Pearson `0.000`, Spearman `0.000`
- `Batch resident KV / KV capacity`: Pearson `0.000`, Spearman `0.000`
- `Batch total compute / 8192`: Pearson `0.000`, Spearman `0.000`

## Interpretation

- If a better axis exists, it should produce both stronger rank correlation and visibly tighter alignment in scatter form.
- Plain `computation tokens` is included as a baseline so it can be compared directly against batch-level and joint axes.
- The most promising candidates are the ones that combine `my compute load` with `global batch pressure`, because prefill is executed inside that shared context.

## Group Split

- Group1, `Batch total computation tokens`: Pearson `0.000`, Spearman `0.000`
- Group1, `Batch total resident KV (MiB)`: Pearson `0.000`, Spearman `0.000`
- Group2, `Batch total computation tokens`: Pearson `0.000`, Spearman `0.000`
- Group2, `Batch total resident KV (MiB)`: Pearson `0.000`, Spearman `0.000`

