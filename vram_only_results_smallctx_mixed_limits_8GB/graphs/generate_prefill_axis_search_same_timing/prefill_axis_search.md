# Prefill Axis Search

## Scope

- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`
- Slice: `tenant_count=32` only
- Samples: 960 successful requests
- Goal: find an x-axis that organizes `prefill time` more consistently than plain computation tokens

## Ranking By Spearman Correlation

- `Batch total resident KV (MiB)`: Pearson `0.818`, Spearman `0.833`
- `Batch resident KV / KV capacity`: Pearson `0.818`, Spearman `0.833`
- `Computation tokens x batch pressure ratio`: Pearson `0.490`, Spearman `0.819`
- `Batch total computation tokens`: Pearson `0.608`, Spearman `0.814`
- `Batch total compute / 8192`: Pearson `0.608`, Spearman `0.814`
- `KV history tokens + computation tokens`: Pearson `0.762`, Spearman `0.805`

## Interpretation

- If a better axis exists, it should produce both stronger rank correlation and visibly tighter alignment in scatter form.
- Plain `computation tokens` is included as a baseline so it can be compared directly against batch-level and joint axes.
- The most promising candidates are the ones that combine `my compute load` with `global batch pressure`, because prefill is executed inside that shared context.

## Group Split

- Group1, `Batch total computation tokens`: Pearson `0.579`, Spearman `0.839`
- Group1, `Batch total resident KV (MiB)`: Pearson `0.842`, Spearman `0.862`
- Group2, `Batch total computation tokens`: Pearson `0.637`, Spearman `0.785`
- Group2, `Batch total resident KV (MiB)`: Pearson `0.795`, Spearman `0.802`

