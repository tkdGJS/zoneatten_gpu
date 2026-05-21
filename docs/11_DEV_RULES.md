# 11. Development Rules

## 1. Preserve Upstream Default Behavior

Every experimental feature must be behind a flag.

Default:

```text
all experimental TTFT/TBT isolation features disabled
```

## 2. Change One Layer at a Time

Recommended order:

1. config only
2. data model only
3. metrics only
4. isolated helper class
5. scheduler hook behind flag
6. policy behavior
7. experiment script

## 3. Avoid Hidden Coupling

Do not:

- mutate request queue while only intending to log
- change scheduling order as side effect of metric collection
- add tenant state directly into unrelated model runner code
- assume GPU timing exists on CPU-only tests
- assume all requests have tenant_id

## 4. Use Explicit Experimental Names

Good:

```python
enable_tbt_debt_priority
tenant_isolation_config
decode_externality_score
```

Bad:

```python
new_policy
better_scheduler
priority2
```

## 5. Tests Required by Change Type

| Change | Required Test |
|---|---|
| config flag | parse/default test |
| helper class | unit test |
| scheduler ordering | deterministic scheduler test |
| metrics | default-off and enabled output test |
| KV reclaim | allocation/free invariant test |
| API metadata | request parsing test |
| benchmark script | dry-run or small mock workload |

## 6. Documentation Required by Change Type

| Change | Required Doc |
|---|---|
| any source patch | `docs/06_PATCH_NOTES.md` |
| new policy | `docs/02_ARCHITECTURE.md` and `docs/15_DECISIONS.md` |
| new metric | `docs/10_METRICS_LOGGING.md` |
| new flag | `docs/13_CONFIG_FLAGS.md` |
| new failure | `docs/07_TROUBLESHOOTING.md` |
| new experiment | `docs/09_EXPERIMENT_DESIGN.md` |

## 7. Performance Rule

Before claiming improvement, compare against:

- vLLM default
- length-aware batching baseline
- static tenant partition
- retention-only baseline
- debt-only baseline

Do not claim isolation from TPS alone.

## 8. Research Integrity Rule

If overload occurs, report overload. Do not hide guarantee failure by dropping requests unless policy explicitly includes admission reject/throttling and reports it.

## 9. Commit Message Format

```text
[area] short description

Examples:
[sched] add default-off tenant TBT metrics
[kv] add elastic KV keep score skeleton
[docs] update patch notes for observability
[tests] add TBT debt manager unit tests
```

## 10. Before PR / Patch Submission

Checklist:

- [ ] `git diff` reviewed
- [ ] default behavior unchanged or explicitly documented
- [ ] relevant tests passed
- [ ] docs updated
- [ ] patch notes updated
- [ ] known limitations written
- [ ] no hardcoded local paths
- [ ] no unguarded experimental behavior
