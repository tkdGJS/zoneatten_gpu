# 06. Patch Notes

이 파일은 vLLM fork 수정 사항을 버전 관리하기 위한 개발 로그다.

## Rules

- 모든 코드 변경 후 entry를 추가한다.
- “무엇을 바꿨는가”보다 “왜 바꿨고 어떤 behavior가 달라졌는가”를 중시한다.
- 파일 경로, class/function, test command, known limitation을 반드시 기록한다.
- 실험용 flag가 default off인지 명시한다.

## Version Format

```text
v0.1.0-observability
v0.2.0-tbt-debt-manager
v0.3.0-externality-meter
v0.4.0-admission-control
v0.5.0-kv-reclaimer
```

## Patch Entry Template

```markdown
## [version] YYYY-MM-DD - short title

### Summary

### Motivation

### Modified Files

| File | Symbol/Class/Function | Change |
|---|---|---|

### Behavior Change

- Default behavior changed: yes/no
- Experimental flag required: yes/no
- Backward compatibility risk:

### Metrics / Logs Added

### Tests

```bash
# commands
```

Result:

### Performance Impact

### Known Limitations

### Follow-up Tasks

### Related Decision / ADR
```

---

## [v0.0.0] 2026-05-21 - documentation bootstrap

### Summary

Created initial project documentation pack for vLLM TTFT/TBT isolation prototype.

### Motivation

Codex and human developer need shared project memory: patch notes, troubleshooting, architecture, code map, API flow, development rules, experiment plan.

### Modified Files

| File | Symbol/Class/Function | Change |
|---|---|---|
| `README.md` | N/A | documentation index |
| `AGENTS.md` | N/A | Codex project rules |
| `docs/*` | N/A | project wiki |
| `codex/*` | N/A | Codex prompts/tasks |
| `templates/*` | N/A | reusable templates |

### Behavior Change

- Default behavior changed: no
- Experimental flag required: no
- Backward compatibility risk: none

### Tests

No source code changed.

### Known Limitations

Actual vLLM file paths must be verified against the target checkout before implementation.

### Follow-up Tasks

- Run repo mapping prompt.
- Update `docs/03_VLLM_CODE_MAP.md` with verified paths.
- Start Phase 1 observability patch.

---

## [v0.0.1] 2026-05-21 - verified vendor vLLM code map

### Summary

Updated the vLLM code map against the current `vendor/vllm` checkout.

### Motivation

The initial code map used top-level `vllm` and `tests` paths. The current checkout stores vLLM source under `vendor/vllm`, reports vLLM `0.14.0`, and does not include a top-level test tree.

### Modified Files

| File | Symbol/Class/Function | Change |
|---|---|---|
| `docs/03_VLLM_CODE_MAP.md` | N/A | Replaced generic paths/search commands with verified `vendor/vllm` paths and noted absent tests |
| `docs/06_PATCH_NOTES.md` | N/A | Added this patch entry |

### Behavior Change

- Default behavior changed: no
- Experimental flag required: no
- Backward compatibility risk: none

### Metrics / Logs Added

None.

### Tests

```bash
git status --short
git branch --show-current
python -V
rg "class Scheduler|def schedule|KVCacheManager|SchedulerConfig" vendor/vllm
```

Result: repository mapping completed. No source code changed.

### Performance Impact

None.

### Known Limitations

The current `vendor/vllm` snapshot does not include tests. Future source patches must either add local tests in an agreed path or restore/use an upstream-compatible test tree.

### Follow-up Tasks

- Decide where local tests for `vendor/vllm` patches should live.
- Start Phase 1 observability with tenant metadata propagation behind default-off behavior.

### Related Decision / ADR

None.

---

## [v0.0.2] 2026-05-21 - KV block guarantee sweep design

### Summary

Documented Experiment A for tenant KV block minimum guarantee sweep.

### Motivation

The first experiment needs a precise design before code changes: sweep each tenant's minimum KV block guarantee and test whether that tenant's request-level `prefill_time_s` decreases in patched vLLM request metrics JSONL.

### Modified Files

| File | Symbol/Class/Function | Change |
|---|---|---|
| `docs/09_EXPERIMENT_DESIGN.md` | Experiment A | Added confirmed scope, workload source explanation, 8-tenant setup, `0/8/16/32` KV block sweep, run matrix, primary metrics, plots, and acceptance evidence |
| `docs/06_PATCH_NOTES.md` | N/A | Added this patch entry |

### Behavior Change

- Default behavior changed: no
- Experimental flag required: no
- Backward compatibility risk: none

### Metrics / Logs Added

None. The design requires future patched vLLM request metrics JSONL to include `prefill_time_s`.

### Tests

```bash
sed -n '1,260p' docs/09_EXPERIMENT_DESIGN.md
```

Result: documentation-only change reviewed.

### Performance Impact

None.

### Known Limitations

The document defines the experiment only. It does not implement tenant KV block guarantees, request metrics JSONL, or the sweep runner.

### Follow-up Tasks

- Implement request metrics JSONL with `tenant_id` and `prefill_time_s`.
- Implement default-off tenant KV block guarantee config.
- Add a sweep runner that emits stable workload configs and summary tables.

### Related Decision / ADR

None.

---

## [v0.0.3] 2026-05-21 - pin KV sweep workload runner

### Summary

Updated Experiment A to use the existing VRAM-only isolation tenant sweep runner as the canonical workload source.

### Motivation

The experiment will rerun `run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh` unchanged on top of patched vLLM, so result changes come from vLLM code changes rather than a new workload generator.

### Modified Files

| File | Symbol/Class/Function | Change |
|---|---|---|
| `docs/09_EXPERIMENT_DESIGN.md` | Experiment A | Replaced synthetic workload recommendation with canonical runner reuse and documented runner variables/env |
| `docs/17_OPEN_QUESTIONS.md` | Implementation Questions | Added open question for how per-tenant `kv_min_budget_blocks` is passed when the runner remains unchanged |
| `docs/06_PATCH_NOTES.md` | N/A | Added this patch entry |

### Behavior Change

- Default behavior changed: no
- Experimental flag required: no
- Backward compatibility risk: none

### Metrics / Logs Added

None.

### Tests

```bash
rg -n "TENANT_VALUES|BLOCK_VALUE|VLLM_REQUEST_METRICS_JSONL" run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh
sed -n '1,220p' docs/09_EXPERIMENT_DESIGN.md
```

Result: documentation-only change reviewed.

### Performance Impact

None.

### Known Limitations

The existing runner sweeps tenant count with `TENANT_VALUES=(32 16 8)` and fixes total GPU blocks with `BLOCK_VALUE=16384`. It does not currently expose per-tenant `kv_min_budget_blocks` as a runner argument.

### Follow-up Tasks

- Decide whether patched vLLM should read per-tenant KV guarantee from env, static config, or request metadata while keeping the runner unchanged.

### Related Decision / ADR

None.

---

## [v0.0.4] 2026-05-21 - add tenant KV minimum sweep runner axis

### Summary

Added `VLLM_TENANT_KV_MIN_BLOCKS` sweep support to the VRAM-only isolation runner and preserved the sweep value in measurement outputs.

### Motivation

Experiment A needs to rerun the same workload while sweeping all-tenant per-tenant KV minimum guarantee values `0`, `8`, `16`, and `32` blocks.

### Modified Files

| File | Symbol/Class/Function | Change |
|---|---|---|
| `run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh` | runner loop | Added `TENANT_KV_MIN_BLOCK_VALUES`, nested sweep loop, `VLLM_TENANT_KV_MIN_BLOCKS` export, and kvmin-specific log/metrics filenames |
| `start_vllm.sh` | env overrides | Logs `VLLM_TENANT_KV_MIN_BLOCKS` as `tenant_kv_min_blocks` for run traceability |
| `measure_vram_only_isolation.py` | `summarize`, `run_tenant`, `main`, `save_request_logs` | Added `--tenant-kv-min-blocks`, raw/summary CSV field, metadata field, and IO log directory partition |
| `docs/09_EXPERIMENT_DESIGN.md` | Experiment A | Updated runner matrix and env var semantics |
| `docs/13_CONFIG_FLAGS.md` | Experiment environment variables | Documented `VLLM_TENANT_KV_MIN_BLOCKS` |
| `docs/15_DECISIONS.md` | ADR-0003 | Recorded env-var based sweep decision |
| `docs/17_OPEN_QUESTIONS.md` | Implementation Questions | Removed resolved question |
| `docs/06_PATCH_NOTES.md` | N/A | Added this patch entry |

### Behavior Change

- Default behavior changed: yes, the runner now executes 21 settings by default instead of 3 because it sweeps `TENANT_KV_MIN_BLOCK_VALUES="0 8 16 32 64 128 256"` across `TENANT_VALUES=(32 16 8)`.
- Experimental flag required: yes, patched vLLM must interpret `VLLM_TENANT_KV_MIN_BLOCKS`.
- Backward compatibility risk: low for experiment scripts; existing raw/summary CSV schemas gain `tenant_kv_min_blocks`.

### Metrics / Logs Added

- `tenant_kv_min_blocks` raw CSV column.
- `tenant_kv_min_blocks` summary CSV grouping column.
- `tenant_kv_min_blocks` request metadata in saved IO logs.
- Metrics/log filenames include `kvmin_<value>`.

### Tests

```bash
bash -n run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh
bash -n start_vllm.sh
python -m py_compile measure_vram_only_isolation.py
pre-commit run --files run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh start_vllm.sh measure_vram_only_isolation.py docs/09_EXPERIMENT_DESIGN.md docs/13_CONFIG_FLAGS.md docs/15_DECISIONS.md docs/17_OPEN_QUESTIONS.md docs/06_PATCH_NOTES.md
```

Result: syntax/compile checks passed. `pre-commit` was not run because the command is not installed in this environment.

### Performance Impact

No runtime path change inside vLLM in this patch. The experiment runner now performs more runs by default.

### Known Limitations

This patch only passes and records the sweep value. The patched vLLM implementation that enforces `VLLM_TENANT_KV_MIN_BLOCKS` is still required.

### Follow-up Tasks

- Implement patched vLLM handling for `VLLM_TENANT_KV_MIN_BLOCKS`.
- Ensure request metrics JSONL records `prefill_time_s` under each sweep setting.

### Related Decision / ADR

ADR-0003

---

## [v0.1.1] 2026-05-21 - extend KV guarantee sweep to 256 blocks

### Summary

Extended Experiment A's per-tenant KV minimum guarantee sweep through 256 blocks.

### Motivation

The sweep should test larger guarantees to expose the point where prefill savings flatten or are outweighed by queueing/preemption costs.

### Modified Files

| File | Symbol/Class/Function | Change |
|---|---|---|
| `run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh` | `TENANT_KV_MIN_BLOCK_VALUES` | Changed default sweep to `0 8 16 32 64 128 256` |
| `docs/09_EXPERIMENT_DESIGN.md` | Experiment A | Updated sweep matrix, run count, all-tenant table, and feasibility note |
| `docs/13_CONFIG_FLAGS.md` | `VLLM_TENANT_KV_MIN_BLOCKS` | Updated documented sweep values |
| `docs/15_DECISIONS.md` | ADR-0003 | Updated sweep values |
| `docs/06_PATCH_NOTES.md` | N/A | Added this patch entry |

### Behavior Change

- Default behavior changed: yes, runner default expands from 12 to 21 runs.
- Experimental flag required: yes, `VLLM_TENANT_KV_MIN_BLOCKS > 0`.
- Backward compatibility risk: low for experiment scripts; runtime increases.

### Metrics / Logs Added

None.

### Tests

```bash
bash -n run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh
```

Result: passed.

### Performance Impact

Runner runtime increases because the default matrix now has 7 KV guarantee points across 3 tenant-count settings.

### Known Limitations

At `tenant_count=32` and `kv_min_budget_blocks=256`, the theoretical protected cached KV upper bound is `8192` blocks. This can substantially reduce evictable KV capacity depending on active KV footprint and may increase queueing/preemption.

### Follow-up Tasks

- Add feasibility warnings to the runner output.
- Add `theoretical_protected_blocks` to summary CSV.

### Related Decision / ADR

ADR-0003

---

## [v0.1.0] 2026-05-21 - tenant KV minimum block protection

### Summary

Implemented tenant ID propagation through `vllm_xargs`, request metrics tagging, and default-off tenant KV minimum block protection using `VLLM_TENANT_KV_MIN_BLOCKS`.

### Motivation

The KV guarantee sweep runner needs patched vLLM to actually receive tenant IDs and protect each tenant's most recently accessed reusable KV blocks up to the configured per-tenant minimum.

### Modified Files

| File | Symbol/Class/Function | Change |
|---|---|---|
| `measure_vram_only_isolation.py` | `stream_completion_with_request_id` | Sends `vllm_xargs={"tenant_id": "tenant_<id>"}` in each completion request |
| `vendor/vllm/v1/request.py` | `Request.__init__` | Extracts `tenant_id` from `SamplingParams.extra_args`, fallback `default` |
| `vendor/vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock` | Adds experimental `tenant_id` and `last_access_seq` metadata |
| `vendor/vllm/v1/core/block_pool.py` | `BlockPool` | Reads `VLLM_TENANT_KV_MIN_BLOCKS`; tracks tenant block access; excludes each tenant's most recent reusable cached blocks from allocation eviction candidates |
| `vendor/vllm/v1/engine/output_processor.py` | `RequestState`, `_write_request_metrics_jsonl` | Carries tenant ID into output state and writes `tenant_id` / `tenant_kv_min_blocks` to request metrics JSONL |
| `docs/10_METRICS_LOGGING.md` | Request-level fields | Documents `tenant_kv_min_blocks` |
| `docs/15_DECISIONS.md` | ADR-0003 | Links this implementation patch |
| `docs/06_PATCH_NOTES.md` | N/A | Added this patch entry |

### Behavior Change

- Default behavior changed: no when `VLLM_TENANT_KV_MIN_BLOCKS=0` or unset.
- Experimental flag required: yes, `VLLM_TENANT_KV_MIN_BLOCKS > 0`.
- Backward compatibility risk: low; request body uses existing `vllm_xargs`, and KV protection is env-gated.

### Metrics / Logs Added

- Request metrics JSONL fields:
  - `tenant_id`
  - `tenant_kv_min_blocks`

### Tests

```bash
python -m py_compile measure_vram_only_isolation.py vendor/vllm/v1/request.py vendor/vllm/v1/core/kv_cache_utils.py vendor/vllm/v1/core/block_pool.py vendor/vllm/v1/engine/output_processor.py
bash -n run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh
bash -n start_vllm.sh
python -m compileall -q vendor/vllm
PYTHONPATH=vendor VLLM_TENANT_KV_MIN_BLOCKS=1 python - <<'PY'
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)
free_blocks = pool.free_block_queue.get_all_free_blocks()
for idx, block in enumerate(free_blocks[:2], start=1):
    block.block_hash = make_block_hash_with_group_id(BlockHash(bytes([idx]) * 32), 0)
    block.tenant_id = "tenant_1"
    block.last_access_seq = idx
selected = pool.get_new_blocks(1)[0]
assert selected.block_id == free_blocks[0].block_id
assert free_blocks[1].block_id in pool._get_protected_free_block_ids()
print("ok")
PY
pre-commit run --files measure_vram_only_isolation.py vendor/vllm/v1/request.py vendor/vllm/v1/core/kv_cache_utils.py vendor/vllm/v1/core/block_pool.py vendor/vllm/v1/engine/output_processor.py docs/10_METRICS_LOGGING.md docs/15_DECISIONS.md docs/06_PATCH_NOTES.md
```

Result: compile, compileall, shell syntax, and in-process BlockPool protection checks passed. `pre-commit` was not run because the command is not installed in this environment.

No pytest target was run because the current `vendor/vllm` snapshot does not include a test tree.

### Performance Impact

No impact when `VLLM_TENANT_KV_MIN_BLOCKS` is unset or `0`. When enabled, `BlockPool.get_num_free_blocks()` and protected allocation scan the free block list to compute per-tenant recent protected cached blocks.

### Known Limitations

- Protection applies to reusable cached blocks in the free queue, not active `ref_cnt > 0` blocks, which are already non-evictable in upstream vLLM.
- The first implementation uses original caching tenant ownership. Cross-tenant prefix hits do not transfer block ownership.
- All tenants receive the same minimum block guarantee from one env value.
- If protected blocks reduce evictable capacity, existing scheduler allocation failure/preemption/wait behavior is used.

### Follow-up Tasks

- Add focused unit tests once a test tree location is finalized.
- Add per-tenant protected/evictable block counters to metrics.
- Consider tenant-specific budgets via config file after the first sweep.

### Related Decision / ADR

ADR-0003
