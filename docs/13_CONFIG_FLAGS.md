# 13. Config and CLI Flags

> 실제 flag 추가 위치는 현재 vLLM의 EngineArgs/SchedulerConfig 구조 확인 후 결정한다.

## 1. Principles

- 모든 실험 기능은 default off.
- 이름에 `experimental` 또는 `tenant_isolation` 의미를 명확히 둔다.
- config parsing, docs, tests를 함께 추가한다.
- public API 안정성이 걱정되면 env var 또는 internal config부터 시작한다.

## 2. Candidate Flags

| Flag | Type | Default | Purpose |
|---|---:|---:|---|
| `--enable-tenant-isolation` | bool | false | umbrella flag |
| `--enable-tenant-isolation-metrics` | bool | false | request/tenant/batch metrics |
| `--tenant-isolation-log-path` | str | none | JSONL trace path |
| `--tenant-slo-config-path` | str | none | tenant SLO yaml/json |
| `--enable-tbt-decomposition` | bool | false | observed TBT breakdown |
| `--enable-tbt-debt-priority` | bool | false | debt-based scheduling priority |
| `--enable-decode-externality-meter` | bool | false | externality estimation |
| `--enable-tbt-aware-admission` | bool | false | admission control |
| `--enable-elastic-kv-reclaim` | bool | false | KV reclaim policy |
| `--tenant-default-tbt-slo-ms` | float | none | fallback TBT SLO |
| `--tenant-default-token-rate` | float | none | fallback service curve rate |
| `--externality-budget-window-sec` | float | 10.0 | externality accounting window |
| `--tbt-debt-window-sec` | float | 10.0 | debt accounting window |
| `--tenant-isolation-policy` | str | `off` | `off`, `debt`, `externality`, `kv`, `full` |

## 3. Experiment Environment Variables

초기 실험 runner는 public CLI를 늘리지 않고 env var로 patched vLLM에 실험 값을 전달한다.

| Env var | Type | Default | Purpose |
|---|---:|---:|---|
| `VLLM_TENANT_KV_MIN_BLOCKS` | int | `0` | all tenants에 적용할 per-tenant minimum KV block guarantee |

`VLLM_TENANT_KV_MIN_BLOCKS=0`은 baseline이며 tenant KV minimum guarantee를 적용하지 않는다. `8`, `16`, `32`, `64`, `128`, `256`은 Experiment A sweep 값이다.

## 4. SLO Config Example

```yaml
tenants:
  premium_a:
    tbt_slo_ms: 40
    ttft_slo_ms: 800
    min_token_rate: 25
    burst_tolerance_tokens: 32
    externality_budget: 1000
    kv_min_budget_blocks: 128
  best_effort:
    tbt_slo_ms: null
    ttft_slo_ms: null
    min_token_rate: null
    burst_tolerance_tokens: null
    externality_budget: null
    kv_min_budget_blocks: 0
defaults:
  tenant_id: default
  policy: best_effort
```

## 5. Validation Rules

- `VLLM_TENANT_KV_MIN_BLOCKS < 0` invalid.
- `enable_tbt_debt_priority` requires tenant SLO or default token rate.
- `enable_tbt_aware_admission` requires TBT metric/debt manager.
- `enable_elastic_kv_reclaim` requires KV manager hook.
- `tenant_default_tbt_slo_ms <= 0` invalid.
- budget windows must be positive.
- if umbrella flag is false, all policy flags must act as false.

## 6. Patch Order

1. Add config fields only.
2. Add parsing tests.
3. Add no-op wiring.
4. Add behavior behind each flag.
5. Add docs and patch notes.
