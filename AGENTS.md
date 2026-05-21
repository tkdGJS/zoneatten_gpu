# AGENTS.md

이 파일은 Codex/agent가 이 저장소에서 작업할 때 항상 먼저 읽어야 하는 최상위 규칙이다.

## Project Goal

vLLM fork에서 multi-turn multi-tenant serving의 TTFT/TBT isolation을 실험한다.

핵심 목표는 다음이다.

- Tenant별 TTFT/TBT SLO 또는 token service curve를 정의한다.
- KV retention을 단순 prefix hit 최적화가 아니라 concurrency/TBT cost를 가진 scheduling resource로 다룬다.
- Decode externality, TBT debt, TBT-aware admission, elastic KV reclaim을 단계적으로 구현한다.
- 모든 코드 변경은 실험 가능하고 되돌릴 수 있어야 한다.

## Required Reading Order

작업 시작 전 다음 문서를 순서대로 읽어라.

1. `docs/00_PROJECT_OVERVIEW.md`
2. `docs/01_RESEARCH_MODEL.md`
3. `docs/02_ARCHITECTURE.md`
4. `docs/03_VLLM_CODE_MAP.md`
5. `docs/04_API_FLOW.md`
6. `docs/05_IMPLEMENTATION_PLAN.md`
7. `docs/11_DEV_RULES.md`

수정 후에는 반드시 다음 문서를 갱신하라.

- `docs/06_PATCH_NOTES.md`
- 필요 시 `docs/07_TROUBLESHOOTING.md`
- 새 결정을 내렸다면 `docs/15_DECISIONS.md`

## Non-Negotiable Rules

- vLLM upstream 파일 구조는 버전마다 달라질 수 있다. 경로를 확신하지 말고 `rg`, `find`, `git grep`로 현재 checkout에서 확인한다.
- line number 기반으로 수정하지 말고 class/function/symbol 기반으로 찾는다.
- 정책 변경 전에는 먼저 로깅/메트릭을 추가한다.
- 성능 최적화보다 correctness와 재현성을 우선한다.
- GPU가 없는 환경에서 통과하기 어려운 테스트는 skip 조건 또는 mock 기반 단위 테스트를 추가한다.
- public API/CLI flag를 바꾸면 docs와 test를 함께 바꾼다.
- 실험 코드와 production path를 섞지 말고 flag로 gated behavior를 만든다.
- default behavior는 vLLM upstream과 최대한 동일하게 유지한다.
- 대규모 변경을 한 번에 하지 말고, 작은 patch 단위로 나눈다.

## Patch Discipline

각 patch는 다음 질문에 답해야 한다.

1. 어떤 파일을 수정했는가?
2. 왜 수정했는가?
3. 어떤 새 상태/메트릭/flag가 생겼는가?
4. 기존 vLLM default behavior가 바뀌었는가?
5. 어떤 테스트를 실행했는가?
6. 실패/한계는 무엇인가?

패치 기록은 `docs/06_PATCH_NOTES.md`에 추가한다.

## Naming Conventions

새로운 실험 기능에는 접두어를 명확히 둔다.

- `tenant_`: tenant-level state
- `tbt_`: time-between-token metric or policy
- `ttft_`: time-to-first-token metric or policy
- `externality_`: batch decode externality metric
- `kv_retention_`: KV keep/reclaim policy
- `isolation_`: SLO/guarantee related policy

## Recommended Modification Areas

현재 가정상 주요 후보는 다음이다. 실제 경로는 반드시 checkout에서 확인한다.

- `vllm/config/scheduler.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/sched/request_queue.py`
- `vllm/v1/core/sched/output.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/request.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/metrics/*`
- `vllm/entrypoints/openai/*`
- `tests/v1/core/sched/*`
- `tests/*scheduler*`, `tests/*metrics*`

## Codex Task Boundary

Codex는 한 번에 하나의 작업만 수행한다.

좋은 작업 예시:

- "tenant_id를 request metadata에서 scheduler까지 전달하고 로그에 포함하라."
- "SchedulerOutput에 per-iteration scheduled token summary를 추가하고 단위 테스트를 작성하라."
- "TBT debt 계산을 별도 class로 만들고 default-disabled flag를 추가하라."

나쁜 작업 예시:

- "전체 isolation 시스템을 구현하라."
- "성능을 최적화하라."
- "논문 아이디어를 코드로 다 바꿔라."

## Safety Checks Before Editing

수정 전 다음 명령으로 현재 구조를 확인한다.

```bash
git status --short
git branch --show-current
python -V
rg "class Scheduler|def schedule|KVCacheManager|SchedulerConfig" vllm tests
```

수정 후 최소 검증:

```bash
python -m pytest -q <relevant-test-file>
pre-commit run --files <changed-files>
python -m compileall vllm
```

## Documentation Update Rule

코드가 바뀌었는데 문서가 바뀌지 않았다면, patch는 incomplete로 간주한다.
