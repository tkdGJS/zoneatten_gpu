# Codex Prompt Templates

## Prompt 1: Repo Mapping

```text
Read AGENTS.md and docs/00_PROJECT_OVERVIEW.md first.
Do not edit code yet.
Map the current vLLM checkout for scheduler, KV cache manager, request metadata, metrics, and OpenAI entrypoint flow.
Update docs/03_VLLM_CODE_MAP.md with verified paths and symbols.
Run only read-only commands such as rg/find/git grep.
```

## Prompt 2: Add Observability

```text
Read AGENTS.md, docs/03_VLLM_CODE_MAP.md, docs/10_METRICS_LOGGING.md.
Implement a default-disabled observability patch for tenant-level TTFT/TBT research.
Add only minimal logging/metrics needed to record tenant_id, scheduler iteration id, scheduled token counts, active batch size, and KV blocks used/free.
Do not change scheduling behavior.
Add or update tests if feasible.
Update docs/06_PATCH_NOTES.md.
```

## Prompt 3: Add TBT Debt Manager Skeleton

```text
Read AGENTS.md and docs/02_ARCHITECTURE.md.
Create a small isolated TBTDebtManager class with unit tests.
It should maintain expected token service, served tokens, and debt by tenant.
Do not wire it into scheduling decisions yet except behind a default-off flag or test-only path.
Update docs/06_PATCH_NOTES.md and docs/15_DECISIONS.md.
```

## Prompt 4: Wire TBT Priority Behind Flag

```text
Read AGENTS.md, docs/02_ARCHITECTURE.md, docs/05_IMPLEMENTATION_PLAN.md.
Wire TBTDebtManager into scheduler priority only when the experimental flag is enabled.
Default behavior must remain upstream-compatible.
Add tests showing default behavior unchanged and enabled behavior changes ordering as expected.
Update docs/06_PATCH_NOTES.md.
```

## Prompt 5: Troubleshoot a Failure

```text
Read docs/07_TROUBLESHOOTING.md.
Given the failing command and stack trace below, identify likely root cause.
Do not make broad refactors.
Make the smallest fix and add a troubleshooting entry.
Update docs/06_PATCH_NOTES.md.
```
