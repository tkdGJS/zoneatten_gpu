# Codex Task Prompt

Read first:

- `AGENTS.md`
- `docs/00_PROJECT_OVERVIEW.md`
- `docs/03_VLLM_CODE_MAP.md`
- relevant docs for this task

Task:

```text
<Describe one narrow coding task here.>
```

Constraints:

- Do not change default vLLM behavior unless explicitly requested.
- Put all behavior behind a config flag.
- Add or update tests.
- Update `docs/06_PATCH_NOTES.md`.
- If you find a recurring failure, update `docs/07_TROUBLESHOOTING.md`.

Validation commands:

```bash
python -m compileall vllm
pytest -q <relevant tests>
pre-commit run --files <changed files>
```
