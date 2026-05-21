# Remaining TTFT Sensitivity Graphs

These graphs preserve the current experiment data and add a visible Remaining TTFT component.

The current request metrics available in this directory are compatibility metrics synthesized from `result_raw.csv`, not the original patched-vLLM metrics. In that synthesized file, `prefill_time_s = TTFT - queued_time_s`, so `Remaining TTFT` becomes zero by construction.

For this additional view, prefill is estimated as a fixed fraction of non-blocking TTFT:

`estimated_prefill = (TTFT - blocking) * fraction`

`estimated_remaining = TTFT - blocking - estimated_prefill`

Generated fractions: `50%`, `70%`, `90%`.

Use these as sensitivity graphs, not as real prefill telemetry. For a factual TTFT decomposition, preserve the original patched-vLLM JSONL with real `scheduled_ts`, `first_token_ts`, and `prefill_time_s`.
