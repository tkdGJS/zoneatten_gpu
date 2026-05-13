#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--tenant-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--kv-capacity-tokens", type=int, default=262144)
    parser.add_argument("--show-first", type=int, default=0)
    return parser.parse_args()


def fmt_ratio(value: int, capacity: int) -> str:
    if capacity <= 0:
        return ""
    return f"{value / capacity:.3f}x"


def summarize_prefix(entries: list[dict], prefix_count: int, kv_capacity_tokens: int) -> None:
    subset = entries[:prefix_count]
    if len(subset) < prefix_count:
        print(
            f"[tenant_count={prefix_count}] not enough entries: "
            f"need {prefix_count}, got {len(subset)}"
        )
        return

    limits = Counter(int(item.get("history_limit_tokens", 0)) for item in subset)
    final_prompt_tokens = [int(item.get("estimated_final_prompt_tokens", 0)) for item in subset]
    final_history_tokens = [
        int((item.get("estimated_history_tokens_after_turn") or [0])[-1]) for item in subset
    ]
    output_budgets = [int(item.get("target_output_budget_tokens", 0)) for item in subset]

    sum_prompt = sum(final_prompt_tokens)
    sum_prompt_plus_budget = sum(
        prompt_tokens + output_budget
        for prompt_tokens, output_budget in zip(final_prompt_tokens, output_budgets)
    )
    sum_history = sum(final_history_tokens)

    print(f"[tenant_count={prefix_count}]")
    print(f"  limits={dict(sorted(limits.items()))}")
    print(
        f"  final_prompt_sum={sum_prompt} "
        f"({fmt_ratio(sum_prompt, kv_capacity_tokens)} of kv_capacity)"
    )
    print(
        f"  final_prompt_plus_budget_sum={sum_prompt_plus_budget} "
        f"({fmt_ratio(sum_prompt_plus_budget, kv_capacity_tokens)} of kv_capacity)"
    )
    print(
        f"  final_history_sum={sum_history} "
        f"({fmt_ratio(sum_history, kv_capacity_tokens)} of kv_capacity)"
    )
    print(
        f"  avg_final_prompt={mean(final_prompt_tokens):.1f} "
        f"avg_final_history={mean(final_history_tokens):.1f}"
    )


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    with dataset_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"dataset_path={dataset_path}")
    print(f"entries={len(entries)}")

    limits = Counter(int(item.get("history_limit_tokens", 0)) for item in entries)
    print(f"history_limit_counts={dict(sorted(limits.items()))}")

    source_indices = [int(item.get("source_index", -1)) for item in entries]
    duplicate_source_indices = sorted(
        source_index
        for source_index, count in Counter(source_indices).items()
        if source_index >= 0 and count > 1
    )
    print(f"duplicate_source_indices={duplicate_source_indices}")

    long_entries = [item for item in entries if int(item.get("history_limit_tokens", 0)) > 12288]
    short_entries = [item for item in entries if int(item.get("history_limit_tokens", 0)) <= 12288]
    if long_entries:
        long_prompts = [int(item.get("estimated_final_prompt_tokens", 0)) for item in long_entries]
        print(
            "long_final_prompt_tokens="
            f"min={min(long_prompts)} max={max(long_prompts)} avg={mean(long_prompts):.1f}"
        )
    if short_entries:
        short_prompts = [int(item.get("estimated_final_prompt_tokens", 0)) for item in short_entries]
        print(
            "short_final_prompt_tokens="
            f"min={min(short_prompts)} max={max(short_prompts)} avg={mean(short_prompts):.1f}"
        )

    if args.show_first > 0:
        print(f"first_{args.show_first}_entries:")
        for index, item in enumerate(entries[: args.show_first], start=1):
            history_tokens_after_turn = item.get("estimated_history_tokens_after_turn") or [0]
            print(
                "  "
                f"{index}: limit={item.get('history_limit_tokens')} "
                f"final_prompt={item.get('estimated_final_prompt_tokens')} "
                f"final_history={history_tokens_after_turn[-1]} "
                f"budget={item.get('target_output_budget_tokens')} "
                f"source_index={item.get('source_index')} "
                f"source_id={item.get('source_id')}"
            )

    for tenant_count in args.tenant_values:
        summarize_prefix(entries, tenant_count, args.kv_capacity_tokens)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
