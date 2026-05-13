#!/usr/bin/env python3
import argparse
from pathlib import Path
from statistics import mean

from build_mixed_history_limit_dataset_offline import (
    add_vendor_dir,
    build_filler_text,
    collect_candidates,
    load_sessions,
    load_tokenizer,
)


def summarize(label: str, candidates: list[dict], limit_tokens: int, target_tokens: int) -> None:
    print(f"[{label}] limit={limit_tokens} candidates={len(candidates)} target={target_tokens}")
    if not candidates:
        return

    final_prompts = [int(item["estimated_final_prompt_tokens"]) for item in candidates]
    final_histories = [int(item["estimated_final_history_tokens"]) for item in candidates]
    print(
        f"  final_prompt min/mean/max="
        f"{min(final_prompts)}/{mean(final_prompts):.1f}/{max(final_prompts)}"
    )
    print(
        f"  final_history min/mean/max="
        f"{min(final_histories)}/{mean(final_histories):.1f}/{max(final_histories)}"
    )

    for tolerance in [256, 512, 1024, 2048, 4096]:
        in_range = [
            item
            for item in candidates
            if abs(int(item["estimated_final_prompt_tokens"]) - target_tokens) <= tolerance
        ]
        print(f"  within_target_tolerance_{tolerance}={len(in_range)}")

    near_limit = [
        item
        for item in candidates
        if int(item["estimated_final_prompt_tokens"]) >= target_tokens
    ]
    print(f"  at_or_above_target={len(near_limit)}")

    ranked = sorted(
        candidates,
        key=lambda item: abs(int(item["estimated_final_prompt_tokens"]) - target_tokens),
    )[:20]
    print("  top20_closest_final_prompt_tokens=")
    for item in ranked:
        print(
            "    "
            f"source_index={item['source_index']} "
            f"source_id={item.get('source_id')} "
            f"final_prompt={item['estimated_final_prompt_tokens']} "
            f"final_history={item['estimated_final_history_tokens']} "
            f"delta={int(item['estimated_final_prompt_tokens']) - target_tokens}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--vendor-dir", default="")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--turns-per-tenant", type=int, default=10)
    parser.add_argument("--short-limit-tokens", type=int, default=12288)
    parser.add_argument("--long-limit-tokens", type=int, default=24576)
    parser.add_argument("--target-output-budget-tokens", type=int, default=1024)
    parser.add_argument("--safety-margin-tokens", type=int, default=64)
    parser.add_argument("--short-target-final-prompt-tokens", type=int, default=11200)
    parser.add_argument("--long-target-final-prompt-tokens", type=int, default=23500)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    add_vendor_dir(args.vendor_dir)
    tokenizer = load_tokenizer(args.model)
    sessions = load_sessions(Path(args.dataset_path), args.turns_per_tenant)
    print(f"[load] sessions_with_{args.turns_per_tenant}_turns={len(sessions)}")

    assistant_filler = build_filler_text(tokenizer, args.target_output_budget_tokens)
    short_candidates = collect_candidates(
        sessions,
        tokenizer,
        args.short_limit_tokens,
        assistant_filler,
        args.safety_margin_tokens,
        progress_label=f"short:{args.short_limit_tokens}",
        progress_every=args.progress_every,
    )
    long_candidates = collect_candidates(
        sessions,
        tokenizer,
        args.long_limit_tokens,
        assistant_filler,
        args.safety_margin_tokens,
        progress_label=f"long:{args.long_limit_tokens}",
        progress_every=args.progress_every,
    )

    summarize(
        "group2_short",
        short_candidates,
        args.short_limit_tokens,
        args.short_target_final_prompt_tokens,
    )
    summarize(
        "group1_long",
        long_candidates,
        args.long_limit_tokens,
        args.long_target_final_prompt_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
