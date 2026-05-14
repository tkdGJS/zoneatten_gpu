#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def render_user_turn(text: str) -> str:
    return f"Human: {text.strip()}\n\n"


def render_assistant_turn(text: str) -> str:
    return f"Assistant: {text.strip()}\n\n"


def add_vendor_dir(vendor_dir: str) -> None:
    if not vendor_dir:
        return
    vendor_path = Path(vendor_dir).resolve()
    if vendor_path.exists():
        sys.path.insert(0, str(vendor_path))


def load_tokenizer(model: str):
    errors: List[str] = []

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return tokenizer
    except Exception as exc:
        errors.append(f"transformers: {exc!r}")

    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer

        return get_tokenizer(model)
    except Exception as exc:
        errors.append(f"vllm: {exc!r}")

    raise RuntimeError("Failed to load tokenizer. " + " / ".join(errors))


def tokenize_text(tokenizer, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def detokenize_ids(tokenizer, token_ids: List[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def load_sessions(dataset_path: Path, turns_per_tenant: int) -> List[Dict[str, object]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sessions: List[Dict[str, object]] = []
    for item in data:
        conversations = item.get("conversations") or []
        user_turns = [
            str(message.get("value", "")).strip()
            for message in conversations
            if str(message.get("from", "")).strip().lower() == "human"
            and str(message.get("value", "")).strip()
        ]
        if len(user_turns) < turns_per_tenant:
            continue
        sessions.append(
            {
                "id": item.get("id"),
                "conversations": conversations,
                "user_turns": user_turns[:turns_per_tenant],
            }
        )
    return sessions


def build_filler_text(tokenizer, target_tokens: int) -> str:
    source_text = (" synthetic assistant output" * max(256, target_tokens)).strip()
    source_ids = tokenize_text(tokenizer, source_text)
    if len(source_ids) < target_tokens:
        raise RuntimeError(
            f"Failed to build filler text: source token count {len(source_ids)} "
            f"is smaller than target {target_tokens}"
        )

    lower = max(1, target_tokens - 32)
    upper = min(len(source_ids), target_tokens + 32)
    for prefix_len in range(target_tokens, lower - 1, -1):
        filler = detokenize_ids(tokenizer, source_ids[:prefix_len])
        if len(tokenize_text(tokenizer, filler)) == target_tokens:
            return filler

    for prefix_len in range(target_tokens + 1, upper + 1):
        filler = detokenize_ids(tokenizer, source_ids[:prefix_len])
        if len(tokenize_text(tokenizer, filler)) == target_tokens:
            return filler

    raise RuntimeError(
        "Failed to build filler text with exact token count "
        f"for target_output_budget_tokens={target_tokens}"
    )


def estimate_session(
    tokenizer,
    user_turns: List[str],
    history_limit_tokens: int,
    assistant_filler: str,
    safety_margin_tokens: int,
) -> Dict[str, object] | None:
    history_text = ""
    prompt_tokens_by_turn: List[int] = []
    history_tokens_after_turn: List[int] = []

    for user_turn in user_turns:
        prompt_text = history_text + render_user_turn(user_turn)
        prompt_tokens = len(tokenize_text(tokenizer, prompt_text))
        if prompt_tokens + safety_margin_tokens > history_limit_tokens:
            return None
        prompt_tokens_by_turn.append(prompt_tokens)

        history_text = history_text + render_user_turn(user_turn) + render_assistant_turn(
            assistant_filler
        )
        history_tokens = len(tokenize_text(tokenizer, history_text))
        if history_tokens + safety_margin_tokens > history_limit_tokens:
            return None
        history_tokens_after_turn.append(history_tokens)

    return {
        "estimated_prompt_tokens_by_turn": prompt_tokens_by_turn,
        "estimated_history_tokens_after_turn": history_tokens_after_turn,
        "estimated_final_prompt_tokens": prompt_tokens_by_turn[-1],
        "estimated_final_history_tokens": history_tokens_after_turn[-1],
    }


def collect_candidates(
    sessions: List[Dict[str, object]],
    tokenizer,
    history_limit_tokens: int,
    assistant_filler: str,
    safety_margin_tokens: int,
    progress_label: str,
    progress_every: int,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    total = len(sessions)
    for index, item in enumerate(sessions, start=1):
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
            print(
                f"[collect:{progress_label}] processed={index}/{total} "
                f"accepted={len(candidates)}"
            )
        estimation = estimate_session(
            tokenizer,
            item["user_turns"],
            history_limit_tokens,
            assistant_filler,
            safety_margin_tokens,
        )
        if estimation is None:
            continue
        candidates.append(
            {
                "source_index": index,
                "source_id": item.get("id"),
                "conversations": item["conversations"],
                "user_turns": item["user_turns"],
                **estimation,
            }
        )
    candidates.sort(key=lambda item: item["estimated_final_history_tokens"])
    return candidates


def select_candidates_near_target(
    candidates: List[Dict[str, object]],
    target_final_prompt_tokens: int,
    final_prompt_tolerance_tokens: int,
    count: int,
    label: str,
    allow_target_fallback: bool,
    excluded_source_indices: set[int] | None = None,
) -> List[Dict[str, object]]:
    excluded_source_indices = excluded_source_indices or set()
    available = [
        item for item in candidates if int(item["source_index"]) not in excluded_source_indices
    ]
    in_range = [
        item
        for item in available
        if abs(item["estimated_final_prompt_tokens"] - target_final_prompt_tokens)
        <= final_prompt_tolerance_tokens
    ]
    print(
        f"[select:{label}] available={len(available)} "
        f"within_target_tolerance={len(in_range)} need={count} "
        f"target={target_final_prompt_tokens} tolerance={final_prompt_tolerance_tokens}"
    )
    if len(in_range) < count and not allow_target_fallback:
        closest = sorted(
            available,
            key=lambda item: abs(
                int(item["estimated_final_prompt_tokens"]) - target_final_prompt_tokens
            ),
        )[: min(10, len(available))]
        closest_text = ", ".join(
            str(item["estimated_final_prompt_tokens"]) for item in closest
        )
        raise RuntimeError(
            f"Not enough {label} candidates near target final prompt: "
            f"need {count}, got {len(in_range)} within ±{final_prompt_tolerance_tokens}. "
            f"Closest final_prompt_tokens=[{closest_text}]. "
            "Lower the target, increase tolerance, or pass --allow-target-fallback "
            "to select the closest candidates anyway."
        )
    ranked_pool = in_range if len(in_range) >= count else available
    ranked = sorted(
        ranked_pool,
        key=lambda item: (
            abs(item["estimated_final_prompt_tokens"] - target_final_prompt_tokens),
            -item["estimated_final_history_tokens"],
        ),
    )
    return ranked[:count]


def build_dataset_entries(
    short_candidates: List[Dict[str, object]],
    long_candidates: List[Dict[str, object]],
    tenant_count: int,
    short_limit_tokens: int,
    long_limit_tokens: int,
    target_output_budget_tokens: int,
    short_target_final_prompt_tokens: int,
    short_final_prompt_tolerance_tokens: int,
    long_target_final_prompt_tokens: int,
    long_final_prompt_tolerance_tokens: int,
    allow_target_fallback: bool,
) -> List[Dict[str, object]]:
    half = tenant_count // 2
    selected_long = select_candidates_near_target(
        long_candidates,
        long_target_final_prompt_tokens,
        long_final_prompt_tolerance_tokens,
        half,
        f"long:{long_limit_tokens}",
        allow_target_fallback,
    )
    used_source_indices = {item["source_index"] for item in selected_long}
    selected_short = select_candidates_near_target(
        short_candidates,
        short_target_final_prompt_tokens,
        short_final_prompt_tolerance_tokens,
        half,
        f"short:{short_limit_tokens}",
        allow_target_fallback,
        used_source_indices,
    )

    if len(selected_long) < half:
        raise RuntimeError(
            f"Not enough candidates for history_limit_tokens={long_limit_tokens}: "
            f"need {half}, got {len(selected_long)}"
        )
    if len(selected_short) < half:
        raise RuntimeError(
            f"Not enough non-overlapping candidates for history_limit_tokens={short_limit_tokens}: "
            f"need {half}, got {len(selected_short)}"
        )

    entries: List[Dict[str, object]] = []
    for candidate in selected_long:
        entries.append(
            {
                "history_limit_tokens": long_limit_tokens,
                "target_output_budget_tokens": target_output_budget_tokens,
                "source_index": candidate["source_index"],
                "source_id": candidate["source_id"],
                "estimated_prompt_tokens_by_turn": candidate["estimated_prompt_tokens_by_turn"],
                "estimated_history_tokens_after_turn": candidate["estimated_history_tokens_after_turn"],
                "estimated_final_prompt_tokens": candidate["estimated_final_prompt_tokens"],
                "conversations": candidate["conversations"],
            }
        )
    for candidate in selected_short:
        entries.append(
            {
                "history_limit_tokens": short_limit_tokens,
                "target_output_budget_tokens": target_output_budget_tokens,
                "source_index": candidate["source_index"],
                "source_id": candidate["source_id"],
                "estimated_prompt_tokens_by_turn": candidate["estimated_prompt_tokens_by_turn"],
                "estimated_history_tokens_after_turn": candidate["estimated_history_tokens_after_turn"],
                "estimated_final_prompt_tokens": candidate["estimated_final_prompt_tokens"],
                "conversations": candidate["conversations"],
            }
        )
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--vendor-dir", default="")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tenant-count", required=True, type=int, choices=[2, 4, 8, 16, 32])
    parser.add_argument("--turns-per-tenant", type=int, default=10)
    parser.add_argument("--short-limit-tokens", type=int, default=2048)
    parser.add_argument("--long-limit-tokens", type=int, default=8192)
    parser.add_argument("--target-output-budget-tokens", type=int, default=64)
    parser.add_argument("--safety-margin-tokens", type=int, default=64)
    parser.add_argument("--short-target-final-prompt-tokens", type=int, default=1800)
    parser.add_argument("--short-final-prompt-tolerance-tokens", type=int, default=400)
    parser.add_argument("--long-target-final-prompt-tokens", type=int, default=7600)
    parser.add_argument("--long-final-prompt-tolerance-tokens", type=int, default=400)
    parser.add_argument(
        "--allow-target-fallback",
        action="store_true",
        help="Select closest candidates when too few sessions fall within the target tolerance.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tenant_count % 2 != 0:
        raise RuntimeError("tenant_count must be even for mixed-limit datasets")

    add_vendor_dir(args.vendor_dir)
    tokenizer = load_tokenizer(args.model)

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    sessions = load_sessions(dataset_path, args.turns_per_tenant)
    if not sessions:
        raise RuntimeError("No qualifying sessions found in source dataset")
    print(f"[load] sessions={len(sessions)} turns_per_tenant={args.turns_per_tenant}")

    assistant_filler = build_filler_text(tokenizer, args.target_output_budget_tokens)
    print(
        f"[filler] target_output_budget_tokens={args.target_output_budget_tokens} "
        f"validated_tokens={len(tokenize_text(tokenizer, assistant_filler))}"
    )
    final_assistant_turn_tokens = len(tokenize_text(tokenizer, render_assistant_turn(assistant_filler)))
    for label, limit_tokens, target_tokens in [
        ("short", args.short_limit_tokens, args.short_target_final_prompt_tokens),
        ("long", args.long_limit_tokens, args.long_target_final_prompt_tokens),
    ]:
        approximate_max_final_prompt = (
            limit_tokens - final_assistant_turn_tokens - args.safety_margin_tokens
        )
        print(
            f"[feasible:{label}] approximate_max_final_prompt_tokens="
            f"{approximate_max_final_prompt} target={target_tokens} "
            f"limit={limit_tokens} final_assistant_turn_tokens={final_assistant_turn_tokens} "
            f"safety_margin={args.safety_margin_tokens}"
        )
        if target_tokens > approximate_max_final_prompt:
            print(
                f"[WARN:{label}] target_final_prompt_tokens={target_tokens} is above "
                f"the approximate feasible maximum {approximate_max_final_prompt}. "
                "The generator will likely fail unless tolerance includes lower candidates."
            )

    short_candidates = collect_candidates(
        sessions,
        tokenizer,
        args.short_limit_tokens,
        assistant_filler,
        args.safety_margin_tokens,
        progress_label=f"short:{args.short_limit_tokens}",
        progress_every=args.progress_every,
    )
    print(f"[collect:short] candidates={len(short_candidates)}")

    long_candidates = collect_candidates(
        sessions,
        tokenizer,
        args.long_limit_tokens,
        assistant_filler,
        args.safety_margin_tokens,
        progress_label=f"long:{args.long_limit_tokens}",
        progress_every=args.progress_every,
    )
    print(f"[collect:long] candidates={len(long_candidates)}")

    entries = build_dataset_entries(
        short_candidates,
        long_candidates,
        args.tenant_count,
        args.short_limit_tokens,
        args.long_limit_tokens,
        args.target_output_budget_tokens,
        args.short_target_final_prompt_tokens,
        args.short_final_prompt_tolerance_tokens,
        args.long_target_final_prompt_tokens,
        args.long_final_prompt_tolerance_tokens,
        args.allow_target_fallback,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(
        f"[DONE] wrote {len(entries)} sessions to {output_path} "
        f"(model={args.model}, long_limit={args.long_limit_tokens}, "
        f"short_limit={args.short_limit_tokens}, "
        f"target_output_budget_tokens={args.target_output_budget_tokens}, "
        f"short_target_final_prompt_tokens={args.short_target_final_prompt_tokens}, "
        f"short_final_prompt_tolerance_tokens={args.short_final_prompt_tolerance_tokens}, "
        f"long_target_final_prompt_tokens={args.long_target_final_prompt_tokens}, "
        f"long_final_prompt_tolerance_tokens={args.long_final_prompt_tolerance_tokens})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
