#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from measure_vllm_run import (
    build_exact_prompt_from_token_ids,
    build_prompt_by_binary_search,
    get_model_id,
    tokenize_text,
)


def render_user_turn(text: str) -> str:
    return f"Human: {text.strip()}\n\n"


def render_assistant_turn(text: str) -> str:
    return f"Assistant: {text.strip()}\n\n"


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


def build_filler_text(
    session: requests.Session,
    base_url: str,
    model: str,
    target_tokens: int,
) -> str:
    source_text = (" synthetic assistant output" * max(256, target_tokens)).strip()
    filler = build_exact_prompt_from_token_ids(
        session, base_url, model, source_text, target_tokens
    )
    if filler is not None:
        return filler
    filler, token_count = build_prompt_by_binary_search(
        session, base_url, model, source_text, target_tokens
    )
    if token_count != target_tokens:
        raise RuntimeError(
            f"Failed to build filler text with exact token count: "
            f"target={target_tokens} actual={token_count}"
        )
    return filler


def estimate_session(
    session_client: requests.Session,
    base_url: str,
    model: str,
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
        prompt_tokens, _ = tokenize_text(session_client, base_url, model, prompt_text)
        if prompt_tokens + safety_margin_tokens > history_limit_tokens:
            return None
        prompt_tokens_by_turn.append(prompt_tokens)

        history_text = history_text + render_user_turn(user_turn) + render_assistant_turn(
            assistant_filler
        )
        history_tokens, _ = tokenize_text(session_client, base_url, model, history_text)
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
    session_client: requests.Session,
    base_url: str,
    model: str,
    history_limit_tokens: int,
    assistant_filler: str,
    safety_margin_tokens: int,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for index, item in enumerate(sessions, start=1):
        estimation = estimate_session(
            session_client,
            base_url,
            model,
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


def select_long_candidates_near_target(
    candidates: List[Dict[str, object]],
    target_final_prompt_tokens: int,
    final_prompt_tolerance_tokens: int,
    count: int,
) -> List[Dict[str, object]]:
    in_range = [
        item
        for item in candidates
        if abs(item["estimated_final_prompt_tokens"] - target_final_prompt_tokens)
        <= final_prompt_tolerance_tokens
    ]
    ranked_pool = in_range if len(in_range) >= count else candidates
    ranked = sorted(
        ranked_pool,
        key=lambda item: (
            abs(item["estimated_final_prompt_tokens"] - target_final_prompt_tokens),
            item["estimated_final_history_tokens"],
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
    long_target_final_prompt_tokens: int,
    long_final_prompt_tolerance_tokens: int,
) -> List[Dict[str, object]]:
    half = tenant_count // 2
    selected_long = select_long_candidates_near_target(
        long_candidates,
        long_target_final_prompt_tokens,
        long_final_prompt_tolerance_tokens,
        half,
    )
    used_source_indices = {item["source_index"] for item in selected_long}
    selected_short = [
        item for item in short_candidates if item["source_index"] not in used_source_indices
    ][:half]

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tenant-count", required=True, type=int, choices=[2, 4, 8, 16, 32])
    parser.add_argument("--turns-per-tenant", type=int, default=10)
    parser.add_argument("--short-limit-tokens", type=int, default=2048)
    parser.add_argument("--long-limit-tokens", type=int, default=8192)
    parser.add_argument("--target-output-budget-tokens", type=int, default=64)
    parser.add_argument("--safety-margin-tokens", type=int, default=64)
    parser.add_argument("--long-target-final-prompt-tokens", type=int, default=7600)
    parser.add_argument("--long-final-prompt-tolerance-tokens", type=int, default=400)
    args = parser.parse_args()

    if args.tenant_count % 2 != 0:
        raise RuntimeError("tenant_count must be even for mixed-limit datasets")

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    session_client = requests.Session()
    model = get_model_id(session_client, args.base_url)
    sessions = load_sessions(dataset_path, args.turns_per_tenant)
    if not sessions:
        raise RuntimeError("No qualifying sessions found in source dataset")

    assistant_filler = build_filler_text(
        session_client,
        args.base_url,
        model,
        args.target_output_budget_tokens,
    )
    short_candidates = collect_candidates(
        sessions,
        session_client,
        args.base_url,
        model,
        args.short_limit_tokens,
        assistant_filler,
        args.safety_margin_tokens,
    )
    long_candidates = collect_candidates(
        sessions,
        session_client,
        args.base_url,
        model,
        args.long_limit_tokens,
        assistant_filler,
        args.safety_margin_tokens,
    )

    entries = build_dataset_entries(
        short_candidates,
        long_candidates,
        args.tenant_count,
        args.short_limit_tokens,
        args.long_limit_tokens,
        args.target_output_budget_tokens,
        args.long_target_final_prompt_tokens,
        args.long_final_prompt_tolerance_tokens,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(
        f"[DONE] wrote {len(entries)} sessions to {output_path} "
        f"(long_limit={args.long_limit_tokens}, short_limit={args.short_limit_tokens}, "
        f"target_output_budget_tokens={args.target_output_budget_tokens}, "
        f"safety_margin_tokens={args.safety_margin_tokens}, "
        f"long_target_final_prompt_tokens={args.long_target_final_prompt_tokens}, "
        f"long_final_prompt_tolerance_tokens={args.long_final_prompt_tolerance_tokens})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
