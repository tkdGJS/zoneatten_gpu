#!/usr/bin/env python3
import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import requests

from measure_vllm_run import (
    build_exact_prompt_from_token_ids,
    build_prompt_by_binary_search,
    derive_tbt_p95_ms,
    format_session_text,
    get_model_id,
    stream_completion,
    tokenize_text,
    utc_now,
)


def append_csv_row(path: str, fieldnames: List[str], row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def mean_or_blank(rows: List[Dict[str, str]], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    if not values:
        return ""
    return str(round(sum(values) / len(values), 4))


def summarize(raw_csv_path: str, summary_csv_path: str) -> None:
    rows: List[Dict[str, str]] = []
    with open(raw_csv_path, "r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["num_gpu_blocks_override"], row["tenant_count"])
        grouped.setdefault(key, []).append(row)

    fieldnames = [
        "num_gpu_blocks_override",
        "tenant_count",
        "successful_requests",
        "failed_requests",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_ttft_ms",
        "avg_p95_tbt_ms",
        "avg_ttlt_ms",
    ]
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for block_value, tenant_count in sorted(grouped, key=lambda x: (int(x[0]), int(x[1]))):
            group_rows = grouped[(block_value, tenant_count)]
            success_rows = [row for row in group_rows if row["status"] == "success"]
            writer.writerow(
                {
                    "num_gpu_blocks_override": block_value,
                    "tenant_count": tenant_count,
                    "successful_requests": len(success_rows),
                    "failed_requests": len(group_rows) - len(success_rows),
                    "avg_prompt_tokens": mean_or_blank(success_rows, "prompt_tokens"),
                    "avg_completion_tokens": mean_or_blank(success_rows, "completion_tokens"),
                    "avg_ttft_ms": mean_or_blank(success_rows, "ttft_ms"),
                    "avg_p95_tbt_ms": mean_or_blank(success_rows, "p95_tbt_ms"),
                    "avg_ttlt_ms": mean_or_blank(success_rows, "ttlt_ms"),
                }
            )


def load_sessions(dataset_path: str, min_turns: int = 4) -> List[List[Dict[str, str]]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sessions: List[List[Dict[str, str]]] = []
    for item in data:
        conversations = item.get("conversations") or []
        if len(conversations) >= min_turns:
            sessions.append(conversations)
    if not sessions:
        raise RuntimeError("No multi-turn ShareGPT sessions found")
    return sessions


def build_source_texts(dataset_path: str, tenant_count: int) -> List[str]:
    sessions = load_sessions(dataset_path)
    stride = max(1, len(sessions) // tenant_count)
    source_texts: List[str] = []
    for tenant_id in range(tenant_count):
        start = min(tenant_id * stride, len(sessions) - 1)
        parts: List[str] = []
        session_idx = 0
        cursor = start
        while cursor < len(sessions) and sum(len(part) for part in parts) < 120000:
            session_idx += 1
            parts.append(format_session_text(sessions[cursor], session_idx))
            cursor += 1
        if not parts:
            raise RuntimeError(f"Unable to build source text for tenant {tenant_id + 1}")
        source_texts.append("".join(parts))
    return source_texts


def ensure_prompt_for_source(
    session: requests.Session,
    base_url: str,
    model: str,
    source_text: str,
    prompt_cache_path: str,
    prompt_meta_path: str,
    target_tokens: int,
    tenant_id: int,
) -> Tuple[str, int]:
    if os.path.exists(prompt_cache_path) and os.path.exists(prompt_meta_path):
        with open(prompt_cache_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        count, _ = tokenize_text(session, base_url, model, prompt)
        if count == target_tokens:
            return prompt, count

    prompt = build_exact_prompt_from_token_ids(session, base_url, model, source_text, target_tokens)
    if prompt is None:
        prompt, prompt_tokens = build_prompt_by_binary_search(
            session, base_url, model, source_text, target_tokens
        )
        if prompt_tokens != target_tokens:
            raise RuntimeError(
                f"Could not construct exact {target_tokens}-token prompt for tenant {tenant_id}; "
                f"nearest prompt has {prompt_tokens} tokens"
            )

    prompt_tokens, _ = tokenize_text(session, base_url, model, prompt)
    if prompt_tokens != target_tokens:
        raise RuntimeError(
            f"Prompt cache validation failed for tenant {tenant_id}: expected {target_tokens} "
            f"tokens, got {prompt_tokens}"
        )

    os.makedirs(os.path.dirname(prompt_cache_path), exist_ok=True)
    with open(prompt_cache_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(prompt_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model,
                "tenant_id": tenant_id,
                "target_tokens": target_tokens,
                "created_at": utc_now(),
            },
            f,
            indent=2,
        )
    return prompt, prompt_tokens


def run_tenant_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    min_tokens: int,
    tenant_id: int,
    start_event: threading.Event,
) -> Dict[str, object]:
    session = requests.Session()
    start_event.wait()
    stream_result = stream_completion(session, base_url, model, prompt, max_tokens, min_tokens)
    p95_tbt_ms, completion_tokens = derive_tbt_p95_ms(
        session,
        base_url,
        model,
        stream_result["completion_text"],
        stream_result["chunk_timings"],
    )
    return {
        "tenant_id": tenant_id,
        "completion_tokens": completion_tokens,
        "ttft_ms": round(float(stream_result["ttft_ms"]), 4),
        "p95_tbt_ms": "" if p95_tbt_ms is None else round(float(p95_tbt_ms), 4),
        "ttlt_ms": round(float(stream_result["ttlt_ms"]), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--num-gpu-blocks-override", required=True, type=int)
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument("--tenant-count", required=True, type=int)
    parser.add_argument("--target-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-tokens", type=int, default=4000)
    parser.add_argument("--pre-request-sleep-sec", type=int, default=180)
    args = parser.parse_args()

    fieldnames = [
        "timestamp_utc",
        "num_gpu_blocks_override",
        "run_index",
        "tenant_count",
        "tenant_id",
        "base_url",
        "status",
        "error_message",
        "prompt_tokens",
        "completion_tokens",
        "ttft_ms",
        "p95_tbt_ms",
        "ttlt_ms",
        "run_log",
    ]

    print(
        f"[INFO] blocks={args.num_gpu_blocks_override} run={args.run_index}: "
        f"sleeping {args.pre_request_sleep_sec}s before {args.tenant_count} tenant requests",
        flush=True,
    )
    time.sleep(args.pre_request_sleep_sec)

    session = requests.Session()
    model = get_model_id(session, args.base_url)
    source_texts = build_source_texts(args.dataset_path, args.tenant_count)

    prompts: List[Tuple[int, str, int]] = []
    for tenant_id, source_text in enumerate(source_texts, start=1):
        prompt_cache_path = os.path.join(args.prompt_dir, f"sharegpt_prompt_4096_tenant_{tenant_id}.txt")
        prompt_meta_path = os.path.join(args.prompt_dir, f"sharegpt_prompt_4096_tenant_{tenant_id}.meta.json")
        prompt, prompt_tokens = ensure_prompt_for_source(
            session,
            args.base_url,
            model,
            source_text,
            prompt_cache_path,
            prompt_meta_path,
            args.target_prompt_tokens,
            tenant_id,
        )
        prompts.append((tenant_id, prompt, prompt_tokens))

    print(
        f"[INFO] blocks={args.num_gpu_blocks_override} run={args.run_index}: "
        f"sending {args.tenant_count} concurrent tenant requests",
        flush=True,
    )

    start_event = threading.Event()
    results: List[Dict[str, object]] = []
    failures: List[Tuple[int, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.tenant_count) as executor:
        future_to_tenant = {
            executor.submit(
                run_tenant_request,
                args.base_url,
                model,
                prompt,
                args.max_tokens,
                args.min_tokens,
                tenant_id,
                start_event,
            ): tenant_id
            for tenant_id, prompt, _ in prompts
        }
        start_event.set()
        for future in as_completed(future_to_tenant):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append((future_to_tenant[future], exc))

    prompt_token_map = {tenant_id: prompt_tokens for tenant_id, _, prompt_tokens in prompts}
    timestamp = utc_now()
    ok = True
    for tenant_id, _, _ in prompts:
        tenant_result = next((row for row in results if row["tenant_id"] == tenant_id), None)
        if tenant_result is not None:
            row = {
                "timestamp_utc": timestamp,
                "num_gpu_blocks_override": args.num_gpu_blocks_override,
                "run_index": args.run_index,
                "tenant_count": args.tenant_count,
                "tenant_id": tenant_id,
                "base_url": args.base_url,
                "status": "success",
                "error_message": "",
                "prompt_tokens": prompt_token_map[tenant_id],
                "completion_tokens": tenant_result["completion_tokens"],
                "ttft_ms": tenant_result["ttft_ms"],
                "p95_tbt_ms": tenant_result["p95_tbt_ms"],
                "ttlt_ms": tenant_result["ttlt_ms"],
                "run_log": os.path.abspath(args.run_log),
            }
            print(
                f"[RESULT] blocks={args.num_gpu_blocks_override} run={args.run_index} "
                f"tenant={tenant_id}: input_tokens={row['prompt_tokens']} "
                f"output_tokens={row['completion_tokens']} ttft_ms={row['ttft_ms']} "
                f"p95_tbt_ms={row['p95_tbt_ms']} ttlt_ms={row['ttlt_ms']}",
                flush=True,
            )
        else:
            ok = False
            failure = next((exc for failed_tenant, exc in failures if failed_tenant == tenant_id), None)
            row = {
                "timestamp_utc": timestamp,
                "num_gpu_blocks_override": args.num_gpu_blocks_override,
                "run_index": args.run_index,
                "tenant_count": args.tenant_count,
                "tenant_id": tenant_id,
                "base_url": args.base_url,
                "status": "failed",
                "error_message": "" if failure is None else str(failure),
                "prompt_tokens": prompt_token_map[tenant_id],
                "completion_tokens": "",
                "ttft_ms": "",
                "p95_tbt_ms": "",
                "ttlt_ms": "",
                "run_log": os.path.abspath(args.run_log),
            }
            print(
                f"[FAIL] blocks={args.num_gpu_blocks_override} run={args.run_index} "
                f"tenant={tenant_id}: {row['error_message']}",
                flush=True,
            )
        append_csv_row(args.raw_csv, fieldnames, row)

    summarize(args.raw_csv, args.summary_csv)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
