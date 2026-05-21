#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

from measure_vllm_run import (
    append_csv_row,
    derive_tbt_p95_ms,
    detokenize_ids,
    get_model_id,
    parse_sse_json,
    tokenize_text,
    utc_now,
)


@dataclass
class TenantState:
    tenant_id: int
    session_index: int
    user_turns: List[str]
    history_limit_tokens: int
    min_tokens: int
    max_tokens: int
    completed_pairs: List[Tuple[str, str]] = field(default_factory=list)


def mean_or_blank(rows: List[Dict[str, str]], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key, "") not in ("", None)]
    if not values:
        return ""
    return str(round(sum(values) / len(values), 4))


def summarize(raw_csv_path: str, summary_csv_path: str) -> None:
    rows: List[Dict[str, str]] = []
    if not os.path.exists(raw_csv_path):
        return
    with open(raw_csv_path, "r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (
            row["num_gpu_blocks_override"],
            row.get("tenant_kv_min_blocks", "0"),
            row["tenant_count"],
            row["tenant_id"],
        )
        grouped.setdefault(key, []).append(row)

    fieldnames = [
        "num_gpu_blocks_override",
        "tenant_kv_min_blocks",
        "tenant_count",
        "tenant_id",
        "history_limit_tokens",
        "successful_requests",
        "failed_requests",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_kv_history_tokens",
        "avg_prefix_hit_tokens",
        "avg_prefix_hit_rate",
        "avg_blocking_time_ms",
        "avg_ttft_ms",
        "avg_p95_tbt_ms",
        "avg_ttlt_ms",
    ]
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (
            num_gpu_blocks_override,
            tenant_kv_min_blocks,
            tenant_count,
            tenant_id,
        ) in sorted(
            grouped,
            key=lambda item: (
                int(item[0][0]),
                int(item[0][1]),
                int(item[0][2]),
                int(item[0][3]),
            ),
        ):
            group_rows = grouped[
                (num_gpu_blocks_override, tenant_kv_min_blocks, tenant_count, tenant_id)
            ]
            success_rows = [row for row in group_rows if row["status"] == "success"]
            writer.writerow(
                {
                    "num_gpu_blocks_override": num_gpu_blocks_override,
                    "tenant_kv_min_blocks": tenant_kv_min_blocks,
                    "tenant_count": tenant_count,
                    "tenant_id": tenant_id,
                    "history_limit_tokens": group_rows[0].get(
                        "history_limit_tokens", ""
                    ),
                    "successful_requests": len(success_rows),
                    "failed_requests": len(group_rows) - len(success_rows),
                    "avg_input_tokens": mean_or_blank(success_rows, "input_tokens"),
                    "avg_output_tokens": mean_or_blank(success_rows, "output_tokens"),
                    "avg_kv_history_tokens": mean_or_blank(
                        success_rows, "kv_history_tokens"
                    ),
                    "avg_prefix_hit_tokens": mean_or_blank(
                        success_rows, "prefix_hit_tokens"
                    ),
                    "avg_prefix_hit_rate": mean_or_blank(
                        success_rows, "prefix_hit_rate"
                    ),
                    "avg_blocking_time_ms": mean_or_blank(
                        success_rows, "blocking_time_ms"
                    ),
                    "avg_ttft_ms": mean_or_blank(success_rows, "ttft_ms"),
                    "avg_p95_tbt_ms": mean_or_blank(success_rows, "p95_tbt_ms"),
                    "avg_ttlt_ms": mean_or_blank(success_rows, "ttlt_ms"),
                }
            )


def render_user_turn(text: str) -> str:
    return f"Human: {text.strip()}\n\n"


def render_assistant_turn(text: str) -> str:
    return f"Assistant: {text.strip()}\n\n"


def resolve_group_max_tokens(
    history_limit_tokens: int,
    default_max_tokens: int,
    short_limit_tokens: int,
    long_limit_tokens: int,
    short_max_tokens: int,
    long_max_tokens: int,
) -> int:
    if long_limit_tokens > 0 and history_limit_tokens >= long_limit_tokens:
        return long_max_tokens
    if short_limit_tokens > 0 and history_limit_tokens <= short_limit_tokens:
        return short_max_tokens
    return default_max_tokens


def resolve_group_min_tokens(
    history_limit_tokens: int,
    default_min_tokens: int,
    short_limit_tokens: int,
    long_limit_tokens: int,
    short_min_tokens: int,
    long_min_tokens: int,
) -> int:
    if long_limit_tokens > 0 and history_limit_tokens >= long_limit_tokens:
        return long_min_tokens
    if short_limit_tokens > 0 and history_limit_tokens <= short_limit_tokens:
        return short_min_tokens
    return default_min_tokens


def load_sessions(
    dataset_path: str,
    min_user_turns: int,
    default_max_tokens: int,
    default_min_tokens: int,
    short_limit_tokens: int,
    long_limit_tokens: int,
    short_min_tokens: int,
    short_max_tokens: int,
    long_min_tokens: int,
    long_max_tokens: int,
) -> List[TenantState]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions: List[TenantState] = []
    session_index = 0
    for item in data:
        conversations = item.get("conversations") or []
        user_turns = [
            str(message.get("value", "")).strip()
            for message in conversations
            if str(message.get("from", "")).strip().lower() == "human"
            and str(message.get("value", "")).strip()
        ]
        if len(user_turns) < min_user_turns:
            continue
        session_index += 1
        history_limit_tokens = int(item.get("history_limit_tokens", 4096))
        max_tokens = resolve_group_max_tokens(
            history_limit_tokens,
            default_max_tokens,
            short_limit_tokens,
            long_limit_tokens,
            short_max_tokens,
            long_max_tokens,
        )
        min_tokens = resolve_group_min_tokens(
            history_limit_tokens,
            default_min_tokens,
            short_limit_tokens,
            long_limit_tokens,
            short_min_tokens,
            long_min_tokens,
        )
        if min_tokens > max_tokens:
            raise RuntimeError(
                f"Invalid decode budget for history_limit_tokens={history_limit_tokens}: "
                f"min_tokens={min_tokens} > max_tokens={max_tokens}"
            )
        sessions.append(
            TenantState(
                tenant_id=0,
                session_index=session_index,
                user_turns=user_turns,
                history_limit_tokens=history_limit_tokens,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
        )
    if not sessions:
        raise RuntimeError("No ShareGPT sessions found with enough user turns")
    return sessions


def assign_sessions(
    dataset_path: str,
    tenant_count: int,
    min_user_turns: int,
    default_max_tokens: int,
    default_min_tokens: int,
    short_limit_tokens: int,
    long_limit_tokens: int,
    short_min_tokens: int,
    short_max_tokens: int,
    long_min_tokens: int,
    long_max_tokens: int,
) -> List[TenantState]:
    sessions = load_sessions(
        dataset_path,
        min_user_turns,
        default_max_tokens,
        default_min_tokens,
        short_limit_tokens,
        long_limit_tokens,
        short_min_tokens,
        short_max_tokens,
        long_min_tokens,
        long_max_tokens,
    )
    if len(sessions) < tenant_count:
        raise RuntimeError(
            f"Not enough qualifying ShareGPT sessions: need {tenant_count}, got {len(sessions)}"
        )
    assigned: List[TenantState] = []
    for tenant_id in range(1, tenant_count + 1):
        source = sessions[tenant_id - 1]
        assigned.append(
            TenantState(
                tenant_id=tenant_id,
                session_index=source.session_index,
                user_turns=list(source.user_turns),
                history_limit_tokens=source.history_limit_tokens,
                min_tokens=source.min_tokens,
                max_tokens=source.max_tokens,
            )
        )
    return assigned


def build_prompt_for_turn(
    session: requests.Session,
    base_url: str,
    model: str,
    tenant_state: TenantState,
    current_user_text: str,
    max_prompt_tokens: int,
) -> Tuple[str, int, int]:
    history_text = "".join(
        render_user_turn(user_text) + render_assistant_turn(assistant_text)
        for user_text, assistant_text in tenant_state.completed_pairs
    )
    current_turn_text = render_user_turn(current_user_text)
    history_tokens = 0
    if history_text:
        history_tokens, _ = tokenize_text(session, base_url, model, history_text)
    full_text = history_text + current_turn_text
    full_count, full_ids = tokenize_text(session, base_url, model, full_text)
    if full_ids is None:
        raise RuntimeError(
            "Tokenize endpoint did not return token ids for prompt trimming"
        )

    kept_ids = (
        full_ids[-max_prompt_tokens:] if full_count > max_prompt_tokens else full_ids
    )
    prompt = detokenize_ids(session, base_url, model, kept_ids)
    if prompt is None:
        raise RuntimeError(
            "Detokenize endpoint did not return prompt text for trimmed prompt"
        )
    prompt_tokens, _ = tokenize_text(session, base_url, model, prompt)
    removed_tokens = full_count - len(kept_ids)
    kv_history_tokens = max(0, history_tokens - removed_tokens)
    return prompt, prompt_tokens, kv_history_tokens


def save_request_logs(
    log_dir: str,
    tenant_count: int,
    tenant_kv_min_blocks: int,
    run_index: int,
    tenant_id: int,
    turn_index: int,
    prompt_text: str,
    output_text: str,
    metadata: Dict[str, object],
) -> Tuple[str, str, str]:
    request_dir = os.path.join(
        log_dir,
        f"tenant_count_{tenant_count}",
        f"tenant_kv_min_blocks_{tenant_kv_min_blocks}",
        f"run_{run_index}",
        f"tenant_{tenant_id}",
    )
    os.makedirs(request_dir, exist_ok=True)
    prompt_path = os.path.join(request_dir, f"turn_{turn_index:02d}_input.txt")
    output_path = os.path.join(request_dir, f"turn_{turn_index:02d}_output.txt")
    meta_path = os.path.join(request_dir, f"turn_{turn_index:02d}.meta.json")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return prompt_path, output_path, meta_path


def wait_for_request_metrics(
    metrics_jsonl_path: str,
    request_id: str,
    timeout_sec: int = 120,
) -> Dict[str, object]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if os.path.exists(metrics_jsonl_path):
            with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("request_id") == request_id:
                        return payload
        time.sleep(0.5)
    raise RuntimeError(
        f"Timed out waiting for patched vLLM metrics for request_id={request_id}"
    )


def stream_completion_with_request_id(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt: str,
    tenant_id: int,
    min_tokens: int,
    max_tokens: int,
    request_id: str,
    request_timeout_sec: int,
) -> Dict[str, object]:
    url = base_url + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "request_id": request_id,
        "vllm_xargs": {"tenant_id": f"tenant_{tenant_id}"},
    }
    if min_tokens > 0:
        payload["min_tokens"] = min_tokens
    headers = {"X-Request-Id": request_id}
    start_ts = time.monotonic()
    response = session.post(
        url,
        json=payload,
        headers=headers,
        stream=True,
        timeout=(10, max(5, request_timeout_sec)),
    )
    response.raise_for_status()

    first_token_ts = None
    prev_text_ts = None
    chunk_timings: List[Tuple[str, float]] = []
    completion_parts: List[str] = []
    response_id = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if time.monotonic() - start_ts > request_timeout_sec:
            response.close()
            raise TimeoutError(
                f"Request exceeded synchronized turn period of {request_timeout_sec}s"
            )
        if not raw_line:
            continue
        message = parse_sse_json(raw_line)
        if not message:
            continue
        if response_id is None and isinstance(message.get("id"), str):
            response_id = message["id"]
        choices = message.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        text = choice.get("text") or ""
        if text:
            now = time.monotonic()
            if first_token_ts is None:
                first_token_ts = now
            if prev_text_ts is not None:
                chunk_timings.append((text, (now - prev_text_ts) * 1000.0))
            prev_text_ts = now
            completion_parts.append(text)

    end_ts = time.monotonic()
    if first_token_ts is None:
        raise RuntimeError("No completion tokens received from streaming response")
    return {
        "completion_text": "".join(completion_parts),
        "ttft_ms": (first_token_ts - start_ts) * 1000.0,
        "ttlt_ms": (end_ts - start_ts) * 1000.0,
        "chunk_timings": chunk_timings,
        "response_id": response_id or f"cmpl-{request_id}",
    }


def run_tenant(
    tenant_state: TenantState,
    base_url: str,
    model: str,
    num_gpu_blocks_override: int,
    tenant_kv_min_blocks: int,
    tenant_count: int,
    run_index: int,
    turns_per_tenant: int,
    max_prompt_tokens: int,
    request_timeout_sec: int,
    metrics_jsonl_path: str,
    io_log_dir: str,
    raw_csv_path: str,
    raw_fieldnames: List[str],
    csv_lock: threading.Lock,
    start_event: threading.Event,
    launch_barrier: threading.Barrier,
    finish_barrier: threading.Barrier,
    abort_event: threading.Event,
    run_log_path: str,
) -> None:
    request_session = requests.Session()
    max_turns = min(turns_per_tenant, len(tenant_state.user_turns))
    start_event.wait()

    for turn_index in range(1, max_turns + 1):
        try:
            launch_barrier.wait()
        except threading.BrokenBarrierError:
            return
        if abort_event.is_set():
            return

        current_user_text = tenant_state.user_turns[turn_index - 1]
        row: Dict[str, object] = {
            "timestamp_utc": utc_now(),
            "num_gpu_blocks_override": num_gpu_blocks_override,
            "tenant_kv_min_blocks": tenant_kv_min_blocks,
            "tenant_count": tenant_count,
            "run_index": run_index,
            "tenant_id": tenant_state.tenant_id,
            "history_limit_tokens": tenant_state.history_limit_tokens,
            "turn_index": turn_index,
            "session_index": tenant_state.session_index,
            "request_id": "",
            "metrics_request_id": "",
            "status": "failed",
            "error_message": "",
            "input_tokens": "",
            "requested_min_tokens": tenant_state.min_tokens,
            "requested_max_tokens": tenant_state.max_tokens,
            "output_tokens": "",
            "kv_history_tokens": "",
            "prefix_hit_tokens": "",
            "prefix_hit_rate": "",
            "blocking_time_ms": "",
            "ttft_ms": "",
            "p95_tbt_ms": "",
            "ttlt_ms": "",
            "prompt_log_path": "",
            "output_log_path": "",
            "run_log": os.path.abspath(run_log_path),
        }
        try:
            prompt, prompt_tokens, kv_history_tokens = build_prompt_for_turn(
                request_session,
                base_url,
                model,
                tenant_state,
                current_user_text,
                tenant_state.history_limit_tokens,
            )
            client_request_id = (
                f"tenant-{tenant_state.tenant_id}-turn-{turn_index}-"
                f"tenants-{tenant_count}-run-{run_index}"
            )
            stream_result = stream_completion_with_request_id(
                request_session,
                base_url,
                model,
                prompt,
                tenant_state.tenant_id,
                tenant_state.min_tokens,
                tenant_state.max_tokens,
                client_request_id,
                request_timeout_sec,
            )
            metrics_request_id = f"{stream_result['response_id']}-0"
            p95_tbt_ms, output_tokens = derive_tbt_p95_ms(
                request_session,
                base_url,
                model,
                stream_result["completion_text"],
                stream_result["chunk_timings"],
            )
            request_metrics = wait_for_request_metrics(
                metrics_jsonl_path, metrics_request_id
            )
            prefix_hit_tokens = int(request_metrics.get("prefix_hit_tokens", 0))
            prefix_hit_rate = ""
            if kv_history_tokens > 0:
                prefix_hit_rate = round(prefix_hit_tokens / kv_history_tokens, 6)

            metadata = {
                "tenant_id": tenant_state.tenant_id,
                "tenant_kv_min_blocks": tenant_kv_min_blocks,
                "history_limit_tokens": tenant_state.history_limit_tokens,
                "turn_index": turn_index,
                "session_index": tenant_state.session_index,
                "request_id": stream_result["response_id"],
                "metrics_request_id": metrics_request_id,
                "client_request_id": client_request_id,
                "input_tokens": prompt_tokens,
                "requested_min_tokens": tenant_state.min_tokens,
                "requested_max_tokens": tenant_state.max_tokens,
                "output_tokens": output_tokens,
                "kv_history_tokens": kv_history_tokens,
                "prefix_hit_tokens": prefix_hit_tokens,
                "prefix_hit_rate": prefix_hit_rate,
                "blocking_time_ms": round(
                    float(request_metrics["queued_time_s"]) * 1000.0, 4
                ),
                "ttft_ms": round(float(stream_result["ttft_ms"]), 4),
                "p95_tbt_ms": "" if p95_tbt_ms is None else round(float(p95_tbt_ms), 4),
                "ttlt_ms": round(float(stream_result["ttlt_ms"]), 4),
            }
            prompt_path, output_path, _ = save_request_logs(
                io_log_dir,
                tenant_count,
                tenant_kv_min_blocks,
                run_index,
                tenant_state.tenant_id,
                turn_index,
                prompt,
                str(stream_result["completion_text"]),
                metadata,
            )
            row.update(
                {
                    "request_id": stream_result["response_id"],
                    "metrics_request_id": metrics_request_id,
                    "status": "success",
                    "input_tokens": prompt_tokens,
                    "requested_min_tokens": tenant_state.min_tokens,
                    "requested_max_tokens": tenant_state.max_tokens,
                    "output_tokens": output_tokens,
                    "kv_history_tokens": kv_history_tokens,
                    "prefix_hit_tokens": prefix_hit_tokens,
                    "prefix_hit_rate": prefix_hit_rate,
                    "blocking_time_ms": round(
                        float(request_metrics["queued_time_s"]) * 1000.0, 4
                    ),
                    "ttft_ms": round(float(stream_result["ttft_ms"]), 4),
                    "p95_tbt_ms": (
                        "" if p95_tbt_ms is None else round(float(p95_tbt_ms), 4)
                    ),
                    "ttlt_ms": round(float(stream_result["ttlt_ms"]), 4),
                    "prompt_log_path": os.path.abspath(prompt_path),
                    "output_log_path": os.path.abspath(output_path),
                }
            )
            tenant_state.completed_pairs.append(
                (current_user_text, str(stream_result["completion_text"]))
            )
            print(
                f"[RESULT] tenant_count={tenant_count} run={run_index} "
                f"tenant_kv_min_blocks={tenant_kv_min_blocks} "
                f"tenant={tenant_state.tenant_id} turn={turn_index} "
                f"history_limit_tokens={tenant_state.history_limit_tokens} "
                f"requested_min_tokens={tenant_state.min_tokens} "
                f"requested_max_tokens={tenant_state.max_tokens} "
                f"input_tokens={prompt_tokens} output_tokens={output_tokens} "
                f"kv_history_tokens={kv_history_tokens} "
                f"prefix_hit_tokens={prefix_hit_tokens} "
                f"blocking_time_ms={row['blocking_time_ms']} "
                f"ttft_ms={row['ttft_ms']} p95_tbt_ms={row['p95_tbt_ms']} "
                f"ttlt_ms={row['ttlt_ms']}",
                flush=True,
            )
        except Exception as exc:
            row["error_message"] = str(exc)
            print(
                f"[FAIL] tenant_count={tenant_count} run={run_index} "
                f"tenant_kv_min_blocks={tenant_kv_min_blocks} "
                f"tenant={tenant_state.tenant_id} turn={turn_index}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            with csv_lock:
                append_csv_row(raw_csv_path, raw_fieldnames, row)
            abort_event.set()
        finally:
            try:
                finish_barrier.wait()
            except threading.BrokenBarrierError:
                return

        with csv_lock:
            append_csv_row(raw_csv_path, raw_fieldnames, row)

        if abort_event.is_set():
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--io-log-dir", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--tenant-count", required=True, type=int)
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument("--num-gpu-blocks-override", required=True, type=int)
    parser.add_argument("--tenant-kv-min-blocks", required=True, type=int)
    parser.add_argument("--turns-per-tenant", type=int, default=10)
    parser.add_argument("--min-session-user-turns", type=int, default=10)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--short-limit-tokens", type=int, default=0)
    parser.add_argument("--long-limit-tokens", type=int, default=0)
    parser.add_argument("--short-min-tokens", type=int, default=0)
    parser.add_argument("--short-max-tokens", type=int, default=0)
    parser.add_argument("--long-min-tokens", type=int, default=0)
    parser.add_argument("--long-max-tokens", type=int, default=0)
    parser.add_argument("--pre-request-sleep-sec", type=int, default=180)
    parser.add_argument("--inter-turn-sleep-sec", type=int, default=30)
    parser.add_argument("--request-timeout-sec", type=int, default=900)
    args = parser.parse_args()

    raw_fieldnames = [
        "timestamp_utc",
        "num_gpu_blocks_override",
        "tenant_kv_min_blocks",
        "tenant_count",
        "run_index",
        "tenant_id",
        "history_limit_tokens",
        "turn_index",
        "session_index",
        "request_id",
        "metrics_request_id",
        "status",
        "error_message",
        "input_tokens",
        "requested_min_tokens",
        "requested_max_tokens",
        "output_tokens",
        "kv_history_tokens",
        "prefix_hit_tokens",
        "prefix_hit_rate",
        "blocking_time_ms",
        "ttft_ms",
        "p95_tbt_ms",
        "ttlt_ms",
        "prompt_log_path",
        "output_log_path",
        "run_log",
    ]

    print(
        f"[INFO] tenant_count={args.tenant_count} run={args.run_index}: "
        f"sleeping {args.pre_request_sleep_sec}s before first synchronized batch",
        flush=True,
    )
    time.sleep(args.pre_request_sleep_sec)

    session = requests.Session()
    model = get_model_id(session, args.base_url)
    short_min_tokens = (
        args.short_min_tokens if args.short_min_tokens > 0 else args.min_tokens
    )
    short_max_tokens = args.short_max_tokens or args.max_tokens
    long_min_tokens = (
        args.long_min_tokens if args.long_min_tokens > 0 else args.min_tokens
    )
    long_max_tokens = args.long_max_tokens or args.max_tokens
    tenant_states = assign_sessions(
        args.dataset_path,
        args.tenant_count,
        args.min_session_user_turns,
        args.max_tokens,
        args.min_tokens,
        args.short_limit_tokens,
        args.long_limit_tokens,
        short_min_tokens,
        short_max_tokens,
        long_min_tokens,
        long_max_tokens,
    )
    print(
        f"[INFO] group decode budgets: short_limit<={args.short_limit_tokens} "
        f"short_min_tokens={short_min_tokens} short_max_tokens={short_max_tokens}; "
        f"long_limit>={args.long_limit_tokens} "
        f"long_min_tokens={long_min_tokens} long_max_tokens={long_max_tokens}; "
        f"default_min_tokens={args.min_tokens} default_max_tokens={args.max_tokens}",
        flush=True,
    )

    csv_lock = threading.Lock()
    start_event = threading.Event()
    launch_barrier = threading.Barrier(args.tenant_count + 1)
    finish_barrier = threading.Barrier(args.tenant_count + 1)
    abort_event = threading.Event()
    print(
        f"[INFO] tenant_count={args.tenant_count} run={args.run_index}: "
        f"starting {args.tenant_count} tenant sessions with synchronized turns; "
        f"sleeping {args.inter_turn_sleep_sec}s after each completed turn",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.tenant_count) as executor:
        futures = [
            executor.submit(
                run_tenant,
                tenant_state,
                args.base_url,
                model,
                args.num_gpu_blocks_override,
                args.tenant_kv_min_blocks,
                args.tenant_count,
                args.run_index,
                args.turns_per_tenant,
                args.max_prompt_tokens,
                args.request_timeout_sec,
                args.metrics_jsonl,
                args.io_log_dir,
                args.raw_csv,
                raw_fieldnames,
                csv_lock,
                start_event,
                launch_barrier,
                finish_barrier,
                abort_event,
                args.run_log,
            )
            for tenant_state in tenant_states
        ]
        start_event.set()
        max_turns = min(
            args.turns_per_tenant,
            min(len(tenant_state.user_turns) for tenant_state in tenant_states),
        )
        for turn_index in range(1, max_turns + 1):
            print(
                f"[INFO] tenant_count={args.tenant_count} run={args.run_index}: "
                f"launching synchronized turn={turn_index}",
                flush=True,
            )
            try:
                launch_barrier.wait()
                finish_barrier.wait(timeout=args.request_timeout_sec + 30)
            except threading.BrokenBarrierError:
                abort_event.set()
                raise RuntimeError(
                    f"Synchronized turn {turn_index} did not complete within "
                    f"{args.request_timeout_sec + 30}s"
                )
            if abort_event.is_set():
                raise RuntimeError(
                    f"Synchronized turn {turn_index} failed or exceeded request timeout "
                    f"{args.request_timeout_sec}s"
                )
            if turn_index < max_turns and args.inter_turn_sleep_sec > 0:
                print(
                    f"[INFO] tenant_count={args.tenant_count} run={args.run_index}: "
                    f"completed turn={turn_index}; sleeping "
                    f"{args.inter_turn_sleep_sec}s before next turn",
                    flush=True,
                )
                time.sleep(args.inter_turn_sleep_sec)
        for future in futures:
            future.result()

    summarize(args.raw_csv, args.summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
