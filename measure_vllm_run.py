#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests


TOKENIZE_PATHS = ["/tokenize", "/v1/tokenize"]
DETOKENIZE_PATHS = ["/detokenize", "/v1/detokenize"]
MODELS_PATH = "/v1/models"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def csv_escape(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(ch in text for ch in [",", "\"", "\n"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def append_csv_row(path: str, fieldnames: List[str], row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def request_json(session: requests.Session, method: str, url: str, **kwargs) -> Dict:
    response = session.request(method, url, timeout=kwargs.pop("timeout", 30), **kwargs)
    response.raise_for_status()
    return response.json()


def get_model_id(session: requests.Session, base_url: str) -> str:
    payload = request_json(session, "GET", base_url + MODELS_PATH, timeout=10)
    data = payload.get("data", [])
    if not data:
        raise RuntimeError("No models returned from /v1/models")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError("Missing model id in /v1/models response")
    return model_id


def _tokenize_request_payload(model: str, text: str) -> Dict[str, object]:
    return {"model": model, "prompt": text}


def tokenize_text(session: requests.Session, base_url: str, model: str, text: str) -> Tuple[int, Optional[List[int]]]:
    errors = []
    for path in TOKENIZE_PATHS:
        try:
            payload = request_json(
                session,
                "POST",
                base_url + path,
                json=_tokenize_request_payload(model, text),
                timeout=60,
            )
            token_ids = None
            for key in ["tokens", "token_ids", "input_ids", "prompt_token_ids"]:
                value = payload.get(key)
                if isinstance(value, list):
                    token_ids = value
                    break
            if token_ids is not None:
                return len(token_ids), token_ids
            for key in ["count", "token_count", "num_tokens"]:
                value = payload.get(key)
                if isinstance(value, int):
                    return value, None
            raise RuntimeError(f"Unsupported tokenize response keys: {sorted(payload.keys())}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    raise RuntimeError("Tokenize request failed: " + "; ".join(errors))


def detokenize_ids(session: requests.Session, base_url: str, model: str, token_ids: List[int]) -> Optional[str]:
    errors = []
    for path in DETOKENIZE_PATHS:
        try:
            payload = request_json(
                session,
                "POST",
                base_url + path,
                json={"model": model, "tokens": token_ids},
                timeout=60,
            )
            for key in ["prompt", "text", "content"]:
                value = payload.get(key)
                if isinstance(value, str):
                    return value
            raise RuntimeError(f"Unsupported detokenize response keys: {sorted(payload.keys())}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return None


def format_session_text(conversations: List[Dict[str, str]], session_idx: int) -> str:
    parts = [f"### Session {session_idx} ###"]
    for message in conversations:
        speaker = message.get("from", "").strip().lower()
        if speaker == "human":
            label = "Human"
        elif speaker == "gpt":
            label = "Assistant"
        else:
            label = speaker.capitalize() or "Unknown"
        value = str(message.get("value", "")).strip()
        if not value:
            continue
        parts.append(f"{label}: {value}")
    return "\n\n".join(parts).strip() + "\n\n"


def load_multi_turn_corpus(dataset_path: str, min_turns: int = 4) -> str:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parts = []
    session_idx = 0
    for item in data:
        conversations = item.get("conversations") or []
        if len(conversations) < min_turns:
            continue
        session_idx += 1
        parts.append(format_session_text(conversations, session_idx))
        if sum(len(part) for part in parts) >= 120000:
            break
    if not parts:
        raise RuntimeError("No multi-turn ShareGPT sessions found")
    return "".join(parts)


def build_exact_prompt_from_token_ids(
    session: requests.Session,
    base_url: str,
    model: str,
    source_text: str,
    target_tokens: int,
) -> Optional[str]:
    token_count, token_ids = tokenize_text(session, base_url, model, source_text)
    if token_ids is None or token_count < target_tokens:
        return None

    # Detokenize/tokenize is not always perfectly round-trippable. Search nearby
    # prefixes and only accept a prompt that validates to the exact target count.
    lower = max(1, target_tokens - 32)
    upper = min(token_count, target_tokens + 32)
    for prefix_len in range(target_tokens, lower - 1, -1):
        prompt = detokenize_ids(session, base_url, model, token_ids[:prefix_len])
        if not prompt:
            continue
        prompt_tokens, _ = tokenize_text(session, base_url, model, prompt)
        if prompt_tokens == target_tokens:
            return prompt

    for prefix_len in range(target_tokens + 1, upper + 1):
        prompt = detokenize_ids(session, base_url, model, token_ids[:prefix_len])
        if not prompt:
            continue
        prompt_tokens, _ = tokenize_text(session, base_url, model, prompt)
        if prompt_tokens == target_tokens:
            return prompt

    return None


def build_prompt_by_binary_search(
    session: requests.Session,
    base_url: str,
    model: str,
    source_text: str,
    target_tokens: int,
) -> Tuple[str, int]:
    low = 1
    high = len(source_text)
    best_text = source_text[:1]
    best_count = 0

    while low <= high:
        mid = (low + high) // 2
        candidate = source_text[:mid]
        count, _ = tokenize_text(session, base_url, model, candidate)
        if count <= target_tokens:
            if count >= best_count:
                best_text = candidate
                best_count = count
            low = mid + 1
        else:
            high = mid - 1

    if best_count == target_tokens:
        return best_text, best_count

    fillers = [
        " a", " the", " and", ".", ",", "\n", "\n\n", " of", " to", " in", " with",
        " hello", " world", " test", " data", " 0", " 1", " 2", " 3", " 4",
    ]
    current = best_text
    current_count = best_count

    for _ in range(32):
        remaining = target_tokens - current_count
        if remaining <= 0:
            break
        progressed = False
        for filler in fillers:
            count, _ = tokenize_text(session, base_url, model, current + filler)
            delta = count - current_count
            if delta <= 0 or delta > remaining:
                continue
            current += filler
            current_count = count
            progressed = True
            if current_count == target_tokens:
                return current, current_count
            break
        if not progressed:
            break
    return current, current_count


def ensure_prompt(
    session: requests.Session,
    base_url: str,
    model: str,
    dataset_path: str,
    prompt_cache_path: str,
    prompt_meta_path: str,
    target_tokens: int,
) -> Tuple[str, int]:
    if os.path.exists(prompt_cache_path) and os.path.exists(prompt_meta_path):
        with open(prompt_cache_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        count, _ = tokenize_text(session, base_url, model, prompt)
        if count == target_tokens:
            return prompt, count

    source_text = load_multi_turn_corpus(dataset_path)
    prompt = build_exact_prompt_from_token_ids(session, base_url, model, source_text, target_tokens)
    if prompt is None:
        prompt, prompt_tokens = build_prompt_by_binary_search(
            session, base_url, model, source_text, target_tokens
        )
        if prompt_tokens != target_tokens:
            raise RuntimeError(
                f"Could not construct exact {target_tokens}-token prompt; nearest prompt has {prompt_tokens} tokens"
            )
    else:
        prompt_tokens, _ = tokenize_text(session, base_url, model, prompt)
    prompt_tokens, _ = tokenize_text(session, base_url, model, prompt)
    if prompt_tokens != target_tokens:
        raise RuntimeError(
            f"Prompt cache validation failed: expected {target_tokens} tokens, got {prompt_tokens}"
        )

    os.makedirs(os.path.dirname(prompt_cache_path), exist_ok=True)
    with open(prompt_cache_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(prompt_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model,
                "target_tokens": target_tokens,
                "created_at": utc_now(),
                "dataset_path": os.path.abspath(dataset_path),
            },
            f,
            indent=2,
        )
    return prompt, prompt_tokens


def parse_sse_json(line: str) -> Optional[Dict]:
    if not line.startswith("data: "):
        return None
    payload = line[6:].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def percentile_95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = 0.95 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def stream_completion(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    min_tokens: int,
) -> Dict[str, object]:
    url = base_url + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "ignore_eos": True,
        "stream": True,
    }
    start_ts = time.monotonic()
    response = session.post(url, json=payload, stream=True, timeout=3600)
    response.raise_for_status()

    first_token_ts = None
    prev_text_ts = None
    chunk_timings: List[Tuple[str, float]] = []
    completion_parts: List[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        message = parse_sse_json(raw_line)
        if not message:
            continue
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

    completion_text = "".join(completion_parts)
    return {
        "completion_text": completion_text,
        "ttft_ms": (first_token_ts - start_ts) * 1000.0,
        "ttlt_ms": (end_ts - start_ts) * 1000.0,
        "chunk_timings": chunk_timings,
    }


def derive_tbt_p95_ms(
    session: requests.Session,
    base_url: str,
    model: str,
    completion_text: str,
    chunk_timings: List[Tuple[str, float]],
) -> Tuple[Optional[float], int]:
    if not completion_text:
        return None, 0

    completion_tokens, _ = tokenize_text(session, base_url, model, completion_text)
    if completion_tokens <= 0:
        return None, 0

    if not chunk_timings:
        return None, completion_tokens

    per_token_gaps: List[float] = []
    for chunk_text, gap_ms in chunk_timings:
        chunk_tokens, _ = tokenize_text(session, base_url, model, chunk_text)
        if chunk_tokens <= 0:
            continue
        per_token_gaps.extend([gap_ms / chunk_tokens] * chunk_tokens)

    if not per_token_gaps:
        return None, completion_tokens

    return percentile_95(per_token_gaps), completion_tokens


def mean_or_blank(rows: List[Dict[str, str]], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    if not values:
        return ""
    return str(round(statistics.mean(values), 4))


def summarize(raw_csv_path: str, summary_csv_path: str) -> None:
    rows: List[Dict[str, str]] = []
    with open(raw_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["num_gpu_blocks_override"], []).append(row)

    fieldnames = [
        "num_gpu_blocks_override",
        "successful_runs",
        "failed_runs",
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
        for block_value in sorted(grouped, key=lambda x: int(x)):
            block_rows = grouped[block_value]
            success_rows = [row for row in block_rows if row["status"] == "success"]
            failed_runs = len(block_rows) - len(success_rows)
            if success_rows:
                writer.writerow(
                    {
                        "num_gpu_blocks_override": block_value,
                        "successful_runs": len(success_rows),
                        "failed_runs": failed_runs,
                        "avg_prompt_tokens": mean_or_blank(success_rows, "prompt_tokens"),
                        "avg_completion_tokens": mean_or_blank(success_rows, "completion_tokens"),
                        "avg_ttft_ms": mean_or_blank(success_rows, "ttft_ms"),
                        "avg_p95_tbt_ms": mean_or_blank(success_rows, "p95_tbt_ms"),
                        "avg_ttlt_ms": mean_or_blank(success_rows, "ttlt_ms"),
                    }
                )
            else:
                writer.writerow(
                    {
                        "num_gpu_blocks_override": block_value,
                        "successful_runs": 0,
                        "failed_runs": failed_runs,
                        "avg_prompt_tokens": "",
                        "avg_completion_tokens": "",
                        "avg_ttft_ms": "",
                        "avg_p95_tbt_ms": "",
                        "avg_ttlt_ms": "",
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--prompt-meta", required=True)
    parser.add_argument("--num-gpu-blocks-override", required=True, type=int)
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument("--target-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-tokens", type=int, default=4000)
    parser.add_argument("--pre-request-sleep-sec", type=int, default=180)
    args = parser.parse_args()

    raw_fieldnames = [
        "timestamp_utc",
        "num_gpu_blocks_override",
        "run_index",
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

    session = requests.Session()
    result_row: Dict[str, object] = {
        "timestamp_utc": utc_now(),
        "num_gpu_blocks_override": args.num_gpu_blocks_override,
        "run_index": args.run_index,
        "base_url": args.base_url,
        "status": "failed",
        "error_message": "",
        "prompt_tokens": "",
        "completion_tokens": "",
        "ttft_ms": "",
        "p95_tbt_ms": "",
        "ttlt_ms": "",
        "run_log": os.path.abspath(args.run_log),
    }

    try:
        print(
            f"[INFO] num_gpu_blocks_override={args.num_gpu_blocks_override} "
            f"run={args.run_index}: sleeping {args.pre_request_sleep_sec}s before request",
            flush=True,
        )
        time.sleep(args.pre_request_sleep_sec)

        model = get_model_id(session, args.base_url)
        prompt, prompt_tokens = ensure_prompt(
            session,
            args.base_url,
            model,
            args.dataset_path,
            args.prompt_cache,
            args.prompt_meta,
            args.target_prompt_tokens,
        )
        print(
            f"[INFO] num_gpu_blocks_override={args.num_gpu_blocks_override} "
            f"run={args.run_index}: sending request with {prompt_tokens} prompt tokens",
            flush=True,
        )
        stream_result = stream_completion(
            session,
            args.base_url,
            model,
            prompt,
            args.max_tokens,
            args.min_tokens,
        )
        p95_tbt_ms, completion_tokens = derive_tbt_p95_ms(
            session,
            args.base_url,
            model,
            stream_result["completion_text"],
            stream_result["chunk_timings"],
        )

        result_row.update(
            {
                "status": "success",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ttft_ms": round(float(stream_result["ttft_ms"]), 4),
                "p95_tbt_ms": "" if p95_tbt_ms is None else round(float(p95_tbt_ms), 4),
                "ttlt_ms": round(float(stream_result["ttlt_ms"]), 4),
            }
        )
        print(
            f"[RESULT] num_gpu_blocks_override={args.num_gpu_blocks_override} "
            f"run={args.run_index}: input_tokens={prompt_tokens} "
            f"output_tokens={completion_tokens} "
            f"ttft_ms={result_row['ttft_ms']} "
            f"p95_tbt_ms={result_row['p95_tbt_ms']} "
            f"ttlt_ms={result_row['ttlt_ms']}",
            flush=True,
        )
    except Exception as exc:
        result_row["error_message"] = str(exc)
        print(
            f"[FAIL] num_gpu_blocks_override={args.num_gpu_blocks_override} run={args.run_index}: {exc}",
            file=sys.stderr,
        )
    finally:
        append_csv_row(args.raw_csv, raw_fieldnames, result_row)
        summarize(args.raw_csv, args.summary_csv)

    return 0 if result_row["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
