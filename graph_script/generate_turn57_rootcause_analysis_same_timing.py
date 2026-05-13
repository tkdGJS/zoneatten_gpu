#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/yuhwa2323/zoneatten")
OUT_DIR = ROOT / "analysis_turn57_rootcause_same_timing"
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
TENANT_COUNT = "32"
TURN_WINDOW = [5, 6, 7]

GROUP_LABELS = {
    "8192": "Group1",
    "2048": "Group2",
}
GROUP_COLORS = {
    "8192": "#2563eb",
    "2048": "#f97316",
}
TURN_COLORS = {
    5: "#16a34a",
    6: "#eab308",
    7: "#dc2626",
}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def corr(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / ((vx * vy) ** 0.5)


def find_model_config():
    cache_root = Path("/home/yuhwa2323/.cache/huggingface/hub")
    matches = list(cache_root.glob("models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/config.json"))
    if not matches:
        return None
    return matches[0]


def load_model_info():
    config_path = find_model_config()
    if config_path is None:
        raise SystemExit("Could not find local model config for meta-llama/Llama-3.2-1B-Instruct")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_layers = int(config["num_hidden_layers"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    bytes_per_element = 2  # vLLM log shows bf16 weights cast to fp16 in this setup.
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
    block_size_tokens = 16
    kv_bytes_per_block = kv_bytes_per_token * block_size_tokens

    snapshot_root = config_path.parent
    model_bytes_on_disk = 0
    for path in snapshot_root.rglob("*"):
        if path.is_file():
            model_bytes_on_disk += path.stat().st_size

    return {
        "config_path": config_path,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": int(config["hidden_size"]),
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_bytes_per_block": kv_bytes_per_block,
        "block_size_tokens": block_size_tokens,
        "model_bytes_on_disk": model_bytes_on_disk,
    }


def mib_from_tokens(token_count, kv_bytes_per_token):
    return token_count * kv_bytes_per_token / (1024.0 * 1024.0)


def gib_from_bytes(byte_count):
    return byte_count / (1024.0 * 1024.0 * 1024.0)


def load_request_metrics(exp_dir):
    metrics = {}
    metrics_dir = ROOT / exp_dir / "vram_only_artifacts_smallctx_mixed_limits" / "request_metrics"
    for path in metrics_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                metrics[payload["request_id"]] = payload
    return metrics


def load_rows(model_info):
    rows = []
    kv_bytes_per_token = model_info["kv_bytes_per_token"]
    for exp_dir, exp_label in EXPERIMENTS:
        request_metrics = load_request_metrics(exp_dir)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success":
                    continue
                if row["tenant_count"] != TENANT_COUNT:
                    continue
                metric = request_metrics.get(row["metrics_request_id"])
                if metric is None:
                    continue

                input_tokens = float(row["input_tokens"] or 0.0)
                output_tokens = float(row["output_tokens"] or 0.0)
                kv_history_tokens = float(row["kv_history_tokens"] or 0.0)
                prefix_hit_tokens = float(row["prefix_hit_tokens"] or 0.0)
                blocking_ms = float(row["blocking_time_ms"] or 0.0)
                ttft_ms = float(row["ttft_ms"] or 0.0)
                prefill_ms = 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0)
                decode_ms = 1000.0 * float(metric.get("decode_time_s", 0.0) or 0.0)
                computation_tokens = max(0.0, input_tokens - prefix_hit_tokens)

                rows.append(
                    {
                        "exp_dir": exp_dir,
                        "exp_label": exp_label,
                        "turn_index": int(row["turn_index"]),
                        "tenant_id": int(row["tenant_id"]),
                        "group": row["history_limit_tokens"],
                        "group_label": GROUP_LABELS.get(row["history_limit_tokens"], row["history_limit_tokens"]),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "kv_history_tokens": kv_history_tokens,
                        "prefix_hit_tokens": prefix_hit_tokens,
                        "prefix_hit_rate": (prefix_hit_tokens / input_tokens) if input_tokens > 0 else 0.0,
                        "computation_tokens": computation_tokens,
                        "blocking_ms": blocking_ms,
                        "prefill_ms": prefill_ms,
                        "ttft_ms": ttft_ms,
                        "remaining_ttft_ms": max(0.0, ttft_ms - blocking_ms - prefill_ms),
                        "decode_ms": decode_ms,
                        "resident_kv_mib": mib_from_tokens(kv_history_tokens, kv_bytes_per_token),
                        "request_kv_mib": mib_from_tokens(input_tokens, kv_bytes_per_token),
                    }
                )
    return rows


def write_enriched_csv(rows, out_path):
    fieldnames = [
        "exp_label",
        "turn_index",
        "tenant_id",
        "group_label",
        "input_tokens",
        "output_tokens",
        "kv_history_tokens",
        "prefix_hit_tokens",
        "prefix_hit_rate",
        "computation_tokens",
        "resident_kv_mib",
        "request_kv_mib",
        "blocking_ms",
        "prefill_ms",
        "remaining_ttft_ms",
        "ttft_ms",
        "decode_ms",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def aggregate_turn(rows, model_info):
    bucket = defaultdict(list)
    for row in rows:
        bucket[row["turn_index"]].append(row)

    summaries = []
    block_capacity = 2048
    token_capacity = block_capacity * model_info["block_size_tokens"]
    kv_gib_capacity = gib_from_bytes(block_capacity * model_info["kv_bytes_per_block"])
    for turn in sorted(bucket):
        subset = bucket[turn]
        per_exp_input = defaultdict(float)
        per_exp_kv = defaultdict(float)
        for item in subset:
            per_exp_input[item["exp_label"]] += item["input_tokens"]
            per_exp_kv[item["exp_label"]] += item["kv_history_tokens"]
        input_sum = sum(r["input_tokens"] for r in subset)
        kv_sum = sum(r["kv_history_tokens"] for r in subset)
        prefix_sum = sum(r["prefix_hit_tokens"] for r in subset)
        summaries.append(
            {
                "turn_index": turn,
                "sample_count": len(subset),
                "mean_blocking_ms": mean([r["blocking_ms"] for r in subset]),
                "p95_blocking_ms": percentile([r["blocking_ms"] for r in subset], 0.95),
                "p99_blocking_ms": percentile([r["blocking_ms"] for r in subset], 0.99),
                "mean_prefill_ms": mean([r["prefill_ms"] for r in subset]),
                "p95_prefill_ms": percentile([r["prefill_ms"] for r in subset], 0.95),
                "p99_prefill_ms": percentile([r["prefill_ms"] for r in subset], 0.99),
                "mean_remaining_ttft_ms": mean([r["remaining_ttft_ms"] for r in subset]),
                "mean_ttft_ms": mean([r["ttft_ms"] for r in subset]),
                "p95_ttft_ms": percentile([r["ttft_ms"] for r in subset], 0.95),
                "p99_ttft_ms": percentile([r["ttft_ms"] for r in subset], 0.99),
                "mean_input_tokens": mean([r["input_tokens"] for r in subset]),
                "mean_output_tokens": mean([r["output_tokens"] for r in subset]),
                "mean_kv_history_tokens": mean([r["kv_history_tokens"] for r in subset]),
                "mean_computation_tokens": mean([r["computation_tokens"] for r in subset]),
                "prefix_hit_rate": (prefix_sum / input_sum) if input_sum > 0 else 0.0,
                "total_input_tokens": input_sum,
                "total_kv_history_tokens": kv_sum,
                "mean_total_resident_kv_gib_per_exp": mean(
                    [gib_from_bytes(value * model_info["kv_bytes_per_token"]) for value in per_exp_kv.values()]
                ),
                "mean_total_request_kv_gib_per_exp": mean(
                    [gib_from_bytes(value * model_info["kv_bytes_per_token"]) for value in per_exp_input.values()]
                ),
                "resident_blocks_required": kv_sum / model_info["block_size_tokens"],
                "request_blocks_required": input_sum / model_info["block_size_tokens"],
                "kv_block_capacity": block_capacity,
                "kv_token_capacity": token_capacity,
                "kv_gib_capacity": kv_gib_capacity,
            }
        )
    return summaries


def aggregate_turn_group(rows, model_info):
    bucket = defaultdict(list)
    for row in rows:
        bucket[(row["turn_index"], row["group"])].append(row)

    summaries = []
    for (turn_index, group), subset in sorted(bucket.items()):
        per_exp_input = defaultdict(float)
        per_exp_kv = defaultdict(float)
        for item in subset:
            per_exp_input[item["exp_label"]] += item["input_tokens"]
            per_exp_kv[item["exp_label"]] += item["kv_history_tokens"]
        summaries.append(
            {
                "turn_index": turn_index,
                "group": group,
                "group_label": GROUP_LABELS[group],
                "sample_count": len(subset),
                "mean_blocking_ms": mean([r["blocking_ms"] for r in subset]),
                "p95_blocking_ms": percentile([r["blocking_ms"] for r in subset], 0.95),
                "p99_blocking_ms": percentile([r["blocking_ms"] for r in subset], 0.99),
                "mean_prefill_ms": mean([r["prefill_ms"] for r in subset]),
                "p95_prefill_ms": percentile([r["prefill_ms"] for r in subset], 0.95),
                "p99_prefill_ms": percentile([r["prefill_ms"] for r in subset], 0.99),
                "mean_ttft_ms": mean([r["ttft_ms"] for r in subset]),
                "mean_input_tokens": mean([r["input_tokens"] for r in subset]),
                "mean_output_tokens": mean([r["output_tokens"] for r in subset]),
                "mean_kv_history_tokens": mean([r["kv_history_tokens"] for r in subset]),
                "mean_prefix_hit_tokens": mean([r["prefix_hit_tokens"] for r in subset]),
                "mean_prefix_hit_rate": mean([r["prefix_hit_rate"] for r in subset]),
                "mean_total_resident_kv_gib_per_exp": mean(
                    [gib_from_bytes(value * model_info["kv_bytes_per_token"]) for value in per_exp_kv.values()]
                ),
                "mean_total_request_kv_gib_per_exp": mean(
                    [gib_from_bytes(value * model_info["kv_bytes_per_token"]) for value in per_exp_input.values()]
                ),
            }
        )
    return summaries


def write_dict_csv(rows, out_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_log_capacity(exp_dir):
    log_path = ROOT / exp_dir / "vram_only_logs_smallctx_mixed_limits" / "block_2048_tenant_32_run_1.log"
    info = {
        "available_kv_cache_memory_gib": None,
        "overridden_num_gpu_blocks": None,
        "gpu_kv_cache_tokens": None,
        "maximum_concurrency_for_12288_tokens": None,
    }
    if not log_path.exists():
        return info
    text = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in text:
        if "Available KV cache memory:" in line:
            info["available_kv_cache_memory_gib"] = float(line.rsplit("Available KV cache memory:", 1)[1].split("GiB")[0].strip())
        elif "Overriding num_gpu_blocks=" in line and "num_gpu_blocks_override=" in line:
            info["overridden_num_gpu_blocks"] = int(line.rsplit("num_gpu_blocks_override=", 1)[1].strip())
        elif "GPU KV cache size:" in line and "tokens" in line:
            info["gpu_kv_cache_tokens"] = int(line.rsplit("GPU KV cache size:", 1)[1].split("tokens")[0].replace(",", "").strip())
        elif "Maximum concurrency for 12,288 tokens per request:" in line:
            value = line.rsplit("Maximum concurrency for 12,288 tokens per request:", 1)[1].split("x")[0].strip()
            info["maximum_concurrency_for_12288_tokens"] = float(value)
    return info


def write_log_capacity_csv(out_path):
    rows = []
    for exp_dir, exp_label in EXPERIMENTS:
        item = parse_log_capacity(exp_dir)
        item["exp_label"] = exp_label
        rows.append(item)
    rows.sort(key=lambda row: row["exp_label"])
    write_dict_csv(rows, out_path)


def plot_turn_transition(plt, turn_summary, model_info, out_path):
    turns = [row["turn_index"] for row in turn_summary]
    mean_blocking = [row["mean_blocking_ms"] for row in turn_summary]
    mean_prefill = [row["mean_prefill_ms"] for row in turn_summary]
    mean_ttft = [row["mean_ttft_ms"] for row in turn_summary]
    total_resident = [row["mean_total_resident_kv_gib_per_exp"] for row in turn_summary]
    total_request = [row["mean_total_request_kv_gib_per_exp"] for row in turn_summary]
    prefix_rates = [100.0 * row["prefix_hit_rate"] for row in turn_summary]
    comp_tokens = [row["mean_computation_tokens"] for row in turn_summary]
    kv_capacity = turn_summary[0]["kv_gib_capacity"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)

    axes[0].plot(turns, mean_blocking, marker="o", color="#2563eb", label="Mean Blocking")
    axes[0].plot(turns, mean_prefill, marker="o", color="#84cc16", label="Mean Prefill")
    axes[0].plot(turns, mean_ttft, marker="o", color="#dc2626", label="Mean TTFT")
    axes[0].axvspan(5.5, 6.5, color="#fef3c7", alpha=0.5)
    axes[0].axvspan(6.5, 7.5, color="#fee2e2", alpha=0.45)
    axes[0].set_title("exp1+exp2+exp3, 32 tenants: turn-by-turn latency transition")
    axes[0].set_xlabel("Turn")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_xticks(turns)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(turns, total_resident, marker="o", color="#1d4ed8", label="Resident KV before request")
    axes[1].plot(turns, total_request, marker="o", color="#ea580c", label="Prompt KV after prefill")
    axes[1].axhline(kv_capacity, color="#111827", linestyle="--", label=f"2048 blocks capacity ({kv_capacity:.2f} GiB)")
    axes[1].axvspan(5.5, 6.5, color="#fef3c7", alpha=0.5)
    axes[1].axvspan(6.5, 7.5, color="#fee2e2", alpha=0.45)
    axes[1].set_xlabel("Turn")
    axes[1].set_ylabel("Estimated KV footprint (GiB)")
    axes[1].set_xticks(turns)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left")

    ax2b = axes[2].twinx()
    axes[2].plot(turns, prefix_rates, marker="o", color="#7c3aed", label="Prefix hit rate")
    ax2b.plot(turns, comp_tokens, marker="o", color="#0f766e", label="Mean computation tokens")
    axes[2].axvspan(5.5, 6.5, color="#fef3c7", alpha=0.5)
    axes[2].axvspan(6.5, 7.5, color="#fee2e2", alpha=0.45)
    axes[2].set_xlabel("Turn")
    axes[2].set_ylabel("Prefix hit rate (%)", color="#7c3aed")
    ax2b.set_ylabel("Mean computation tokens", color="#0f766e")
    axes[2].set_xticks(turns)
    axes[2].grid(axis="y", alpha=0.25)
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prefill_tail(plt, turn_summary, out_path):
    turns = [row["turn_index"] for row in turn_summary]
    p50 = [row["mean_prefill_ms"] for row in turn_summary]
    p95 = [row["p95_prefill_ms"] for row in turn_summary]
    p99 = [row["p99_prefill_ms"] for row in turn_summary]

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(turns, p50, marker="o", color="#84cc16", label="Prefill mean")
    ax.plot(turns, p95, marker="o", color="#f59e0b", label="Prefill p95")
    ax.plot(turns, p99, marker="o", color="#dc2626", label="Prefill p99")
    ax.axvspan(5.5, 6.5, color="#fef3c7", alpha=0.5)
    ax.axvspan(6.5, 7.5, color="#fee2e2", alpha=0.45)
    ax.set_title("exp1+exp2+exp3, 32 tenants: prefill tail growth by turn")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Prefill time (ms)")
    ax.set_xticks(turns)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_turn_window_group_panels(plt, group_summary, out_path):
    filtered = [row for row in group_summary if row["turn_index"] in TURN_WINDOW]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, sharex=True)
    x_positions = list(range(len(TURN_WINDOW)))
    width = 0.35
    metrics = [
        ("mean_blocking_ms", "Mean Blocking (ms)"),
        ("mean_prefill_ms", "Mean Prefill (ms)"),
        ("mean_total_resident_kv_gib_per_exp", "Mean Total Resident KV Per Exp (GiB)"),
    ]
    for ax, (metric_key, title) in zip(axes, metrics):
        for offset, group in [(-width / 2, "8192"), (width / 2, "2048")]:
            values = []
            for turn in TURN_WINDOW:
                match = next(row for row in filtered if row["turn_index"] == turn and row["group"] == group)
                values.append(match[metric_key])
            ax.bar(
                [x + offset for x in x_positions],
                values,
                width=width,
                color=GROUP_COLORS[group],
                label=GROUP_LABELS[group],
                alpha=0.85,
            )
        ax.set_title(title)
        ax.set_xticks(x_positions, [f"Turn {turn}" for turn in TURN_WINDOW])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_request_relationships(plt, rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    x_keys = [
        ("resident_kv_mib", "Estimated resident KV per tenant (MiB)"),
        ("request_kv_mib", "Estimated prompt KV per tenant (MiB)"),
        ("input_tokens", "Input tokens"),
        ("output_tokens", "Output tokens"),
    ]
    y_values = [row["ttft_ms"] for row in rows]
    for ax, (x_key, x_label) in zip(axes.flatten(), x_keys):
        for turn in TURN_WINDOW:
            subset = [row for row in rows if row["turn_index"] == turn]
            ax.scatter(
                [row[x_key] for row in subset],
                [row["ttft_ms"] for row in subset],
                s=32,
                alpha=0.6,
                color=TURN_COLORS[turn],
                label=f"Turn {turn} (n={len(subset)})",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("TTFT (ms)")
        ax.grid(alpha=0.2)
    axes[0][0].legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_turn5_turn7_per_request(plt, rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharey=True)
    for ax, turn in zip(axes, [5, 7]):
        subset = [row for row in rows if row["turn_index"] == turn]
        for group in ["8192", "2048"]:
            group_rows = [row for row in subset if row["group"] == group]
            ax.scatter(
                [row["resident_kv_mib"] for row in group_rows],
                [row["ttft_ms"] for row in group_rows],
                s=[max(30.0, row["input_tokens"] * 0.04) for row in group_rows],
                alpha=0.6,
                color=GROUP_COLORS[group],
                label=f"{GROUP_LABELS[group]} (n={len(group_rows)})",
            )
        ax.set_title(f"Turn {turn}")
        ax.set_xlabel("Estimated resident KV per tenant (MiB)")
        ax.set_ylabel("TTFT (ms)")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(rows, turn_summary, group_summary, model_info, out_path):
    log_items = [parse_log_capacity(exp_dir) for exp_dir, _ in EXPERIMENTS]
    available_kv_values = [item["available_kv_cache_memory_gib"] for item in log_items if item["available_kv_cache_memory_gib"]]
    gpu_kv_tokens_values = [item["gpu_kv_cache_tokens"] for item in log_items if item["gpu_kv_cache_tokens"]]
    block_override_values = [item["overridden_num_gpu_blocks"] for item in log_items if item["overridden_num_gpu_blocks"]]
    max_concurrency_values = [
        item["maximum_concurrency_for_12288_tokens"] for item in log_items if item["maximum_concurrency_for_12288_tokens"]
    ]

    turn_map = {row["turn_index"]: row for row in turn_summary}
    turn5 = turn_map[5]
    turn6 = turn_map[6]
    turn7 = turn_map[7]

    corr_input_ttft = corr([row["input_tokens"] for row in rows], [row["ttft_ms"] for row in rows])
    corr_output_ttft = corr([row["output_tokens"] for row in rows], [row["ttft_ms"] for row in rows])
    corr_kv_ttft = corr([row["resident_kv_mib"] for row in rows], [row["ttft_ms"] for row in rows])
    corr_kv_prefill = corr([row["resident_kv_mib"] for row in rows], [row["prefill_ms"] for row in rows])

    lines = []
    lines.append("# Turn 5/6/7 Root Cause Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append("- Slice: `tenant_count=32` only")
    lines.append("- Samples: 3 experiments x 32 tenants x 10 turns = 960 successful requests")
    lines.append("- Group1: `history_limit_tokens=8192`")
    lines.append("- Group2: `history_limit_tokens=2048`")
    lines.append("")
    lines.append("## What Happened At Turn 5, Turn 6, And Turn 7")
    lines.append("")
    lines.append(
        f"- `turn 5` itself is not the real queueing onset. Combined mean blocking at turn 5 is `{turn5['mean_blocking_ms']:.1f} ms`, still effectively zero."
    )
    lines.append(
        f"- The first structural queueing starts at `turn 6`: mean blocking rises to `{turn6['mean_blocking_ms']:.1f} ms` while mean resident KV per experiment reaches `{turn6['mean_total_resident_kv_gib_per_exp']:.3f} GiB`."
    )
    lines.append(
        f"- By `turn 7`, mean blocking reaches `{turn7['mean_blocking_ms']:.1f} ms`, mean prefill reaches `{turn7['mean_prefill_ms']:.1f} ms`, and mean TTFT reaches `{turn7['mean_ttft_ms']:.1f} ms`."
    )
    lines.append(
        f"- Prefix reuse collapses across the same transition: global prefix-hit rate drops from `{100.0 * turn5['prefix_hit_rate']:.1f}%` at turn 5 to `{100.0 * turn6['prefix_hit_rate']:.1f}%` at turn 6 and `{100.0 * turn7['prefix_hit_rate']:.1f}%` at turn 7."
    )
    lines.append(
        f"- Tail prefill also starts to break at `turn 6`: p99 prefill is `{turn5['p99_prefill_ms']:.1f} ms` at turn 5, `{turn6['p99_prefill_ms']:.1f} ms` at turn 6, and `{turn7['p99_prefill_ms']:.1f} ms` at turn 7."
    )
    lines.append("")
    lines.append("## vLLM Capacity Evidence")
    lines.append("")
    if available_kv_values:
        lines.append(
            f"- vLLM log reports raw available KV cache memory around `{mean(available_kv_values):.2f} GiB` before applying the manual block override."
        )
    if block_override_values:
        lines.append(f"- The run forces `num_gpu_blocks_override={int(mean(block_override_values))}`.")
    if gpu_kv_tokens_values:
        lines.append(f"- vLLM log reports `GPU KV cache size = {int(mean(gpu_kv_tokens_values)):,} tokens` after the override.")
    if max_concurrency_values:
        lines.append(
            f"- vLLM log reports `maximum concurrency for 12,288 tokens/request = {mean(max_concurrency_values):.2f}x`, which is far below the 32 synchronized tenants used here."
        )
    lines.append(
        f"- From the local model config, KV cache cost is `2 x layers x KV heads x head_dim x bytes = {model_info['kv_bytes_per_token']:,} bytes/token`, about `{model_info['kv_bytes_per_token'] / 1024.0:.1f} KiB/token`."
    )
    lines.append(
        f"- With `block_size={model_info['block_size_tokens']}`, one GPU block is `{model_info['kv_bytes_per_block'] / 1024.0:.1f} KiB` and `2048` blocks correspond to about `{gib_from_bytes(2048 * model_info['kv_bytes_per_block']):.3f} GiB` of active KV capacity."
    )
    lines.append("")
    lines.append("## Capacity Crossing")
    lines.append("")
    lines.append(
        f"- Mean total resident KV before prefill is `{turn5['mean_total_resident_kv_gib_per_exp']:.3f} GiB` at turn 5, `{turn6['mean_total_resident_kv_gib_per_exp']:.3f} GiB` at turn 6, and `{turn7['mean_total_resident_kv_gib_per_exp']:.3f} GiB` at turn 7."
    )
    lines.append(
        f"- This means the system is still below the effective `~1.0 GiB` block budget at turn 5, sits right on the boundary at turn 6, and exceeds it clearly at turn 7."
    )
    lines.append(
        f"- Mean total prompt KV after prefill is even larger: `{turn5['mean_total_request_kv_gib_per_exp']:.3f} GiB` at turn 5 and `{turn7['mean_total_request_kv_gib_per_exp']:.3f} GiB` at turn 7."
    )
    lines.append("")
    lines.append("## Group-Level Difference At Turn 7")
    lines.append("")
    turn7_groups = [row for row in group_summary if row["turn_index"] == 7]
    for row in sorted(turn7_groups, key=lambda item: item["group"]):
        lines.append(
            f"- {row['group_label']}: mean input `{row['mean_input_tokens']:.1f}`, mean prefix-hit `{row['mean_prefix_hit_tokens']:.1f}`, mean blocking `{row['mean_blocking_ms']:.1f} ms`, "
            f"mean prefill `{row['mean_prefill_ms']:.1f} ms`, p99 prefill `{row['p99_prefill_ms']:.1f} ms`, mean resident KV per experiment `{row['mean_total_resident_kv_gib_per_exp']:.3f} GiB`."
        )
    lines.append(
        "- Group2 carries less resident history per tenant, but it also loses more reusable prefix. That is why its prompt footprint can be smaller while its prefill pressure still remains high."
    )
    lines.append("")
    lines.append("## TTFT Relationship")
    lines.append("")
    lines.append(f"- Correlation, estimated resident KV per tenant vs TTFT: `{corr_kv_ttft:.3f}`")
    lines.append(f"- Correlation, estimated resident KV per tenant vs prefill: `{corr_kv_prefill:.3f}`")
    lines.append(f"- Correlation, input tokens vs TTFT: `{corr_input_ttft:.3f}`")
    lines.append(f"- Correlation, output tokens vs TTFT: `{corr_output_ttft:.3f}`")
    lines.append(
        "- Output tokens have weak correlation because TTFT is dominated by queueing plus prefill, not by decode length."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The turn-7 prefill jump is not best explained by VRAM shortage alone. It is better explained as `effective KV-capacity saturation + lower prefix reuse + 32-way synchronized arrival`, which forces more real prompt computation into the same prefill window."
    )
    lines.append(
        "- Once the resident KV sum approaches the 2048-block budget, the scheduler cannot admit all 32 requests into active execution immediately. That shows up first as blocking at turn 6."
    )
    lines.append(
        "- After that, prefix-hit quality drops sharply, so each admitted request also carries more computation tokens. That pushes prefill time up at turn 7."
    )
    lines.append(
        "- The combined effect is `blocking increase first, then prefill increase on top`, which matches the measured TTFT breakdown."
    )
    lines.append("")
    lines.append("## Measurement Limits")
    lines.append("")
    lines.append(
        "- This repository does not contain per-request `nvidia-smi` samples or allocator traces. Per-tenant VRAM numbers in the CSV are therefore estimates derived from KV token footprint, not direct GPU telemetry."
    )
    lines.append(
        f"- The local model snapshot size on disk is about `{gib_from_bytes(model_info['model_bytes_on_disk']):.3f} GiB`, but runtime VRAM for weights/activations can differ from on-disk size."
    )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_info = load_model_info()
    rows = load_rows(model_info)
    turn_summary = aggregate_turn(rows, model_info)
    group_summary = aggregate_turn_group(rows, model_info)

    write_enriched_csv(rows, OUT_DIR / "exp123_32tenant_enriched_requests.csv")
    write_dict_csv(turn_summary, OUT_DIR / "exp123_32tenant_turn_summary.csv")
    write_dict_csv(group_summary, OUT_DIR / "exp123_32tenant_turn_group_summary.csv")
    write_log_capacity_csv(OUT_DIR / "exp123_32tenant_vllm_capacity_from_logs.csv")

    plot_turn_transition(plt, turn_summary, model_info, OUT_DIR / "exp123_32tenant_turn_transition_metrics.png")
    plot_prefill_tail(plt, turn_summary, OUT_DIR / "exp123_32tenant_prefill_tail_by_turn.png")
    plot_turn_window_group_panels(plt, group_summary, OUT_DIR / "exp123_32tenant_turn5_6_7_group_panels.png")
    plot_request_relationships(plt, rows, OUT_DIR / "exp123_32tenant_ttft_relationships_turn5_6_7.png")
    plot_turn5_turn7_per_request(plt, rows, OUT_DIR / "exp123_32tenant_turn5_vs_turn7_resident_kv_ttft.png")
    write_markdown(rows, turn_summary, group_summary, model_info, OUT_DIR / "turn57_rootcause_analysis.md")


if __name__ == "__main__":
    main()
