#!/usr/bin/env python3
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/yuhwa2323/zoneatten")
OUT_DIR = ROOT / "analysis_prefill_cost_drivers_same_timing"
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
TENANT_COUNT = "32"
GROUP_LABELS = {
    "8192": "Group1",
    "2048": "Group2",
}
TURN_COLORS = {
    5: "#16a34a",
    6: "#eab308",
    7: "#dc2626",
}
BATCH_BIN_COLORS = {
    "Low": "#93c5fd",
    "High": "#2563eb",
    "Extra": "#1e3a8a",
}
MAX_NUM_BATCHED_TOKENS = 8192.0
BASELINE_SLOPE_MS_PER_TOKEN = 1.0 / 6.0


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
        raise SystemExit("Could not find local model config for meta-llama/Llama-3.2-1B-Instruct")
    return matches[0]


def load_model_info():
    config_path = find_model_config()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_layers = int(config["num_hidden_layers"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    bytes_per_element = 2
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
    return {"kv_bytes_per_token": kv_bytes_per_token}


def mib_from_tokens(token_count, kv_bytes_per_token):
    return token_count * kv_bytes_per_token / (1024.0 * 1024.0)


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
    kv_bytes_per_token = model_info["kv_bytes_per_token"]
    rows = []
    for exp_dir, exp_label in EXPERIMENTS:
        metrics = load_request_metrics(exp_dir)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success" or row["tenant_count"] != TENANT_COUNT:
                    continue
                metric = metrics.get(row["metrics_request_id"])
                if metric is None:
                    continue
                input_tokens = float(row["input_tokens"] or 0.0)
                prefix_hit_tokens = float(row["prefix_hit_tokens"] or 0.0)
                kv_history_tokens = float(row["kv_history_tokens"] or 0.0)
                prefill_ms = 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0)
                computation_tokens = max(0.0, input_tokens - prefix_hit_tokens)
                baseline_prefill_ms = computation_tokens * BASELINE_SLOPE_MS_PER_TOKEN
                rows.append(
                    {
                        "exp_label": exp_label,
                        "turn_index": int(row["turn_index"]),
                        "tenant_id": int(row["tenant_id"]),
                        "group_label": GROUP_LABELS[row["history_limit_tokens"]],
                        "input_tokens": input_tokens,
                        "prefix_hit_tokens": prefix_hit_tokens,
                        "computation_tokens": computation_tokens,
                        "kv_history_tokens": kv_history_tokens,
                        "resident_kv_mib": mib_from_tokens(kv_history_tokens, kv_bytes_per_token),
                        "prefill_ms": prefill_ms,
                        "baseline_prefill_ms": baseline_prefill_ms,
                        "prefill_residual_ms": prefill_ms - baseline_prefill_ms,
                        "prefill_cost_per_token": (prefill_ms / computation_tokens) if computation_tokens > 0 else 0.0,
                    }
                )
    return rows


def enrich_batch_context(rows):
    bucket = defaultdict(list)
    for row in rows:
        bucket[(row["exp_label"], row["turn_index"])].append(row)

    for key, subset in bucket.items():
        total_input = sum(r["input_tokens"] for r in subset)
        total_computation = sum(r["computation_tokens"] for r in subset)
        total_kv = sum(r["resident_kv_mib"] for r in subset)
        total_group1 = sum(r["resident_kv_mib"] for r in subset if r["group_label"] == "Group1")
        total_group2 = sum(r["resident_kv_mib"] for r in subset if r["group_label"] == "Group2")
        est_rounds = math.ceil(total_computation / MAX_NUM_BATCHED_TOKENS) if total_computation > 0 else 0
        for row in subset:
            row["batch_total_input_tokens"] = total_input
            row["batch_total_computation_tokens"] = total_computation
            row["batch_total_resident_kv_mib"] = total_kv
            row["batch_group1_resident_kv_mib"] = total_group1
            row["batch_group2_resident_kv_mib"] = total_group2
            row["batch_estimated_prefill_rounds"] = est_rounds
            row["peer_mean_resident_kv_mib"] = (total_kv - row["resident_kv_mib"]) / max(1, len(subset) - 1)
            row["peer_mean_computation_tokens"] = (total_computation - row["computation_tokens"]) / max(1, len(subset) - 1)


def assign_pressure_bins(rows):
    for row in rows:
        value = row["batch_total_resident_kv_mib"]
        if value <= 512.0:
            row["batch_pressure_bin"] = "Low"
        elif value <= 1024.0:
            row["batch_pressure_bin"] = "High"
        else:
            row["batch_pressure_bin"] = "Extra"
    return 512.0, 1024.0


def assign_computation_bins(rows):
    values = sorted(row["computation_tokens"] for row in rows if row["turn_index"] in {5, 6, 7})
    q1 = percentile(values, 1.0 / 3.0)
    q2 = percentile(values, 2.0 / 3.0)
    for row in rows:
        value = row["computation_tokens"]
        if value <= q1:
            row["computation_bin"] = "Low compute"
        elif value <= q2:
            row["computation_bin"] = "Mid compute"
        else:
            row["computation_bin"] = "High compute"
    return q1, q2


def write_enriched_csv(rows, out_path):
    fieldnames = [
        "exp_label",
        "turn_index",
        "tenant_id",
        "group_label",
        "input_tokens",
        "prefix_hit_tokens",
        "computation_tokens",
        "kv_history_tokens",
        "resident_kv_mib",
        "prefill_ms",
        "baseline_prefill_ms",
        "prefill_residual_ms",
        "prefill_cost_per_token",
        "batch_total_input_tokens",
        "batch_total_computation_tokens",
        "batch_total_resident_kv_mib",
        "batch_group1_resident_kv_mib",
        "batch_group2_resident_kv_mib",
        "batch_estimated_prefill_rounds",
        "peer_mean_resident_kv_mib",
        "peer_mean_computation_tokens",
        "batch_pressure_bin",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_batch_summary(rows, out_path):
    batch_rows = []
    seen = set()
    for row in rows:
        key = (row["exp_label"], row["turn_index"])
        if key in seen:
            continue
        seen.add(key)
        batch_rows.append(
            {
                "exp_label": row["exp_label"],
                "turn_index": row["turn_index"],
                "batch_total_computation_tokens": row["batch_total_computation_tokens"],
                "batch_total_resident_kv_mib": row["batch_total_resident_kv_mib"],
                "batch_group1_resident_kv_mib": row["batch_group1_resident_kv_mib"],
                "batch_group2_resident_kv_mib": row["batch_group2_resident_kv_mib"],
                "batch_estimated_prefill_rounds": row["batch_estimated_prefill_rounds"],
                "mean_prefill_ms": mean(
                    [
                        item["prefill_ms"]
                        for item in rows
                        if item["exp_label"] == row["exp_label"] and item["turn_index"] == row["turn_index"]
                    ]
                ),
                "mean_prefill_cost_per_token": mean(
                    [
                        item["prefill_cost_per_token"]
                        for item in rows
                        if item["exp_label"] == row["exp_label"]
                        and item["turn_index"] == row["turn_index"]
                        and item["prefill_cost_per_token"] > 0
                    ]
                ),
            }
        )

    fieldnames = list(batch_rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(batch_rows, key=lambda item: (item["exp_label"], item["turn_index"])):
            writer.writerow(row)


def write_summary(rows, q1, q2, out_path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scope",
                "count",
                "mean_prefill_ms",
                "mean_prefill_cost_per_token",
                "mean_prefill_residual_ms",
                "corr_resident_kv_cost_per_token",
                "corr_batch_kv_cost_per_token",
                "corr_batch_compute_cost_per_token",
                "corr_batch_kv_residual",
                "corr_batch_compute_residual",
            ]
        )

        def emit(scope, subset):
            positive = [r for r in subset if r["prefill_cost_per_token"] > 0]
            writer.writerow(
                [
                    scope,
                    len(subset),
                    f"{mean([r['prefill_ms'] for r in subset]):.4f}",
                    f"{mean([r['prefill_cost_per_token'] for r in positive]):.6f}",
                    f"{mean([r['prefill_residual_ms'] for r in subset]):.4f}",
                    f"{corr([r['resident_kv_mib'] for r in positive], [r['prefill_cost_per_token'] for r in positive]):.6f}",
                    f"{corr([r['batch_total_resident_kv_mib'] for r in positive], [r['prefill_cost_per_token'] for r in positive]):.6f}",
                    f"{corr([r['batch_total_computation_tokens'] for r in positive], [r['prefill_cost_per_token'] for r in positive]):.6f}",
                    f"{corr([r['batch_total_resident_kv_mib'] for r in subset], [r['prefill_residual_ms'] for r in subset]):.6f}",
                    f"{corr([r['batch_total_computation_tokens'] for r in subset], [r['prefill_residual_ms'] for r in subset]):.6f}",
                ]
            )

        emit("all", rows)
        focus = [r for r in rows if r["turn_index"] in {5, 6, 7}]
        emit("focus_turns_5_6_7", focus)
        for turn in [5, 6, 7]:
            emit(f"turn={turn}", [r for r in rows if r["turn_index"] == turn])
        for label in ["Low", "High", "Extra"]:
            emit(f"pressure_bin={label}", [r for r in rows if r["batch_pressure_bin"] == label])
        writer.writerow([])
        writer.writerow(["batch_pressure_cutoff_mib", f"q1={q1:.4f}", f"q2={q2:.4f}"])


def plot_cost_vs_resident_kv_by_turn(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    filtered = [r for r in rows if r["turn_index"] in {5, 6, 7} and r["prefill_cost_per_token"] > 0]
    for turn in [5, 6, 7]:
        subset = [r for r in filtered if r["turn_index"] == turn]
        ax.scatter(
            [r["resident_kv_mib"] for r in subset],
            [r["prefill_cost_per_token"] for r in subset],
            s=32,
            alpha=0.65,
            color=TURN_COLORS[turn],
            label=f"Turn {turn} (n={len(subset)})",
        )
    ax.set_title("Prefill cost per token vs resident KV per tenant")
    ax.set_xlabel("Resident KV per tenant (MiB)")
    ax.set_ylabel("Prefill time / computation tokens (ms per token)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cost_vs_batch_pressure(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    filtered = [r for r in rows if r["turn_index"] in {5, 6, 7} and r["prefill_cost_per_token"] > 0]
    for label in ["Low", "High", "Extra"]:
        subset = [r for r in filtered if r["batch_pressure_bin"] == label]
        ax.scatter(
            [r["batch_total_resident_kv_mib"] for r in subset],
            [r["prefill_cost_per_token"] for r in subset],
            s=32,
            alpha=0.6,
            color=BATCH_BIN_COLORS[label],
            label=f"{label} (n={len(subset)})",
        )
    ax.set_title("Prefill cost per token vs batch total resident KV")
    ax.set_xlabel("Batch total resident KV (MiB)")
    ax.set_ylabel("Prefill time / computation tokens (ms per token)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual_vs_batch_compute(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    filtered = [r for r in rows if r["turn_index"] in {5, 6, 7}]
    for turn in [5, 6, 7]:
        subset = [r for r in filtered if r["turn_index"] == turn]
        ax.scatter(
            [r["batch_total_computation_tokens"] for r in subset],
            [r["prefill_residual_ms"] for r in subset],
            s=32,
            alpha=0.65,
            color=TURN_COLORS[turn],
            label=f"Turn {turn} (n={len(subset)})",
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.4, color="#111827")
    ax.set_title("Prefill residual vs batch total computation tokens")
    ax.set_xlabel("Batch total computation tokens")
    ax.set_ylabel("Prefill residual (ms)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_turn_batch_driver_bars(plt, rows, out_path):
    batch_rows = []
    seen = set()
    for row in rows:
        if row["turn_index"] not in {5, 6, 7}:
            continue
        key = (row["exp_label"], row["turn_index"])
        if key in seen:
            continue
        seen.add(key)
        batch_rows.append(row)

    batch_rows.sort(key=lambda r: (r["turn_index"], r["exp_label"]))
    labels = [f"{r['exp_label']}-t{r['turn_index']}" for r in batch_rows]
    x = list(range(len(batch_rows)))
    comp = [r["batch_total_computation_tokens"] for r in batch_rows]
    kv = [r["batch_total_resident_kv_mib"] for r in batch_rows]
    rounds = [r["batch_estimated_prefill_rounds"] for r in batch_rows]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True, sharex=True)
    axes[0].bar(x, comp, color="#0f766e")
    axes[0].set_ylabel("Batch computation tokens")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, kv, color="#2563eb")
    axes[1].set_ylabel("Batch resident KV (MiB)")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, rounds, color="#7c3aed")
    axes[2].set_ylabel("Estimated prefill rounds")
    axes[2].set_xlabel("Experiment-turn batch")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_xticks(x, labels, rotation=45, ha="right")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prefill_box_by_compute_and_pressure(plt, rows, out_path):
    focus = [r for r in rows if r["turn_index"] in {5, 6, 7}]
    compute_bins = ["Low compute", "Mid compute", "High compute"]
    pressure_bins = ["Low", "High", "Extra"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, sharey=True)

    for ax, compute_bin in zip(axes, compute_bins):
        data = []
        labels = []
        colors = []
        for pressure_bin in pressure_bins:
            subset = [
                r["prefill_ms"]
                for r in focus
                if r["computation_bin"] == compute_bin and r["batch_pressure_bin"] == pressure_bin
            ]
            data.append(subset if subset else [0.0])
            labels.append(pressure_bin)
            colors.append(BATCH_BIN_COLORS[pressure_bin])
        bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(compute_bin)
        ax.set_xlabel("Batch pressure")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Prefill time (ms)")
    fig.suptitle("Prefill distribution within the same computation-token bin")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_exp_batch_kv_vs_residual(plt, rows, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, sharey=True)
    for ax, exp_label in zip(axes, [label for _, label in EXPERIMENTS]):
        subset = [r for r in rows if r["turn_index"] in {5, 6, 7} and r["exp_label"] == exp_label]
        for turn in [5, 6, 7]:
            turn_subset = [r for r in subset if r["turn_index"] == turn]
            ax.scatter(
                [r["batch_total_resident_kv_mib"] for r in turn_subset],
                [r["prefill_residual_ms"] for r in turn_subset],
                s=30,
                alpha=0.65,
                color=TURN_COLORS[turn],
                label=f"Turn {turn} (n={len(turn_subset)})",
            )
        ax.set_title(exp_label)
        ax.set_xlabel("Batch total resident KV (MiB)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Prefill residual (ms)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Per-experiment batch total KV vs prefill residual")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prefill_vs_computation_all_with_pressure(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for label in ["Low", "High", "Extra"]:
        subset = [r for r in rows if r["batch_pressure_bin"] == label]
        ax.scatter(
            [r["computation_tokens"] for r in subset],
            [r["prefill_ms"] for r in subset],
            s=24,
            alpha=0.45,
            color=BATCH_BIN_COLORS[label],
            label=f"{label} pressure (n={len(subset)})",
        )
    x_max = max((r["computation_tokens"] for r in rows), default=1.0)
    line_x = [0.0, x_max]
    line_y = [BASELINE_SLOPE_MS_PER_TOKEN * x for x in line_x]
    ax.plot(line_x, line_y, linestyle="--", linewidth=1.5, color="#111827", label="baseline y = x / 6")
    ax.set_title("All turns: prefill vs computation tokens, colored by batch pressure")
    ax.set_xlabel("Computation tokens = input tokens - prefix hit")
    ax.set_ylabel("Prefill time (ms)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prefill_residual_vs_computation_all_with_pressure(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for label in ["Low", "High", "Extra"]:
        subset = [r for r in rows if r["batch_pressure_bin"] == label]
        ax.scatter(
            [r["computation_tokens"] for r in subset],
            [r["prefill_residual_ms"] for r in subset],
            s=24,
            alpha=0.45,
            color=BATCH_BIN_COLORS[label],
            label=f"{label} pressure (n={len(subset)})",
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.5, color="#111827")
    ax.set_title("All turns: prefill residual vs computation tokens, colored by batch pressure")
    ax.set_xlabel("Computation tokens = input tokens - prefix hit")
    ax.set_ylabel("Prefill residual (ms)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prefill_vs_computation_all_by_turn(plt, rows, out_path):
    turns = list(range(1, 11))
    cmap = plt.get_cmap("viridis", len(turns))
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for idx, turn in enumerate(turns):
        subset = [r for r in rows if r["turn_index"] == turn]
        ax.scatter(
            [r["computation_tokens"] for r in subset],
            [r["prefill_ms"] for r in subset],
            s=20,
            alpha=0.42,
            color=cmap(idx),
            label=f"Turn {turn}",
        )
    x_max = max((r["computation_tokens"] for r in rows), default=1.0)
    line_x = [0.0, x_max]
    line_y = [BASELINE_SLOPE_MS_PER_TOKEN * x for x in line_x]
    ax.plot(line_x, line_y, linestyle="--", linewidth=1.5, color="#111827", label="baseline y = x / 6")
    ax.set_title("All turns: prefill vs computation tokens, colored by turn")
    ax.set_xlabel("Computation tokens = input tokens - prefix hit")
    ax.set_ylabel("Prefill time (ms)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(rows, q1, q2, cq1, cq2, out_path):
    focus = [r for r in rows if r["turn_index"] in {5, 6, 7}]
    positive = [r for r in focus if r["prefill_cost_per_token"] > 0]
    all_positive = [r for r in rows if r["prefill_cost_per_token"] > 0]
    corr_all_comp_prefill = corr([r["computation_tokens"] for r in rows], [r["prefill_ms"] for r in rows])
    corr_all_comp_residual = corr([r["computation_tokens"] for r in rows], [r["prefill_residual_ms"] for r in rows])
    corr_all_batch_kv_residual = corr([r["batch_total_resident_kv_mib"] for r in rows], [r["prefill_residual_ms"] for r in rows])
    corr_all_batch_kv_cost = corr(
        [r["batch_total_resident_kv_mib"] for r in all_positive],
        [r["prefill_cost_per_token"] for r in all_positive],
    )

    corr_resident_cost = corr([r["resident_kv_mib"] for r in positive], [r["prefill_cost_per_token"] for r in positive])
    corr_batch_kv_cost = corr([r["batch_total_resident_kv_mib"] for r in positive], [r["prefill_cost_per_token"] for r in positive])
    corr_batch_compute_cost = corr(
        [r["batch_total_computation_tokens"] for r in positive],
        [r["prefill_cost_per_token"] for r in positive],
    )
    corr_batch_kv_residual = corr([r["batch_total_resident_kv_mib"] for r in focus], [r["prefill_residual_ms"] for r in focus])
    corr_batch_compute_residual = corr(
        [r["batch_total_computation_tokens"] for r in focus],
        [r["prefill_residual_ms"] for r in focus],
    )

    lines = []
    lines.append("# Prefill Cost Driver Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append("- Slice: `tenant_count=32` only")
    lines.append("- Main dataset: all turns 1~10")
    lines.append("- Additional focus view: turns 5, 6, 7")
    lines.append("- Residual definition: `prefill residual = prefill - computation_tokens / 6`")
    lines.append("")
    lines.append("## Question 1: Why is the cost of one miss token not constant?")
    lines.append("")
    lines.append(
        f"- Correlation between per-tenant resident KV and prefill cost per token is `{corr_resident_cost:.3f}` in turns 5/6/7."
    )
    lines.append(
        "- This means the same number of computation tokens can still cost different time depending on how long the already-resident history is."
    )
    lines.append(
        "- The practical interpretation is that new miss tokens are not appended into identical execution context. They are appended on top of different history length and different active-cache state."
    )
    lines.append("")
    lines.append("## Full-Range View")
    lines.append("")
    lines.append(
        f"- Across all turns, correlation between computation tokens and prefill is `{corr_all_comp_prefill:.3f}`."
    )
    lines.append(
        f"- Across all turns, correlation between computation tokens and prefill residual is `{corr_all_comp_residual:.3f}`."
    )
    lines.append(
        f"- Across all turns, correlation between batch total resident KV and prefill residual is `{corr_all_batch_kv_residual:.3f}`."
    )
    lines.append(
        f"- Across all turns, correlation between batch total resident KV and prefill cost per token is `{corr_all_batch_kv_cost:.3f}`."
    )
    lines.append(
        "- The new all-range scatter plots are meant to show the global shape directly instead of splitting the x-axis into computation bins."
    )
    lines.append("")
    lines.append("## Question 2: Does cost per token change near capacity?")
    lines.append("")
    lines.append(
        f"- Correlation between batch total resident KV and prefill cost per token is `{corr_batch_kv_cost:.3f}`."
    )
    lines.append(
        f"- Correlation between batch total computation tokens and prefill cost per token is `{corr_batch_compute_cost:.3f}`."
    )
    lines.append(
        f"- Correlation between batch total resident KV and residual is `{corr_batch_kv_residual:.3f}`."
    )
    lines.append(
        f"- Correlation between batch total computation tokens and residual is `{corr_batch_compute_residual:.3f}`."
    )
    lines.append(
        "- This supports a batch-level explanation: once the synchronized batch itself becomes heavier, token cost and unexplained residual both move."
    )
    lines.append("")
    lines.append("## Estimated Chunk/Batch Pressure")
    lines.append("")
    lines.append(
        "- Batch total computation tokens are compared against `max_num_batched_tokens=8192` to estimate how many prefill rounds would be needed if all computation tokens had to be served through the chunked-prefill budget."
    )
    lines.append(
        "- This is an estimate, not a direct vLLM internal trace. The repository does not contain per-iteration scheduler dumps."
    )
    lines.append("")
    lines.append("## Batch Pressure Bins")
    lines.append("")
    lines.append(f"- Low: `0 <= batch_total_resident_kv_mib <= {q1:.1f}`")
    lines.append(f"- High: `{q1:.1f} < batch_total_resident_kv_mib <= {q2:.1f}`")
    lines.append(f"- Extra: `batch_total_resident_kv_mib > {q2:.1f}`")
    lines.append("")
    lines.append("## Computation Bins")
    lines.append("")
    lines.append(f"- Low compute: `computation_tokens <= {cq1:.1f}`")
    lines.append(f"- Mid compute: `{cq1:.1f} < computation_tokens <= {cq2:.1f}`")
    lines.append(f"- High compute: `computation_tokens > {cq2:.1f}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The data does not support a single fixed ms-per-token model. Token cost depends on both local context length and the global synchronized batch state."
    )
    lines.append(
        "- In other words, `computation tokens` are the base load, but `existing KV length` and `batch pressure near capacity` act like extra modifiers on that load."
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
    enrich_batch_context(rows)
    q1, q2 = assign_pressure_bins(rows)
    cq1, cq2 = assign_computation_bins(rows)

    write_enriched_csv(rows, OUT_DIR / "prefill_cost_driver_enriched_32tenants.csv")
    write_batch_summary(rows, OUT_DIR / "prefill_cost_driver_batch_summary_32tenants.csv")
    write_summary(rows, q1, q2, OUT_DIR / "prefill_cost_driver_summary_32tenants.csv")
    plot_cost_vs_resident_kv_by_turn(plt, rows, OUT_DIR / "prefill_cost_per_token_vs_resident_kv_turn5_6_7_32tenants.png")
    plot_cost_vs_batch_pressure(plt, rows, OUT_DIR / "prefill_cost_per_token_vs_batch_total_kv_turn5_6_7_32tenants.png")
    plot_residual_vs_batch_compute(plt, rows, OUT_DIR / "prefill_residual_vs_batch_total_computation_turn5_6_7_32tenants.png")
    plot_turn_batch_driver_bars(plt, rows, OUT_DIR / "prefill_batch_driver_bars_turn5_6_7_32tenants.png")
    plot_prefill_box_by_compute_and_pressure(
        plt,
        rows,
        OUT_DIR / "prefill_box_by_computation_bin_and_batch_pressure_32tenants.png",
    )
    plot_exp_batch_kv_vs_residual(
        plt,
        rows,
        OUT_DIR / "prefill_residual_vs_batch_total_kv_by_exp_turn5_6_7_32tenants.png",
    )
    plot_prefill_vs_computation_all_with_pressure(
        plt,
        rows,
        OUT_DIR / "prefill_vs_computation_all_turns_by_batch_pressure_32tenants.png",
    )
    plot_prefill_residual_vs_computation_all_with_pressure(
        plt,
        rows,
        OUT_DIR / "prefill_residual_vs_computation_all_turns_by_batch_pressure_32tenants.png",
    )
    plot_prefill_vs_computation_all_by_turn(
        plt,
        rows,
        OUT_DIR / "prefill_vs_computation_all_turns_by_turn_32tenants.png",
    )
    write_markdown(rows, q1, q2, cq1, cq2, OUT_DIR / "prefill_cost_driver_analysis.md")


if __name__ == "__main__":
    main()
