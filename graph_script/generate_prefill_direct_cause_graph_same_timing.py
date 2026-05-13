#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "graphs" / Path(__file__).stem
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
TENANT_COUNT = "32"
BLOCKING_THRESHOLD_MS = 100.0
MAX_NUM_BATCHED_TOKENS = 8192.0
KV_CAPACITY_MIB = 1024.0
TURN_COLORS = {
    1: "#440154",
    2: "#482777",
    3: "#3f4a8a",
    4: "#31678e",
    5: "#26838f",
    6: "#1f9d8a",
    7: "#6cce5a",
    8: "#b6de2b",
    9: "#fee825",
    10: "#f9f871",
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


def load_batch_rows(model_info):
    kv_bytes_per_token = model_info["kv_bytes_per_token"]
    by_batch = defaultdict(list)
    for exp_dir, exp_label in EXPERIMENTS:
        metrics = load_request_metrics(exp_dir)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success" or row["tenant_count"] != TENANT_COUNT:
                    continue
                blocking_ms = float(row["blocking_time_ms"] or 0.0)
                if blocking_ms >= BLOCKING_THRESHOLD_MS:
                    continue
                metric = metrics.get(row["metrics_request_id"])
                if metric is None:
                    continue
                input_tokens = float(row["input_tokens"] or 0.0)
                prefix_hit_tokens = float(row["prefix_hit_tokens"] or 0.0)
                kv_history_tokens = float(row["kv_history_tokens"] or 0.0)
                by_batch[(exp_label, int(row["turn_index"]))].append(
                    {
                        "prefill_ms": 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0),
                        "computation_tokens": max(0.0, input_tokens - prefix_hit_tokens),
                        "resident_kv_mib": mib_from_tokens(kv_history_tokens, kv_bytes_per_token),
                    }
                )
    batch_rows = []
    for (exp_label, turn_index), subset in sorted(by_batch.items()):
        batch_rows.append(
            {
                "exp_label": exp_label,
                "turn_index": turn_index,
                "filtered_batch_size": len(subset),
                "batch_total_computation_tokens": sum(r["computation_tokens"] for r in subset),
                "batch_compute_ratio_8192": sum(r["computation_tokens"] for r in subset) / MAX_NUM_BATCHED_TOKENS,
                "batch_total_resident_kv_mib": sum(r["resident_kv_mib"] for r in subset),
                "batch_kv_ratio_1024": sum(r["resident_kv_mib"] for r in subset) / KV_CAPACITY_MIB,
                "batch_mean_prefill_ms": mean([r["prefill_ms"] for r in subset]),
                "batch_p95_prefill_ms": percentile([r["prefill_ms"] for r in subset], 0.95),
                "batch_p99_prefill_ms": percentile([r["prefill_ms"] for r in subset], 0.99),
                "batch_max_prefill_ms": max(r["prefill_ms"] for r in subset),
            }
        )
    return batch_rows


def write_csv(rows, out_path):
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_direct_cause(plt, rows, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.98, wspace=0.22)

    specs = [
        (
            "batch_total_computation_tokens",
            "batch_mean_prefill_ms",
            "Batch total computation tokens",
            "Prefill time average (ms)",
        ),
        (
            "batch_compute_ratio_8192",
            "batch_p95_prefill_ms",
            "Batch total compute / 8192",
            "Batch p95 prefill (ms)",
        ),
        (
            "batch_total_resident_kv_mib",
            "batch_mean_prefill_ms",
            "Batch total resident KV (MiB)",
            "Prefill time average (ms)",
        ),
    ]

    turn_handles = []
    for ax, (x_key, y_key, x_label, y_label) in zip(axes, specs):
        for turn in range(1, 11):
            subset = [r for r in rows if r["turn_index"] == turn]
            sc = ax.scatter(
                [r[x_key] for r in subset],
                [r[y_key] for r in subset],
                s=55,
                alpha=0.75,
                color=TURN_COLORS[turn],
                edgecolors="white",
                linewidths=0.4,
            )
            if len(turn_handles) < 10:
                turn_handles.append((sc, f"Turn {turn}"))
        pear = corr([r[x_key] for r in rows], [r[y_key] for r in rows])
        ax.set_title(f"r = {pear:.3f}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)
        if x_key == "batch_compute_ratio_8192":
            ax.axvline(1.0, linestyle="--", linewidth=1.3, color="#111827")
            ax.text(1.0, ax.get_ylim()[1], "8192 budget", ha="left", va="top", fontsize=9)
        if x_key == "batch_total_resident_kv_mib":
            ax.axvline(1024.0, linestyle="--", linewidth=1.3, color="#111827")
            ax.text(1024.0, ax.get_ylim()[1], "1 GiB KV capacity", ha="left", va="top", fontsize=9)

    fig.legend(
        [h for h, _ in turn_handles],
        [label for _, label in turn_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        title="Color = turn",
        frameon=True,
    )
    fig.suptitle("Batch-level direct-cause view for prefill, blocking time < 100 ms")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_single_metric(
    plt,
    rows,
    x_key,
    y_key,
    x_label,
    y_label,
    title,
    out_path,
    vertical_line_x=None,
    vertical_line_label=None,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.11, right=0.82)
    turn_handles = []
    for turn in range(1, 11):
        subset = [r for r in rows if r["turn_index"] == turn]
        sc = ax.scatter(
            [r[x_key] for r in subset],
            [r[y_key] for r in subset],
            s=60,
            alpha=0.78,
            color=TURN_COLORS[turn],
            edgecolors="white",
            linewidths=0.4,
        )
        if len(turn_handles) < 10:
            turn_handles.append((sc, f"Turn {turn}"))
    pear = corr([r[x_key] for r in rows], [r[y_key] for r in rows])
    ax.set_title(f"{title}\nr = {pear:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)
    if vertical_line_x is not None:
        ax.axvline(vertical_line_x, linestyle="--", linewidth=1.3, color="#111827")
        if vertical_line_label:
            ax.text(vertical_line_x, ax.get_ylim()[1], vertical_line_label, ha="left", va="top", fontsize=9)
    fig.legend(
        [h for h, _ in turn_handles],
        [label for _, label in turn_handles],
        loc="upper left",
        bbox_to_anchor=(0.84, 0.98),
        ncol=1,
        title="Color = turn",
        frameon=True,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(rows, out_path):
    lines = []
    lines.append("# Prefill Direct Cause Graph")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append("- Slice: `tenant_count=32`, `blocking_time_ms < 100`")
    lines.append(f"- Batch samples: `{len(rows)}`")
    lines.append("")
    lines.append("## Why These Axes")
    lines.append("")
    lines.append("- One point is one filtered `(exp, turn)` batch, not one request.")
    lines.append("- This is closer to direct cause because the x-axis is a batch-level load variable and the y-axis is a batch-level prefill outcome.")
    lines.append("- It avoids mixing request-level outputs with batch-level cause in the same point.")
    lines.append("")
    lines.append("## Correlations")
    lines.append("")
    lines.append(
        f"- `batch_total_computation_tokens` vs `prefill time average (ms)`: `{corr([r['batch_total_computation_tokens'] for r in rows], [r['batch_mean_prefill_ms'] for r in rows]):.3f}`"
    )
    lines.append(
        f"- `batch_compute_ratio_8192` vs `batch_p95_prefill_ms`: `{corr([r['batch_compute_ratio_8192'] for r in rows], [r['batch_p95_prefill_ms'] for r in rows]):.3f}`"
    )
    lines.append(
        f"- `batch_total_resident_kv_mib` vs `prefill time average (ms)`: `{corr([r['batch_total_resident_kv_mib'] for r in rows], [r['batch_mean_prefill_ms'] for r in rows]):.3f}`"
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
    rows = load_batch_rows(model_info)
    write_csv(rows, OUT_DIR / "prefill_direct_cause_batches_blocking_lt_100ms_32tenants.csv")
    plot_direct_cause(plt, rows, OUT_DIR / "prefill_direct_cause_batches_blocking_lt_100ms_32tenants.png")
    plot_single_metric(
        plt,
        rows,
        "batch_total_computation_tokens",
        "batch_mean_prefill_ms",
        "Batch total computation tokens",
        "Prefill time average (ms)",
        "Batch total computation tokens vs prefill time average\nblocking time < 100 ms",
        OUT_DIR / "batch_total_computation_tokens_vs_batch_mean_prefill_blocking_lt_100ms_32tenants.png",
    )
    plot_single_metric(
        plt,
        rows,
        "batch_total_computation_tokens",
        "batch_p95_prefill_ms",
        "Batch total computation tokens",
        "Batch p95 prefill (ms)",
        "Batch total computation tokens vs batch p95 prefill\nblocking time < 100 ms",
        OUT_DIR / "batch_total_computation_tokens_vs_batch_p95_prefill_blocking_lt_100ms_32tenants.png",
    )
    plot_single_metric(
        plt,
        rows,
        "batch_total_computation_tokens",
        "batch_p99_prefill_ms",
        "Batch total computation tokens",
        "Batch p99 prefill (ms)",
        "Batch total computation tokens vs batch p99 prefill\nblocking time < 100 ms",
        OUT_DIR / "batch_total_computation_tokens_vs_batch_p99_prefill_blocking_lt_100ms_32tenants.png",
    )
    plot_single_metric(
        plt,
        rows,
        "batch_compute_ratio_8192",
        "batch_p95_prefill_ms",
        "Total compute tokens / 8192",
        "Batch p95 prefill (ms)",
        "Batch total compute / 8192 vs batch p95 prefill\nblocking time < 100 ms",
        OUT_DIR / "batch_compute_ratio_8192_vs_batch_p95_prefill_blocking_lt_100ms_32tenants.png",
        vertical_line_x=1.0,
        vertical_line_label="8192 budget",
    )
    plot_single_metric(
        plt,
        rows,
        "batch_compute_ratio_8192",
        "batch_mean_prefill_ms",
        "Total compute tokens / 8192",
        "Prefill time average (ms)",
        "Batch total compute / 8192 vs prefill time average\nblocking time < 100 ms",
        OUT_DIR / "batch_compute_ratio_8192_vs_batch_mean_prefill_blocking_lt_100ms_32tenants.png",
        vertical_line_x=1.0,
        vertical_line_label="8192 budget",
    )
    plot_single_metric(
        plt,
        rows,
        "batch_compute_ratio_8192",
        "batch_p99_prefill_ms",
        "Total compute tokens / 8192",
        "Batch p99 prefill (ms)",
        "Batch total compute / 8192 vs batch p99 prefill\nblocking time < 100 ms",
        OUT_DIR / "batch_compute_ratio_8192_vs_batch_p99_prefill_blocking_lt_100ms_32tenants.png",
        vertical_line_x=1.0,
        vertical_line_label="8192 budget",
    )
    plot_single_metric(
        plt,
        rows,
        "batch_total_resident_kv_mib",
        "batch_mean_prefill_ms",
        "Batch total resident KV (MiB)",
        "Prefill time average (ms)",
        "Batch total resident KV vs prefill time average\nblocking time < 100 ms",
        OUT_DIR / "batch_total_resident_kv_mib_vs_batch_mean_prefill_blocking_lt_100ms_32tenants.png",
        vertical_line_x=1024.0,
        vertical_line_label="1 GiB KV capacity",
    )
    write_markdown(rows, OUT_DIR / "prefill_direct_cause_batches_blocking_lt_100ms.md")


if __name__ == "__main__":
    main()
