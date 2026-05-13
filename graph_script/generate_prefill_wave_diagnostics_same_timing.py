#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/yuhwa2323/zoneatten")
OUT_DIR = ROOT / "analysis_prefill_wave_diagnostics_same_timing"
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
TENANT_COUNT = "32"
BLOCKING_THRESHOLD_MS = 100.0
GROUP_COLORS = {
    "Group1": "#2563eb",
    "Group2": "#f97316",
}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def load_request_metrics(exp_dir):
    metrics = {}
    metrics_dir = ROOT / exp_dir / "vram_only_artifacts_smallctx_mixed_limits" / "request_metrics"
    for path in metrics_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                metrics[payload["request_id"]] = payload
    return metrics


def load_rows():
    rows = []
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
                rows.append(
                    {
                        "exp_label": exp_label,
                        "turn_index": int(row["turn_index"]),
                        "tenant_id": int(row["tenant_id"]),
                        "group_label": "Group1" if row["history_limit_tokens"] == "8192" else "Group2",
                        "prefill_ms": 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0),
                        "computation_tokens": max(0.0, input_tokens - prefix_hit_tokens),
                        "first_token_ts": float(metric.get("first_token_ts", 0.0) or 0.0),
                    }
                )
    return rows


def enrich(rows):
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[(row["exp_label"], row["turn_index"])].append(row)

    batch_summary = []
    for (exp_label, turn_index), subset in by_batch.items():
        subset.sort(key=lambda item: item["first_token_ts"])
        first_min = subset[0]["first_token_ts"]
        prev_delay = 0.0
        for idx, row in enumerate(subset, start=1):
            delay_ms = (row["first_token_ts"] - first_min) * 1000.0
            row["rank_in_batch"] = idx
            row["delay_from_earliest_first_token_ms"] = delay_ms
            row["delta_from_prev_delay_ms"] = 0.0 if idx == 1 else delay_ms - prev_delay
            prev_delay = delay_ms
        spread = subset[-1]["delay_from_earliest_first_token_ms"]
        batch_summary.append(
            {
                "exp_label": exp_label,
                "turn_index": turn_index,
                "filtered_batch_size": len(subset),
                "delay_spread_ms": spread,
                "mean_prefill_ms": mean([r["prefill_ms"] for r in subset]),
                "mean_computation_tokens": mean([r["computation_tokens"] for r in subset]),
                "max_delta_gap_ms": max(r["delta_from_prev_delay_ms"] for r in subset),
            }
        )
    return batch_summary


def write_csv(rows, out_path):
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_top_batches(rows, top_n=6):
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[(row["exp_label"], row["turn_index"])].append(row)
    ranked = sorted(
        by_batch.items(),
        key=lambda item: max(r["delay_from_earliest_first_token_ms"] for r in item[1]),
        reverse=True,
    )
    return ranked[:top_n]


def plot_wave_plot(plt, rows, out_path):
    selected = select_top_batches(rows, top_n=6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.07, right=0.86, hspace=0.30, wspace=0.18)
    for ax, ((exp_label, turn_index), subset) in zip(axes.flatten(), selected):
        subset = sorted(subset, key=lambda r: r["rank_in_batch"])
        x = [r["rank_in_batch"] for r in subset]
        y = [r["delay_from_earliest_first_token_ms"] for r in subset]
        sizes = [max(24.0, r["computation_tokens"] * 0.08) for r in subset]
        colors = [GROUP_COLORS[r["group_label"]] for r in subset]
        ax.plot(x, y, color="#9ca3af", alpha=0.55, linewidth=1.0)
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.75)
        ax.set_title(f"{exp_label}, turn {turn_index}")
        ax.set_xlabel("Completion rank in filtered batch")
        ax.set_ylabel("Relative first-token delay (ms)")
        ax.grid(alpha=0.2)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLORS["Group1"], label="Group1"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLORS["Group2"], label="Group2"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#6b7280", markersize=10, label="Larger marker = more computation tokens"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.87, 0.98), ncol=1, title="Legend", frameon=True)
    fig.suptitle("Wave plot: top-6 widest filtered batches")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_timeline_strip(plt, rows, out_path):
    selected = select_top_batches(rows, top_n=6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False, sharey=False)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.07, right=0.86, hspace=0.30, wspace=0.18)
    for ax, ((exp_label, turn_index), subset) in zip(axes.flatten(), selected):
        subset = sorted(subset, key=lambda r: r["delay_from_earliest_first_token_ms"])
        x = [r["delay_from_earliest_first_token_ms"] for r in subset]
        y = list(range(1, len(subset) + 1))
        sizes = [max(24.0, r["computation_tokens"] * 0.08) for r in subset]
        colors = [GROUP_COLORS[r["group_label"]] for r in subset]
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.75)
        ax.set_title(f"{exp_label}, turn {turn_index}")
        ax.set_xlabel("Relative first-token delay (ms)")
        ax.set_ylabel("Sorted request index")
        ax.grid(alpha=0.2)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLORS["Group1"], label="Group1"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLORS["Group2"], label="Group2"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#6b7280", markersize=10, label="Larger marker = more computation tokens"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.87, 0.98), ncol=1, title="Legend", frameon=True)
    fig.suptitle("Timeline strip: completion clusters in top-6 widest filtered batches")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_delta_gap(plt, rows, out_path):
    selected = select_top_batches(rows, top_n=6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.07, right=0.86, hspace=0.30, wspace=0.18)
    for ax, ((exp_label, turn_index), subset) in zip(axes.flatten(), selected):
        subset = sorted(subset, key=lambda r: r["rank_in_batch"])
        x = [r["rank_in_batch"] for r in subset]
        y = [r["delta_from_prev_delay_ms"] for r in subset]
        colors = [GROUP_COLORS[r["group_label"]] for r in subset]
        ax.bar(x, y, color=colors, alpha=0.8)
        ax.set_title(f"{exp_label}, turn {turn_index}")
        ax.set_xlabel("Completion rank in filtered batch")
        ax.set_ylabel("Delta gap from previous completion (ms)")
        ax.grid(axis="y", alpha=0.2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["Group1"], alpha=0.8, label="Group1"),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["Group2"], alpha=0.8, label="Group2"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.87, 0.98), ncol=1, title="Color = group", frameon=True)
    fig.suptitle("Delta-gap plot: possible wave boundaries in top-6 widest filtered batches")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(rows, batch_summary, out_path):
    lines = []
    lines.append("# Prefill Wave Diagnostics")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append("- Slice: `tenant_count=32`, `blocking_time_ms < 100`")
    lines.append(f"- Request samples: `{len(rows)}`")
    lines.append(f"- Filtered batches: `{len(batch_summary)}`")
    lines.append("- Plots focus on the top-6 batches with the widest relative first-token spread")
    lines.append("")
    lines.append("## Terms")
    lines.append("")
    lines.append("- `filtered batch`: same `(exp, turn)` after applying `blocking < 100 ms`")
    lines.append("- `completion rank`: order of `first_token_ts` inside the filtered batch")
    lines.append("- `relative first-token delay`: `first_token_ts - min(first_token_ts in same filtered batch)`")
    lines.append("- `delta gap`: difference in relative first-token delay between adjacent completion ranks")
    lines.append("")
    lines.append("## What To Look For")
    lines.append("")
    lines.append("- Wave plot: stair-step patterns indicate multiple completion waves")
    lines.append("- Timeline strip: clustered points indicate groups of requests completing near the same time")
    lines.append("- Delta-gap plot: large bars indicate likely wave boundaries")
    lines.append("")
    lines.append("## Batch Summary")
    lines.append("")
    lines.append(
        f"- Mean delay spread across filtered batches: `{mean([r['delay_spread_ms'] for r in batch_summary]):.1f} ms`"
    )
    lines.append(
        f"- Mean max delta gap across filtered batches: `{mean([r['max_delta_gap_ms'] for r in batch_summary]):.1f} ms`"
    )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    batch_summary = enrich(rows)
    write_csv(rows, OUT_DIR / "prefill_wave_enriched_blocking_lt_100ms_32tenants.csv")
    write_csv(batch_summary, OUT_DIR / "prefill_wave_batch_summary_blocking_lt_100ms_32tenants.csv")
    plot_wave_plot(plt, rows, OUT_DIR / "prefill_wave_plot_top6_blocking_lt_100ms_32tenants.png")
    plot_timeline_strip(plt, rows, OUT_DIR / "prefill_timeline_strip_top6_blocking_lt_100ms_32tenants.png")
    plot_delta_gap(plt, rows, OUT_DIR / "prefill_delta_gap_top6_blocking_lt_100ms_32tenants.png")
    write_markdown(rows, batch_summary, OUT_DIR / "prefill_wave_diagnostics_blocking_lt_100ms.md")


if __name__ == "__main__":
    main()
