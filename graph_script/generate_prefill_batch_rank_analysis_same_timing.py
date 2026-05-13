#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB" / "graphs" / Path(__file__).stem
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


def rankdata(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    return corr(rankdata(xs), rankdata(ys))


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
                computation_tokens = max(0.0, input_tokens - prefix_hit_tokens)
                kv_history_tokens = float(row["kv_history_tokens"] or 0.0)
                rows.append(
                    {
                        "exp_label": exp_label,
                        "turn_index": int(row["turn_index"]),
                        "tenant_id": int(row["tenant_id"]),
                        "group_label": "Group1" if row["history_limit_tokens"] == "8192" else "Group2",
                        "blocking_ms": blocking_ms,
                        "input_tokens": input_tokens,
                        "prefix_hit_tokens": prefix_hit_tokens,
                        "prefix_hit_rate": (prefix_hit_tokens / input_tokens) if input_tokens > 0 else 0.0,
                        "computation_tokens": computation_tokens,
                        "kv_history_tokens": kv_history_tokens,
                        "prefill_ms": 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0),
                        "first_token_ts": float(metric.get("first_token_ts", 0.0) or 0.0),
                    }
                )
    return rows


def enrich_rank_and_delay(rows):
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[(row["exp_label"], row["turn_index"])].append(row)

    batch_summary = []
    fastest_slowest_rows = []
    for (exp_label, turn_index), subset in by_batch.items():
        subset.sort(key=lambda item: item["first_token_ts"])
        earliest = subset[0]["first_token_ts"]
        latest = subset[-1]["first_token_ts"]
        spread_ms = (latest - earliest) * 1000.0
        n = len(subset)
        batch_comp = sum(r["computation_tokens"] for r in subset)
        batch_prefill_mean = mean([r["prefill_ms"] for r in subset])
        for idx, row in enumerate(subset, start=1):
            row["rank_in_batch"] = idx
            row["rank_frac"] = idx / n
            row["delay_from_earliest_first_token_ms"] = (row["first_token_ts"] - earliest) * 1000.0

        k = max(1, min(4, n // 4 if n >= 4 else 1))
        fastest = subset[:k]
        slowest = subset[-k:]
        batch_summary.append(
            {
                "exp_label": exp_label,
                "turn_index": turn_index,
                "filtered_batch_size": n,
                "batch_total_computation_tokens": batch_comp,
                "batch_mean_prefill_ms": batch_prefill_mean,
                "batch_first_token_spread_ms": spread_ms,
                "fastest_mean_prefill_ms": mean([r["prefill_ms"] for r in fastest]),
                "slowest_mean_prefill_ms": mean([r["prefill_ms"] for r in slowest]),
                "fastest_mean_computation_tokens": mean([r["computation_tokens"] for r in fastest]),
                "slowest_mean_computation_tokens": mean([r["computation_tokens"] for r in slowest]),
                "fastest_mean_kv_history_tokens": mean([r["kv_history_tokens"] for r in fastest]),
                "slowest_mean_kv_history_tokens": mean([r["kv_history_tokens"] for r in slowest]),
                "fastest_group1_ratio": mean([1.0 if r["group_label"] == "Group1" else 0.0 for r in fastest]),
                "slowest_group1_ratio": mean([1.0 if r["group_label"] == "Group1" else 0.0 for r in slowest]),
            }
        )

        for bucket_name, bucket in [("fastest", fastest), ("slowest", slowest)]:
            for row in bucket:
                fastest_slowest_rows.append(
                    {
                        "exp_label": exp_label,
                        "turn_index": turn_index,
                        "bucket": bucket_name,
                        "tenant_id": row["tenant_id"],
                        "group_label": row["group_label"],
                        "rank_in_batch": row["rank_in_batch"],
                        "delay_from_earliest_first_token_ms": row["delay_from_earliest_first_token_ms"],
                        "prefill_ms": row["prefill_ms"],
                        "computation_tokens": row["computation_tokens"],
                        "kv_history_tokens": row["kv_history_tokens"],
                        "prefix_hit_rate": row["prefix_hit_rate"],
                    }
                )

    return batch_summary, fastest_slowest_rows


def write_csv_dicts(rows, out_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rank_vs_prefill(plt, rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.86, wspace=0.22)

    specs = [
        ("rank_in_batch", "Rank in filtered batch"),
        ("rank_frac", "Rank fraction in filtered batch"),
    ]
    for ax, (key, label) in zip(axes, specs):
        for group_label in ["Group1", "Group2"]:
            subset = [r for r in rows if r["group_label"] == group_label]
            ax.scatter(
                [r[key] for r in subset],
                [r["prefill_ms"] for r in subset],
                s=18,
                alpha=0.4,
                color=GROUP_COLORS[group_label],
                label=f"{group_label} (n={len(subset)})",
            )
        pear = corr([r[key] for r in rows], [r["prefill_ms"] for r in rows])
        spear = spearman([r[key] for r in rows], [r["prefill_ms"] for r in rows])
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Prefill time (ms)")
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.97,
            f"all r={pear:.3f}\nall ρ={spear:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLORS["Group1"], label="Group1"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLORS["Group2"], label="Group2"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.87, 0.98),
        ncol=1,
        title="Color = group",
        frameon=True,
    )
    fig.suptitle("Prefill vs batch-completion rank, blocking time < 100 ms")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_wave_patterns(plt, rows, out_path):
    batches = defaultdict(list)
    for row in rows:
        batches[(row["exp_label"], row["turn_index"])].append(row)
    selected = sorted(
        batches.items(),
        key=lambda item: max(r["delay_from_earliest_first_token_ms"] for r in item[1]),
        reverse=True,
    )[:6]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.07, right=0.86, hspace=0.28, wspace=0.18)
    for ax, ((exp_label, turn_index), subset) in zip(axes.flatten(), selected):
        subset = sorted(subset, key=lambda r: r["delay_from_earliest_first_token_ms"])
        x = list(range(1, len(subset) + 1))
        y = [r["delay_from_earliest_first_token_ms"] for r in subset]
        colors = [GROUP_COLORS[r["group_label"]] for r in subset]
        ax.scatter(x, y, s=28, c=colors, alpha=0.75)
        ax.plot(x, y, color="#9ca3af", alpha=0.5, linewidth=1.0)
        ax.set_title(f"{exp_label}, turn {turn_index}")
        ax.set_xlabel("Rank by first-token completion")
        ax.set_ylabel("Relative first-token delay (ms)")
        ax.grid(alpha=0.2)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLORS["Group1"], label="Group1"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLORS["Group2"], label="Group2"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.87, 0.98),
        ncol=1,
        title="Color = group",
        frameon=True,
    )
    fig.suptitle("Wave patterns in the six widest filtered batches")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_fastest_vs_slowest_bars(plt, batch_summary, out_path):
    metrics = [
        ("fastest_mean_prefill_ms", "slowest_mean_prefill_ms", "Prefill time (ms)"),
        ("fastest_mean_computation_tokens", "slowest_mean_computation_tokens", "Computation tokens"),
        ("fastest_mean_kv_history_tokens", "slowest_mean_kv_history_tokens", "KV history tokens"),
        ("fastest_group1_ratio", "slowest_group1_ratio", "Group1 ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.86, hspace=0.32, wspace=0.24)
    for ax, (fast_key, slow_key, title) in zip(axes.flatten(), metrics):
        fast_mean = mean([row[fast_key] for row in batch_summary])
        slow_mean = mean([row[slow_key] for row in batch_summary])
        ax.bar(["Fastest", "Slowest"], [fast_mean, slow_mean], color=["#10b981", "#ef4444"], alpha=0.8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate([fast_mean, slow_mean]):
            ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Fastest vs slowest requests within each filtered batch")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#10b981", alpha=0.8, label="Fastest subset mean"),
        plt.Rectangle((0, 0), 1, 1, color="#ef4444", alpha=0.8, label="Slowest subset mean"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.87, 0.98),
        ncol=1,
        title="Bar meaning",
        frameon=True,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(rows, batch_summary, out_path):
    lines = []
    lines.append("# Prefill Batch Rank Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append("- Slice: `tenant_count=32`, `blocking_time_ms < 100`")
    lines.append(f"- Request samples: `{len(rows)}`")
    lines.append(f"- Filtered batches: `{len(batch_summary)}`")
    lines.append("")
    lines.append("## Rank Interpretation")
    lines.append("")
    lines.append("- `rank_in_batch=1` means the earliest first-token completion inside the filtered batch.")
    lines.append("- Larger rank means the request completed its first-token later inside the same filtered batch.")
    lines.append("")
    lines.append("## Key Correlations")
    lines.append("")
    lines.append(
        f"- `rank_in_batch` vs `prefill_ms`: Pearson `{corr([r['rank_in_batch'] for r in rows], [r['prefill_ms'] for r in rows]):.3f}`, "
        f"Spearman `{spearman([r['rank_in_batch'] for r in rows], [r['prefill_ms'] for r in rows]):.3f}`"
    )
    lines.append(
        f"- `rank_frac` vs `prefill_ms`: Pearson `{corr([r['rank_frac'] for r in rows], [r['prefill_ms'] for r in rows]):.3f}`, "
        f"Spearman `{spearman([r['rank_frac'] for r in rows], [r['prefill_ms'] for r in rows]):.3f}`"
    )
    lines.append("")
    lines.append("## Fastest vs Slowest Mean Comparison")
    lines.append("")
    lines.append(
        f"- Prefill: fastest `{mean([r['fastest_mean_prefill_ms'] for r in batch_summary]):.1f} ms`, "
        f"slowest `{mean([r['slowest_mean_prefill_ms'] for r in batch_summary]):.1f} ms`"
    )
    lines.append(
        f"- Computation tokens: fastest `{mean([r['fastest_mean_computation_tokens'] for r in batch_summary]):.1f}`, "
        f"slowest `{mean([r['slowest_mean_computation_tokens'] for r in batch_summary]):.1f}`"
    )
    lines.append(
        f"- KV history tokens: fastest `{mean([r['fastest_mean_kv_history_tokens'] for r in batch_summary]):.1f}`, "
        f"slowest `{mean([r['slowest_mean_kv_history_tokens'] for r in batch_summary]):.1f}`"
    )
    lines.append(
        f"- Group1 ratio: fastest `{mean([r['fastest_group1_ratio'] for r in batch_summary]):.2f}`, "
        f"slowest `{mean([r['slowest_group1_ratio'] for r in batch_summary]):.2f}`"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If slowest requests also have higher computation or longer KV history, request shape contributes to late completion.")
    lines.append("- If wave plots show stair-step clusters, that is evidence for internal chunk/interleaving waves inside the filtered batch.")
    lines.append("- If fast/slow gaps remain large even when request-shape gaps are small, scheduler/interleaving effects are likely dominant.")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    batch_summary, fastest_slowest_rows = enrich_rank_and_delay(rows)

    write_csv_dicts(rows, OUT_DIR / "prefill_batch_rank_enriched_blocking_lt_100ms_32tenants.csv")
    write_csv_dicts(batch_summary, OUT_DIR / "prefill_batch_rank_summary_blocking_lt_100ms_32tenants.csv")
    write_csv_dicts(
        fastest_slowest_rows,
        OUT_DIR / "prefill_fastest_vs_slowest_rows_blocking_lt_100ms_32tenants.csv",
    )
    plot_rank_vs_prefill(plt, rows, OUT_DIR / "prefill_vs_batch_rank_blocking_lt_100ms_32tenants.png")
    plot_wave_patterns(plt, rows, OUT_DIR / "prefill_wave_patterns_top6_blocking_lt_100ms_32tenants.png")
    plot_fastest_vs_slowest_bars(
        plt,
        batch_summary,
        OUT_DIR / "prefill_fastest_vs_slowest_summary_blocking_lt_100ms_32tenants.png",
    )
    write_markdown(rows, batch_summary, OUT_DIR / "prefill_batch_rank_analysis_blocking_lt_100ms.md")


if __name__ == "__main__":
    main()
