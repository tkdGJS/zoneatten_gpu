#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/yuhwa2323/zoneatten")
OUT_DIR = ROOT / "analysis_prefill_direct_cause_same_timing"
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
TENANT_COUNTS = ["8", "16", "32"]
THRESHOLD_MS = 100.0
MAX_NUM_BATCHED_TOKENS = 8192.0
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


def load_request_metrics(exp_dir):
    metrics = {}
    metrics_dir = ROOT / exp_dir / "vram_only_artifacts_smallctx_mixed_limits" / "request_metrics"
    for path in metrics_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                metrics[payload["request_id"]] = payload
    return metrics


def load_batch_rows(blocking_mode, tenant_count):
    by_batch = defaultdict(list)
    for exp_dir, exp_label in EXPERIMENTS:
        metrics = load_request_metrics(exp_dir)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success" or row["tenant_count"] != tenant_count:
                    continue
                blocking_ms = float(row["blocking_time_ms"] or 0.0)
                if blocking_mode == "lt100" and not (blocking_ms < THRESHOLD_MS):
                    continue
                if blocking_mode == "gt100" and not (blocking_ms > THRESHOLD_MS):
                    continue
                metric = metrics.get(row["metrics_request_id"])
                if metric is None:
                    continue
                input_tokens = float(row["input_tokens"] or 0.0)
                prefix_hit_tokens = float(row["prefix_hit_tokens"] or 0.0)
                by_batch[(exp_label, int(row["turn_index"]))].append(
                    {
                        "prefill_ms": 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0),
                        "computation_tokens": max(0.0, input_tokens - prefix_hit_tokens),
                    }
                )

    batch_rows = []
    for (exp_label, turn_index), subset in sorted(by_batch.items()):
        if not subset:
            continue
        batch_rows.append(
            {
                "exp_label": exp_label,
                "turn_index": turn_index,
                "filtered_batch_size": len(subset),
                "batch_total_computation_tokens": sum(r["computation_tokens"] for r in subset),
                "batch_compute_ratio_8192": sum(r["computation_tokens"] for r in subset) / MAX_NUM_BATCHED_TOKENS,
                "batch_mean_prefill_ms": mean([r["prefill_ms"] for r in subset]),
            }
        )
    return batch_rows


def write_csv(rows_lt, rows_gt, out_path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "blocking_slice",
                "exp_label",
                "turn_index",
                "filtered_batch_size",
                "batch_total_computation_tokens",
                "batch_compute_ratio_8192",
                "batch_mean_prefill_ms",
            ]
        )
        for label, rows in [("lt100", rows_lt), ("gt100", rows_gt)]:
            for row in rows:
                writer.writerow(
                    [
                        label,
                        row["exp_label"],
                        row["turn_index"],
                        row["filtered_batch_size"],
                        f"{row['batch_total_computation_tokens']:.4f}",
                        f"{row['batch_compute_ratio_8192']:.6f}",
                        f"{row['batch_mean_prefill_ms']:.4f}",
                    ]
                )


def plot_compare(plt, rows_lt, rows_gt, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.86, wspace=0.18)
    handles = []
    for ax, rows, title in [
        (axes[0], rows_lt, "blocking < 100 ms"),
        (axes[1], rows_gt, "blocking > 100 ms"),
    ]:
        for turn in range(1, 11):
            subset = [r for r in rows if r["turn_index"] == turn]
            sc = ax.scatter(
                [r["batch_compute_ratio_8192"] for r in subset],
                [r["batch_mean_prefill_ms"] for r in subset],
                s=55,
                alpha=0.78,
                color=TURN_COLORS[turn],
                edgecolors="white",
                linewidths=0.4,
            )
            if len(handles) < 10:
                handles.append((sc, f"Turn {turn}"))
        pear = corr([r["batch_compute_ratio_8192"] for r in rows], [r["batch_mean_prefill_ms"] for r in rows]) if rows else 0.0
        ax.axvline(1.0, linestyle="--", linewidth=1.3, color="#111827")
        ax.set_title(f"{title}\nr = {pear:.3f}")
        ax.set_xlabel("Total compute tokens / 8192")
        ax.set_ylabel("Prefill time average (ms)")
        ax.grid(alpha=0.2)
    fig.legend(
        [h for h, _ in handles],
        [label for _, label in handles],
        loc="upper left",
        bbox_to_anchor=(0.87, 0.98),
        ncol=1,
        title="Color = turn",
        frameon=True,
    )
    fig.suptitle("Prefill vs batch total compute / 8192 by blocking slice")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(rows_lt, rows_gt, out_path, tenant_count):
    r_lt = corr([r["batch_compute_ratio_8192"] for r in rows_lt], [r["batch_mean_prefill_ms"] for r in rows_lt]) if rows_lt else 0.0
    r_gt = corr([r["batch_compute_ratio_8192"] for r in rows_gt], [r["batch_mean_prefill_ms"] for r in rows_gt]) if rows_gt else 0.0
    lines = []
    lines.append("# Prefill vs Compute Ratio By Blocking Slice")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append(f"- Slice: `tenant_count={tenant_count}`, one point = one filtered `(exp, turn)` batch")
    lines.append("- Comparison:")
    lines.append("  - left panel: requests with `blocking < 100 ms`")
    lines.append("  - right panel: requests with `blocking > 100 ms`")
    lines.append("")
    lines.append("## Correlation")
    lines.append("")
    lines.append(f"- `blocking < 100 ms`: Pearson `{r_lt:.3f}`, batches `{len(rows_lt)}`")
    lines.append(f"- `blocking > 100 ms`: Pearson `{r_gt:.3f}`, batches `{len(rows_gt)}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If the `blocking > 100 ms` panel still shows a clear positive trend, then prefill remains strongly tied to execution load even after queueing is already present.")
    lines.append("- If the trend weakens substantially, that means queueing/memory pressure dominates and compute ratio becomes a less direct driver in that slice.")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tenant_count in TENANT_COUNTS:
        rows_lt = load_batch_rows("lt100", tenant_count)
        rows_gt = load_batch_rows("gt100", tenant_count)
        suffix = f"{tenant_count}tenants"
        write_csv(
            rows_lt,
            rows_gt,
            OUT_DIR / f"prefill_vs_compute_ratio_by_blocking_slice_{suffix}.csv",
        )
        plot_compare(
            plt,
            rows_lt,
            rows_gt,
            OUT_DIR / f"prefill_vs_compute_ratio_by_blocking_slice_{suffix}.png",
        )
        write_markdown(
            rows_lt,
            rows_gt,
            OUT_DIR / f"prefill_vs_compute_ratio_by_blocking_slice_{suffix}.md",
            tenant_count,
        )


if __name__ == "__main__":
    main()
