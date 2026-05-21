#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "vram_only_results_smallctx_mixed_limits_8GB" / "graphs" / Path(__file__).stem
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
TENANT_COUNTS = []
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
GROUP_COLORS = {
    "8192": "#2563eb",
    "2048": "#f97316",
}
DISPLAY_LIMIT_MAP = {
    "8192": "24576",
    "2048": "12288",
}
GROUP1_COLOR = "#2563eb"
GROUP2_COLOR = "#f97316"
BLOCKING_COLOR = "#2563eb"
PREFILL_COLOR = "#84cc16"
REMAINING_COLOR = "#dc2626"
THRESHOLD_MS = 100.0
MAX_NUM_BATCHED_TOKENS = 16384.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    frac = idx - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


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


def padded_limits(values):
    if not values:
        return (0.0, 1.0)
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        pad = max(1.0, abs(vmax) * 0.05)
        return (vmin - pad, vmax + pad)
    pad = (vmax - vmin) * 0.05
    return (vmin - pad, vmax + pad)


def actual_tenant_counts(rows):
    return [str(value) for value in sorted({int(row["tenant_count"]) for row in rows})]


def actual_group_limits(rows):
    limits = [str(value) for value in sorted({int(row["history_limit_tokens"]) for row in rows})]
    if len(limits) >= 2:
        return limits[-1], limits[0]
    if len(limits) == 1:
        return limits[0], limits[0]
    return "", ""


def group_label_for_limit(history_limit_tokens, group1_limit, group2_limit):
    if history_limit_tokens == group1_limit:
        return f"Group1 (limit={DISPLAY_LIMIT_MAP.get(group1_limit, group1_limit)})"
    if history_limit_tokens == group2_limit:
        return f"Group2 (limit={DISPLAY_LIMIT_MAP.get(group2_limit, group2_limit)})"
    return f"limit={DISPLAY_LIMIT_MAP.get(history_limit_tokens, history_limit_tokens)}"


def group_color_for_limit(history_limit_tokens, group1_limit, group2_limit):
    if history_limit_tokens == group1_limit:
        return GROUP1_COLOR
    if history_limit_tokens == group2_limit:
        return GROUP2_COLOR
    return GROUP_COLORS.get(history_limit_tokens, "#6b7280")


def load_request_metrics(exp_dir):
    metrics = {}
    metrics_dir = ROOT / exp_dir / "vram_only_artifacts_smallctx_mixed_limits" / "request_metrics"
    for path in metrics_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                metrics[payload["request_id"]] = payload
    return metrics


def load_success_rows():
    rows = []
    for exp_dir, exp_label in EXPERIMENTS:
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success":
                    continue
                rows.append(
                    {
                        "exp_label": exp_label,
                        "tenant_count": row["tenant_count"],
                        "turn_index": int(row["turn_index"]),
                        "history_limit_tokens": row["history_limit_tokens"],
                        "prefix_hit_rate": float(row["prefix_hit_rate"] or 0.0),
                        "ttft_ms": float(row["ttft_ms"] or 0.0),
                        "blocking_time_ms": float(row["blocking_time_ms"] or 0.0),
                        "metrics_request_id": row["metrics_request_id"],
                        "input_tokens": float(row["input_tokens"] or 0.0),
                        "output_tokens": float(row["output_tokens"] or 0.0),
                        "p95_tbt_ms": float(row["p95_tbt_ms"] or 0.0),
                        "prefix_hit_tokens": float(row["prefix_hit_tokens"] or 0.0),
                    }
                )
    return rows


def load_breakdown_points(tenant_count_filter="32"):
    points = []
    for exp_dir, exp_label in EXPERIMENTS:
        metrics = load_request_metrics(exp_dir)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success" or row["tenant_count"] != tenant_count_filter:
                    continue
                metric = metrics.get(row["metrics_request_id"])
                if metric is None:
                    continue
                ttft_ms = float(row["ttft_ms"] or 0.0)
                blocking_ms = 1000.0 * float(metric.get("queued_time_s", 0.0) or 0.0)
                prefill_ms = 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0)
                remaining_ms = max(0.0, ttft_ms - blocking_ms - prefill_ms)
                points.append(
                    {
                        "exp_label": exp_label,
                        "turn_index": int(row["turn_index"]),
                        "history_limit_tokens": row["history_limit_tokens"],
                        "blocking_ms": blocking_ms,
                        "prefill_ms": prefill_ms,
                        "remaining_ms": remaining_ms,
                    }
                )
    return points


def load_prefill_batches(blocking_mode, tenant_count="32"):
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
    for (_, turn_index), subset in sorted(by_batch.items()):
        if not subset:
            continue
        total_compute = sum(r["computation_tokens"] for r in subset)
        batch_rows.append(
            {
                "turn_index": turn_index,
                "batch_compute_ratio": total_compute / MAX_NUM_BATCHED_TOKENS,
                "batch_mean_prefill_ms": mean([r["prefill_ms"] for r in subset]),
            }
        )
    return batch_rows


def plot_ttft_vs_prefix_hit_rate(plt, rows, out_path):
    tenant_counts = actual_tenant_counts(rows)
    fig, axes = plt.subplots(
        1,
        max(1, len(tenant_counts)),
        figsize=(6 * max(1, len(tenant_counts)), 6.2),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.97, wspace=0.16)
    xlim = padded_limits([r["prefix_hit_rate"] for r in rows])
    ylim = padded_limits([r["ttft_ms"] / 1000.0 for r in rows])
    handles = []
    labels = []

    for idx, (ax, tenant_count) in enumerate(zip(axes, tenant_counts)):
        subset = [r for r in rows if r["tenant_count"] == tenant_count]
        for turn in range(1, 11):
            turn_rows = [r for r in subset if r["turn_index"] == turn]
            sc = ax.scatter(
                [r["prefix_hit_rate"] for r in turn_rows],
                [r["ttft_ms"] / 1000.0 for r in turn_rows],
                s=34,
                alpha=0.58,
                color=TURN_COLORS[turn],
                edgecolors="none",
            )
            if len(handles) < 10:
                handles.append(sc)
                labels.append(f"Turn {turn}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Prefix Hit Rate")
        ax.set_box_aspect(2.0 / 3.0)
        ax.grid(alpha=0.2)
        ax.text(0.03, 0.97, f"tenants={tenant_count}", transform=ax.transAxes, va="top", ha="left")
        ax.text(0.5, -0.30, f"({chr(ord('a') + idx)})", transform=ax.transAxes, ha="center", va="top")
    axes[0].set_ylabel("TTFT (sec)")
    axes[0].legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        ncol=2,
        fontsize=13,
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=1.5,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_p99_breakdown_by_turn(plt, points, out_path, fig_height=6.2):
    group1_limit, group2_limit = actual_group_limits(points)
    groups = [
        (
            group_label_for_limit(group1_limit, group1_limit, group2_limit),
            [p for p in points if p["history_limit_tokens"] == group1_limit],
        ),
    ]
    if group2_limit != group1_limit:
        groups.append(
            (
                group_label_for_limit(group2_limit, group1_limit, group2_limit),
                [p for p in points if p["history_limit_tokens"] == group2_limit],
            )
        )
    fig, axes = plt.subplots(1, len(groups), figsize=(9 * len(groups), fig_height), sharey=True)
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.21, top=0.97, wspace=0.16)
    ymax = 0.0
    stacked_data = []
    for _, group_points in groups:
        grouped = defaultdict(list)
        for point in group_points:
            grouped[point["turn_index"]].append(point)
        turns = list(range(1, 11))
        blocking = [percentile([p["blocking_ms"] for p in grouped[t]], 0.99) / 1000.0 for t in turns]
        prefill = [percentile([p["prefill_ms"] for p in grouped[t]], 0.99) / 1000.0 for t in turns]
        remaining = [percentile([p["remaining_ms"] for p in grouped[t]], 0.99) / 1000.0 for t in turns]
        ymax = max(ymax, max(b + p + r for b, p, r in zip(blocking, prefill, remaining)))
        stacked_data.append((turns, blocking, prefill, remaining))

    for idx, (ax, (group_label, _), (turns, blocking, prefill, remaining)) in enumerate(zip(axes, groups, stacked_data)):
        ax.stackplot(
            turns,
            blocking,
            prefill,
            remaining,
            colors=[BLOCKING_COLOR, PREFILL_COLOR, REMAINING_COLOR],
            alpha=0.92,
        )
        ax.set_xlim(1, 10)
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Turn")
        ax.set_box_aspect(0.48)
        ax.grid(axis="y", alpha=0.2)
        ax.text(0.03, 0.97, group_label, transform=ax.transAxes, va="top", ha="left")
        ax.text(
            0.5,
            -0.24,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            clip_on=False,
        )
    axes[0].set_ylabel("P99 Time (sec)")
    axes[0].legend(
        ["Blocking Time", "Prefill Time", "Remaining TTFT"],
        loc="upper right",
        frameon=True,
        fontsize=14,
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=1.7,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_p99_blocking_prefill_sum_by_turn(points):
    group1_limit, group2_limit = actual_group_limits(points)
    groups = [
        (
            group_label_for_limit(group1_limit, group1_limit, group2_limit),
            [p for p in points if p["history_limit_tokens"] == group1_limit],
        ),
    ]
    if group2_limit != group1_limit:
        groups.append(
            (
                group_label_for_limit(group2_limit, group1_limit, group2_limit),
                [p for p in points if p["history_limit_tokens"] == group2_limit],
            )
        )

    rows = []
    for group_label, group_points in groups:
        grouped = defaultdict(list)
        for point in group_points:
            grouped[point["turn_index"]].append(point)
        for turn in range(1, 11):
            turn_points = grouped[turn]
            p99_blocking_ms = percentile([p["blocking_ms"] for p in turn_points], 0.99)
            p99_prefill_ms = percentile([p["prefill_ms"] for p in turn_points], 0.99)
            p99_remaining_ms = percentile([p["remaining_ms"] for p in turn_points], 0.99)
            rows.append(
                {
                    "group_label": group_label,
                    "turn_index": turn,
                    "p99_blocking_ms": p99_blocking_ms,
                    "p99_prefill_ms": p99_prefill_ms,
                    "p99_blocking_plus_prefill_ms": p99_blocking_ms + p99_prefill_ms,
                    "p99_remaining_ms": p99_remaining_ms,
                    "p99_total_components_ms": p99_blocking_ms
                    + p99_prefill_ms
                    + p99_remaining_ms,
                }
            )
    return rows


def write_p99_blocking_prefill_sum_csv(rows, out_path):
    fieldnames = [
        "group_label",
        "turn_index",
        "p99_blocking_ms",
        "p99_prefill_ms",
        "p99_blocking_plus_prefill_ms",
        "p99_remaining_ms",
        "p99_total_components_ms",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_p99_blocking_prefill_sum_by_turn(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.18, top=0.96)
    group_labels = []
    for row in rows:
        if row["group_label"] not in group_labels:
            group_labels.append(row["group_label"])

    markers = ["o", "s", "^", "D"]
    colors = [GROUP1_COLOR, GROUP2_COLOR, "#059669", "#7c3aed"]
    for index, group_label in enumerate(group_labels):
        subset = [row for row in rows if row["group_label"] == group_label]
        ax.plot(
            [row["turn_index"] for row in subset],
            [row["p99_blocking_plus_prefill_ms"] / 1000.0 for row in subset],
            marker=markers[index % len(markers)],
            linewidth=2.6,
            markersize=7,
            color=colors[index % len(colors)],
            label=group_label,
        )

    ax.set_xlim(1, 10)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Turn")
    ax.set_ylabel("P99 Blocking + P99 Prefill (sec)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_p99_component_totals_by_group(rows):
    totals = []
    group_labels = []
    for row in rows:
        if row["group_label"] not in group_labels:
            group_labels.append(row["group_label"])
    for group_label in group_labels:
        subset = [row for row in rows if row["group_label"] == group_label]
        blocking_ms = sum(row["p99_blocking_ms"] for row in subset)
        prefill_ms = sum(row["p99_prefill_ms"] for row in subset)
        remaining_ms = sum(row["p99_remaining_ms"] for row in subset)
        totals.append(
            {
                "group_label": group_label,
                "sum_p99_blocking_ms": blocking_ms,
                "sum_p99_prefill_ms": prefill_ms,
                "sum_p99_remaining_ms": remaining_ms,
                "sum_p99_total_components_ms": blocking_ms + prefill_ms + remaining_ms,
            }
        )
    return totals


def write_p99_component_totals_csv(rows, out_path):
    fieldnames = [
        "group_label",
        "sum_p99_blocking_ms",
        "sum_p99_prefill_ms",
        "sum_p99_remaining_ms",
        "sum_p99_total_components_ms",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_p99_component_totals_by_group(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.2, top=0.94)

    labels = [row["group_label"] for row in rows]
    xs = list(range(len(rows)))
    blocking = [row["sum_p99_blocking_ms"] / 1000.0 for row in rows]
    prefill = [row["sum_p99_prefill_ms"] / 1000.0 for row in rows]
    remaining = [row["sum_p99_remaining_ms"] / 1000.0 for row in rows]

    ax.bar(xs, blocking, color=BLOCKING_COLOR, label="Blocking Time")
    ax.bar(xs, prefill, bottom=blocking, color=PREFILL_COLOR, label="Prefill Time")
    ax.bar(
        xs,
        remaining,
        bottom=[b + p for b, p in zip(blocking, prefill)],
        color=REMAINING_COLOR,
        label="Remaining TTFT",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Sum of component time (sec)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=13)

    for x, total in zip(xs, [b + p + r for b, p, r in zip(blocking, prefill, remaining)]):
        ax.text(x, total, f"{total:.1f}s", ha="center", va="bottom", fontsize=12)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_group_tbt_by_turn(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["tenant_count"],
                row["history_limit_tokens"],
                row["turn_index"],
            )
        ].append(row["p95_tbt_ms"])

    out_rows = []
    group1_limit, group2_limit = actual_group_limits(rows)
    for (tenant_count, history_limit, turn_index), values in sorted(grouped.items()):
        out_rows.append(
            {
                "tenant_count": tenant_count,
                "group_label": group_label_for_limit(history_limit, group1_limit, group2_limit),
                "turn_index": turn_index,
                "mean_p95_tbt_ms": mean(values),
                "p95_p95_tbt_ms": percentile(values, 0.95),
                "n": len(values),
            }
        )
    return out_rows


def write_group_tbt_by_turn_csv(rows, out_path):
    fieldnames = [
        "tenant_count",
        "group_label",
        "turn_index",
        "mean_p95_tbt_ms",
        "p95_p95_tbt_ms",
        "n",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_group_tbt_by_turn(plt, rows, out_path):
    tenant_counts = [str(value) for value in sorted({int(row["tenant_count"]) for row in rows})]
    fig, axes = plt.subplots(
        1,
        max(1, len(tenant_counts)),
        figsize=(6.2 * max(1, len(tenant_counts)), 5.8),
        sharey=True,
    )
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.94, wspace=0.12)

    group_labels = []
    for row in rows:
        if row["group_label"] not in group_labels:
            group_labels.append(row["group_label"])

    colors = [GROUP1_COLOR, GROUP2_COLOR, "#059669", "#7c3aed"]
    markers = ["o", "s", "^", "D"]
    ymax = max((row["mean_p95_tbt_ms"] / 1000.0 for row in rows), default=1.0)

    for idx, (ax, tenant_count) in enumerate(zip(axes, tenant_counts)):
        for group_index, group_label in enumerate(group_labels):
            subset = [
                row
                for row in rows
                if row["tenant_count"] == tenant_count and row["group_label"] == group_label
            ]
            ax.plot(
                [row["turn_index"] for row in subset],
                [row["mean_p95_tbt_ms"] / 1000.0 for row in subset],
                marker=markers[group_index % len(markers)],
                linewidth=2.4,
                markersize=6,
                color=colors[group_index % len(colors)],
                label=group_label,
            )
        ax.set_xlim(1, 10)
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Turn")
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.03, 0.95, f"tenants={tenant_count}", transform=ax.transAxes, va="top", ha="left")
        ax.text(0.5, -0.28, f"({chr(ord('a') + idx)})", transform=ax.transAxes, ha="center", va="top")
    axes[0].set_ylabel("Mean p95 TBT (sec)")
    axes[0].legend(loc="upper left", frameon=True, fontsize=12)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_group_tbt_pair_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["exp_label"],
                row["tenant_count"],
                row["turn_index"],
                row["history_limit_tokens"],
            )
        ].append(row["p95_tbt_ms"])

    group1_limit, group2_limit = actual_group_limits(rows)
    pair_rows = []
    keys = sorted({(row["exp_label"], row["tenant_count"], row["turn_index"]) for row in rows})
    for exp_label, tenant_count, turn_index in keys:
        group1_values = grouped[(exp_label, tenant_count, turn_index, group1_limit)]
        group2_values = grouped[(exp_label, tenant_count, turn_index, group2_limit)]
        if not group1_values or not group2_values:
            continue
        pair_rows.append(
            {
                "exp_label": exp_label,
                "tenant_count": tenant_count,
                "turn_index": turn_index,
                "group1_mean_p95_tbt_ms": mean(group1_values),
                "group2_mean_p95_tbt_ms": mean(group2_values),
            }
        )
    return pair_rows


def write_group_tbt_pair_csv(rows, out_path):
    fieldnames = [
        "exp_label",
        "tenant_count",
        "turn_index",
        "group1_mean_p95_tbt_ms",
        "group2_mean_p95_tbt_ms",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_group_tbt_pair_scatter(plt, rows, out_path):
    tenant_counts = [str(value) for value in sorted({int(row["tenant_count"]) for row in rows})]
    fig, axes = plt.subplots(
        1,
        max(1, len(tenant_counts)),
        figsize=(6.2 * max(1, len(tenant_counts)), 5.8),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.94, wspace=0.12)

    all_values = [
        value / 1000.0
        for row in rows
        for value in (row["group1_mean_p95_tbt_ms"], row["group2_mean_p95_tbt_ms"])
    ]
    xlim = padded_limits(all_values)
    ylim = xlim
    handles = {}

    for idx, (ax, tenant_count) in enumerate(zip(axes, tenant_counts)):
        subset = [row for row in rows if row["tenant_count"] == tenant_count]
        for turn in range(1, 11):
            turn_rows = [row for row in subset if row["turn_index"] == turn]
            sc = ax.scatter(
                [row["group1_mean_p95_tbt_ms"] / 1000.0 for row in turn_rows],
                [row["group2_mean_p95_tbt_ms"] / 1000.0 for row in turn_rows],
                s=68,
                alpha=0.82,
                color=TURN_COLORS[turn],
                edgecolors="white",
                linewidths=0.45,
            )
            handles.setdefault(turn, sc)
        ax.plot(xlim, ylim, color="#111827", linewidth=1.2, linestyle="--", alpha=0.75)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Group1 mean p95 TBT (sec)")
        ax.set_box_aspect(1.0)
        ax.grid(alpha=0.22)
        ax.text(
            0.03,
            0.95,
            f"tenants={tenant_count}\nr={corr([r['group1_mean_p95_tbt_ms'] for r in subset], [r['group2_mean_p95_tbt_ms'] for r in subset]):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        ax.text(0.5, -0.28, f"({chr(ord('a') + idx)})", transform=ax.transAxes, ha="center", va="top")
    axes[0].set_ylabel("Group2 mean p95 TBT (sec)")
    axes[-1].legend(
        [handles[t] for t in range(1, 11)],
        [f"Turn {t}" for t in range(1, 11)],
        loc="lower right",
        frameon=True,
        ncol=2,
        fontsize=11,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tbt_vs_tokens(plt, rows, token_field, xlabel, out_path):
    tenant_counts = [str(value) for value in sorted({int(row["tenant_count"]) for row in rows})]
    group1_limit, group2_limit = actual_group_limits(rows)
    fig, axes = plt.subplots(
        1,
        max(1, len(tenant_counts)),
        figsize=(6.2 * max(1, len(tenant_counts)), 5.8),
        sharey=True,
    )
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.94, wspace=0.12)

    xlim = padded_limits([row[token_field] for row in rows])
    ylim = padded_limits([row["p95_tbt_ms"] / 1000.0 for row in rows])
    limits = [group1_limit]
    if group2_limit != group1_limit:
        limits.append(group2_limit)

    for idx, (ax, tenant_count) in enumerate(zip(axes, tenant_counts)):
        subset = [row for row in rows if row["tenant_count"] == tenant_count]
        for limit in limits:
            group_rows = [row for row in subset if row["history_limit_tokens"] == limit]
            ax.scatter(
                [row[token_field] for row in group_rows],
                [row["p95_tbt_ms"] / 1000.0 for row in group_rows],
                s=30,
                alpha=0.42,
                color=group_color_for_limit(limit, group1_limit, group2_limit),
                edgecolors="none",
                label=group_label_for_limit(limit, group1_limit, group2_limit),
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_box_aspect(2.0 / 3.0)
        ax.grid(alpha=0.22)
        ax.text(
            0.03,
            0.95,
            f"tenants={tenant_count}\nr={corr([r[token_field] for r in subset], [r['p95_tbt_ms'] for r in subset]):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        ax.text(0.5, -0.28, f"({chr(ord('a') + idx)})", transform=ax.transAxes, ha="center", va="top")
    axes[0].set_ylabel("p95 TBT (sec)")
    axes[0].legend(loc="upper left", frameon=True, fontsize=11)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_total_tbt_sum_by_turn(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["exp_label"], row["tenant_count"], row["turn_index"])].append(row)

    detail_rows = []
    for (exp_label, tenant_count, turn_index), subset in sorted(grouped.items()):
        total_tbt_ms = sum(row["p95_tbt_ms"] for row in subset)
        detail_rows.append(
            {
                "exp_label": exp_label,
                "tenant_count": tenant_count,
                "turn_index": turn_index,
                "sum_p95_tbt_ms": total_tbt_ms,
                "mean_p95_tbt_ms": mean([row["p95_tbt_ms"] for row in subset]),
                "sum_input_tokens": sum(row["input_tokens"] for row in subset),
                "sum_output_tokens": sum(row["output_tokens"] for row in subset),
                "n": len(subset),
            }
        )

    summary_bucket = defaultdict(list)
    for row in detail_rows:
        summary_bucket[(row["tenant_count"], row["turn_index"])].append(row)

    summary_rows = []
    for (tenant_count, turn_index), subset in sorted(summary_bucket.items()):
        values = [row["sum_p95_tbt_ms"] for row in subset]
        summary_rows.append(
            {
                "tenant_count": tenant_count,
                "turn_index": turn_index,
                "mean_sum_p95_tbt_ms": mean(values),
                "min_sum_p95_tbt_ms": min(values),
                "max_sum_p95_tbt_ms": max(values),
                "mean_sum_input_tokens": mean([row["sum_input_tokens"] for row in subset]),
                "mean_sum_output_tokens": mean([row["sum_output_tokens"] for row in subset]),
                "runs": len(subset),
            }
        )
    return detail_rows, summary_rows


def write_total_tbt_sum_csv(rows, out_path):
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_total_tbt_sum_by_turn(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.18, top=0.96)

    tenant_counts = [str(value) for value in sorted({int(row["tenant_count"]) for row in rows})]
    colors = {
        "8": "#16a34a",
        "16": "#f97316",
        "32": "#dc2626",
    }
    turns = list(range(1, 11))
    bar_width = 0.22
    offsets = {
        tenant_count: (index - (len(tenant_counts) - 1) / 2.0) * bar_width
        for index, tenant_count in enumerate(tenant_counts)
    }

    for tenant_count in tenant_counts:
        subset = [row for row in rows if row["tenant_count"] == tenant_count]
        row_by_turn = {row["turn_index"]: row for row in subset}
        xs = [turn + offsets[tenant_count] for turn in turns]
        means = [row["mean_sum_p95_tbt_ms"] / 1000.0 for row in subset]
        mins = [row["min_sum_p95_tbt_ms"] / 1000.0 for row in subset]
        maxs = [row["max_sum_p95_tbt_ms"] / 1000.0 for row in subset]
        color = colors.get(tenant_count, "#2563eb")
        means = [row_by_turn[turn]["mean_sum_p95_tbt_ms"] / 1000.0 for turn in turns]
        mins = [row_by_turn[turn]["min_sum_p95_tbt_ms"] / 1000.0 for turn in turns]
        maxs = [row_by_turn[turn]["max_sum_p95_tbt_ms"] / 1000.0 for turn in turns]
        lower_err = [max(0.0, mean_value - min_value) for mean_value, min_value in zip(means, mins)]
        upper_err = [max(0.0, max_value - mean_value) for mean_value, max_value in zip(means, maxs)]
        ax.bar(
            xs,
            means,
            width=bar_width * 0.92,
            color=color,
            alpha=0.88,
            label=f"{tenant_count} tenants",
        )
        ax.errorbar(
            xs,
            means,
            yerr=[lower_err, upper_err],
            fmt="none",
            ecolor="#111827",
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.75,
        )

    ax.set_xlim(0.45, 10.55)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Turn")
    ax.set_ylabel("Sum of tenant p95 TBT (sec)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=13)
    ax.text(
        0.03,
        0.06,
        "Bar: exp1-3 mean, error bar: min-max",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_figure_f_rows(rows, points, total_tbt_summary, tenant_count="32"):
    tbt_by_turn = {
        row["turn_index"]: row
        for row in total_tbt_summary
        if row["tenant_count"] == tenant_count
    }
    ttft_by_turn = defaultdict(list)
    for row in rows:
        if row["tenant_count"] == tenant_count:
            ttft_by_turn[row["turn_index"]].append(row["ttft_ms"])

    component_by_turn = defaultdict(list)
    for point in points:
        component_by_turn[point["turn_index"]].append(point)

    out_rows = []
    for turn in range(1, 11):
        tbt_row = tbt_by_turn.get(turn, {})
        component_rows = component_by_turn[turn]
        out_rows.append(
            {
                "tenant_count": tenant_count,
                "turn_index": turn,
                "mean_sum_p95_tbt_ms": tbt_row.get("mean_sum_p95_tbt_ms", 0.0),
                "min_sum_p95_tbt_ms": tbt_row.get("min_sum_p95_tbt_ms", 0.0),
                "max_sum_p95_tbt_ms": tbt_row.get("max_sum_p95_tbt_ms", 0.0),
                "p99_blocking_ms": percentile([p["blocking_ms"] for p in component_rows], 0.99),
                "p99_prefill_ms": percentile([p["prefill_ms"] for p in component_rows], 0.99),
                "p99_ttft_ms": percentile(ttft_by_turn[turn], 0.99),
            }
        )
    return out_rows


def write_figure_f_csv(rows, out_path):
    fieldnames = [
        "tenant_count",
        "turn_index",
        "mean_sum_p95_tbt_ms",
        "min_sum_p95_tbt_ms",
        "max_sum_p95_tbt_ms",
        "p99_blocking_ms",
        "p99_prefill_ms",
        "p99_ttft_ms",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_figure_f_tbt_vs_ttft_components(plt, rows, out_path):
    turns = [row["turn_index"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.96, wspace=0.24)

    ax0, ax1 = axes
    tbt_mean = [row["mean_sum_p95_tbt_ms"] / 1000.0 for row in rows]
    tbt_min = [row["min_sum_p95_tbt_ms"] / 1000.0 for row in rows]
    tbt_max = [row["max_sum_p95_tbt_ms"] / 1000.0 for row in rows]
    lower_err = [max(0.0, mean_value - min_value) for mean_value, min_value in zip(tbt_mean, tbt_min)]
    upper_err = [max(0.0, max_value - mean_value) for mean_value, max_value in zip(tbt_mean, tbt_max)]

    ax0.bar(turns, tbt_mean, color="#dc2626", alpha=0.88, width=0.68)
    ax0.errorbar(
        turns,
        tbt_mean,
        yerr=[lower_err, upper_err],
        fmt="none",
        ecolor="#111827",
        elinewidth=1.0,
        capsize=2.5,
        alpha=0.75,
    )
    ax0.set_xlim(0.45, 10.55)
    ax0.set_xticks(turns)
    ax0.set_xlabel("Turn")
    ax0.set_ylabel("Sum of tenant p95 TBT (sec)")
    ax0.grid(axis="y", alpha=0.25)
    ax0.text(0.03, 0.95, "Decode step latency proxy", transform=ax0.transAxes, va="top", ha="left")
    ax0.text(0.5, -0.24, "(a)", transform=ax0.transAxes, ha="center", va="top")

    ax1.plot(
        turns,
        [row["p99_blocking_ms"] / 1000.0 for row in rows],
        marker="o",
        linewidth=2.5,
        color=BLOCKING_COLOR,
        label="P99 Blocking Time",
    )
    ax1.plot(
        turns,
        [row["p99_prefill_ms"] / 1000.0 for row in rows],
        marker="s",
        linewidth=2.5,
        color=PREFILL_COLOR,
        label="P99 Prefill Time",
    )
    ax1.plot(
        turns,
        [row["p99_ttft_ms"] / 1000.0 for row in rows],
        marker="^",
        linewidth=2.5,
        color=REMAINING_COLOR,
        label="P99 TTFT",
    )
    ax1.set_xlim(0.45, 10.55)
    ax1.set_xticks(turns)
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("P99 Time (sec)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", frameon=True, fontsize=12)
    ax1.text(0.03, 0.95, "Queue/prefill dominated latency", transform=ax1.transAxes, va="top", ha="left")
    ax1.text(0.5, -0.24, "(b)", transform=ax1.transAxes, ha="center", va="top")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_context_tbt_by_turn(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["exp_label"], row["tenant_count"], row["turn_index"])].append(row)

    detail_rows = []
    for (exp_label, tenant_count, turn_index), subset in sorted(grouped.items()):
        detail_rows.append(
            {
                "exp_label": exp_label,
                "tenant_count": tenant_count,
                "turn_index": turn_index,
                "sum_input_tokens": sum(row["input_tokens"] for row in subset),
                "sum_output_tokens": sum(row["output_tokens"] for row in subset),
                "sum_uncached_context_tokens": sum(
                    max(0.0, row["input_tokens"] - row["prefix_hit_tokens"]) for row in subset
                ),
                "sum_p95_tbt_ms": sum(row["p95_tbt_ms"] for row in subset),
                "mean_p95_tbt_ms": mean([row["p95_tbt_ms"] for row in subset]),
                "n": len(subset),
            }
        )

    summary_bucket = defaultdict(list)
    for row in detail_rows:
        summary_bucket[(row["tenant_count"], row["turn_index"])].append(row)

    summary_rows = []
    for (tenant_count, turn_index), subset in sorted(summary_bucket.items()):
        summary_rows.append(
            {
                "tenant_count": tenant_count,
                "turn_index": turn_index,
                "mean_sum_input_tokens": mean([row["sum_input_tokens"] for row in subset]),
                "mean_sum_output_tokens": mean([row["sum_output_tokens"] for row in subset]),
                "mean_sum_uncached_context_tokens": mean(
                    [row["sum_uncached_context_tokens"] for row in subset]
                ),
                "mean_sum_p95_tbt_ms": mean([row["sum_p95_tbt_ms"] for row in subset]),
                "min_sum_p95_tbt_ms": min(row["sum_p95_tbt_ms"] for row in subset),
                "max_sum_p95_tbt_ms": max(row["sum_p95_tbt_ms"] for row in subset),
                "mean_p95_tbt_ms": mean([row["mean_p95_tbt_ms"] for row in subset]),
                "runs": len(subset),
            }
        )
    return detail_rows, summary_rows


def write_context_tbt_csv(rows, out_path):
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_context_tbt_by_turn(plt, rows, out_path):
    tenant_counts = [str(value) for value in sorted({int(row["tenant_count"]) for row in rows})]
    fig, axes = plt.subplots(
        1,
        max(1, len(tenant_counts)),
        figsize=(7.8 * max(1, len(tenant_counts)), 6.2),
        sharey=False,
    )
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.2, top=0.94, wspace=0.5)

    for idx, (ax, tenant_count) in enumerate(zip(axes, tenant_counts)):
        subset = [row for row in rows if row["tenant_count"] == tenant_count]
        turns = [row["turn_index"] for row in subset]
        input_k = [row["mean_sum_input_tokens"] / 1000.0 for row in subset]
        uncached_k = [row["mean_sum_uncached_context_tokens"] / 1000.0 for row in subset]
        tbt_s = [row["mean_sum_p95_tbt_ms"] / 1000.0 for row in subset]
        lower_err = [
            max(0.0, row["mean_sum_p95_tbt_ms"] - row["min_sum_p95_tbt_ms"]) / 1000.0
            for row in subset
        ]
        upper_err = [
            max(0.0, row["max_sum_p95_tbt_ms"] - row["mean_sum_p95_tbt_ms"]) / 1000.0
            for row in subset
        ]

        ax.bar(
            [turn - 0.16 for turn in turns],
            input_k,
            width=0.32,
            color="#94a3b8",
            alpha=0.82,
            label="Total input context",
        )
        ax.bar(
            [turn + 0.16 for turn in turns],
            uncached_k,
            width=0.32,
            color="#64748b",
            alpha=0.82,
            label="Uncached context",
        )
        ax.set_xlim(0.45, 10.55)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Turn")
        ax.set_ylabel("Context tokens in batch (K)")
        ax.grid(axis="y", alpha=0.22)
        ax.text(0.03, 0.95, f"tenants={tenant_count}", transform=ax.transAxes, va="top", ha="left")
        ax.text(0.5, -0.26, f"({chr(ord('a') + idx)})", transform=ax.transAxes, ha="center", va="top")

        ax_tbt = ax.twinx()
        ax_tbt.plot(
            turns,
            tbt_s,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color="#dc2626",
            label="Sum p95 TBT",
        )
        ax_tbt.errorbar(
            turns,
            tbt_s,
            yerr=[lower_err, upper_err],
            fmt="none",
            ecolor="#7f1d1d",
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.75,
        )
        ax_tbt.tick_params(axis="y", colors="#dc2626", pad=2)
        ax_tbt.spines["right"].set_color("#dc2626")
        if idx == len(tenant_counts) - 1:
            ax_tbt.set_ylabel("Sum of tenant p95 TBT (sec)", color="#dc2626", labelpad=10)
        else:
            ax_tbt.set_ylabel("")

        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax_tbt.get_legend_handles_labels()
            ax.legend(handles + handles2, labels + labels2, loc="upper left", frameon=True, fontsize=10)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_context_tbt_normalized_by_turn(plt, rows, out_path):
    tenant_counts = [str(value) for value in sorted({int(row["tenant_count"]) for row in rows})]
    fig, axes = plt.subplots(
        1,
        max(1, len(tenant_counts)),
        figsize=(6.4 * max(1, len(tenant_counts)), 5.8),
        sharey=False,
    )
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.94, wspace=0.26)

    for idx, (ax, tenant_count) in enumerate(zip(axes, tenant_counts)):
        subset = [row for row in rows if row["tenant_count"] == tenant_count]
        subset.sort(key=lambda row: row["turn_index"])
        turns = [row["turn_index"] for row in subset]
        base_input = subset[0]["mean_sum_input_tokens"] if subset else 1.0
        base_tbt = subset[0]["mean_sum_p95_tbt_ms"] if subset else 1.0
        input_norm = [
            row["mean_sum_input_tokens"] / base_input if base_input else 0.0 for row in subset
        ]
        tbt_norm = [row["mean_sum_p95_tbt_ms"] / base_tbt if base_tbt else 0.0 for row in subset]

        ax.plot(
            turns,
            input_norm,
            marker="s",
            linewidth=2.4,
            markersize=6,
            color="#64748b",
            label="Total input context / turn1",
        )
        ax.plot(
            turns,
            tbt_norm,
            marker="o",
            linewidth=2.4,
            markersize=6,
            color="#dc2626",
            label="Sum p95 TBT / turn1",
        )
        ax.set_xlim(1, 10)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Turn")
        ax.set_ylabel("Normalized value")
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.03, 0.95, f"tenants={tenant_count}", transform=ax.transAxes, va="top", ha="left")
        ax.text(0.5, -0.26, f"({chr(ord('a') + idx)})", transform=ax.transAxes, ha="center", va="top")
        if idx == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=10)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_context_per_sec_rows(context_rows, figure_f_rows):
    figure_f_by_turn = {
        (row["tenant_count"], row["turn_index"]): row
        for row in figure_f_rows
    }
    out_rows = []
    for row in context_rows:
        tenant_count = row["tenant_count"]
        turn_index = row["turn_index"]
        sum_tbt_s = row["mean_sum_p95_tbt_ms"] / 1000.0
        ttft_row = figure_f_by_turn.get((tenant_count, turn_index), {})
        p99_ttft_s = float(ttft_row.get("p99_ttft_ms", 0.0)) / 1000.0
        p99_blocking_s = float(ttft_row.get("p99_blocking_ms", 0.0)) / 1000.0
        p99_prefill_s = float(ttft_row.get("p99_prefill_ms", 0.0)) / 1000.0
        out_rows.append(
            {
                "tenant_count": tenant_count,
                "turn_index": turn_index,
                "mean_sum_input_tokens": row["mean_sum_input_tokens"],
                "mean_sum_uncached_context_tokens": row["mean_sum_uncached_context_tokens"],
                "mean_sum_p95_tbt_ms": row["mean_sum_p95_tbt_ms"],
                "input_context_per_sum_tbt_sec": (
                    row["mean_sum_input_tokens"] / sum_tbt_s if sum_tbt_s > 0 else 0.0
                ),
                "uncached_context_per_sum_tbt_sec": (
                    row["mean_sum_uncached_context_tokens"] / sum_tbt_s if sum_tbt_s > 0 else 0.0
                ),
                "input_context_per_p99_ttft_sec": (
                    row["mean_sum_input_tokens"] / p99_ttft_s if p99_ttft_s > 0 else 0.0
                ),
                "input_context_per_p99_blocking_plus_prefill_sec": (
                    row["mean_sum_input_tokens"] / (p99_blocking_s + p99_prefill_s)
                    if (p99_blocking_s + p99_prefill_s) > 0
                    else 0.0
                ),
            }
        )
    return out_rows


def write_context_per_sec_csv(rows, out_path):
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_context_per_sec_by_turn(plt, rows, out_path):
    tenant_counts = [str(value) for value in sorted({int(row["tenant_count"]) for row in rows})]
    fig, axes = plt.subplots(
        1,
        max(1, len(tenant_counts)),
        figsize=(6.4 * max(1, len(tenant_counts)), 5.8),
        sharey=False,
    )
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.94, wspace=0.28)

    for idx, (ax, tenant_count) in enumerate(zip(axes, tenant_counts)):
        subset = [row for row in rows if row["tenant_count"] == tenant_count]
        subset.sort(key=lambda row: row["turn_index"])
        turns = [row["turn_index"] for row in subset]
        ax.plot(
            turns,
            [row["input_context_per_sum_tbt_sec"] / 1000.0 for row in subset],
            marker="o",
            linewidth=2.5,
            markersize=6,
            color="#dc2626",
            label="Input context / sum p95 TBT",
        )
        ax.plot(
            turns,
            [row["uncached_context_per_sum_tbt_sec"] / 1000.0 for row in subset],
            marker="s",
            linewidth=2.3,
            markersize=6,
            color="#64748b",
            label="Uncached context / sum p95 TBT",
        )
        ax.set_xlim(1, 10)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Turn")
        ax.set_ylabel("Context processed per TBT-sec (K tokens/sec)")
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.03, 0.95, f"tenants={tenant_count}", transform=ax.transAxes, va="top", ha="left")
        ax.text(0.5, -0.26, f"({chr(ord('a') + idx)})", transform=ax.transAxes, ha="center", va="top")
        if idx == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=10)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_blocking_and_prefill(plt, rows, rows_lt, rows_gt, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.97, wspace=0.22)

    ax0 = axes[0]
    group1_limit, group2_limit = actual_group_limits(rows)
    x0 = [r["blocking_time_ms"] / 1000.0 for r in rows]
    y0 = [r["ttft_ms"] / 1000.0 for r in rows]
    x0lim = padded_limits(x0)
    y0lim = padded_limits(y0)
    ax0.scatter(
        [
            r["blocking_time_ms"] / 1000.0
            for r in rows
            if r["history_limit_tokens"] == group1_limit
        ],
        [
            r["ttft_ms"] / 1000.0
            for r in rows
            if r["history_limit_tokens"] == group1_limit
        ],
        s=26,
        alpha=0.42,
        color=group_color_for_limit(group1_limit, group1_limit, group2_limit),
        edgecolors="none",
        label=group_label_for_limit(group1_limit, group1_limit, group2_limit),
    )
    if group2_limit != group1_limit:
        ax0.scatter(
            [
                r["blocking_time_ms"] / 1000.0
                for r in rows
                if r["history_limit_tokens"] == group2_limit
            ],
            [
                r["ttft_ms"] / 1000.0
                for r in rows
                if r["history_limit_tokens"] == group2_limit
            ],
            s=26,
            alpha=0.42,
            color=group_color_for_limit(group2_limit, group1_limit, group2_limit),
            edgecolors="none",
            label=group_label_for_limit(group2_limit, group1_limit, group2_limit),
        )
    ax0.set_xlim(*x0lim)
    ax0.set_ylim(*y0lim)
    ax0.set_xlabel("Blocking Time (sec)")
    ax0.set_ylabel("TTFT (sec)")
    ax0.set_box_aspect(2.0 / 3.0)
    ax0.grid(alpha=0.2)
    ax0.text(0.03, 0.97, "All tenants", transform=ax0.transAxes, va="top", ha="left")
    ax0.text(0.5, -0.30, "(a)", transform=ax0.transAxes, ha="center", va="top")

    all_prefill_rows = rows_lt + rows_gt
    xlim = padded_limits([r["batch_compute_ratio"] for r in all_prefill_rows])
    ylim = padded_limits([r["batch_mean_prefill_ms"] / 1000.0 for r in all_prefill_rows])
    turn_handles = {}
    for ax, panel_rows, label in [
        (axes[1], rows_lt, "blocking < 100 ms"),
        (axes[2], rows_gt, "blocking > 100 ms"),
    ]:
        for turn in range(1, 11):
            subset = [r for r in panel_rows if r["turn_index"] == turn]
            sc = ax.scatter(
                [r["batch_compute_ratio"] for r in subset],
                [r["batch_mean_prefill_ms"] / 1000.0 for r in subset],
                s=62,
                alpha=0.78,
                color=TURN_COLORS[turn],
                edgecolors="white",
                linewidths=0.4,
            )
            turn_handles.setdefault(turn, sc)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Normalized compute-token load")
        ax.set_box_aspect(2.0 / 3.0)
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.97,
            f"{label}\nr = {corr([r['batch_compute_ratio'] for r in panel_rows], [r['batch_mean_prefill_ms'] for r in panel_rows]):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
    axes[1].set_ylabel("Prefill time average (sec)")
    axes[1].text(0.5, -0.30, "(b)", transform=axes[1].transAxes, ha="center", va="top")
    axes[2].text(0.5, -0.30, "(c)", transform=axes[2].transAxes, ha="center", va="top")

    ax0.legend(
        loc="lower right",
        frameon=True,
        fontsize=14,
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=1.7,
    )
    axes[1].legend(
        [turn_handles[t] for t in range(1, 11)],
        [f"Turn {t}" for t in range(1, 11)],
        loc="lower right",
        frameon=True,
        ncol=2,
        fontsize=13,
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=1.5,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    plt.rcParams.update(
        {
            "font.size": 19,
            "axes.labelsize": 21,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "legend.title_fontsize": 17,
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_success_rows()
    points_32 = load_breakdown_points("32")
    prefill_lt = load_prefill_batches("lt100", "32")
    prefill_gt = load_prefill_batches("gt100", "32")

    plot_ttft_vs_prefix_hit_rate(
        plt,
        rows,
        OUT_DIR / "paper_ttft_vs_prefix_hit_rate_by_tenantcount.png",
    )
    plot_p99_breakdown_by_turn(
        plt,
        points_32,
        OUT_DIR / "paper_ttft_breakdown_group_p99_by_turn_32tenants.png",
        fig_height=6.8,
    )
    plot_p99_breakdown_by_turn(
        plt,
        points_32,
        OUT_DIR / "paper_ttft_breakdown_group_p99_by_turn_32tenants_shorter_height.png",
        fig_height=5.6,
    )
    p99_blocking_prefill_sum = build_p99_blocking_prefill_sum_by_turn(points_32)
    write_p99_blocking_prefill_sum_csv(
        p99_blocking_prefill_sum,
        OUT_DIR / "paper_p99_blocking_prefill_sum_by_turn_32tenants.csv",
    )
    plot_p99_blocking_prefill_sum_by_turn(
        plt,
        p99_blocking_prefill_sum,
        OUT_DIR / "paper_p99_blocking_prefill_sum_by_turn_32tenants.png",
    )
    p99_component_totals = build_p99_component_totals_by_group(p99_blocking_prefill_sum)
    write_p99_component_totals_csv(
        p99_component_totals,
        OUT_DIR / "paper_p99_component_totals_by_group_32tenants.csv",
    )
    plot_p99_component_totals_by_group(
        plt,
        p99_component_totals,
        OUT_DIR / "paper_p99_component_totals_by_group_32tenants.png",
    )

    group_tbt_by_turn = build_group_tbt_by_turn(rows)
    write_group_tbt_by_turn_csv(
        group_tbt_by_turn,
        OUT_DIR / "paper_figure_a_group_tbt_by_turn.csv",
    )
    plot_group_tbt_by_turn(
        plt,
        group_tbt_by_turn,
        OUT_DIR / "paper_figure_a_group_tbt_by_turn.png",
    )

    group_tbt_pairs = build_group_tbt_pair_rows(rows)
    write_group_tbt_pair_csv(
        group_tbt_pairs,
        OUT_DIR / "paper_figure_b_group1_vs_group2_tbt.csv",
    )
    plot_group_tbt_pair_scatter(
        plt,
        group_tbt_pairs,
        OUT_DIR / "paper_figure_b_group1_vs_group2_tbt.png",
    )

    plot_tbt_vs_tokens(
        plt,
        rows,
        "input_tokens",
        "Input tokens",
        OUT_DIR / "paper_figure_c_tbt_vs_input_tokens.png",
    )
    plot_tbt_vs_tokens(
        plt,
        rows,
        "output_tokens",
        "Output tokens",
        OUT_DIR / "paper_figure_d_tbt_vs_output_tokens.png",
    )

    total_tbt_detail, total_tbt_summary = build_total_tbt_sum_by_turn(rows)
    write_total_tbt_sum_csv(
        total_tbt_detail,
        OUT_DIR / "paper_figure_e_total_tbt_sum_by_turn_detail.csv",
    )
    write_total_tbt_sum_csv(
        total_tbt_summary,
        OUT_DIR / "paper_figure_e_total_tbt_sum_by_turn_summary.csv",
    )
    plot_total_tbt_sum_by_turn(
        plt,
        total_tbt_summary,
        OUT_DIR / "paper_figure_e_total_tbt_sum_by_turn.png",
    )
    figure_f_rows = build_figure_f_rows(rows, points_32, total_tbt_summary, "32")
    write_figure_f_csv(
        figure_f_rows,
        OUT_DIR / "paper_figure_f_tbt_vs_ttft_components_32tenants.csv",
    )
    plot_figure_f_tbt_vs_ttft_components(
        plt,
        figure_f_rows,
        OUT_DIR / "paper_figure_f_tbt_vs_ttft_components_32tenants.png",
    )
    context_tbt_detail, context_tbt_summary = build_context_tbt_by_turn(rows)
    write_context_tbt_csv(
        context_tbt_detail,
        OUT_DIR / "paper_figure_g_context_tbt_by_turn_detail.csv",
    )
    write_context_tbt_csv(
        context_tbt_summary,
        OUT_DIR / "paper_figure_g_context_tbt_by_turn_summary.csv",
    )
    plot_context_tbt_by_turn(
        plt,
        context_tbt_summary,
        OUT_DIR / "paper_figure_g_context_tbt_by_turn.png",
    )
    plot_context_tbt_normalized_by_turn(
        plt,
        context_tbt_summary,
        OUT_DIR / "paper_figure_h_normalized_context_tbt_by_turn.png",
    )
    context_per_sec_rows = build_context_per_sec_rows(context_tbt_summary, figure_f_rows)
    write_context_per_sec_csv(
        context_per_sec_rows,
        OUT_DIR / "paper_figure_i_context_per_sec_by_turn.csv",
    )
    plot_context_per_sec_by_turn(
        plt,
        context_per_sec_rows,
        OUT_DIR / "paper_figure_i_context_per_sec_by_turn.png",
    )
    plot_blocking_and_prefill(
        plt,
        rows,
        prefill_lt,
        prefill_gt,
        OUT_DIR / "paper_ttft_vs_blocking_and_prefill_vs_compute_ratio.png",
    )


if __name__ == "__main__":
    main()
