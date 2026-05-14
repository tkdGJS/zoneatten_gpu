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
GROUP1_COLOR = "#2563eb"
GROUP2_COLOR = "#f97316"
BLOCKING_COLOR = "#2563eb"
PREFILL_COLOR = "#84cc16"
REMAINING_COLOR = "#dc2626"
THRESHOLD_MS = 100.0
MAX_NUM_BATCHED_TOKENS = 8192.0


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
        return "Group1"
    if history_limit_tokens == group2_limit:
        return "Group2"
    return f"limit={history_limit_tokens}"


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
                "batch_compute_ratio_8192": total_compute / MAX_NUM_BATCHED_TOKENS,
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
            f"Group1 (limit={group1_limit})",
            [p for p in points if p["history_limit_tokens"] == group1_limit],
        ),
    ]
    if group2_limit != group1_limit:
        groups.append(
            (
                f"Group2 (limit={group2_limit})",
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
        label=f"Group1 (limit={group1_limit})",
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
            label=f"Group2 (limit={group2_limit})",
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
    xlim = padded_limits([r["batch_compute_ratio_8192"] for r in all_prefill_rows])
    ylim = padded_limits([r["batch_mean_prefill_ms"] / 1000.0 for r in all_prefill_rows])
    turn_handles = {}
    for ax, panel_rows, label in [
        (axes[1], rows_lt, "blocking < 100 ms"),
        (axes[2], rows_gt, "blocking > 100 ms"),
    ]:
        for turn in range(1, 11):
            subset = [r for r in panel_rows if r["turn_index"] == turn]
            sc = ax.scatter(
                [r["batch_compute_ratio_8192"] for r in subset],
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
            f"{label}\nr = {corr([r['batch_compute_ratio_8192'] for r in panel_rows], [r['batch_mean_prefill_ms'] for r in panel_rows]):.3f}",
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
    plot_blocking_and_prefill(
        plt,
        rows,
        prefill_lt,
        prefill_gt,
        OUT_DIR / "paper_ttft_vs_blocking_and_prefill_vs_compute_ratio.png",
    )


if __name__ == "__main__":
    main()
