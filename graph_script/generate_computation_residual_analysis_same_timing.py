#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
OUT_DIR = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB" / "graphs" / Path(__file__).stem
TENANT_COUNTS = ["8", "16", "32"]
BLOCKING_THRESHOLD_MS = 100.0
BASELINE_SLOPE = 1.0 / 6.0  # y = x / 6

GROUP_COLORS = {
    "8192": "#2563eb",
    "2048": "#f97316",
}
GROUP_LABELS = {
    "8192": "Group1",
    "2048": "Group2",
}


def mean(values):
    return sum(values) / len(values) if values else 0.0


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


def percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = idx - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def load_rows(exp_dir: str, exp_label: str):
    raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
    rows = []
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "success":
                continue
            if row["tenant_count"] not in TENANT_COUNTS:
                continue
            blocking_time_ms = float(row["blocking_time_ms"] or 0.0)
            if blocking_time_ms >= BLOCKING_THRESHOLD_MS:
                continue
            input_tokens_f = float(row["input_tokens"] or 0.0)
            prefix_hit_tokens_f = float(row["prefix_hit_tokens"] or 0.0)
            computation_tokens_f = max(0.0, input_tokens_f - prefix_hit_tokens_f)
            ttft_ms_f = float(row["ttft_ms"] or 0.0)
            baseline_ttft_f = computation_tokens_f * BASELINE_SLOPE
            residual_ttft_f = ttft_ms_f - baseline_ttft_f
            row["exp_label"] = exp_label
            row["ttft_ms_f"] = ttft_ms_f
            row["input_tokens_f"] = input_tokens_f
            row["prefix_hit_tokens_f"] = prefix_hit_tokens_f
            row["prefix_hit_rate_f"] = float(row["prefix_hit_rate"] or 0.0)
            row["kv_history_tokens_f"] = float(row["kv_history_tokens"] or 0.0)
            row["blocking_time_ms_f"] = blocking_time_ms
            row["computation_tokens_f"] = computation_tokens_f
            row["baseline_ttft_f"] = baseline_ttft_f
            row["residual_ttft_f"] = residual_ttft_f
            rows.append(row)
    return rows


def write_summary(rows):
    out_path = OUT_DIR / "residual_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scope",
                "count",
                "mean_ttft_ms",
                "mean_computation_tokens",
                "mean_baseline_ttft_ms",
                "mean_residual_ttft_ms",
                "min_residual_ttft_ms",
                "max_residual_ttft_ms",
            ]
        )

        def emit(scope, subset):
            if not subset:
                writer.writerow([scope, 0, "", "", "", "", "", ""])
                return
            writer.writerow(
                [
                    scope,
                    len(subset),
                    f"{mean([r['ttft_ms_f'] for r in subset]):.4f}",
                    f"{mean([r['computation_tokens_f'] for r in subset]):.4f}",
                    f"{mean([r['baseline_ttft_f'] for r in subset]):.4f}",
                    f"{mean([r['residual_ttft_f'] for r in subset]):.4f}",
                    f"{min(r['residual_ttft_f'] for r in subset):.4f}",
                    f"{max(r['residual_ttft_f'] for r in subset):.4f}",
                ]
            )

        emit("all", rows)
        for tenant_count in TENANT_COUNTS:
            emit(
                f"tenant_count={tenant_count}",
                [r for r in rows if r["tenant_count"] == tenant_count],
            )
        for history_limit in ["8192", "2048"]:
            emit(
                f"group={GROUP_LABELS[history_limit]}",
                [r for r in rows if r["history_limit_tokens"] == history_limit],
            )
        for _, exp_label in EXPERIMENTS:
            emit(
                f"experiment={exp_label}",
                [r for r in rows if r["exp_label"] == exp_label],
            )


def write_correlations(rows):
    out_path = OUT_DIR / "residual_correlations.txt"
    lines = []
    targets = [
        ("tenant_count", [float(r["tenant_count"]) for r in rows]),
        ("blocking_time_ms", [r["blocking_time_ms_f"] for r in rows]),
        ("prefix_hit_rate", [r["prefix_hit_rate_f"] for r in rows]),
        ("prefix_hit_tokens", [r["prefix_hit_tokens_f"] for r in rows]),
        ("kv_history_tokens", [r["kv_history_tokens_f"] for r in rows]),
        ("computation_tokens", [r["computation_tokens_f"] for r in rows]),
    ]
    residuals = [r["residual_ttft_f"] for r in rows]
    for name, values in targets:
        lines.append(f"{name}: corr={corr(values, residuals):.6f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_percentile_summary(rows):
    out_path = OUT_DIR / "residual_percentiles_by_tenant.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tenant_count",
                "count",
                "mean_residual_ttft_ms",
                "p50_residual_ttft_ms",
                "p95_residual_ttft_ms",
                "min_residual_ttft_ms",
                "max_residual_ttft_ms",
            ]
        )
        for tenant_count in TENANT_COUNTS:
            subset = [r for r in rows if r["tenant_count"] == tenant_count]
            residuals = [r["residual_ttft_f"] for r in subset]
            if not residuals:
                writer.writerow([tenant_count, 0, "", "", "", "", ""])
                continue
            writer.writerow(
                [
                    tenant_count,
                    len(subset),
                    f"{mean(residuals):.4f}",
                    f"{percentile(residuals, 0.5):.4f}",
                    f"{percentile(residuals, 0.95):.4f}",
                    f"{min(residuals):.4f}",
                    f"{max(residuals):.4f}",
                ]
            )


def load_request_metrics():
    metrics = {}
    for exp_dir, _ in EXPERIMENTS:
        metrics_dir = ROOT / exp_dir / "vram_only_artifacts_smallctx_mixed_limits" / "request_metrics"
        for path in metrics_dir.glob("*.jsonl"):
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    item = __import__("json").loads(line)
                    metrics[(exp_dir, item["request_id"])] = item
    return metrics


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is not installed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    request_metrics = load_request_metrics()
    rows = []
    for exp_dir, exp_label in EXPERIMENTS:
        exp_rows = load_rows(exp_dir, exp_label)
        for row in exp_rows:
            metric = request_metrics.get((exp_dir, row["metrics_request_id"]))
            row["queued_ms_f"] = 1000.0 * float((metric or {}).get("queued_time_s", 0.0) or 0.0)
            row["prefill_ms_f"] = 1000.0 * float((metric or {}).get("prefill_time_s", 0.0) or 0.0)
            row["decode_ms_f"] = 1000.0 * float((metric or {}).get("decode_time_s", 0.0) or 0.0)
            row["extra_prefill_ms_f"] = row["prefill_ms_f"] - row["baseline_ttft_f"]
        rows.extend(exp_rows)

    write_summary(rows)
    write_correlations(rows)
    write_percentile_summary(rows)

    x_all = [r["computation_tokens_f"] for r in rows]
    ttft_all = [r["ttft_ms_f"] for r in rows]
    residual_all = [r["residual_ttft_f"] for r in rows]
    blocking_all = [r["blocking_time_ms_f"] for r in rows]
    baseline_all = [r["baseline_ttft_f"] for r in rows]
    prefill_all = [r["prefill_ms_f"] for r in rows]
    extra_prefill_all = [r["extra_prefill_ms_f"] for r in rows]

    xlim = padded_limits(x_all)
    ttft_ylim = padded_limits(ttft_all + baseline_all)
    residual_ylim = padded_limits(residual_all)
    blocking_xlim = padded_limits(blocking_all)
    prefill_xlim = padded_limits(prefill_all)
    extra_prefill_ylim = padded_limits(extra_prefill_all)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (_, exp_label) in zip(axes, EXPERIMENTS):
        exp_rows = [r for r in rows if r["exp_label"] == exp_label]
        for history_limit in ["8192", "2048"]:
            group = [r for r in exp_rows if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [r["computation_tokens_f"] for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=f"{GROUP_LABELS[history_limit]} (n={len(group)})",
            )
        line_x = [xlim[0], xlim[1]]
        line_y = [value * BASELINE_SLOPE for value in line_x]
        ax.plot(line_x, line_y, color="black", linewidth=1.2, linestyle="--", label="TTFT = computation / 6")
        ax.set_title(f"{exp_label}: TTFT vs Computation Tokens")
        ax.set_xlabel("Computation Tokens = Input Tokens - Prefix Hit Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ttft_ylim)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.savefig(OUT_DIR / "ttft_vs_computation_tokens_with_baseline.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0][0]
    for tenant_count in TENANT_COUNTS:
        subset = [r for r in rows if r["tenant_count"] == tenant_count]
        ax.scatter(
            [float(tenant_count)] * len(subset),
            [r["residual_ttft_f"] for r in subset],
            s=18,
            alpha=0.35,
            label=f"tenant={tenant_count} (n={len(subset)})",
        )
    ax.set_title("Residual vs Tenant Count")
    ax.set_xlabel("Tenant Count")
    ax.set_ylabel("Residual TTFT (ms)")
    ax.set_ylim(*residual_ylim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[0][1]
    for history_limit in ["8192", "2048"]:
        subset = [r for r in rows if r["history_limit_tokens"] == history_limit]
        ax.scatter(
            [r["blocking_time_ms_f"] for r in subset],
            [r["residual_ttft_f"] for r in subset],
            s=18,
            alpha=0.4,
            color=GROUP_COLORS[history_limit],
            label=f"{GROUP_LABELS[history_limit]} (n={len(subset)})",
        )
    ax.set_title("Residual vs Blocking Time")
    ax.set_xlabel("Blocking Time (ms)")
    ax.set_ylabel("Residual TTFT (ms)")
    ax.set_xlim(*blocking_xlim)
    ax.set_ylim(*residual_ylim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1][0]
    for history_limit in ["8192", "2048"]:
        subset = [r for r in rows if r["history_limit_tokens"] == history_limit]
        ax.scatter(
            [r["prefix_hit_rate_f"] for r in subset],
            [r["residual_ttft_f"] for r in subset],
            s=18,
            alpha=0.4,
            color=GROUP_COLORS[history_limit],
            label=f"{GROUP_LABELS[history_limit]} (n={len(subset)})",
        )
    ax.set_title("Residual vs Prefix Hit Rate")
    ax.set_xlabel("Prefix Hit Rate")
    ax.set_ylabel("Residual TTFT (ms)")
    ax.set_ylim(*residual_ylim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1][1]
    labels = ["Group1", "Group2"]
    means = [
        mean([r["residual_ttft_f"] for r in rows if r["history_limit_tokens"] == "8192"]),
        mean([r["residual_ttft_f"] for r in rows if r["history_limit_tokens"] == "2048"]),
    ]
    colors = [GROUP_COLORS["8192"], GROUP_COLORS["2048"]]
    ax.bar(labels, means, color=colors)
    ax.set_title("Mean Residual by Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Residual TTFT (ms)")
    ax.grid(axis="y", alpha=0.25)

    fig.savefig(OUT_DIR / "residual_drivers_overview.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, tenant_count in zip(axes, TENANT_COUNTS):
        subset = [r for r in rows if r["tenant_count"] == tenant_count]
        for history_limit in ["8192", "2048"]:
            group = [r for r in subset if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [r["computation_tokens_f"] for r in group],
                [r["residual_ttft_f"] for r in group],
                s=18,
                alpha=0.4,
                color=GROUP_COLORS[history_limit],
                label=f"{GROUP_LABELS[history_limit]} (n={len(group)})",
            )
        ax.set_title(f"Residual vs Computation Tokens\ntenant={tenant_count}")
        ax.set_xlabel("Computation Tokens")
        ax.set_ylabel("Residual TTFT (ms)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*residual_ylim)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.savefig(OUT_DIR / "residual_vs_computation_tokens_by_tenant.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (_, exp_label) in zip(axes, EXPERIMENTS):
        exp_rows = [r for r in rows if r["exp_label"] == exp_label]
        for history_limit in ["8192", "2048"]:
            group = [r for r in exp_rows if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [r["computation_tokens_f"] for r in group],
                [r["residual_ttft_f"] for r in group],
                s=18,
                alpha=0.4,
                color=GROUP_COLORS[history_limit],
                label=f"{GROUP_LABELS[history_limit]} (n={len(group)})",
            )
        ax.set_title(f"{exp_label}: Residual vs Computation Tokens")
        ax.set_xlabel("Computation Tokens")
        ax.set_ylabel("Residual TTFT (ms)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*residual_ylim)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.savefig(OUT_DIR / "residual_vs_computation_tokens_by_experiment.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    tenant_labels = TENANT_COUNTS
    p50_values = []
    p95_values = []
    mean_values = []
    for tenant_count in TENANT_COUNTS:
        residuals = [r["residual_ttft_f"] for r in rows if r["tenant_count"] == tenant_count]
        mean_values.append(mean(residuals))
        p50_values.append(percentile(residuals, 0.5))
        p95_values.append(percentile(residuals, 0.95))

    ax = axes[0]
    width = 0.25
    xs = list(range(len(tenant_labels)))
    ax.bar([x - width for x in xs], mean_values, width=width, label="mean", color="#94a3b8")
    ax.bar(xs, p50_values, width=width, label="p50", color="#2563eb")
    ax.bar([x + width for x in xs], p95_values, width=width, label="p95", color="#dc2626")
    ax.set_title("Residual Summary by Tenant Count")
    ax.set_xlabel("Tenant Count")
    ax.set_ylabel("Residual TTFT (ms)")
    ax.set_xticks(xs, tenant_labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    for tenant_count, color in zip(TENANT_COUNTS, ["#2563eb", "#f59e0b", "#dc2626"]):
        subset = [r["residual_ttft_f"] for r in rows if r["tenant_count"] == tenant_count]
        ax.hist(subset, bins=30, alpha=0.4, label=f"tenant={tenant_count}", color=color)
    ax.set_title("Residual Distribution by Tenant Count")
    ax.set_xlabel("Residual TTFT (ms)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    fig.savefig(OUT_DIR / "residual_percentiles_by_tenant.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    for tenant_count, color in zip(TENANT_COUNTS, ["#2563eb", "#f59e0b", "#dc2626"]):
        subset = [r for r in rows if r["tenant_count"] == tenant_count]
        ax.scatter(
            [r["prefill_ms_f"] for r in subset],
            [r["residual_ttft_f"] for r in subset],
            s=18,
            alpha=0.35,
            color=color,
            label=f"tenant={tenant_count} (n={len(subset)})",
        )
    line_x = [prefill_xlim[0], prefill_xlim[1]]
    ax.plot(line_x, line_x, color="black", linestyle="--", linewidth=1.2, label="residual = prefill")
    ax.set_title("Residual vs Prefill Time")
    ax.set_xlabel("Prefill Time (ms)")
    ax.set_ylabel("Residual TTFT (ms)")
    ax.set_xlim(*prefill_xlim)
    ax.set_ylim(*residual_ylim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    p50_prefill = []
    p95_prefill = []
    mean_prefill = []
    for tenant_count in TENANT_COUNTS:
        values = [r["prefill_ms_f"] for r in rows if r["tenant_count"] == tenant_count]
        mean_prefill.append(mean(values))
        p50_prefill.append(percentile(values, 0.5))
        p95_prefill.append(percentile(values, 0.95))
    width = 0.25
    xs = list(range(len(TENANT_COUNTS)))
    ax.bar([x - width for x in xs], mean_prefill, width=width, label="mean", color="#94a3b8")
    ax.bar(xs, p50_prefill, width=width, label="p50", color="#2563eb")
    ax.bar([x + width for x in xs], p95_prefill, width=width, label="p95", color="#dc2626")
    ax.set_title("Prefill Time by Tenant Count")
    ax.set_xlabel("Tenant Count")
    ax.set_ylabel("Prefill Time (ms)")
    ax.set_xticks(xs, TENANT_COUNTS)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    fig.savefig(OUT_DIR / "residual_vs_prefill_time.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    for tenant_count, color in zip(TENANT_COUNTS, ["#2563eb", "#f59e0b", "#dc2626"]):
        subset = [r for r in rows if r["tenant_count"] == tenant_count]
        ax.scatter(
            [r["prefill_ms_f"] for r in subset],
            [r["ttft_ms_f"] for r in subset],
            s=18,
            alpha=0.35,
            color=color,
            label=f"tenant={tenant_count} (n={len(subset)})",
        )
    line_x = [prefill_xlim[0], prefill_xlim[1]]
    ax.plot(line_x, line_x, color="black", linestyle="--", linewidth=1.2, label="TTFT = prefill")
    ax.set_title("TTFT vs Prefill Time")
    ax.set_xlabel("Prefill Time (ms)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_xlim(*prefill_xlim)
    ax.set_ylim(*ttft_ylim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    for tenant_count, color in zip(TENANT_COUNTS, ["#2563eb", "#f59e0b", "#dc2626"]):
        subset = [r for r in rows if r["tenant_count"] == tenant_count]
        ax.scatter(
            [r["prefill_ms_f"] for r in subset],
            [r["extra_prefill_ms_f"] for r in subset],
            s=18,
            alpha=0.35,
            color=color,
            label=f"tenant={tenant_count} (n={len(subset)})",
        )
    ax.set_title("Extra Prefill vs Prefill Time")
    ax.set_xlabel("Prefill Time (ms)")
    ax.set_ylabel("Extra Prefill (ms) = Prefill - Computation/6")
    ax.set_xlim(*prefill_xlim)
    ax.set_ylim(*extra_prefill_ylim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    fig.savefig(OUT_DIR / "ttft_and_extra_prefill_vs_prefill_time.png", dpi=200)
    plt.close(fig)

    print(f"[DONE] residual analysis saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
