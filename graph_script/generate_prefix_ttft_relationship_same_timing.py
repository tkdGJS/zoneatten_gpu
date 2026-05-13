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

GROUP_COLORS = {
    "8192": "#2563eb",
    "2048": "#f97316",
}
GROUP_LABELS = {
    "8192": "Group1",
    "2048": "Group2",
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


def load_rows(exp_dir: str):
    raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["status"] == "success"]
    return rows


def save_group_summary_plot(plt, experiments, out_path, metrics, tenant_count=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    for ax, (metric_key, title) in zip(axes, metrics):
        labels = []
        values = []
        colors = []
        for exp in experiments:
            for history_limit in ["8192", "2048"]:
                group = [
                    r
                    for r in exp["rows"]
                    if r["history_limit_tokens"] == history_limit
                    and (tenant_count is None or r["tenant_count"] == tenant_count)
                ]
                labels.append(f"{exp['label']}\n{GROUP_LABELS[history_limit]}")
                values.append(mean([r[metric_key] for r in group]))
                colors.append(GROUP_COLORS[history_limit])
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        if "ttft" in metric_key:
            ax.set_ylabel("Time (ms)")
        else:
            ax.set_ylabel("Tokens")
        ax.grid(axis="y", alpha=0.25)
    if tenant_count is None:
        fig.suptitle("All Tenant Counts", fontsize=14)
    else:
        fig.suptitle(f"Tenant Count = {tenant_count}", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_aggregated_ttft_bar_plot(plt, experiments, out_path, title, row_filter=None):
    tenant_counts = ["8", "16", "32"]
    means = []
    for tenant_count in tenant_counts:
        tenant_rows = []
        for exp in experiments:
            for row in exp["rows"]:
                if row["tenant_count"] != tenant_count:
                    continue
                if row_filter is not None and not row_filter(row):
                    continue
                tenant_rows.append(row)
        means.append(mean([r["ttft_ms_f"] for r in tenant_rows]))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = ["#2563eb", "#f59e0b", "#dc2626"]
    bars = ax.bar(tenant_counts, means, color=colors, width=0.6)
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title(title)
    ax.set_xlabel("Tenant Count")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_combined_group_scatter(plt, rows, out_path, title, x_getter, x_label):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    all_xs = [x_getter(r) for r in rows]
    all_ys = [r["ttft_ms_f"] for r in rows]
    ax.set_xlim(*padded_limits(all_xs))
    ax.set_ylim(*padded_limits(all_ys))
    for history_limit in ["8192", "2048"]:
        group = [r for r in rows if r["history_limit_tokens"] == history_limit]
        ax.scatter(
            [x_getter(r) for r in group],
            [r["ttft_ms_f"] for r in group],
            s=18,
            alpha=0.45,
            color=GROUP_COLORS[history_limit],
            label=GROUP_LABELS[history_limit],
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("TTFT (ms)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Group")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_combined_group_scatter_by_tenantcount(plt, experiments, out_path, title_prefix, x_getter, x_label):
    tenant_counts = ["8", "16", "32"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    all_rows = [
        r
        for exp in experiments
        for r in exp["rows"]
        if r["tenant_count"] in tenant_counts
    ]
    xlim = padded_limits([x_getter(r) for r in all_rows])
    ylim = padded_limits([r["ttft_ms_f"] for r in all_rows])
    for ax, tenant_count in zip(axes, tenant_counts):
        subset = [
            r
            for exp in experiments
            for r in exp["rows"]
            if r["tenant_count"] == tenant_count
        ]
        for history_limit in ["8192", "2048"]:
            group = [r for r in subset if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [x_getter(r) for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=GROUP_LABELS[history_limit],
            )
        ax.set_title(f"{title_prefix}\ntenants={tenant_count}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("TTFT (ms)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Group")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_combined_turn_scatter_by_tenantcount(plt, experiments, out_path, title_prefix, x_getter, x_label):
    tenant_counts = ["8", "16", "32"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    all_rows = [
        r
        for exp in experiments
        for r in exp["rows"]
        if r["tenant_count"] in tenant_counts
    ]
    xlim = padded_limits([x_getter(r) for r in all_rows])
    ylim = padded_limits([r["ttft_ms_f"] for r in all_rows])
    for ax, tenant_count in zip(axes, tenant_counts):
        subset = [
            r
            for exp in experiments
            for r in exp["rows"]
            if r["tenant_count"] == tenant_count
        ]
        for turn in range(1, 11):
            turn_rows = [r for r in subset if int(r["turn_index"]) == turn]
            ax.scatter(
                [x_getter(r) for r in turn_rows],
                [r["ttft_ms_f"] for r in turn_rows],
                s=18,
                alpha=0.45,
                color=TURN_COLORS[turn],
                label=f"Turn {turn}",
            )
        ax.set_title(f"{title_prefix}\ntenants={tenant_count}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("TTFT (ms)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Turn", fontsize=8)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_combined_group_scatter_by_tenantcount_filtered(
    plt,
    experiments,
    out_path,
    title_prefix,
    x_getter,
    x_label,
    row_filter,
):
    tenant_counts = ["8", "16", "32"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    all_rows = [
        r
        for exp in experiments
        for r in exp["rows"]
        if r["tenant_count"] in tenant_counts and row_filter(r)
    ]
    xlim = padded_limits([x_getter(r) for r in all_rows])
    ylim = padded_limits([r["ttft_ms_f"] for r in all_rows])
    for ax, tenant_count in zip(axes, tenant_counts):
        subset = [
            r
            for exp in experiments
            for r in exp["rows"]
            if r["tenant_count"] == tenant_count and row_filter(r)
        ]
        for history_limit in ["8192", "2048"]:
            group = [r for r in subset if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [x_getter(r) for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=GROUP_LABELS[history_limit],
            )
        ax.set_title(f"{title_prefix}\ntenants={tenant_count}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("TTFT (ms)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Group")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is not installed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = []
    for exp_dir, label in EXPERIMENTS:
        rows = load_rows(exp_dir)
        for row in rows:
            row["ttft_ms_f"] = float(row["ttft_ms"])
            row["input_tokens_f"] = float(row["input_tokens"] or 0.0)
            row["prefix_hit_rate_f"] = float(row["prefix_hit_rate"] or 0.0)
            row["prefix_hit_tokens_f"] = float(row["prefix_hit_tokens"] or 0.0)
            row["kv_history_tokens_f"] = float(row["kv_history_tokens"] or 0.0)
            row["miss_tokens_f"] = max(
                0.0, row["kv_history_tokens_f"] - row["prefix_hit_tokens_f"]
            )
            row["computation_tokens_f"] = max(
                0.0, row["input_tokens_f"] - row["prefix_hit_tokens_f"]
            )
        experiments.append({"dir": exp_dir, "label": label, "rows": rows})

    tenant_counts = ["8", "16", "32"]

    def scatter_by_tenantcount(
        filename,
        title_prefix,
        x_getter,
        x_label,
        row_filter=None,
        use_blocking_color=False,
    ):
        all_rows = []
        for exp in experiments:
            for row in exp["rows"]:
                if row["tenant_count"] not in tenant_counts:
                    continue
                if row_filter is not None and not row_filter(row):
                    continue
                all_rows.append(row)
        xlim = padded_limits([x_getter(r) for r in all_rows])
        ylim = padded_limits([r["ttft_ms_f"] for r in all_rows])
        clim = padded_limits([float(r["blocking_time_ms"] or 0.0) for r in all_rows]) if use_blocking_color else None

        fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
        for row_idx, tenant_count in enumerate(tenant_counts):
            for col_idx, exp in enumerate(experiments):
                ax = axes[row_idx][col_idx]
                exp_rows = [
                    r for r in exp["rows"]
                    if r["tenant_count"] == tenant_count and (row_filter(r) if row_filter is not None else True)
                ]
                if use_blocking_color:
                    scatter_rows = exp_rows
                    scatter = ax.scatter(
                        [x_getter(r) for r in scatter_rows],
                        [r["ttft_ms_f"] for r in scatter_rows],
                        c=[float(r["blocking_time_ms"] or 0.0) for r in scatter_rows],
                        cmap="viridis",
                        vmin=clim[0],
                        vmax=clim[1],
                        s=20,
                        alpha=0.6,
                    )
                    for history_limit in ["8192", "2048"]:
                        group = [r for r in exp_rows if r["history_limit_tokens"] == history_limit]
                        if group:
                            ax.scatter(
                                [mean([x_getter(r) for r in group])],
                                [mean([r["ttft_ms_f"] for r in group])],
                                s=140,
                                marker="X",
                                color=GROUP_COLORS[history_limit],
                                edgecolors="black",
                                linewidths=0.8,
                                label=f"{GROUP_LABELS[history_limit]} mean",
                            )
                    fig.colorbar(scatter, ax=ax, label="Blocking Time (ms)")
                else:
                    for history_limit in ["8192", "2048"]:
                        group = [r for r in exp_rows if r["history_limit_tokens"] == history_limit]
                        ax.scatter(
                            [x_getter(r) for r in group],
                            [r["ttft_ms_f"] for r in group],
                            s=18,
                            alpha=0.45,
                            color=GROUP_COLORS[history_limit],
                            label=f"{GROUP_LABELS[history_limit]} (n={len(group)})",
                        )
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_title(f"{exp['label']}, tenants={tenant_count}")
                ax.set_xlabel(x_label)
                ax.set_ylabel("TTFT (ms)")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8, loc="best")
        fig.suptitle(title_prefix, fontsize=16)
        fig.savefig(OUT_DIR / filename, dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, exp in zip(axes, experiments):
        for history_limit in ["8192", "2048"]:
            group = [r for r in exp["rows"] if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [r["prefix_hit_rate_f"] for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=GROUP_LABELS[history_limit],
            )
        ax.set_title(f"{exp['label']}: TTFT vs Prefix Hit Rate")
        ax.set_xlabel("Prefix Hit Rate")
        ax.set_ylabel("TTFT (ms)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(OUT_DIR / "ttft_vs_prefix_hit_rate.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, exp in zip(axes, experiments):
        for history_limit in ["8192", "2048"]:
            group = [r for r in exp["rows"] if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [r["miss_tokens_f"] for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=GROUP_LABELS[history_limit],
            )
        ax.set_title(f"{exp['label']}: TTFT vs Miss Tokens")
        ax.set_xlabel("Miss Tokens = KV History - Prefix Hit Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(OUT_DIR / "ttft_vs_miss_tokens.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, exp in zip(axes, experiments):
        for history_limit in ["8192", "2048"]:
            group = [r for r in exp["rows"] if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [r["computation_tokens_f"] for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=GROUP_LABELS[history_limit],
            )
        ax.set_title(f"{exp['label']}: TTFT vs Computation Tokens")
        ax.set_xlabel("Computation Tokens = Input Tokens - Prefix Hit Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(OUT_DIR / "ttft_vs_computation_tokens.png", dpi=200)
    plt.close(fig)

    threshold_ms = 100.0
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    filtered_counts = []
    for ax, exp in zip(axes, experiments):
        exp_counts = {"exp": exp["label"]}
        for history_limit in ["8192", "2048"]:
            group = [
                r
                for r in exp["rows"]
                if r["history_limit_tokens"] == history_limit
                and float(r["blocking_time_ms"] or 0.0) < threshold_ms
            ]
            exp_counts[history_limit] = len(group)
            ax.scatter(
                [r["miss_tokens_f"] for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=f"{GROUP_LABELS[history_limit]} (n={len(group)})",
            )
        filtered_counts.append(exp_counts)
        ax.set_title(f"{exp['label']}: TTFT vs Miss Tokens\n(blocking < {int(threshold_ms)} ms)")
        ax.set_xlabel("Miss Tokens = KV History - Prefix Hit Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(OUT_DIR / "ttft_vs_miss_tokens_blocking_lt_100ms.png", dpi=200)
    plt.close(fig)

    counts_path = OUT_DIR / "ttft_vs_miss_tokens_blocking_lt_100ms_counts.txt"
    counts_lines = []
    for item in filtered_counts:
        counts_lines.append(
            f"{item['exp']}: Group1={item['8192']}, Group2={item['2048']}, total={item['8192'] + item['2048']}"
        )
    counts_path.write_text("\n".join(counts_lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, exp in zip(axes, experiments):
        for history_limit in ["8192", "2048"]:
            group = [
                r
                for r in exp["rows"]
                if r["history_limit_tokens"] == history_limit
                and float(r["blocking_time_ms"] or 0.0) < threshold_ms
            ]
            ax.scatter(
                [r["computation_tokens_f"] for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=f"{GROUP_LABELS[history_limit]} (n={len(group)})",
            )
        ax.set_title(f"{exp['label']}: TTFT vs Computation Tokens\n(blocking < {int(threshold_ms)} ms)")
        ax.set_xlabel("Computation Tokens = Input Tokens - Prefix Hit Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(OUT_DIR / "ttft_vs_computation_tokens_blocking_lt_100ms.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, exp in zip(axes, experiments):
        for history_limit in ["8192", "2048"]:
            group = [
                r
                for r in exp["rows"]
                if r["history_limit_tokens"] == history_limit
                and float(r["blocking_time_ms"] or 0.0) < threshold_ms
            ]
            ax.scatter(
                [float(r["input_tokens"] or 0.0) for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=f"{GROUP_LABELS[history_limit]} (n={len(group)})",
            )
        ax.set_title(f"{exp['label']}: TTFT vs Input Tokens\n(blocking < {int(threshold_ms)} ms)")
        ax.set_xlabel("Input Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(OUT_DIR / "ttft_vs_input_tokens_blocking_lt_100ms.png", dpi=200)
    plt.close(fig)

    scatter_by_tenantcount(
        "ttft_vs_prefix_hit_rate_by_tenantcount_unified.png",
        "TTFT vs Prefix Hit Rate by Tenant Count",
        lambda r: r["prefix_hit_rate_f"],
        "Prefix Hit Rate",
    )
    scatter_by_tenantcount(
        "ttft_vs_miss_tokens_by_tenantcount_unified.png",
        "TTFT vs Miss Tokens by Tenant Count",
        lambda r: r["miss_tokens_f"],
        "Miss Tokens",
    )
    scatter_by_tenantcount(
        "ttft_vs_computation_tokens_by_tenantcount_unified.png",
        "TTFT vs Computation Tokens by Tenant Count",
        lambda r: r["computation_tokens_f"],
        "Computation Tokens = Input Tokens - Prefix Hit Tokens",
    )
    scatter_by_tenantcount(
        "ttft_vs_blocking_time_by_tenantcount_unified.png",
        "TTFT vs Blocking Time by Tenant Count",
        lambda r: float(r["blocking_time_ms"] or 0.0),
        "Blocking Time (ms)",
    )
    scatter_by_tenantcount(
        "ttft_vs_miss_tokens_blocking_lt_100ms_by_tenantcount_unified.png",
        "TTFT vs Miss Tokens by Tenant Count (blocking < 100 ms)",
        lambda r: r["miss_tokens_f"],
        "Miss Tokens",
        row_filter=lambda r: float(r["blocking_time_ms"] or 0.0) < threshold_ms,
    )
    scatter_by_tenantcount(
        "ttft_vs_computation_tokens_blocking_lt_100ms_by_tenantcount_unified.png",
        "TTFT vs Computation Tokens by Tenant Count (blocking < 100 ms)",
        lambda r: r["computation_tokens_f"],
        "Computation Tokens = Input Tokens - Prefix Hit Tokens",
        row_filter=lambda r: float(r["blocking_time_ms"] or 0.0) < threshold_ms,
    )
    scatter_by_tenantcount(
        "ttft_vs_input_tokens_blocking_lt_100ms_by_tenantcount_unified.png",
        "TTFT vs Input Tokens by Tenant Count (blocking < 100 ms)",
        lambda r: float(r["input_tokens"] or 0.0),
        "Input Tokens",
        row_filter=lambda r: float(r["blocking_time_ms"] or 0.0) < threshold_ms,
    )
    scatter_by_tenantcount(
        "ttft_vs_miss_tokens_with_blocking_by_tenantcount_unified.png",
        "TTFT vs Miss Tokens by Tenant Count (color = blocking)",
        lambda r: r["miss_tokens_f"],
        "Miss Tokens",
        use_blocking_color=True,
    )
    scatter_by_tenantcount(
        "ttft_vs_computation_tokens_with_blocking_by_tenantcount_unified.png",
        "TTFT vs Computation Tokens by Tenant Count (color = blocking)",
        lambda r: r["computation_tokens_f"],
        "Computation Tokens = Input Tokens - Prefix Hit Tokens",
        use_blocking_color=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    all_xs = [r["miss_tokens_f"] for exp in experiments for r in exp["rows"]]
    all_ys = [r["ttft_ms_f"] for exp in experiments for r in exp["rows"]]
    all_cs = [float(r["blocking_time_ms"] or 0.0) for exp in experiments for r in exp["rows"]]
    xlim = padded_limits(all_xs)
    ylim = padded_limits(all_ys)
    clim = padded_limits(all_cs)
    for ax, exp in zip(axes, experiments):
        xs = [r["miss_tokens_f"] for r in exp["rows"]]
        ys = [r["ttft_ms_f"] for r in exp["rows"]]
        cs = [float(r["blocking_time_ms"] or 0.0) for r in exp["rows"]]
        scatter = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=clim[0], vmax=clim[1], s=20, alpha=0.6)
        for history_limit in ["8192", "2048"]:
            group = [r for r in exp["rows"] if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [mean([r["miss_tokens_f"] for r in group])],
                [mean([r["ttft_ms_f"] for r in group])],
                s=140,
                marker="X",
                color=GROUP_COLORS[history_limit],
                edgecolors="black",
                linewidths=0.8,
                label=f"{GROUP_LABELS[history_limit]} mean",
            )
        ax.set_title(f"{exp['label']}: TTFT vs Miss Tokens\n(color = blocking time)")
        ax.set_xlabel("Miss Tokens = KV History - Prefix Hit Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(scatter, ax=ax, label="Blocking Time (ms)")
    fig.savefig(OUT_DIR / "ttft_vs_miss_tokens_with_blocking.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    all_xs = [r["computation_tokens_f"] for exp in experiments for r in exp["rows"]]
    all_ys = [r["ttft_ms_f"] for exp in experiments for r in exp["rows"]]
    all_cs = [float(r["blocking_time_ms"] or 0.0) for exp in experiments for r in exp["rows"]]
    xlim = padded_limits(all_xs)
    ylim = padded_limits(all_ys)
    clim = padded_limits(all_cs)
    for ax, exp in zip(axes, experiments):
        xs = [r["computation_tokens_f"] for r in exp["rows"]]
        ys = [r["ttft_ms_f"] for r in exp["rows"]]
        cs = [float(r["blocking_time_ms"] or 0.0) for r in exp["rows"]]
        scatter = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=clim[0], vmax=clim[1], s=20, alpha=0.6)
        for history_limit in ["8192", "2048"]:
            group = [r for r in exp["rows"] if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [mean([r["computation_tokens_f"] for r in group])],
                [mean([r["ttft_ms_f"] for r in group])],
                s=140,
                marker="X",
                color=GROUP_COLORS[history_limit],
                edgecolors="black",
                linewidths=0.8,
                label=f"{GROUP_LABELS[history_limit]} mean",
            )
        ax.set_title(f"{exp['label']}: TTFT vs Computation Tokens\n(color = blocking time)")
        ax.set_xlabel("Computation Tokens = Input Tokens - Prefix Hit Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(scatter, ax=ax, label="Blocking Time (ms)")
    fig.savefig(OUT_DIR / "ttft_vs_computation_tokens_with_blocking.png", dpi=200)
    plt.close(fig)

    metrics = [
        ("kv_history_tokens_f", "Mean KV History Tokens"),
        ("prefix_hit_tokens_f", "Mean Prefix Hit Tokens"),
        ("miss_tokens_f", "Mean Miss Tokens"),
        ("ttft_ms_f", "Mean TTFT (ms)"),
    ]
    save_group_summary_plot(
        plt,
        experiments,
        OUT_DIR / "group_mean_prefix_kv_ttft_summary.png",
        metrics,
    )
    for tenant_count in tenant_counts:
        save_group_summary_plot(
            plt,
            experiments,
            OUT_DIR / f"group_mean_prefix_kv_ttft_summary_{tenant_count}tenants.png",
            metrics,
            tenant_count=tenant_count,
        )

    save_aggregated_ttft_bar_plot(
        plt,
        experiments,
        OUT_DIR / "mean_ttft_by_tenantcount_all_experiments_bar.png",
        "Mean TTFT by Tenant Count (exp1+exp2+exp3, all data)",
    )
    save_aggregated_ttft_bar_plot(
        plt,
        experiments,
        OUT_DIR / "mean_ttft_by_tenantcount_blocking_lt_100ms_all_experiments_bar.png",
        "Mean TTFT by Tenant Count (exp1+exp2+exp3, blocking < 100 ms)",
        row_filter=lambda r: float(r["blocking_time_ms"] or 0.0) < threshold_ms,
    )

    combined_rows = [r for exp in experiments for r in exp["rows"]]
    save_combined_group_scatter(
        plt,
        combined_rows,
        OUT_DIR / "exp123_ttft_vs_prefix_hit_rate.png",
        "exp1+exp2+exp3: TTFT vs Prefix Hit Rate",
        lambda r: r["prefix_hit_rate_f"],
        "Prefix Hit Rate",
    )
    save_combined_group_scatter(
        plt,
        combined_rows,
        OUT_DIR / "exp123_ttft_vs_prefix_hit_tokens.png",
        "exp1+exp2+exp3: TTFT vs Prefix Hit Tokens",
        lambda r: r["prefix_hit_tokens_f"],
        "Prefix Hit Tokens",
    )
    save_combined_group_scatter(
        plt,
        combined_rows,
        OUT_DIR / "exp123_ttft_vs_blocking_time.png",
        "exp1+exp2+exp3: TTFT vs Blocking Time",
        lambda r: float(r["blocking_time_ms"] or 0.0),
        "Blocking Time (ms)",
    )
    save_combined_group_scatter_by_tenantcount(
        plt,
        experiments,
        OUT_DIR / "exp123_ttft_vs_prefix_hit_rate_by_tenantcount.png",
        "exp1+exp2+exp3: TTFT vs Prefix Hit Rate",
        lambda r: r["prefix_hit_rate_f"],
        "Prefix Hit Rate",
    )
    save_combined_turn_scatter_by_tenantcount(
        plt,
        experiments,
        OUT_DIR / "exp123_ttft_vs_prefix_hit_rate_by_tenantcount_turn_color.png",
        "exp1+exp2+exp3: TTFT vs Prefix Hit Rate",
        lambda r: r["prefix_hit_rate_f"],
        "Prefix Hit Rate",
    )
    save_combined_group_scatter_by_tenantcount(
        plt,
        experiments,
        OUT_DIR / "exp123_ttft_vs_prefix_hit_tokens_by_tenantcount.png",
        "exp1+exp2+exp3: TTFT vs Prefix Hit Tokens",
        lambda r: r["prefix_hit_tokens_f"],
        "Prefix Hit Tokens",
    )
    save_combined_group_scatter_by_tenantcount_filtered(
        plt,
        experiments,
        OUT_DIR / "exp123_ttft_vs_prefix_hit_rate_blocking_lt_100ms_by_tenantcount.png",
        "exp1+exp2+exp3: TTFT vs Prefix Hit Rate (blocking < 100 ms)",
        lambda r: r["prefix_hit_rate_f"],
        "Prefix Hit Rate",
        row_filter=lambda r: float(r["blocking_time_ms"] or 0.0) < threshold_ms,
    )
    save_combined_group_scatter_by_tenantcount_filtered(
        plt,
        experiments,
        OUT_DIR / "exp123_ttft_vs_prefix_hit_tokens_blocking_lt_100ms_by_tenantcount.png",
        "exp1+exp2+exp3: TTFT vs Prefix Hit Tokens (blocking < 100 ms)",
        lambda r: r["prefix_hit_tokens_f"],
        "Prefix Hit Tokens",
        row_filter=lambda r: float(r["blocking_time_ms"] or 0.0) < threshold_ms,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    all_xs = [float(r["blocking_time_ms"] or 0.0) for exp in experiments for r in exp["rows"]]
    all_ys = [r["ttft_ms_f"] for exp in experiments for r in exp["rows"]]
    xlim = padded_limits(all_xs)
    ylim = padded_limits(all_ys)
    for ax, exp in zip(axes, experiments):
        for history_limit in ["8192", "2048"]:
            group = [r for r in exp["rows"] if r["history_limit_tokens"] == history_limit]
            ax.scatter(
                [float(r["blocking_time_ms"] or 0.0) for r in group],
                [r["ttft_ms_f"] for r in group],
                s=18,
                alpha=0.45,
                color=GROUP_COLORS[history_limit],
                label=GROUP_LABELS[history_limit],
            )
        ax.set_title(f"{exp['label']}: TTFT vs Blocking Time")
        ax.set_xlabel("Blocking Time (ms)")
        ax.set_ylabel("TTFT (ms)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(OUT_DIR / "ttft_vs_blocking_time.png", dpi=200)
    plt.close(fig)

    print(f"[DONE] plots saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
