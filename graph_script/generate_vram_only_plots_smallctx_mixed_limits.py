#!/usr/bin/env python3
import csv
import os
from collections import defaultdict
from pathlib import Path

from matplotlib.patches import Patch


ROOT = Path(
    os.environ.get(
        "VRAM_RESULTS_ROOT",
        str(Path(__file__).resolve().parents[1] / "block_2048_limit_same_timing1" / "vram_only_results_smallctx_mixed_limits"),
    )
)
GROUP1_LABEL = os.environ.get("GROUP1_LABEL", "Blue tenants: history_limit=8192")
GROUP2_LABEL = os.environ.get("GROUP2_LABEL", "Orange tenants: history_limit=2048")
GROUP1_LINE_LABEL = os.environ.get("GROUP1_LINE_LABEL", "history_limit=8192")
GROUP2_LINE_LABEL = os.environ.get("GROUP2_LINE_LABEL", "history_limit=2048")


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib is not installed. Activate the experiment venv and run: "
            "pip install matplotlib"
        )

    raw_csv = ROOT / "result_summary.csv"
    out_dir = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "graphs" / Path(__file__).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(raw_csv.open("r", encoding="utf-8", newline="")))
    by_tenant_count = defaultdict(list)
    for row in rows:
        by_tenant_count[int(row["tenant_count"])].append(row)

    metric_specs = [
        ("avg_ttft_ms", "TTFT (ms)", "tenant_ttft.png"),
        ("avg_prefix_hit_rate", "Prefix Hit Rate", "tenant_prefix_hit_rate.png"),
        ("avg_ttlt_ms", "TTLT (ms)", "tenant_ttlt.png"),
    ]
    colors = {"2048": "#f97316", "8192": "#2563eb"}
    legend_handles = [
        Patch(facecolor=colors["8192"], label=GROUP1_LABEL),
        Patch(facecolor=colors["2048"], label=GROUP2_LABEL),
    ]

    for metric_key, ylabel, filename in metric_specs:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
        axes = axes.flatten()
        for ax, tenant_count in zip(axes, sorted(by_tenant_count)):
            group = sorted(
                by_tenant_count[tenant_count],
                key=lambda row: (int(row["history_limit_tokens"]), int(row["tenant_id"])),
            )
            xs = list(range(1, len(group) + 1))
            ys = [float(row[metric_key]) for row in group]
            bar_colors = [colors.get(row["history_limit_tokens"], "#6b7280") for row in group]
            ax.bar(xs, ys, color=bar_colors)
            ax.set_title(f"tenant_count={tenant_count}")
            ax.set_xlabel("Tenant Slot")
            ax.set_ylabel(ylabel)
            ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
        fig.suptitle(f"VRAM-only Isolation (smallctx mixed limits): {ylabel}", fontsize=16)
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    tenant_counts = sorted(by_tenant_count)
    ttft_means = []
    ttlt_means = []
    tbt_means = []
    for tenant_count in tenant_counts:
        group = by_tenant_count[tenant_count]
        ttft_means.append(sum(float(r["avg_ttft_ms"]) for r in group) / len(group))
        ttlt_means.append(sum(float(r["avg_ttlt_ms"]) for r in group) / len(group))
        tbt_means.append(sum(float(r["avg_p95_tbt_ms"]) for r in group) / len(group))
    ax.plot(tenant_counts, ttft_means, marker="o", label="avg TTFT")
    ax.plot(tenant_counts, tbt_means, marker="o", label="avg p95(TBT)")
    ax.plot(tenant_counts, ttlt_means, marker="o", label="avg TTLT")
    ax.set_xlabel("Tenant Count")
    ax.set_ylabel("Milliseconds")
    ax.set_title("VRAM-only Isolation (smallctx mixed limits): Mean Metric by Tenant Count")
    ax.legend()
    fig.savefig(out_dir / "tenant_count_overview.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(tenant_counts, ttft_means, marker="o", linewidth=2.5, markersize=7, label="TTFT", color="#2563eb")
    ax.plot(tenant_counts, tbt_means, marker="s", linewidth=2.5, markersize=7, label="p95(TBT)", color="#dc2626")
    ax.plot(tenant_counts, ttlt_means, marker="^", linewidth=2.5, markersize=7, label="TTLT", color="#059669")
    ax.set_xlabel("Tenant Count")
    ax.set_ylabel("Milliseconds")
    ax.set_title("Mean TTFT / p95(TBT) / TTLT Across Tenant Sweep")
    ax.set_xticks(tenant_counts)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "tenant_sweep_mean_latency.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    group_metric_specs = [
        ("avg_ttft_ms", "TTFT (ms)", axes[0]),
        ("avg_p95_tbt_ms", "p95(TBT) (ms)", axes[1]),
        ("avg_ttlt_ms", "TTLT (ms)", axes[2]),
    ]
    for metric_key, ylabel, ax in group_metric_specs:
        for history_limit, color, label in [
            ("8192", "#2563eb", "Group 1"),
            ("2048", "#f97316", "Group 2"),
        ]:
            ys = []
            for tenant_count in tenant_counts:
                group = [
                    row
                    for row in by_tenant_count[tenant_count]
                    if row["history_limit_tokens"] == history_limit
                ]
                ys.append(sum(float(r[metric_key]) for r in group) / len(group))
            ax.plot(tenant_counts, ys, marker="o", linewidth=2.5, markersize=7, label=label, color=color)
        ax.set_xlabel("Tenant Count")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(tenant_counts)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.savefig(out_dir / "tenant_sweep_group_mean_latency.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    overview_specs = [
        ("avg_ttft_ms", "TTFT (ms)", axes[0]),
        ("avg_prefix_hit_rate", "Prefix Hit Rate", axes[1]),
        ("avg_ttlt_ms", "TTLT (ms)", axes[2]),
    ]
    for metric_key, ylabel, ax in overview_specs:
        for history_limit in ["2048", "8192"]:
            ys = []
            for tenant_count in tenant_counts:
                group = [
                    row
                    for row in by_tenant_count[tenant_count]
                    if row["history_limit_tokens"] == history_limit
                ]
                ys.append(sum(float(r[metric_key]) for r in group) / len(group))
            ax.plot(
                tenant_counts,
                ys,
                marker="o",
                label=GROUP1_LINE_LABEL if history_limit == "8192" else GROUP2_LINE_LABEL,
                color=colors[history_limit],
            )
        ax.set_xlabel("Tenant Count")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "history_limit_overview.png", dpi=200)
    plt.close(fig)

    print(f"[DONE] plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
