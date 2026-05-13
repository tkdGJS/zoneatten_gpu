#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib is not installed. Activate the experiment venv and run: "
            "pip install matplotlib"
        )

    root = Path(__file__).resolve().parents[1]
    raw_csv = root / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "result_summary.csv"
    out_dir = root / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "graphs" / Path(__file__).stem
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

    for metric_key, ylabel, filename in metric_specs:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
        axes = axes.flatten()
        for ax, tenant_count in zip(axes, sorted(by_tenant_count)):
            group = sorted(by_tenant_count[tenant_count], key=lambda row: int(row["tenant_id"]))
            xs = [int(row["tenant_id"]) for row in group]
            ys = [float(row[metric_key]) for row in group]
            ax.bar(xs, ys, color="#3b82f6")
            ax.set_title(f"tenant_count={tenant_count}")
            ax.set_xlabel("Tenant ID")
            ax.set_ylabel(ylabel)
        fig.suptitle(f"VRAM-only Isolation (smallctx): {ylabel} by Tenant", fontsize=16)
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
    ax.set_title("VRAM-only Isolation (smallctx): Mean Metric by Tenant Count")
    ax.legend()
    fig.savefig(out_dir / "tenant_count_overview.png", dpi=200)
    plt.close(fig)

    print(f"[DONE] plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
