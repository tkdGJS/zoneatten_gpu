#!/usr/bin/env python3
import csv
from pathlib import Path


EXPERIMENTS = [
    ("block_2048_not_saturation_diff_8192max2", "exp1", "#2563eb"),
    ("block_2048_not_saturation_diff_8192max3", "exp2", "#dc2626"),
    ("block_2048_not_saturation_diff_8192max4", "exp3", "#059669"),
]

ROOT = Path("/home/yuhwa2323/zoneatten")
OUT_DIR = ROOT / "analysis_32tenant_compare"


def load_rows(exp_dir: str):
    raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row["tenant_count"] == "32" and row["status"] == "success"
        ]
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is not installed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = []
    for exp_dir, label, color in EXPERIMENTS:
        rows = load_rows(exp_dir)
        experiments.append(
            {
                "dir": exp_dir,
                "label": label,
                "color": color,
                "rows": rows,
            }
        )

    metric_specs = [
        ("ttft_ms", "TTFT (ms)", "ttft"),
        ("p95_tbt_ms", "p95(TBT) (ms)", "tbt"),
        ("ttlt_ms", "TTLT (ms)", "ttlt"),
    ]

    turns = list(range(1, 11))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (metric_key, ylabel, slug) in zip(axes, metric_specs):
        for exp in experiments:
            ys = []
            for turn in turns:
                turn_rows = [r for r in exp["rows"] if int(r["turn_index"]) == turn]
                ys.append(mean([float(r[metric_key]) for r in turn_rows]))
            ax.plot(turns, ys, marker="o", linewidth=2, label=exp["label"], color=exp["color"])
        ax.set_title(f"32 Tenants: {ylabel} by Turn")
        ax.set_xlabel("Turn Index")
        ax.set_ylabel(ylabel)
        ax.set_xticks(turns)
        ax.legend()
        ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "turn_by_turn_latency_32tenants.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (metric_key, ylabel, slug) in zip(axes, metric_specs):
        data = [[float(r[metric_key]) for r in exp["rows"]] for exp in experiments]
        labels = [exp["label"] for exp in experiments]
        bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
        for patch, exp in zip(bp["boxes"], experiments):
            patch.set_facecolor(exp["color"])
            patch.set_alpha(0.5)
        ax.set_title(f"32 Tenants: {ylabel} Distribution")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "distribution_latency_32tenants.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (metric_key, ylabel, slug) in zip(axes, metric_specs):
        values = []
        labels = []
        colors = []
        for exp in experiments:
            turn_rows = [r for r in exp["rows"] if r["turn_index"] == "10"]
            values.append(mean([float(r[metric_key]) for r in turn_rows]))
            labels.append(exp["label"])
            colors.append(exp["color"])
        ax.bar(labels, values, color=colors)
        ax.set_title(f"32 Tenants Turn 10: {ylabel}")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "turn10_latency_32tenants.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for exp in experiments:
        ys = []
        for turn in turns:
            turn_rows = [r for r in exp["rows"] if int(r["turn_index"]) == turn]
            values = [float(r["ttft_ms"]) - float(r["blocking_time_ms"]) for r in turn_rows]
            ys.append(mean(values))
        ax.plot(turns, ys, marker="o", linewidth=2, label=exp["label"], color=exp["color"])
    ax.set_title("32 Tenants: Service TTFT by Turn")
    ax.set_xlabel("Turn Index")
    ax.set_ylabel("TTFT - Blocking Time (ms)")
    ax.set_xticks(turns)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "service_ttft_by_turn_32tenants.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for exp in experiments:
        xs = [float(r["blocking_time_ms"]) for r in exp["rows"]]
        ys = [float(r["ttft_ms"]) for r in exp["rows"]]
        ax.scatter(xs, ys, s=18, alpha=0.45, label=exp["label"], color=exp["color"])
    ax.set_title("32 Tenants: Blocking Time vs TTFT")
    ax.set_xlabel("Blocking Time (ms)")
    ax.set_ylabel("TTFT (ms)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "blocking_vs_ttft_32tenants.png", dpi=200)
    plt.close(fig)

    group_specs = [("8192", "Group 1", "#2563eb"), ("2048", "Group 2", "#f97316")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (metric_key, ylabel, slug) in zip(axes, metric_specs):
        xlabels = []
        values = []
        colors = []
        for exp in experiments:
            for history_limit, group_label, group_color in group_specs:
                group_rows = [r for r in exp["rows"] if r["history_limit_tokens"] == history_limit]
                xlabels.append(f"{exp['label']}\n{group_label}")
                values.append(mean([float(r[metric_key]) for r in group_rows]))
                colors.append(group_color)
        ax.bar(xlabels, values, color=colors)
        ax.set_title(f"32 Tenants: Mean {ylabel} by Group")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "group_mean_latency_32tenants.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (metric_key, ylabel, slug) in zip(axes, metric_specs):
        xlabels = []
        values = []
        colors = []
        for exp in experiments:
            for history_limit, group_label, group_color in group_specs:
                group_rows = [
                    r
                    for r in exp["rows"]
                    if r["history_limit_tokens"] == history_limit and r["turn_index"] == "10"
                ]
                xlabels.append(f"{exp['label']}\n{group_label}")
                values.append(mean([float(r[metric_key]) for r in group_rows]))
                colors.append(group_color)
        ax.bar(xlabels, values, color=colors)
        ax.set_title(f"32 Tenants Turn 10: Mean {ylabel} by Group")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "group_turn10_latency_32tenants.png", dpi=200)
    plt.close(fig)

    print(f"[DONE] comparison plots saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
