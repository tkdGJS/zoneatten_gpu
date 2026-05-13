#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


TENANT_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
}


def make_bar_plot(subset: pd.DataFrame, metric: str, ylabel: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [TENANT_COLORS.get(int(tenant_id), "#2F6BFF") for tenant_id in subset["tenant_id"]]
    ax.bar(subset["tenant_id"].astype(str), subset[metric], color=colors)
    ax.set_xlabel("Tenant ID")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Tenant ID (tenant_count={int(subset['tenant_count'].iloc[0])})")
    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df[df["status"] == "success"].copy()
    if df.empty:
        raise SystemExit("No successful rows found in input CSV")

    df["tenant_count"] = df["tenant_count"].astype(int)
    df["tenant_id"] = df["tenant_id"].astype(int)
    df["ttft_ms"] = df["ttft_ms"].astype(float)
    df["p95_tbt_ms"] = df["p95_tbt_ms"].astype(float)
    df["ttlt_ms"] = df["ttlt_ms"].astype(float)

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = [
        ("ttft_ms", "TTFT (ms)", "ttft"),
        ("p95_tbt_ms", "p95(TBT) (ms)", "tbt"),
        ("ttlt_ms", "TTLT (ms)", "ttlt"),
    ]

    metric_ranges = {}
    for metric, _, _ in metrics:
        values = df[metric].astype(float)
        metric_ranges[metric] = (0.0, float(values.max()) * 1.05)

    for tenant_count in sorted(df["tenant_count"].unique()):
        subset = df[df["tenant_count"] == tenant_count].sort_values("tenant_id")
        for metric, ylabel, slug in metrics:
            output_path = os.path.join(
                args.output_dir,
                f"tenant_count_{tenant_count}_{slug}.png",
            )
            fig, ax = make_bar_plot(subset, metric, ylabel, output_path)
            ax.set_ylim(metric_ranges[metric])
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
