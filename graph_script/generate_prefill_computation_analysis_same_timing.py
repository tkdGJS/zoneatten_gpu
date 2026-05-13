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
GROUP_LABELS = {
    "8192": "Group1",
    "2048": "Group2",
}
TURN_COLORS = {
    5: "#16a34a",
    6: "#eab308",
    7: "#dc2626",
}
KV_BIN_COLORS = {
    "Low KV": "#93c5fd",
    "Mid KV": "#2563eb",
    "High KV": "#1e3a8a",
}
BASELINE_SLOPE_MS_PER_TOKEN = 1.0 / 6.0


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


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def find_model_config():
    cache_root = Path("/home/yuhwa2323/.cache/huggingface/hub")
    matches = list(cache_root.glob("models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/config.json"))
    if not matches:
        raise SystemExit("Could not find local model config for meta-llama/Llama-3.2-1B-Instruct")
    return matches[0]


def load_model_info():
    config_path = find_model_config()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_layers = int(config["num_hidden_layers"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    bytes_per_element = 2
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
    return {
        "kv_bytes_per_token": kv_bytes_per_token,
    }


def mib_from_tokens(token_count, kv_bytes_per_token):
    return token_count * kv_bytes_per_token / (1024.0 * 1024.0)


def load_request_metrics(exp_dir):
    metrics = {}
    metrics_dir = ROOT / exp_dir / "vram_only_artifacts_smallctx_mixed_limits" / "request_metrics"
    for path in metrics_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                metrics[payload["request_id"]] = payload
    return metrics


def load_rows(model_info):
    rows = []
    kv_bytes_per_token = model_info["kv_bytes_per_token"]
    for exp_dir, exp_label in EXPERIMENTS:
        request_metrics = load_request_metrics(exp_dir)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success":
                    continue
                if row["tenant_count"] != TENANT_COUNT:
                    continue
                metric = request_metrics.get(row["metrics_request_id"])
                if metric is None:
                    continue
                input_tokens = float(row["input_tokens"] or 0.0)
                prefix_hit_tokens = float(row["prefix_hit_tokens"] or 0.0)
                blocking_ms = float(row["blocking_time_ms"] or 0.0)
                prefill_ms = 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0)
                kv_history_tokens = float(row["kv_history_tokens"] or 0.0)
                computation_tokens = max(0.0, input_tokens - prefix_hit_tokens)
                prefill_baseline_ms = computation_tokens * BASELINE_SLOPE_MS_PER_TOKEN
                rows.append(
                    {
                        "exp_label": exp_label,
                        "turn_index": int(row["turn_index"]),
                        "group_label": GROUP_LABELS[row["history_limit_tokens"]],
                        "input_tokens": input_tokens,
                        "prefix_hit_tokens": prefix_hit_tokens,
                        "computation_tokens": computation_tokens,
                        "blocking_ms": blocking_ms,
                        "prefill_ms": prefill_ms,
                        "kv_history_tokens": kv_history_tokens,
                        "resident_kv_mib": mib_from_tokens(kv_history_tokens, kv_bytes_per_token),
                        "prefill_baseline_ms": prefill_baseline_ms,
                        "prefill_residual_ms": prefill_ms - prefill_baseline_ms,
                        "prefill_cost_per_token": (prefill_ms / computation_tokens) if computation_tokens > 0 else 0.0,
                    }
                )
    return rows


def assign_kv_bins(rows):
    values = sorted(row["resident_kv_mib"] for row in rows)
    q1 = percentile(values, 1.0 / 3.0)
    q2 = percentile(values, 2.0 / 3.0)
    for row in rows:
        value = row["resident_kv_mib"]
        if value <= q1:
            row["kv_bin"] = "Low KV"
        elif value <= q2:
            row["kv_bin"] = "Mid KV"
        else:
            row["kv_bin"] = "High KV"
    return q1, q2


def write_csv(rows, out_path):
    fieldnames = [
        "exp_label",
        "turn_index",
        "group_label",
        "input_tokens",
        "prefix_hit_tokens",
        "computation_tokens",
        "blocking_ms",
        "prefill_ms",
        "kv_history_tokens",
        "resident_kv_mib",
        "prefill_baseline_ms",
        "prefill_residual_ms",
        "prefill_cost_per_token",
        "kv_bin",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_summary(rows, q1, q2, out_path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scope",
                "count",
                "mean_prefill_ms",
                "mean_computation_tokens",
                "mean_resident_kv_mib",
                "mean_prefill_residual_ms",
                "mean_prefill_cost_per_token",
                "corr_computation_prefill",
                "corr_kv_prefill",
                "corr_kv_residual",
            ]
        )

        def emit(scope, subset):
            writer.writerow(
                [
                    scope,
                    len(subset),
                    f"{mean([r['prefill_ms'] for r in subset]):.4f}",
                    f"{mean([r['computation_tokens'] for r in subset]):.4f}",
                    f"{mean([r['resident_kv_mib'] for r in subset]):.4f}",
                    f"{mean([r['prefill_residual_ms'] for r in subset]):.4f}",
                    f"{mean([r['prefill_cost_per_token'] for r in subset if r['prefill_cost_per_token'] > 0]):.6f}",
                    f"{corr([r['computation_tokens'] for r in subset], [r['prefill_ms'] for r in subset]):.6f}",
                    f"{corr([r['resident_kv_mib'] for r in subset], [r['prefill_ms'] for r in subset]):.6f}",
                    f"{corr([r['resident_kv_mib'] for r in subset], [r['prefill_residual_ms'] for r in subset]):.6f}",
                ]
            )

        emit("all", rows)
        for turn in range(1, 11):
            emit(f"turn={turn}", [r for r in rows if r["turn_index"] == turn])
        for turn in [5, 6, 7]:
            emit(f"focus_turn={turn}", [r for r in rows if r["turn_index"] == turn])
        for kv_bin in ["Low KV", "Mid KV", "High KV"]:
            emit(f"kv_bin={kv_bin}", [r for r in rows if r["kv_bin"] == kv_bin])
        writer.writerow([])
        writer.writerow(["kv_bin_cutoff_mib", f"q1={q1:.4f}", f"q2={q2:.4f}"])


def plot_prefill_vs_computation_by_turn(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    xs = [row["computation_tokens"] for row in rows]
    x_max = max(xs) if xs else 1.0
    line_x = [0.0, x_max]
    line_y = [BASELINE_SLOPE_MS_PER_TOKEN * x for x in line_x]
    ax.plot(line_x, line_y, linestyle="--", color="#111827", linewidth=1.6, label="baseline y = x / 6")

    for turn in [5, 6, 7]:
        subset = [row for row in rows if row["turn_index"] == turn]
        ax.scatter(
            [row["computation_tokens"] for row in subset],
            [row["prefill_ms"] for row in subset],
            s=28,
            alpha=0.6,
            color=TURN_COLORS[turn],
            label=f"Turn {turn} (n={len(subset)})",
        )

    ax.set_title("32 tenants: prefill time vs computation tokens")
    ax.set_xlabel("Computation tokens = input tokens - prefix hit")
    ax.set_ylabel("Prefill time (ms)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prefill_vs_computation_by_kv_bin(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    xs = [row["computation_tokens"] for row in rows]
    x_max = max(xs) if xs else 1.0
    line_x = [0.0, x_max]
    line_y = [BASELINE_SLOPE_MS_PER_TOKEN * x for x in line_x]
    ax.plot(line_x, line_y, linestyle="--", color="#111827", linewidth=1.6, label="baseline y = x / 6")

    for kv_bin in ["Low KV", "Mid KV", "High KV"]:
        subset = [row for row in rows if row["kv_bin"] == kv_bin]
        ax.scatter(
            [row["computation_tokens"] for row in subset],
            [row["prefill_ms"] for row in subset],
            s=26,
            alpha=0.5,
            color=KV_BIN_COLORS[kv_bin],
            label=f"{kv_bin} (n={len(subset)})",
        )

    ax.set_title("32 tenants: prefill time vs computation tokens, colored by resident KV bin")
    ax.set_xlabel("Computation tokens = input tokens - prefix hit")
    ax.set_ylabel("Prefill time (ms)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual_vs_kv(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for turn in [5, 6, 7]:
        subset = [row for row in rows if row["turn_index"] == turn]
        ax.scatter(
            [row["resident_kv_mib"] for row in subset],
            [row["prefill_residual_ms"] for row in subset],
            s=30,
            alpha=0.6,
            color=TURN_COLORS[turn],
            label=f"Turn {turn} (n={len(subset)})",
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.4, color="#111827")
    ax.set_title("32 tenants: prefill residual vs resident KV usage")
    ax.set_xlabel("Resident KV usage per tenant (MiB)")
    ax.set_ylabel("Prefill residual (ms) = prefill - computation_tokens / 6")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cost_per_token_vs_kv(plt, rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    filtered = [row for row in rows if row["computation_tokens"] > 0]
    for turn in [5, 6, 7]:
        subset = [row for row in filtered if row["turn_index"] == turn]
        ax.scatter(
            [row["resident_kv_mib"] for row in subset],
            [row["prefill_cost_per_token"] for row in subset],
            s=30,
            alpha=0.6,
            color=TURN_COLORS[turn],
            label=f"Turn {turn} (n={len(subset)})",
        )
    ax.set_title("32 tenants: prefill cost per token vs resident KV usage")
    ax.set_xlabel("Resident KV usage per tenant (MiB)")
    ax.set_ylabel("Prefill time / computation tokens (ms per token)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(rows, q1, q2, out_path):
    focus = [row for row in rows if row["turn_index"] in {5, 6, 7}]
    corr_comp_prefill = corr([r["computation_tokens"] for r in focus], [r["prefill_ms"] for r in focus])
    corr_kv_prefill = corr([r["resident_kv_mib"] for r in focus], [r["prefill_ms"] for r in focus])
    corr_kv_residual = corr([r["resident_kv_mib"] for r in focus], [r["prefill_residual_ms"] for r in focus])
    corr_comp_residual = corr([r["computation_tokens"] for r in focus], [r["prefill_residual_ms"] for r in focus])

    lines = []
    lines.append("# Prefill vs Computation Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append("- Slice: `tenant_count=32` only")
    lines.append("- Samples: 960 successful requests")
    lines.append("- Main focus for visual inspection: turns 5, 6, 7")
    lines.append("")
    lines.append("## What This Checks")
    lines.append("")
    lines.append("- Whether `prefill time` is explained only by `computation tokens = input tokens - prefix hit`.")
    lines.append("- Whether the remaining unexplained part grows with `resident KV usage`.")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append(
        f"- In turns 5/6/7, correlation between computation tokens and prefill is `{corr_comp_prefill:.3f}`."
    )
    lines.append(
        f"- In the same turns, correlation between resident KV per tenant and prefill is `{corr_kv_prefill:.3f}`."
    )
    lines.append(
        f"- After subtracting the baseline `prefill = computation_tokens / 6`, correlation between resident KV and prefill residual is `{corr_kv_residual:.3f}`."
    )
    lines.append(
        f"- Correlation between computation tokens and that residual is `{corr_comp_residual:.3f}`."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If computation tokens alone explained prefill, the scatter would collapse toward a single narrow curve. It does not."
    )
    lines.append(
        "- Higher resident KV bins tend to sit higher for the same computation-token range, which means the same compute load costs more under higher KV pressure."
    )
    lines.append(
        "- The residual plot shows the same point more directly: even after subtracting a simple token-count baseline, higher resident KV still tends to push requests upward."
    )
    lines.append(
        "- This supports the stricter claim: `computation tokens create the base prefill load, but KV pressure / batching / scheduling state add extra spread and extra delay`."
    )
    lines.append("")
    lines.append("## KV Bins")
    lines.append("")
    lines.append(f"- Low KV: `resident_kv_mib <= {q1:.1f}`")
    lines.append(f"- Mid KV: `{q1:.1f} < resident_kv_mib <= {q2:.1f}`")
    lines.append(f"- High KV: `resident_kv_mib > {q2:.1f}`")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_info = load_model_info()
    rows = load_rows(model_info)
    q1, q2 = assign_kv_bins(rows)

    write_csv(rows, OUT_DIR / "prefill_computation_enriched_32tenants.csv")
    write_summary(rows, q1, q2, OUT_DIR / "prefill_computation_summary_32tenants.csv")
    plot_prefill_vs_computation_by_turn(plt, rows, OUT_DIR / "prefill_vs_computation_turn5_6_7_32tenants.png")
    plot_prefill_vs_computation_by_kv_bin(plt, rows, OUT_DIR / "prefill_vs_computation_kv_bins_32tenants.png")
    plot_residual_vs_kv(plt, rows, OUT_DIR / "prefill_residual_vs_resident_kv_turn5_6_7_32tenants.png")
    plot_cost_per_token_vs_kv(plt, rows, OUT_DIR / "prefill_cost_per_token_vs_resident_kv_turn5_6_7_32tenants.png")
    write_markdown(rows, q1, q2, OUT_DIR / "prefill_vs_computation_analysis.md")


if __name__ == "__main__":
    main()
