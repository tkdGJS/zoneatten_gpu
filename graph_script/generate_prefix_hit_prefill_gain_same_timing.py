#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "graphs" / Path(__file__).stem
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
TENANT_COUNT = "32"
BLOCKING_THRESHOLD_MS = 100.0


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


def fit_line(xs, ys):
    if len(xs) < 2:
        return 0.0, 0.0
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    if vx <= 0.0:
        return 0.0, my
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = cov / vx
    intercept = my - slope * mx
    return slope, intercept


def padded_limits(values, pad_ratio=0.05):
    if not values:
        return None
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = max(1.0, abs(lo) * pad_ratio, 0.1)
        return (lo - pad, hi + pad)
    pad = (hi - lo) * pad_ratio
    return (lo - pad, hi + pad)


def load_request_metrics(exp_dir):
    metrics = {}
    metrics_dir = ROOT / exp_dir / "vram_only_artifacts_smallctx_mixed_limits" / "request_metrics"
    for path in metrics_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                metrics[payload["request_id"]] = payload
    return metrics


def blocking_matches(blocking_ms, blocking_mode):
    if blocking_mode == "lt100":
        return blocking_ms < BLOCKING_THRESHOLD_MS
    if blocking_mode == "gt100":
        return blocking_ms > BLOCKING_THRESHOLD_MS
    raise ValueError(f"unknown blocking mode: {blocking_mode}")


def blocking_label(blocking_mode):
    return "blocking time < 100 ms" if blocking_mode == "lt100" else "blocking time > 100 ms"


def blocking_suffix(blocking_mode):
    return "blocking_lt_100ms" if blocking_mode == "lt100" else "blocking_gt_100ms"


def group_label_from_history_limit(history_limit_tokens):
    return "Group1" if str(history_limit_tokens) == "8192" else "Group2"


def load_batch_rows(blocking_mode):
    by_batch = defaultdict(list)
    for exp_dir, exp_label in EXPERIMENTS:
        metrics = load_request_metrics(exp_dir)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success" or row["tenant_count"] != TENANT_COUNT:
                    continue
                blocking_ms = float(row["blocking_time_ms"] or 0.0)
                if not blocking_matches(blocking_ms, blocking_mode):
                    continue
                metric = metrics.get(row["metrics_request_id"])
                if metric is None:
                    continue
                group_label = group_label_from_history_limit(row["history_limit_tokens"])
                input_tokens = float(row["input_tokens"] or 0.0)
                prefix_hit_tokens = float(row["prefix_hit_tokens"] or 0.0)
                computation_tokens = max(0.0, input_tokens - prefix_hit_tokens)
                by_batch[(exp_label, int(row["turn_index"]), group_label)].append(
                    {
                        "prefill_ms": 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0),
                        "input_tokens": input_tokens,
                        "prefix_hit_tokens": prefix_hit_tokens,
                        "computation_tokens": computation_tokens,
                    }
                )
    rows = []
    for (exp_label, turn_index, group_label), subset in sorted(by_batch.items()):
        total_input = sum(r["input_tokens"] for r in subset)
        total_prefix_hit = sum(r["prefix_hit_tokens"] for r in subset)
        total_compute = sum(r["computation_tokens"] for r in subset)
        mean_prefill = mean([r["prefill_ms"] for r in subset])
        rows.append(
            {
                "exp_label": exp_label,
                "group_label": group_label,
                "turn_index": turn_index,
                "filtered_batch_size": len(subset),
                "batch_total_input_tokens": total_input,
                "batch_total_prefix_hit_tokens": total_prefix_hit,
                "batch_total_computation_tokens": total_compute,
                "batch_prefix_hit_rate": (total_prefix_hit / total_input) if total_input > 0 else 0.0,
                "batch_mean_prefill_ms": mean_prefill,
                "batch_mean_prefill_per_input_token_ms": (mean_prefill / total_input) if total_input > 0 else 0.0,
            }
        )
    xs = [r["batch_total_computation_tokens"] for r in rows]
    ys = [r["batch_mean_prefill_ms"] for r in rows]
    slope, intercept = fit_line(xs, ys)
    for row in rows:
        row["fit_prefill_from_compute_ms"] = slope * row["batch_total_computation_tokens"] + intercept
        row["fit_prefill_no_hit_ms"] = slope * row["batch_total_input_tokens"] + intercept
        row["estimated_prefix_hit_gain_ms"] = row["fit_prefill_no_hit_ms"] - row["batch_mean_prefill_ms"]
    return rows, slope, intercept


def write_csv(rows, out_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_input_vs_prefill_by_hit_rate(plt, rows, out_path, blocking_mode, xlim=None, ylim=None, clim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.88)
    xs = [r["batch_total_input_tokens"] for r in rows]
    ys = [r["batch_mean_prefill_ms"] for r in rows]
    cs = [r["batch_prefix_hit_rate"] for r in rows]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=70, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_title(f"Batch total input tokens vs prefill time average\n{blocking_label(blocking_mode)}")
    ax.set_xlabel("Batch total input tokens")
    ax.set_ylabel("Prefill time average (ms)")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    if clim:
        sc.set_clim(*clim)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Batch prefix hit rate")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_input_vs_compute_with_hit_rate(plt, rows, out_path, blocking_mode, xlim=None, ylim=None, clim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.88)
    xs = [r["batch_total_input_tokens"] for r in rows]
    ys = [r["batch_total_computation_tokens"] for r in rows]
    cs = [r["batch_prefix_hit_rate"] for r in rows]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=70, alpha=0.8, edgecolors="white", linewidths=0.4)
    lo = min(min(xs), min(ys)) if rows else 0.0
    hi = max(max(xs), max(ys)) if rows else 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#6b7280", linewidth=1.2, label="y = x")
    ax.set_title(f"Batch total input tokens vs total compute tokens\n{blocking_label(blocking_mode)}")
    ax.set_xlabel("Batch total input tokens")
    ax.set_ylabel("Batch total computation tokens")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    if clim:
        sc.set_clim(*clim)
    ax.legend(loc="upper left")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Batch prefix hit rate")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hit_rate_vs_prefill_per_input(plt, rows, out_path, blocking_mode, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.82)
    for group_label, color in [("Group1", "#2563eb"), ("Group2", "#f97316")]:
        subset = [r for r in rows if r["group_label"] == group_label]
        ax.scatter(
            [r["batch_prefix_hit_rate"] for r in subset],
            [r["batch_mean_prefill_per_input_token_ms"] for r in subset],
            s=70,
            alpha=0.8,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            label=group_label,
        )
    pear = corr([r["batch_prefix_hit_rate"] for r in rows], [r["batch_mean_prefill_per_input_token_ms"] for r in rows])
    ax.set_title(f"Prefix hit rate vs prefill cost per input token\n{blocking_label(blocking_mode)}, r = {pear:.3f}")
    ax.set_xlabel("Batch prefix hit rate")
    ax.set_ylabel("Prefill time average / batch total input tokens")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Color = group", frameon=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hit_tokens_vs_prefill_per_input(plt, rows, out_path, blocking_mode, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.82)
    for group_label, color in [("Group1", "#2563eb"), ("Group2", "#f97316")]:
        subset = [r for r in rows if r["group_label"] == group_label]
        ax.scatter(
            [r["batch_total_prefix_hit_tokens"] for r in subset],
            [r["batch_mean_prefill_per_input_token_ms"] for r in subset],
            s=70,
            alpha=0.8,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            label=group_label,
        )
    pear = corr([r["batch_total_prefix_hit_tokens"] for r in rows], [r["batch_mean_prefill_per_input_token_ms"] for r in rows])
    ax.set_title(f"Prefix hit tokens vs prefill cost per input token\n{blocking_label(blocking_mode)}, r = {pear:.3f}")
    ax.set_xlabel("Batch total prefix hit tokens")
    ax.set_ylabel("Prefill time average / batch total input tokens")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Color = group", frameon=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hit_rate_vs_estimated_gain(plt, rows, out_path, blocking_mode, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.82)
    for group_label, color in [("Group1", "#2563eb"), ("Group2", "#f97316")]:
        subset = [r for r in rows if r["group_label"] == group_label]
        ax.scatter(
            [r["batch_prefix_hit_rate"] for r in subset],
            [r["estimated_prefix_hit_gain_ms"] for r in subset],
            s=70,
            alpha=0.8,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            label=group_label,
        )
    pear = corr([r["batch_prefix_hit_rate"] for r in rows], [r["estimated_prefix_hit_gain_ms"] for r in rows])
    ax.set_title(f"Prefix hit rate vs estimated prefill gain\n{blocking_label(blocking_mode)}, r = {pear:.3f}")
    ax.set_xlabel("Batch prefix hit rate")
    ax.set_ylabel("Estimated prefill gain from prefix hit (ms)")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Color = group", frameon=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_saved_tokens_vs_estimated_gain(plt, rows, out_path, blocking_mode, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.82)
    for group_label, color in [("Group1", "#2563eb"), ("Group2", "#f97316")]:
        subset = [r for r in rows if r["group_label"] == group_label]
        ax.scatter(
            [r["batch_total_prefix_hit_tokens"] for r in subset],
            [r["estimated_prefix_hit_gain_ms"] for r in subset],
            s=70,
            alpha=0.8,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            label=group_label,
        )
    pear = corr([r["batch_total_prefix_hit_tokens"] for r in rows], [r["estimated_prefix_hit_gain_ms"] for r in rows])
    ax.set_title(f"Saved tokens vs estimated prefill gain\n{blocking_label(blocking_mode)}, r = {pear:.3f}")
    ax.set_xlabel("Batch total prefix hit tokens")
    ax.set_ylabel("Estimated prefill gain from prefix hit (ms)")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Color = group", frameon=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(rows, slope, intercept, out_path, blocking_mode):
    lines = []
    lines.append("# Prefix Hit vs Prefill Gain")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append(f"- Slice: `tenant_count=32`, `{blocking_label(blocking_mode)}`")
    lines.append("- One point = one filtered `(exp, turn, group)` batch")
    lines.append("")
    lines.append("## Why This Analysis")
    lines.append("")
    lines.append("- `Prefill vs total compute / 8192` explains execution pressure, but it does not directly explain why prefix hit itself matters.")
    lines.append("- These graphs add the missing counterfactual view: how much extra prefill would have been needed if the same input had not benefited from prefix reuse.")
    lines.append("")
    lines.append("## Fitted Baseline")
    lines.append("")
    lines.append(
        f"- A batch-level linear fit was estimated from `batch_total_computation_tokens -> batch_mean_prefill_ms`: `prefill ≈ {slope:.6f} * compute + {intercept:.3f}`"
    )
    lines.append("- This fit is then used to estimate a no-hit counterfactual by replacing `compute` with `input`.")
    lines.append("")
    lines.append("## Estimated Gain Definition")
    lines.append("")
    lines.append("- `fit_prefill_no_hit_ms = slope * batch_total_input_tokens + intercept`")
    lines.append("- `estimated_prefix_hit_gain_ms = fit_prefill_no_hit_ms - actual_batch_mean_prefill_ms`")
    lines.append("- This is an estimate, not a directly observed vLLM counterfactual run.")
    lines.append("")
    lines.append("## Key Readout")
    lines.append("")
    lines.append(
        f"- `batch_prefix_hit_rate` vs `prefill time average / input token`: r = {corr([r['batch_prefix_hit_rate'] for r in rows], [r['batch_mean_prefill_per_input_token_ms'] for r in rows]):.3f}"
    )
    lines.append(
        f"- `batch_prefix_hit_rate` vs `estimated_prefix_hit_gain_ms`: r = {corr([r['batch_prefix_hit_rate'] for r in rows], [r['estimated_prefix_hit_gain_ms'] for r in rows]):.3f}"
    )
    lines.append(
        f"- `batch_total_prefix_hit_tokens` vs `estimated_prefix_hit_gain_ms`: r = {corr([r['batch_total_prefix_hit_tokens'] for r in rows], [r['estimated_prefix_hit_gain_ms'] for r in rows]):.3f}"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If higher prefix-hit rate lowers prefill cost per input token, then prefix reuse is not just reducing token count on paper; it is translating into lower execution cost.")
    lines.append("- If saved tokens track estimated gain, then prefix hit can be framed as a concrete prefill-latency optimization rather than only a cache statistic.")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for mode in ("lt100", "gt100"):
        rows, slope, intercept = load_batch_rows(mode)
        loaded[mode] = {"rows": rows, "slope": slope, "intercept": intercept}

    all_rows = loaded["lt100"]["rows"] + loaded["gt100"]["rows"]
    input_xlim = padded_limits([r["batch_total_input_tokens"] for r in all_rows])
    compute_ylim = padded_limits([r["batch_total_computation_tokens"] for r in all_rows])
    prefill_ylim = padded_limits([r["batch_mean_prefill_ms"] for r in all_rows])
    hit_rate_xlim = padded_limits([r["batch_prefix_hit_rate"] for r in all_rows])
    hit_tokens_xlim = padded_limits([r["batch_total_prefix_hit_tokens"] for r in all_rows])
    prefill_per_input_ylim = padded_limits([r["batch_mean_prefill_per_input_token_ms"] for r in all_rows])
    estimated_gain_ylim = padded_limits([r["estimated_prefix_hit_gain_ms"] for r in all_rows])
    saved_tokens_xlim = padded_limits([r["batch_total_prefix_hit_tokens"] for r in all_rows])
    hit_rate_clim = padded_limits([r["batch_prefix_hit_rate"] for r in all_rows], pad_ratio=0.0)

    for mode in ("lt100", "gt100"):
        rows = loaded[mode]["rows"]
        slope = loaded[mode]["slope"]
        intercept = loaded[mode]["intercept"]
        suffix = blocking_suffix(mode)
        write_csv(rows, OUT_DIR / f"prefix_hit_prefill_gain_batches_{suffix}_32tenants.csv")
        plot_input_vs_prefill_by_hit_rate(
            plt,
            rows,
            OUT_DIR / f"batch_total_input_tokens_vs_prefill_by_prefix_hit_rate_{suffix}_32tenants.png",
            mode,
            input_xlim,
            prefill_ylim,
            hit_rate_clim,
        )
        plot_input_vs_compute_with_hit_rate(
            plt,
            rows,
            OUT_DIR / f"batch_total_input_tokens_vs_total_compute_tokens_by_prefix_hit_rate_{suffix}_32tenants.png",
            mode,
            input_xlim,
            compute_ylim,
            hit_rate_clim,
        )
        plot_hit_rate_vs_prefill_per_input(
            plt,
            rows,
            OUT_DIR / f"batch_prefix_hit_rate_vs_prefill_per_input_token_{suffix}_32tenants.png",
            mode,
            hit_rate_xlim,
            prefill_per_input_ylim,
        )
        plot_hit_tokens_vs_prefill_per_input(
            plt,
            rows,
            OUT_DIR / f"batch_prefix_hit_tokens_vs_prefill_per_input_token_{suffix}_32tenants.png",
            mode,
            hit_tokens_xlim,
            prefill_per_input_ylim,
        )
        plot_hit_rate_vs_estimated_gain(
            plt,
            rows,
            OUT_DIR / f"batch_prefix_hit_rate_vs_estimated_prefill_gain_{suffix}_32tenants.png",
            mode,
            hit_rate_xlim,
            estimated_gain_ylim,
        )
        plot_saved_tokens_vs_estimated_gain(
            plt,
            rows,
            OUT_DIR / f"batch_saved_tokens_vs_estimated_prefill_gain_{suffix}_32tenants.png",
            mode,
            saved_tokens_xlim,
            estimated_gain_ylim,
        )
        write_markdown(
            rows,
            slope,
            intercept,
            OUT_DIR / ("prefix_hit_prefill_gain_analysis.md" if mode == "lt100" else "prefix_hit_prefill_gain_analysis_blocking_gt_100ms.md"),
            mode,
        )


if __name__ == "__main__":
    main()
