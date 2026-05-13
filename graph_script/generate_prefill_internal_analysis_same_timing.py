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
MAX_NUM_BATCHED_TOKENS = 8192.0
GROUP_COLORS = {
    "Group1": "#2563eb",
    "Group2": "#f97316",
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


def rankdata(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    return corr(rankdata(xs), rankdata(ys))


def load_request_metrics(exp_dir):
    metrics = {}
    metrics_dir = ROOT / exp_dir / "vram_only_artifacts_smallctx_mixed_limits" / "request_metrics"
    for path in metrics_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                metrics[payload["request_id"]] = payload
    return metrics


def load_rows():
    rows = []
    for exp_dir, exp_label in EXPERIMENTS:
        metrics = load_request_metrics(exp_dir)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success" or row["tenant_count"] != TENANT_COUNT:
                    continue
                blocking_ms = float(row["blocking_time_ms"] or 0.0)
                if blocking_ms >= BLOCKING_THRESHOLD_MS:
                    continue
                metric = metrics.get(row["metrics_request_id"])
                if metric is None:
                    continue
                input_tokens = float(row["input_tokens"] or 0.0)
                prefix_hit_tokens = float(row["prefix_hit_tokens"] or 0.0)
                computation_tokens = max(0.0, input_tokens - prefix_hit_tokens)
                rows.append(
                    {
                        "exp_label": exp_label,
                        "turn_index": int(row["turn_index"]),
                        "tenant_id": int(row["tenant_id"]),
                        "group_label": "Group1" if row["history_limit_tokens"] == "8192" else "Group2",
                        "blocking_ms": blocking_ms,
                        "input_tokens": input_tokens,
                        "prefix_hit_tokens": prefix_hit_tokens,
                        "computation_tokens": computation_tokens,
                        "prefill_ms": 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0),
                        "first_token_ts": float(metric.get("first_token_ts", 0.0) or 0.0),
                        "queued_ts": float(metric.get("queued_ts", 0.0) or 0.0),
                        "scheduled_ts": float(metric.get("scheduled_ts", 0.0) or 0.0),
                    }
                )
    return rows


def enrich_batch_context(rows):
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[(row["exp_label"], row["turn_index"])].append(row)

    for _, subset in by_batch.items():
        batch_total_computation = sum(r["computation_tokens"] for r in subset)
        batch_total_input = sum(r["input_tokens"] for r in subset)
        first_token_values = [r["first_token_ts"] for r in subset if r["first_token_ts"] > 0]
        first_token_min = min(first_token_values) if first_token_values else 0.0
        first_token_max = max(first_token_values) if first_token_values else 0.0
        first_token_spread_ms = (first_token_max - first_token_min) * 1000.0
        estimated_rounds = batch_total_computation / MAX_NUM_BATCHED_TOKENS
        for row in subset:
            row["batch_size_filtered"] = len(subset)
            row["batch_total_computation_tokens"] = batch_total_computation
            row["batch_total_input_tokens"] = batch_total_input
            row["batch_estimated_prefill_rounds_ratio"] = estimated_rounds
            row["batch_first_token_spread_ms"] = first_token_spread_ms
            row["time_from_batch_first_token_ms"] = max(0.0, (row["first_token_ts"] - first_token_min) * 1000.0)


AXES = [
    ("computation_tokens", "Computation tokens"),
    ("batch_total_computation_tokens", "Batch total computation tokens"),
    ("batch_estimated_prefill_rounds_ratio", "Batch total compute / 8192"),
    ("batch_first_token_spread_ms", "Batch first-token spread (ms)"),
    ("time_from_batch_first_token_ms", "Delay from first request in batch (ms)"),
]


def write_enriched_csv(rows, out_path):
    fieldnames = [
        "exp_label",
        "turn_index",
        "tenant_id",
        "group_label",
        "blocking_ms",
        "input_tokens",
        "prefix_hit_tokens",
        "computation_tokens",
        "prefill_ms",
        "first_token_ts",
        "queued_ts",
        "scheduled_ts",
        "batch_size_filtered",
        "batch_total_computation_tokens",
        "batch_total_input_tokens",
        "batch_estimated_prefill_rounds_ratio",
        "batch_first_token_spread_ms",
        "time_from_batch_first_token_ms",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_axis_summary(rows, out_path):
    ys = [r["prefill_ms"] for r in rows]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["axis_key", "axis_label", "pearson_prefill", "spearman_prefill"])
        scored = []
        for key, label in AXES:
            xs = [r[key] for r in rows]
            pear = corr(xs, ys)
            spear = spearman(xs, ys)
            scored.append((abs(spear), key, label, pear, spear))
        for _, key, label, pear, spear in sorted(scored, reverse=True):
            writer.writerow([key, label, f"{pear:.6f}", f"{spear:.6f}"])


def plot_axis_panel(plt, rows, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.08, right=0.98, hspace=0.30, wspace=0.24)
    turn_handles = []
    ys = [r["prefill_ms"] for r in rows]
    for ax, (key, label) in zip(axes.flatten(), AXES):
        for turn in range(1, 11):
            subset = [r for r in rows if r["turn_index"] == turn]
            sc = ax.scatter(
                [r[key] for r in subset],
                [r["prefill_ms"] for r in subset],
                s=16,
                alpha=0.42,
                color=TURN_COLORS[turn],
            )
            if len(turn_handles) < 10:
                turn_handles.append((sc, f"Turn {turn}"))
        pear = corr([r[key] for r in rows], ys)
        spear = spearman([r[key] for r in rows], ys)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Prefill time (ms)")
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.97,
            f"r={pear:.2f}\nρ={spear:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    axes.flatten()[-1].axis("off")
    fig.legend(
        [h for h, _ in turn_handles],
        [label for _, label in turn_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        title="Color = turn",
        frameon=True,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_group_split(plt, rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.08, right=0.98, wspace=0.22)
    specs = [
        ("batch_total_computation_tokens", "Batch total computation tokens"),
        ("batch_first_token_spread_ms", "Batch first-token spread (ms)"),
    ]
    for ax, (key, label) in zip(axes, specs):
        for group_label in ["Group1", "Group2"]:
            subset = [r for r in rows if r["group_label"] == group_label]
            ax.scatter(
                [r[key] for r in subset],
                [r["prefill_ms"] for r in subset],
                s=18,
                alpha=0.42,
                color=GROUP_COLORS[group_label],
                label=f"{group_label} (n={len(subset)})",
            )
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Prefill time (ms)")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", title="Color = group", frameon=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_batch_level_summary(plt, rows, out_path):
    batch_rows = []
    seen = set()
    for row in rows:
        key = (row["exp_label"], row["turn_index"])
        if key in seen:
            continue
        seen.add(key)
        batch_subset = [r for r in rows if r["exp_label"] == row["exp_label"] and r["turn_index"] == row["turn_index"]]
        batch_rows.append(
            {
                "exp_label": row["exp_label"],
                "turn_index": row["turn_index"],
                "batch_total_computation_tokens": row["batch_total_computation_tokens"],
                "batch_first_token_spread_ms": row["batch_first_token_spread_ms"],
                "mean_prefill_ms": mean([r["prefill_ms"] for r in batch_subset]),
                "p95_prefill_ms": sorted([r["prefill_ms"] for r in batch_subset])[int(max(0, len(batch_subset) * 0.95 - 1))],
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.08, right=0.98, wspace=0.22)
    for ax, (key, label) in zip(
        axes,
        [
            ("batch_total_computation_tokens", "Batch total computation tokens"),
            ("batch_first_token_spread_ms", "Batch first-token spread (ms)"),
        ],
    ):
        ax.scatter(
            [r[key] for r in batch_rows],
            [r["mean_prefill_ms"] for r in batch_rows],
            s=45,
            alpha=0.7,
            color="#2563eb",
            label="Batch mean prefill",
        )
        ax.scatter(
            [r[key] for r in batch_rows],
            [r["p95_prefill_ms"] for r in batch_rows],
            s=45,
            alpha=0.7,
            color="#dc2626",
            label="Batch p95 prefill",
        )
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Prefill time (ms)")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", frameon=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(rows, out_path):
    lines = []
    lines.append("# Prefill Internal Analysis With Blocking < 100 ms")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append("- Slice: `tenant_count=32` only")
    lines.append(f"- Filter: `blocking_time_ms < {BLOCKING_THRESHOLD_MS}`")
    lines.append(f"- Samples after filtering: `{len(rows)}`")
    lines.append("")
    lines.append("## Why This Filter")
    lines.append("")
    lines.append("- The goal is to remove most queueing effect and focus on internal prefill execution differences.")
    lines.append("- In this slice, prefill variation should be interpreted mainly as execution/context difference inside the engine, not API waiting.")
    lines.append("")
    lines.append("## Axis Ranking")
    lines.append("")
    ys = [r["prefill_ms"] for r in rows]
    scored = []
    for key, label in AXES:
        xs = [r[key] for r in rows]
        scored.append((abs(spearman(xs, ys)), label, corr(xs, ys), spearman(xs, ys)))
    for _, label, pear, spear in sorted(scored, reverse=True):
        lines.append(f"- `{label}`: Pearson `{pear:.3f}`, Spearman `{spear:.3f}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If `time_from_batch_first_token_ms` or `batch_first_token_spread_ms` still tracks prefill after queueing is filtered out, that is evidence for internal stagger/chunk/interleaving effects.")
    lines.append("- If `batch_total_computation_tokens` remains the strongest axis, then batch-wide prefill load is still the most direct driver even without explicit blocking.")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    enrich_batch_context(rows)
    write_enriched_csv(rows, OUT_DIR / "prefill_internal_enriched_blocking_lt_100ms_32tenants.csv")
    write_axis_summary(rows, OUT_DIR / "prefill_internal_axis_summary_blocking_lt_100ms_32tenants.csv")
    plot_axis_panel(plt, rows, OUT_DIR / "prefill_internal_axis_panel_blocking_lt_100ms_32tenants.png")
    plot_group_split(plt, rows, OUT_DIR / "prefill_internal_group_split_blocking_lt_100ms_32tenants.png")
    plot_batch_level_summary(plt, rows, OUT_DIR / "prefill_internal_batch_level_summary_blocking_lt_100ms_32tenants.png")
    write_markdown(rows, OUT_DIR / "prefill_internal_analysis_blocking_lt_100ms.md")


if __name__ == "__main__":
    main()
