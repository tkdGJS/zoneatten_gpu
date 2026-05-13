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
GROUP_COLORS = {
    "Group1": "#2563eb",
    "Group2": "#f97316",
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
                rows.append(
                    {
                        "exp_label": exp_label,
                        "turn_index": int(row["turn_index"]),
                        "group_label": "Group1" if row["history_limit_tokens"] == "8192" else "Group2",
                        "prefill_ms": 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0),
                        "first_token_ts": float(metric.get("first_token_ts", 0.0) or 0.0),
                    }
                )
    return rows


def enrich_delay(rows):
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[(row["exp_label"], row["turn_index"])].append(row)
    for subset in by_batch.values():
        min_first_token = min(r["first_token_ts"] for r in subset if r["first_token_ts"] > 0)
        for row in subset:
            row["delay_from_earliest_first_token_ms"] = max(0.0, (row["first_token_ts"] - min_first_token) * 1000.0)


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    enrich_delay(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.98, wspace=0.18)

    for ax, group_label in zip(axes, ["Group1", "Group2"]):
        subset = [r for r in rows if r["group_label"] == group_label]
        ax.scatter(
            [r["delay_from_earliest_first_token_ms"] for r in subset],
            [r["prefill_ms"] for r in subset],
            s=18,
            alpha=0.45,
            color=GROUP_COLORS[group_label],
        )
        pear = corr(
            [r["delay_from_earliest_first_token_ms"] for r in subset],
            [r["prefill_ms"] for r in subset],
        )
        ax.set_title(f"{group_label}")
        ax.set_xlabel("Delay from earliest first-token in filtered batch (ms)")
        ax.set_ylabel("Prefill time (ms)")
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.97,
            f"r={pear:.3f}\nn={len(subset)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig.suptitle("Prefill vs earliest first-token delay by group, blocking time < 100 ms")
    out_path = OUT_DIR / "prefill_vs_earliest_first_token_delay_group_panels_blocking_lt_100ms_32tenants.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
