#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "vram_only_results_smallctx_mixed_limits_8GB" / "result_raw.csv"
OUT_DIR = ROOT / "vram_only_results_smallctx_mixed_limits_8GB" / "graphs" / Path(__file__).stem

BLOCKING_COLOR = "#2563eb"
PREFILL_COLOR = "#84cc16"
REMAINING_COLOR = "#dc2626"
PREFILL_FRACTIONS = [0.5, 0.7, 0.9]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    frac = idx - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def load_rows():
    rows = []
    with RAW_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "success":
                continue
            ttft_ms = float(row["ttft_ms"] or 0.0)
            blocking_ms = float(row["blocking_time_ms"] or 0.0)
            nonblocking_ms = max(0.0, ttft_ms - blocking_ms)
            rows.append(
                {
                    "tenant_count": int(row["tenant_count"]),
                    "turn_index": int(row["turn_index"]),
                    "tenant_id": int(row["tenant_id"]),
                    "history_limit_tokens": int(row["history_limit_tokens"]),
                    "ttft_ms": ttft_ms,
                    "blocking_ms": blocking_ms,
                    "nonblocking_ms": nonblocking_ms,
                    "input_tokens": float(row["input_tokens"] or 0.0),
                    "prefix_hit_tokens": float(row["prefix_hit_tokens"] or 0.0),
                    "computation_tokens": max(
                        0.0,
                        float(row["input_tokens"] or 0.0)
                        - float(row["prefix_hit_tokens"] or 0.0),
                    ),
                }
            )
    return rows


def add_estimated_breakdown(rows, prefill_fraction):
    enriched = []
    for row in rows:
        prefill_ms = row["nonblocking_ms"] * prefill_fraction
        remaining_ms = max(0.0, row["ttft_ms"] - row["blocking_ms"] - prefill_ms)
        enriched.append(
            {
                **row,
                "estimated_prefill_fraction": prefill_fraction,
                "estimated_prefill_ms": prefill_ms,
                "estimated_remaining_ttft_ms": remaining_ms,
            }
        )
    return enriched


def group_name(limit, group1_limit, group2_limit):
    if limit == group1_limit:
        return f"Group1 (limit={limit})"
    if limit == group2_limit:
        return f"Group2 (limit={limit})"
    return f"limit={limit}"


def write_enriched_csv(rows, out_path):
    fieldnames = [
        "estimated_prefill_fraction",
        "tenant_count",
        "turn_index",
        "tenant_id",
        "history_limit_tokens",
        "ttft_ms",
        "blocking_ms",
        "estimated_prefill_ms",
        "estimated_remaining_ttft_ms",
        "input_tokens",
        "prefix_hit_tokens",
        "computation_tokens",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def plot_group_p99_by_turn(plt, rows, prefill_fraction, out_path):
    limits = sorted({row["history_limit_tokens"] for row in rows})
    group2_limit = limits[0]
    group1_limit = limits[-1]
    groups = [
        (
            group_name(group1_limit, group1_limit, group2_limit),
            [row for row in rows if row["history_limit_tokens"] == group1_limit],
        ),
        (
            group_name(group2_limit, group1_limit, group2_limit),
            [row for row in rows if row["history_limit_tokens"] == group2_limit],
        ),
    ]

    fig, axes = plt.subplots(1, len(groups), figsize=(18, 6.2), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.9, wspace=0.16)
    ymax = 0.0
    stacked = []
    for _, group_rows in groups:
        by_turn = defaultdict(list)
        for row in group_rows:
            by_turn[row["turn_index"]].append(row)
        turns = list(range(1, 11))
        blocking = [percentile([r["blocking_ms"] for r in by_turn[t]], 0.99) / 1000.0 for t in turns]
        prefill = [
            percentile([r["estimated_prefill_ms"] for r in by_turn[t]], 0.99) / 1000.0
            for t in turns
        ]
        remaining = [
            percentile([r["estimated_remaining_ttft_ms"] for r in by_turn[t]], 0.99) / 1000.0
            for t in turns
        ]
        ymax = max(ymax, max(b + p + r for b, p, r in zip(blocking, prefill, remaining)))
        stacked.append((turns, blocking, prefill, remaining))

    for idx, (ax, (label, _), (turns, blocking, prefill, remaining)) in enumerate(
        zip(axes, groups, stacked)
    ):
        ax.stackplot(
            turns,
            blocking,
            prefill,
            remaining,
            colors=[BLOCKING_COLOR, PREFILL_COLOR, REMAINING_COLOR],
            alpha=0.92,
        )
        ax.set_xlim(1, 10)
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Turn")
        ax.grid(axis="y", alpha=0.2)
        ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top", ha="left")
        ax.text(
            0.5,
            -0.22,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            clip_on=False,
        )
    axes[0].set_ylabel("P99 Time (sec)")
    axes[0].legend(
        ["Blocking Time", f"Estimated Prefill ({prefill_fraction:.0%})", "Remaining TTFT"],
        loc="upper right",
        frameon=True,
        fontsize=13,
    )
    fig.suptitle(
        "Estimated TTFT Breakdown With Visible Remaining TTFT "
        f"(prefill={prefill_fraction:.0%} of non-blocking TTFT)",
        fontsize=16,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tenant_sweep_mean_by_turn(plt, rows, prefill_fraction, out_path):
    tenant_counts = sorted({row["tenant_count"] for row in rows})
    fig, axes = plt.subplots(1, len(tenant_counts), figsize=(6 * len(tenant_counts), 5.8), sharey=True)
    if len(tenant_counts) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.88, wspace=0.16)
    ymax = 0.0
    stacked = []
    for tenant_count in tenant_counts:
        subset = [row for row in rows if row["tenant_count"] == tenant_count]
        by_turn = defaultdict(list)
        for row in subset:
            by_turn[row["turn_index"]].append(row)
        turns = list(range(1, 11))
        blocking = [mean([r["blocking_ms"] for r in by_turn[t]]) / 1000.0 for t in turns]
        prefill = [mean([r["estimated_prefill_ms"] for r in by_turn[t]]) / 1000.0 for t in turns]
        remaining = [
            mean([r["estimated_remaining_ttft_ms"] for r in by_turn[t]]) / 1000.0
            for t in turns
        ]
        ymax = max(ymax, max(b + p + r for b, p, r in zip(blocking, prefill, remaining)))
        stacked.append((tenant_count, turns, blocking, prefill, remaining))

    for idx, (ax, (tenant_count, turns, blocking, prefill, remaining)) in enumerate(
        zip(axes, stacked)
    ):
        ax.stackplot(
            turns,
            blocking,
            prefill,
            remaining,
            colors=[BLOCKING_COLOR, PREFILL_COLOR, REMAINING_COLOR],
            alpha=0.92,
        )
        ax.set_xlim(1, 10)
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Turn")
        ax.grid(axis="y", alpha=0.2)
        ax.text(0.03, 0.97, f"tenants={tenant_count}", transform=ax.transAxes, va="top", ha="left")
        ax.text(
            0.5,
            -0.22,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            clip_on=False,
        )
    axes[0].set_ylabel("Mean Time (sec)")
    axes[0].legend(
        ["Blocking Time", f"Estimated Prefill ({prefill_fraction:.0%})", "Remaining TTFT"],
        loc="upper right",
        frameon=True,
        fontsize=12,
    )
    fig.suptitle(
        "Mean TTFT Breakdown By Tenant Count With Estimated Remaining TTFT "
        f"(prefill={prefill_fraction:.0%} of non-blocking TTFT)",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_markdown(out_path):
    lines = [
        "# Remaining TTFT Sensitivity Graphs",
        "",
        "These graphs preserve the current experiment data and add a visible Remaining TTFT component.",
        "",
        "The current request metrics available in this directory are compatibility metrics synthesized from `result_raw.csv`, not the original patched-vLLM metrics. In that synthesized file, `prefill_time_s = TTFT - queued_time_s`, so `Remaining TTFT` becomes zero by construction.",
        "",
        "For this additional view, prefill is estimated as a fixed fraction of non-blocking TTFT:",
        "",
        "`estimated_prefill = (TTFT - blocking) * fraction`",
        "",
        "`estimated_remaining = TTFT - blocking - estimated_prefill`",
        "",
        "Generated fractions: `50%`, `70%`, `90%`.",
        "",
        "Use these as sensitivity graphs, not as real prefill telemetry. For a factual TTFT decomposition, preserve the original patched-vLLM JSONL with real `scheduled_ts`, `first_token_ts`, and `prefill_time_s`.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    all_enriched = []
    for fraction in PREFILL_FRACTIONS:
        enriched = add_estimated_breakdown(rows, fraction)
        all_enriched.extend(enriched)
        slug = int(fraction * 100)
        plot_group_p99_by_turn(
            plt,
            [row for row in enriched if row["tenant_count"] == 32],
            fraction,
            OUT_DIR / f"estimated_remaining_ttft_group_p99_by_turn_32tenants_prefill{slug}.png",
        )
        plot_tenant_sweep_mean_by_turn(
            plt,
            enriched,
            fraction,
            OUT_DIR / f"estimated_remaining_ttft_mean_by_turn_tenant_sweep_prefill{slug}.png",
        )
    write_enriched_csv(all_enriched, OUT_DIR / "estimated_remaining_ttft_sensitivity.csv")
    write_markdown(OUT_DIR / "remaining_ttft_sensitivity.md")


if __name__ == "__main__":
    main()
