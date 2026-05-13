#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "graphs" / Path(__file__).stem
CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "vram_only_results_smallctx_mixed_limits_8GB_ori"
    / "graphs"
    / "generate_input_kv_usage_by_turn_same_timing"
    / "input_kv_usage_by_turn_32tenants.csv"
)


def load_rows():
    rows = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "scope": row["scope"],
                    "turn_index": int(row["turn_index"]),
                    "group_label": row["group_label"],
                    "input_kv_usage_mib": float(row["input_kv_usage_mib"]),
                }
            )
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def total_for_scope(rows, scope, turn):
    subset = [r for r in rows if r["scope"] == scope and r["turn_index"] == turn]
    return sum(r["input_kv_usage_mib"] for r in subset)


def group_for_scope(rows, scope, turn, group_label):
    subset = [
        r["input_kv_usage_mib"]
        for r in rows
        if r["scope"] == scope and r["turn_index"] == turn and r["group_label"] == group_label
    ]
    return subset[0] if subset else 0.0


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing CSV: {CSV_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    scopes = ["exp1", "exp2", "exp3", "mean"]

    lines = []
    lines.append("# Input KV Usage By Turn Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append("- Slice: `tenant_count=32` only")
    lines.append("- Outputs:")
    lines.append("  - `exp1/exp2/exp3_input_kv_usage_by_turn_32tenants.png`")
    lines.append("  - `exp123_mean_input_kv_usage_by_turn_32tenants.png`")
    lines.append("  - `exp1/exp2/exp3_total_input_kv_usage_by_turn_32tenants.png`")
    lines.append("  - `exp123_mean_total_input_kv_usage_by_turn_32tenants.png`")
    lines.append("")
    lines.append("## What These Graphs Measure")
    lines.append("")
    lines.append("- The graphs convert `input_tokens` into a KV-style footprint in MiB using the same per-token KV size used elsewhere in the analysis.")
    lines.append("- This is not `resident KV` already stored from history. It is the footprint of the prompt tokens presented in that turn.")
    lines.append("- Group1 corresponds to `history_limit_tokens=8192` and Group2 corresponds to `history_limit_tokens=2048`.")
    lines.append("- The dashed `1024 MiB` line is the reference KV capacity used in the rest of the study.")
    lines.append("")
    lines.append("## How To Read The Graphs")
    lines.append("")
    lines.append("- The stacked graphs show how much of the turn-level input footprint comes from Group1 versus Group2.")
    lines.append("- The total-only graphs collapse Group1+Group2 into a single bar so the turn-level aggregate is easier to compare against the `1024 MiB` reference line.")
    lines.append("- Because this is based on `input_tokens`, it is a prompt-size view, not an active-cache view.")
    lines.append("")
    lines.append("## Mean Pattern Across exp1+exp2+exp3")
    lines.append("")
    for turn in [1, 5, 6, 7, 8, 9, 10]:
        total = total_for_scope(rows, "mean", turn)
        group1 = group_for_scope(rows, "mean", turn, "Group1")
        group2 = group_for_scope(rows, "mean", turn, "Group2")
        lines.append(
            f"- Turn {turn}: total `{total:.1f} MiB`, Group1 `{group1:.1f} MiB`, Group2 `{group2:.1f} MiB`"
        )
    lines.append("")
    lines.append("## Main Observations")
    lines.append("")
    mean_turn5 = total_for_scope(rows, "mean", 5)
    mean_turn6 = total_for_scope(rows, "mean", 6)
    mean_turn7 = total_for_scope(rows, "mean", 7)
    mean_turn10 = total_for_scope(rows, "mean", 10)
    lines.append(
        f"- Mean total input-KV footprint is already high by turn 5 (`{mean_turn5:.1f} MiB`) and exceeds the `1024 MiB` reference by turn 6 (`{mean_turn6:.1f} MiB`)."
    )
    lines.append(
        f"- By turn 7 the mean total input-KV footprint grows further to `{mean_turn7:.1f} MiB`, and by turn 10 it reaches `{mean_turn10:.1f} MiB`."
    )
    lines.append(
        "- Group1 contributes the larger share of input-KV footprint in every later turn, which is expected because the longer-history group builds larger prompts."
    )
    lines.append(
        "- Group2 still contributes a meaningful fraction, so later-turn batch pressure is not only a Group1 phenomenon; it is a mixed-batch load problem."
    )
    lines.append("")
    lines.append("## Comparison To Resident KV Graphs")
    lines.append("")
    lines.append(
        "- `Resident KV` graphs answer: how much history is already active in cache before the new turn starts."
    )
    lines.append(
        "- `Input KV` graphs answer: how large the current prompt payload is when translated into KV-sized token footprint."
    )
    lines.append(
        "- In practice, `resident KV` is more relevant to admission/blocking pressure, while `input KV` is more relevant to prefill execution load."
    )
    lines.append(
        "- This is why a turn can show strong prefill growth even before the resident-KV sum alone obviously exceeds the capacity line."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- These graphs support the interpretation that later turns increase prefill pressure mainly because prompt payload grows substantially across tenants."
    )
    lines.append(
        "- The input-KV view therefore complements the resident-KV view: one captures prompt compute load, the other captures active-cache pressure."
    )
    lines.append(
        "- Together they explain why prefill can begin degrading even before blocking becomes dominant."
    )
    lines.append("")

    out_path = OUT_DIR / "input_kv_usage_analysis.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
