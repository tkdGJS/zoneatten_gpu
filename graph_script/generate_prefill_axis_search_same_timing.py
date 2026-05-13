#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "graphs" / Path(__file__).stem
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
TENANT_COUNT = "32"
MAX_NUM_BATCHED_TOKENS = 8192.0
KV_CAPACITY_MIB = 1024.0
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
    return {"kv_bytes_per_token": kv_bytes_per_token}


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
        metrics = load_request_metrics(exp_dir)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success" or row["tenant_count"] != TENANT_COUNT:
                    continue
                metric = metrics.get(row["metrics_request_id"])
                if metric is None:
                    continue
                input_tokens = float(row["input_tokens"] or 0.0)
                prefix_hit_tokens = float(row["prefix_hit_tokens"] or 0.0)
                kv_history_tokens = float(row["kv_history_tokens"] or 0.0)
                prefill_ms = 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0)
                computation_tokens = max(0.0, input_tokens - prefix_hit_tokens)
                rows.append(
                    {
                        "exp_label": exp_label,
                        "turn_index": int(row["turn_index"]),
                        "tenant_id": int(row["tenant_id"]),
                        "group_label": "Group1" if row["history_limit_tokens"] == "8192" else "Group2",
                        "input_tokens": input_tokens,
                        "prefix_hit_tokens": prefix_hit_tokens,
                        "prefix_hit_rate": (prefix_hit_tokens / input_tokens) if input_tokens > 0 else 0.0,
                        "kv_history_tokens": kv_history_tokens,
                        "resident_kv_mib": mib_from_tokens(kv_history_tokens, kv_bytes_per_token),
                        "computation_tokens": computation_tokens,
                        "prefill_ms": prefill_ms,
                    }
                )
    return rows


def enrich_batch_context(rows):
    by_batch = {}
    for row in rows:
        key = (row["exp_label"], row["turn_index"])
        by_batch.setdefault(key, []).append(row)
    for subset in by_batch.values():
        total_compute = sum(r["computation_tokens"] for r in subset)
        total_resident_kv = sum(r["resident_kv_mib"] for r in subset)
        estimated_rounds = total_compute / MAX_NUM_BATCHED_TOKENS
        for row in subset:
            row["batch_total_computation_tokens"] = total_compute
            row["batch_total_resident_kv_mib"] = total_resident_kv
            row["batch_pressure_ratio"] = total_resident_kv / KV_CAPACITY_MIB
            row["estimated_prefill_rounds_ratio"] = estimated_rounds
            row["joint_compute_x_pressure"] = row["computation_tokens"] * row["batch_pressure_ratio"]
            row["joint_compute_x_rounds"] = row["computation_tokens"] * estimated_rounds
            row["compute_over_hit_ratio"] = (
                row["computation_tokens"] / max(1.0, row["prefix_hit_tokens"])
            )
            row["context_plus_compute"] = row["kv_history_tokens"] + row["computation_tokens"]


AXES = [
    ("computation_tokens", "Computation tokens"),
    ("resident_kv_mib", "Resident KV per tenant (MiB)"),
    ("batch_total_computation_tokens", "Batch total computation tokens"),
    ("batch_total_resident_kv_mib", "Batch total resident KV (MiB)"),
    ("batch_pressure_ratio", "Batch resident KV / KV capacity"),
    ("estimated_prefill_rounds_ratio", "Batch total compute / 8192"),
    ("joint_compute_x_pressure", "Computation tokens x batch pressure ratio"),
    ("joint_compute_x_rounds", "Computation tokens x estimated prefill rounds"),
    ("compute_over_hit_ratio", "Computation / prefix-hit ratio"),
    ("context_plus_compute", "KV history tokens + computation tokens"),
]


def write_enriched_csv(rows, out_path):
    fieldnames = [
        "exp_label",
        "turn_index",
        "tenant_id",
        "group_label",
        "input_tokens",
        "prefix_hit_tokens",
        "prefix_hit_rate",
        "kv_history_tokens",
        "resident_kv_mib",
        "computation_tokens",
        "prefill_ms",
        "batch_total_computation_tokens",
        "batch_total_resident_kv_mib",
        "batch_pressure_ratio",
        "estimated_prefill_rounds_ratio",
        "joint_compute_x_pressure",
        "joint_compute_x_rounds",
        "compute_over_hit_ratio",
        "context_plus_compute",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_axis_ranking(rows, out_path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["axis_key", "axis_label", "pearson_prefill", "spearman_prefill"])
        scored = []
        ys = [r["prefill_ms"] for r in rows]
        for key, label in AXES:
            xs = [r[key] for r in rows]
            pear = corr(xs, ys)
            spear = spearman(xs, ys)
            scored.append((abs(spear), key, label, pear, spear))
        for _, key, label, pear, spear in sorted(scored, reverse=True):
            writer.writerow([key, label, f"{pear:.6f}", f"{spear:.6f}"])


def plot_axis_grid(plt, rows, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.98, hspace=0.32, wspace=0.24)
    turn_handles = []
    for ax, (key, label) in zip(axes.flatten(), AXES[:9]):
        for turn in range(1, 11):
            subset = [r for r in rows if r["turn_index"] == turn]
            sc = ax.scatter(
                [r[key] for r in subset],
                [r["prefill_ms"] for r in subset],
                s=10,
                alpha=0.35,
                color=TURN_COLORS[turn],
            )
            if len(turn_handles) < 10:
                turn_handles.append((sc, f"Turn {turn}"))
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Prefill time (ms)")
        ax.grid(alpha=0.2)
        pear = corr([r[key] for r in rows], [r["prefill_ms"] for r in rows])
        spear = spearman([r[key] for r in rows], [r["prefill_ms"] for r in rows])
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


def plot_top_axes(plt, rows, out_path):
    ys = [r["prefill_ms"] for r in rows]
    scored = []
    for key, label in AXES:
        spear = spearman([r[key] for r in rows], ys)
        scored.append((abs(spear), key, label, spear))
    top = sorted(scored, reverse=True)[:4]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.08, right=0.98, hspace=0.28, wspace=0.22)
    turn_handles = []
    for ax, (_, key, label, _) in zip(axes.flatten(), top):
        for turn in range(1, 11):
            subset = [r for r in rows if r["turn_index"] == turn]
            sc = ax.scatter(
                [r[key] for r in subset],
                [r["prefill_ms"] for r in subset],
                s=14,
                alpha=0.38,
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


def plot_joint_axis_focus(plt, rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.98, wspace=0.22)
    focus_specs = [
        ("joint_compute_x_pressure", "Computation tokens x batch pressure ratio"),
        ("joint_compute_x_rounds", "Computation tokens x estimated prefill rounds"),
    ]
    turn_handles = []
    for ax, (key, label) in zip(axes, focus_specs):
        for turn in range(1, 11):
            subset = [r for r in rows if r["turn_index"] == turn]
            sc = ax.scatter(
                [r[key] for r in subset],
                [r["prefill_ms"] for r in subset],
                s=16,
                alpha=0.4,
                color=TURN_COLORS[turn],
            )
            if len(turn_handles) < 10:
                turn_handles.append((sc, f"Turn {turn}"))
        pear = corr([r[key] for r in rows], [r["prefill_ms"] for r in rows])
        spear = spearman([r[key] for r in rows], [r["prefill_ms"] for r in rows])
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


def plot_top_axis_by_group(plt, rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.07, right=0.98, wspace=0.22)
    focus_specs = [
        ("batch_total_computation_tokens", "Batch total computation tokens"),
        ("batch_total_resident_kv_mib", "Batch total resident KV (MiB)"),
    ]
    for ax, (key, label) in zip(axes, focus_specs):
        for group_label in ["Group1", "Group2"]:
            subset = [r for r in rows if r["group_label"] == group_label]
            ax.scatter(
                [r[key] for r in subset],
                [r["prefill_ms"] for r in subset],
                s=16,
                alpha=0.4,
                color=GROUP_COLORS[group_label],
                label=f"{group_label} (n={len(subset)})",
            )
        pear = corr([r[key] for r in rows], [r["prefill_ms"] for r in rows])
        spear = spearman([r[key] for r in rows], [r["prefill_ms"] for r in rows])
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Prefill time (ms)")
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.97,
            f"all r={pear:.2f}\nall ρ={spear:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        ax.legend(loc="upper left", title="Color = group", frameon=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_top_axis_group_panels(plt, rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.08, right=0.98, hspace=0.28, wspace=0.22)
    specs = [
        ("Group1", "batch_total_computation_tokens", "Group1: batch total computation tokens"),
        ("Group2", "batch_total_computation_tokens", "Group2: batch total computation tokens"),
        ("Group1", "batch_total_resident_kv_mib", "Group1: batch total resident KV"),
        ("Group2", "batch_total_resident_kv_mib", "Group2: batch total resident KV"),
    ]
    turn_handles = []
    for ax, (group_label, key, title) in zip(axes.flatten(), specs):
        subset = [r for r in rows if r["group_label"] == group_label]
        for turn in range(1, 11):
            turn_subset = [r for r in subset if r["turn_index"] == turn]
            sc = ax.scatter(
                [r[key] for r in turn_subset],
                [r["prefill_ms"] for r in turn_subset],
                s=14,
                alpha=0.4,
                color=TURN_COLORS[turn],
            )
            if len(turn_handles) < 10:
                turn_handles.append((sc, f"Turn {turn}"))
        pear = corr([r[key] for r in subset], [r["prefill_ms"] for r in subset])
        spear = spearman([r[key] for r in subset], [r["prefill_ms"] for r in subset])
        ax.set_title(title)
        ax.set_xlabel(key.replace("_", " "))
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


def write_markdown(rows, out_path):
    ys = [r["prefill_ms"] for r in rows]
    scored = []
    for key, label in AXES:
        xs = [r[key] for r in rows]
        scored.append(
            {
                "key": key,
                "label": label,
                "pearson": corr(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )
    scored.sort(key=lambda item: abs(item["spearman"]), reverse=True)

    lines = []
    lines.append("# Prefill Axis Search")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Data: `block_2048_limit_same_timing1`, `block_2048_limit_same_timing2`, `block_2048_limit_same_timing3`")
    lines.append("- Slice: `tenant_count=32` only")
    lines.append("- Samples: 960 successful requests")
    lines.append("- Goal: find an x-axis that organizes `prefill time` more consistently than plain computation tokens")
    lines.append("")
    lines.append("## Ranking By Spearman Correlation")
    lines.append("")
    for item in scored[:6]:
        lines.append(
            f"- `{item['label']}`: Pearson `{item['pearson']:.3f}`, Spearman `{item['spearman']:.3f}`"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If a better axis exists, it should produce both stronger rank correlation and visibly tighter alignment in scatter form."
    )
    lines.append(
        "- Plain `computation tokens` is included as a baseline so it can be compared directly against batch-level and joint axes."
    )
    lines.append(
        "- The most promising candidates are the ones that combine `my compute load` with `global batch pressure`, because prefill is executed inside that shared context."
    )
    lines.append("")
    lines.append("## Group Split")
    lines.append("")
    for group_label in ["Group1", "Group2"]:
        subset = [r for r in rows if r["group_label"] == group_label]
        lines.append(
            f"- {group_label}, `Batch total computation tokens`: Pearson `{corr([r['batch_total_computation_tokens'] for r in subset], [r['prefill_ms'] for r in subset]):.3f}`, "
            f"Spearman `{spearman([r['batch_total_computation_tokens'] for r in subset], [r['prefill_ms'] for r in subset]):.3f}`"
        )
        lines.append(
            f"- {group_label}, `Batch total resident KV (MiB)`: Pearson `{corr([r['batch_total_resident_kv_mib'] for r in subset], [r['prefill_ms'] for r in subset]):.3f}`, "
            f"Spearman `{spearman([r['batch_total_resident_kv_mib'] for r in subset], [r['prefill_ms'] for r in subset]):.3f}`"
        )
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
    enrich_batch_context(rows)
    write_enriched_csv(rows, OUT_DIR / "prefill_axis_search_enriched_32tenants.csv")
    write_axis_ranking(rows, OUT_DIR / "prefill_axis_ranking_32tenants.csv")
    plot_axis_grid(plt, rows, OUT_DIR / "prefill_axis_search_grid_32tenants.png")
    plot_top_axes(plt, rows, OUT_DIR / "prefill_axis_search_top_axes_32tenants.png")
    plot_joint_axis_focus(plt, rows, OUT_DIR / "prefill_axis_search_joint_axes_32tenants.png")
    plot_top_axis_by_group(plt, rows, OUT_DIR / "prefill_axis_search_top_axes_by_group_32tenants.png")
    plot_top_axis_group_panels(plt, rows, OUT_DIR / "prefill_axis_search_group_panels_32tenants.png")
    write_markdown(rows, OUT_DIR / "prefill_axis_search.md")


if __name__ == "__main__":
    main()
