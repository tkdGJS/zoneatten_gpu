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
GROUP_COLORS = {
    "8192": "#2563eb",
    "2048": "#f97316",
}
CAPACITY_MIB = 1024.0


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


def load_experiment_turn_group_usage(model_info):
    kv_bytes_per_token = model_info["kv_bytes_per_token"]
    usage = {}
    for exp_dir, exp_label in EXPERIMENTS:
        turn_group_sum = defaultdict(float)
        raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "success":
                    continue
                if row["tenant_count"] != TENANT_COUNT:
                    continue
                turn_index = int(row["turn_index"])
                group = row["history_limit_tokens"]
                kv_history_tokens = float(row["kv_history_tokens"] or 0.0)
                turn_group_sum[(turn_index, group)] += mib_from_tokens(kv_history_tokens, kv_bytes_per_token)
        usage[exp_label] = turn_group_sum
    return usage


def build_mean_usage(exp_usage):
    mean_usage = defaultdict(float)
    for turn in range(1, 11):
        for group in GROUP_LABELS:
            values = [exp_usage[exp_label][(turn, group)] for _, exp_label in EXPERIMENTS]
            mean_usage[(turn, group)] = sum(values) / len(values)
    return mean_usage


def write_usage_csv(rows, out_path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "turn_index", "group_label", "kv_usage_mib"])
        for row in rows:
            writer.writerow(row)


def plot_scope(plt, scope_name, turn_group_usage, out_path):
    turns = list(range(1, 11))
    group1 = [turn_group_usage[(turn, "8192")] for turn in turns]
    group2 = [turn_group_usage[(turn, "2048")] for turn in turns]
    totals = [a + b for a, b in zip(group1, group2)]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    width = 0.72
    bars1 = ax.bar(
        turns,
        group1,
        width=width,
        color=GROUP_COLORS["8192"],
        label="Group1",
    )
    bars2 = ax.bar(
        turns,
        group2,
        width=width,
        bottom=group1,
        color=GROUP_COLORS["2048"],
        label="Group2",
    )

    ax.axhline(CAPACITY_MIB, color="#111827", linestyle="--", linewidth=1.6, label="KV capacity (1024 MiB)")
    ax.text(
        10.45,
        CAPACITY_MIB,
        "KV capacity",
        color="#111827",
        va="bottom",
        ha="left",
        fontsize=10,
    )

    for turn, total in zip(turns, totals):
        ax.text(turn, total + 18.0, f"{int(round(total))}", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"{scope_name}: resident KV usage by turn (32 tenants)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("KV usage (MiB)")
    ax.set_xticks(turns)
    ymax = max(max(totals) * 1.12, CAPACITY_MIB * 1.08)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_info = load_model_info()
    exp_usage = load_experiment_turn_group_usage(model_info)
    mean_usage = build_mean_usage(exp_usage)

    csv_rows = []
    for _, exp_label in EXPERIMENTS:
        for turn in range(1, 11):
            for group in ["8192", "2048"]:
                csv_rows.append((exp_label, turn, GROUP_LABELS[group], f"{exp_usage[exp_label][(turn, group)]:.4f}"))
    for turn in range(1, 11):
        for group in ["8192", "2048"]:
            csv_rows.append(("mean", turn, GROUP_LABELS[group], f"{mean_usage[(turn, group)]:.4f}"))
    write_usage_csv(csv_rows, OUT_DIR / "kv_usage_by_turn_32tenants.csv")

    for _, exp_label in EXPERIMENTS:
        plot_scope(
            plt,
            exp_label,
            exp_usage[exp_label],
            OUT_DIR / f"{exp_label}_kv_usage_by_turn_32tenants.png",
        )
    plot_scope(
        plt,
        "exp1+exp2+exp3 mean",
        mean_usage,
        OUT_DIR / "exp123_mean_kv_usage_by_turn_32tenants.png",
    )


if __name__ == "__main__":
    main()
