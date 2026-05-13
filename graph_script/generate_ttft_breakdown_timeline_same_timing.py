#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    ("block_2048_limit_same_timing1", "exp1"),
    ("block_2048_limit_same_timing2", "exp2"),
    ("block_2048_limit_same_timing3", "exp3"),
]
OUT_DIR = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB" / "graphs" / Path(__file__).stem
TENANT_COUNT_FILTER = "32"
TENANT_COUNTS = ["8", "16", "32"]

BLOCKING_COLOR = "#2563eb"
PREFILL_COLOR = "#84cc16"
REMAINING_COLOR = "#dc2626"


def load_metric_map(exp_dir: str):
    metric_map = {}
    metrics_dir = ROOT / exp_dir / "vram_only_artifacts_smallctx_mixed_limits" / "request_metrics"
    for path in metrics_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                metric_map[payload["request_id"]] = payload
    return metric_map


def load_points(exp_dir: str, tenant_count_filter=None):
    metric_map = load_metric_map(exp_dir)
    raw_csv = ROOT / exp_dir / "vram_only_results_smallctx_mixed_limits" / "result_raw.csv"
    points = []
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "success":
                continue
            if tenant_count_filter is not None and row["tenant_count"] != tenant_count_filter:
                continue
            metric = metric_map.get(row["metrics_request_id"])
            if metric is None:
                continue
            ttft_ms = float(row["ttft_ms"] or 0.0)
            blocking_ms = 1000.0 * float(metric.get("queued_time_s", 0.0) or 0.0)
            prefill_ms = 1000.0 * float(metric.get("prefill_time_s", 0.0) or 0.0)
            remaining_ms = max(0.0, ttft_ms - blocking_ms - prefill_ms)
            points.append(
                {
                    "arrival_time": float(metric["arrival_time"]),
                    "blocking_ms": blocking_ms,
                    "prefill_ms": prefill_ms,
                    "remaining_ms": remaining_ms,
                    "ttft_ms": ttft_ms,
                    "tenant_count": row["tenant_count"],
                    "tenant_id": row["tenant_id"],
                    "history_limit_tokens": row["history_limit_tokens"],
                    "turn_index": row["turn_index"],
                }
            )
    points.sort(key=lambda item: item["arrival_time"])
    return points


def build_elapsed_ms(points):
    if not points:
        return []
    first = points[0]["arrival_time"]
    return [(point["arrival_time"] - first) * 1000.0 for point in points]


def rounded_sum(values):
    return int(round(sum(values)))


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


def write_points_csv(points, compressed_xs_ms, out_path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "arrival_time_epoch_s",
                "compressed_elapsed_ms",
                "tenant_count",
                "tenant_id",
                "turn_index",
                "blocking_ms",
                "prefill_ms",
                "remaining_ttft_ms",
                "ttft_ms",
            ]
        )
        for point, compressed_ms in zip(points, compressed_xs_ms):
            writer.writerow(
                [
                    f"{point['arrival_time']:.6f}",
                    f"{compressed_ms:.3f}",
                    point["tenant_count"],
                    point["tenant_id"],
                    point["turn_index"],
                    f"{point['blocking_ms']:.4f}",
                    f"{point['prefill_ms']:.4f}",
                    f"{point['remaining_ms']:.4f}",
                    f"{point['ttft_ms']:.4f}",
                ]
            )


def render_turn_breakdown_by_tenant_continuous(plt, points, title, out_path, y_limits=None):
    if not points:
        return

    tenant_ids = sorted({int(p["tenant_id"]) for p in points})
    ncols = 4
    nrows = (len(tenant_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, max(3 * nrows, 8)), constrained_layout=True)
    axes = axes.flatten()

    for ax, tenant_id in zip(axes, tenant_ids):
        tenant_points = [p for p in points if int(p["tenant_id"]) == tenant_id]
        tenant_points.sort(key=lambda p: int(p["turn_index"]))
        turns = [int(p["turn_index"]) for p in tenant_points]
        blocking = [p["blocking_ms"] for p in tenant_points]
        prefill = [p["prefill_ms"] for p in tenant_points]
        remaining = [p["remaining_ms"] for p in tenant_points]
        ax.stackplot(
            turns,
            blocking,
            prefill,
            remaining,
            colors=[BLOCKING_COLOR, PREFILL_COLOR, REMAINING_COLOR],
            alpha=0.9,
        )
        ax.set_title(f"tenant {tenant_id}")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Time (ms)")
        ax.set_xticks(range(1, 11))
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(tenant_ids):]:
        ax.axis("off")

    total_blocking = rounded_sum([p["blocking_ms"] for p in points])
    total_prefill = rounded_sum([p["prefill_ms"] for p in points])
    total_remaining = rounded_sum([p["remaining_ms"] for p in points])
    handles = [
        plt.Line2D([0], [0], color=BLOCKING_COLOR, lw=6),
        plt.Line2D([0], [0], color=PREFILL_COLOR, lw=6),
        plt.Line2D([0], [0], color=REMAINING_COLOR, lw=6),
    ]
    labels = [
        f"Blocking Time (sum={total_blocking})",
        f"Prefill Time (sum={total_prefill})",
        f"Remaining TTFT (sum={total_remaining})",
    ]
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title, fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_turn_breakdown_mean_continuous(plt, points, title, out_path, y_limits=None):
    if not points:
        return
    grouped = {turn: [] for turn in range(1, 11)}
    for point in points:
        grouped[int(point["turn_index"])].append(point)

    turns = []
    blocking = []
    prefill = []
    remaining = []
    for turn in range(1, 11):
        subset = grouped[turn]
        if not subset:
            continue
        turns.append(turn)
        blocking.append(sum(p["blocking_ms"] for p in subset) / len(subset))
        prefill.append(sum(p["prefill_ms"] for p in subset) / len(subset))
        remaining.append(sum(p["remaining_ms"] for p in subset) / len(subset))

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    total_blocking = rounded_sum(blocking)
    total_prefill = rounded_sum(prefill)
    total_remaining = rounded_sum(remaining)
    ax.stackplot(
        turns,
        blocking,
        prefill,
        remaining,
        colors=[BLOCKING_COLOR, PREFILL_COLOR, REMAINING_COLOR],
        labels=[
            f"Blocking Time (sum={total_blocking})",
            f"Prefill Time (sum={total_prefill})",
            f"Remaining TTFT (sum={total_remaining})",
        ],
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean Time (ms)")
    ax.set_xticks(range(1, 11))
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_turn_breakdown_group_means(plt, points, title, out_path, y_limits=None):
    if not points:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True, sharey=True)
    groups = [
        ("Group1", [p for p in points if p["history_limit_tokens"] == "8192"]),
        ("Group2", [p for p in points if p["history_limit_tokens"] == "2048"]),
    ]

    for ax, (group_label, group_points) in zip(axes, groups):
        grouped = {turn: [] for turn in range(1, 11)}
        for point in group_points:
            grouped[int(point["turn_index"])].append(point)

        turns = []
        blocking = []
        prefill = []
        remaining = []
        for turn in range(1, 11):
            subset = grouped[turn]
            if not subset:
                continue
            turns.append(turn)
            blocking.append(sum(p["blocking_ms"] for p in subset) / len(subset))
            prefill.append(sum(p["prefill_ms"] for p in subset) / len(subset))
            remaining.append(sum(p["remaining_ms"] for p in subset) / len(subset))

        blocking_sum = rounded_sum([p["blocking_ms"] for p in group_points])
        prefill_sum = rounded_sum([p["prefill_ms"] for p in group_points])
        remaining_sum = rounded_sum([p["remaining_ms"] for p in group_points])
        ax.stackplot(
            turns,
            blocking,
            prefill,
            remaining,
            colors=[BLOCKING_COLOR, PREFILL_COLOR, REMAINING_COLOR],
            labels=[
                f"Blocking Time ({blocking_sum})",
                f"Prefill Time ({prefill_sum})",
                f"Remaining TTFT ({remaining_sum})",
            ],
            alpha=0.9,
        )
        ax.set_title(group_label)
        ax.set_xlabel("Turn")
        ax.set_xticks(range(1, 11))
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("Mean Time (ms)")
    fig.suptitle(title, fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_turn_breakdown_group_percentiles(
    plt,
    points,
    title,
    out_path,
    quantile,
    y_limits=None,
):
    if not points:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True, sharey=True)
    groups = [
        ("Group1", [p for p in points if p["history_limit_tokens"] == "8192"]),
        ("Group2", [p for p in points if p["history_limit_tokens"] == "2048"]),
    ]

    for ax, (group_label, group_points) in zip(axes, groups):
        grouped = {turn: [] for turn in range(1, 11)}
        for point in group_points:
            grouped[int(point["turn_index"])].append(point)

        turns = []
        blocking = []
        prefill = []
        remaining = []
        for turn in range(1, 11):
            subset = grouped[turn]
            if not subset:
                continue
            turns.append(turn)
            blocking.append(percentile([p["blocking_ms"] for p in subset], quantile))
            prefill.append(percentile([p["prefill_ms"] for p in subset], quantile))
            remaining.append(percentile([p["remaining_ms"] for p in subset], quantile))

        blocking_sum = rounded_sum([p["blocking_ms"] for p in group_points])
        prefill_sum = rounded_sum([p["prefill_ms"] for p in group_points])
        remaining_sum = rounded_sum([p["remaining_ms"] for p in group_points])
        ax.stackplot(
            turns,
            blocking,
            prefill,
            remaining,
            colors=[BLOCKING_COLOR, PREFILL_COLOR, REMAINING_COLOR],
            labels=[
                f"Blocking Time ({blocking_sum})",
                f"Prefill Time ({prefill_sum})",
                f"Remaining TTFT ({remaining_sum})",
            ],
            alpha=0.9,
        )
        ax.set_title(group_label)
        ax.set_xlabel("Turn")
        ax.set_xticks(range(1, 11))
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("Time (ms)")
    fig.suptitle(title, fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_group_metric_means_bar(plt, points, title, out_path, y_limits=None):
    if not points:
        return

    categories = ["Blocking Time", "Prefill Time", "Remaining TTFT"]
    group1 = [p for p in points if p["history_limit_tokens"] == "8192"]
    group2 = [p for p in points if p["history_limit_tokens"] == "2048"]

    group1_means = [
        sum(p["blocking_ms"] for p in group1) / len(group1),
        sum(p["prefill_ms"] for p in group1) / len(group1),
        sum(p["remaining_ms"] for p in group1) / len(group1),
    ]
    group2_means = [
        sum(p["blocking_ms"] for p in group2) / len(group2),
        sum(p["prefill_ms"] for p in group2) / len(group2),
        sum(p["remaining_ms"] for p in group2) / len(group2),
    ]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    xs = list(range(len(categories)))
    width = 0.35
    bars1 = ax.bar(
        [x - width / 2 for x in xs],
        group1_means,
        width=width,
        color="#2563eb",
        label="Group1",
    )
    bars2 = ax.bar(
        [x + width / 2 for x in xs],
        group2_means,
        width=width,
        color="#f97316",
        label="Group2",
    )
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{round(height):.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.set_title(title)
    ax.set_xlabel("Breakdown Component")
    ax.set_ylabel("Mean Time (ms)")
    ax.set_xticks(xs, categories)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_group_metric_percentile_bar(plt, points, title, out_path, quantile, y_limits=None):
    if not points:
        return

    categories = ["Blocking Time", "Prefill Time", "Remaining TTFT"]
    group1 = [p for p in points if p["history_limit_tokens"] == "8192"]
    group2 = [p for p in points if p["history_limit_tokens"] == "2048"]

    group1_values = [
        percentile([p["blocking_ms"] for p in group1], quantile),
        percentile([p["prefill_ms"] for p in group1], quantile),
        percentile([p["remaining_ms"] for p in group1], quantile),
    ]
    group2_values = [
        percentile([p["blocking_ms"] for p in group2], quantile),
        percentile([p["prefill_ms"] for p in group2], quantile),
        percentile([p["remaining_ms"] for p in group2], quantile),
    ]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    xs = list(range(len(categories)))
    width = 0.35
    bars1 = ax.bar(
        [x - width / 2 for x in xs],
        group1_values,
        width=width,
        color="#2563eb",
        label="Group1",
    )
    bars2 = ax.bar(
        [x + width / 2 for x in xs],
        group2_values,
        width=width,
        color="#f97316",
        label="Group2",
    )
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{round(height):.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.set_title(title)
    ax.set_xlabel("Breakdown Component")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(xs, categories)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_group_means_by_turn_csv(points_by_tenant, out_path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tenant_count",
                "group",
                "turn",
                "mean_blocking_ms",
                "mean_prefill_ms",
                "mean_remaining_ttft_ms",
                "mean_total_ttft_ms",
                "sample_count",
            ]
        )
        for tenant_count in TENANT_COUNTS:
            combined_points = []
            for points in points_by_tenant[tenant_count].values():
                combined_points.extend(points)
            for history_limit_tokens, group_label in [("8192", "Group1"), ("2048", "Group2")]:
                group_points = [
                    p for p in combined_points if p["history_limit_tokens"] == history_limit_tokens
                ]
                for turn in range(1, 11):
                    turn_points = [p for p in group_points if int(p["turn_index"]) == turn]
                    if not turn_points:
                        continue
                    mean_blocking = mean([p["blocking_ms"] for p in turn_points])
                    mean_prefill = mean([p["prefill_ms"] for p in turn_points])
                    mean_remaining = mean([p["remaining_ms"] for p in turn_points])
                    writer.writerow(
                        [
                            tenant_count,
                            group_label,
                            turn,
                            f"{mean_blocking:.4f}",
                            f"{mean_prefill:.4f}",
                            f"{mean_remaining:.4f}",
                            f"{(mean_blocking + mean_prefill + mean_remaining):.4f}",
                            len(turn_points),
                        ]
                    )


def write_group_mean_breakdown_bar_csv(points_by_tenant, out_path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tenant_count",
                "group",
                "mean_blocking_ms",
                "mean_prefill_ms",
                "mean_remaining_ttft_ms",
                "mean_total_ttft_ms",
                "sample_count",
            ]
        )
        for tenant_count in TENANT_COUNTS:
            combined_points = []
            for points in points_by_tenant[tenant_count].values():
                combined_points.extend(points)
            for history_limit_tokens, group_label in [("8192", "Group1"), ("2048", "Group2")]:
                group_points = [
                    p for p in combined_points if p["history_limit_tokens"] == history_limit_tokens
                ]
                mean_blocking = mean([p["blocking_ms"] for p in group_points])
                mean_prefill = mean([p["prefill_ms"] for p in group_points])
                mean_remaining = mean([p["remaining_ms"] for p in group_points])
                writer.writerow(
                    [
                        tenant_count,
                        group_label,
                        f"{mean_blocking:.4f}",
                        f"{mean_prefill:.4f}",
                        f"{mean_remaining:.4f}",
                        f"{(mean_blocking + mean_prefill + mean_remaining):.4f}",
                        len(group_points),
                    ]
                )


def write_group_means_markdown(points_by_tenant, out_path):
    lines = [
        "# TTFT Breakdown Analysis",
        "",
        "exp1, exp2, exp3를 합친 뒤 tenant 8/16/32 조건에서 Group1과 Group2의 mean breakdown을 비교한 결과입니다.",
        "",
        "## Group Definition",
        "",
        "- Group1: `history_limit_tokens = 8192`",
        "- Group2: `history_limit_tokens = 2048`",
        "- 각 요청의 TTFT를 `blocking time + prefill time + remaining TTFT`로 분해해 비교합니다.",
        "",
        "## Experiment Goal",
        "",
        "- tenant 수가 `8`, `16`, `32`로 증가할 때 TTFT breakdown이 어떻게 변하는지 확인합니다.",
        "- 같은 tenant 조건에서 Group1과 Group2의 차이가 주로 `blocking`, `prefill`, `remaining TTFT` 중 어디에서 발생하는지 봅니다.",
        "- `exp1`, `exp2`, `exp3`를 합쳐 개별 실험 편차보다 전체 경향을 우선 확인합니다.",
        "",
        "## Method",
        "",
        "- 입력 데이터는 `exp1`, `exp2`, `exp3`의 성공한 요청만 사용했습니다.",
        "- tenant_count를 `8`, `16`, `32`로 나누고, 각 조건에서 Group1과 Group2를 분리했습니다.",
        "- Breakdown 항목은 다음처럼 계산했습니다.",
        "  - Blocking Time: request metrics의 `queued_time_s * 1000`",
        "  - Prefill Time: request metrics의 `prefill_time_s * 1000`",
        "  - Remaining TTFT: `ttft_ms - blocking_ms - prefill_ms`",
        "- `ttft_breakdown_group_means_by_turn` 결과는 같은 turn에 속한 요청들의 평균입니다.",
        "- `mean_breakdown_bar` 결과는 turn 구분 없이 전체 요청 샘플 평균입니다.",
        "",
        "## KV History Control",
        "",
        "- 각 tenant는 고정된 ShareGPT 세션 하나를 배정받고, 같은 세션의 user turn을 순서대로 사용합니다.",
        "- prompt는 이전에 완료된 `(user, assistant)` 쌍을 누적해 구성합니다.",
        "- 누적 history가 `history_limit_tokens`를 초과하면, 토큰 기준으로 뒤쪽 최근 내용만 남기도록 잘라냅니다.",
        "- 따라서 KV history는 `Group1=8192`, `Group2=2048` 한도 안에서 최근 대화 이력만 유지되도록 통제됩니다.",
        "- 데이터셋 생성 단계에서도 각 세션이 해당 history limit 안에 들어오도록 사전 필터링합니다.",
        "",
        "## Request Schedule",
        "",
        "- 각 tenant는 `10 turns` 동안 요청을 보냅니다.",
        "- 첫 동기화 배치 전에는 `30초` 대기합니다 (`pre_request_sleep_sec=30`).",
        "- 이후 모든 tenant가 같은 turn에서 동시에 요청을 보내도록 barrier로 동기화합니다.",
        "- turn 간 주기는 `30초`입니다 (`inter_turn_sleep_sec=30`).",
        "- 즉 한 turn의 동시 요청을 보낸 뒤, 다음 동시 요청 배치는 `30초` 뒤에 시작되도록 스케줄됩니다.",
        "- 각 tenant 수 실험은 `1 run`씩 수행했습니다.",
        "",
        "## vLLM Configuration",
        "",
        "- Model: `meta-llama/Llama-3.2-1B-Instruct`",
        "- dtype: `half`",
        "- kv-cache-dtype: `auto`",
        "- block-size: `16`",
        "- max-model-len: `12288`",
        "- max-num-seqs: `32`",
        "- max-num-batched-tokens: `8192`",
        "- num-gpu-blocks-override: `2048`",
        "- cpu-offload-gb: `0`",
        "- scheduling-policy: `fcfs`",
        "- chunked prefill: enabled",
        "- prefix caching: enabled",
        "- max output tokens per request: `128`",
        "",
        "## Environment",
        "",
        "- Source dataset: `ShareGPT_V3_unfiltered_cleaned_split.json`",
        "- tenant별 dataset은 실험 시작 전에 생성된 mixed-history-limit dataset을 사용했습니다.",
        "- GPU 스펙은 현재 실행 환경에서 `nvidia-smi`가 NVIDIA driver와 통신하지 못해 자동 수집하지 못했습니다.",
        "",
    ]
    for tenant_count in TENANT_COUNTS:
        combined_points = []
        for points in points_by_tenant[tenant_count].values():
            combined_points.extend(points)
        g1 = [p for p in combined_points if p["history_limit_tokens"] == "8192"]
        g2 = [p for p in combined_points if p["history_limit_tokens"] == "2048"]
        g1_block = mean([p["blocking_ms"] for p in g1])
        g1_prefill = mean([p["prefill_ms"] for p in g1])
        g1_rem = mean([p["remaining_ms"] for p in g1])
        g2_block = mean([p["blocking_ms"] for p in g2])
        g2_prefill = mean([p["prefill_ms"] for p in g2])
        g2_rem = mean([p["remaining_ms"] for p in g2])
        lines.extend(
            [
                f"## Tenant {tenant_count}",
                "",
                f"- Group1 mean total TTFT: `{g1_block + g1_prefill + g1_rem:.2f} ms`",
                f"- Group2 mean total TTFT: `{g2_block + g2_prefill + g2_rem:.2f} ms`",
                f"- Group1 breakdown: blocking `{g1_block:.2f} ms`, prefill `{g1_prefill:.2f} ms`, remaining `{g1_rem:.2f} ms`",
                f"- Group2 breakdown: blocking `{g2_block:.2f} ms`, prefill `{g2_prefill:.2f} ms`, remaining `{g2_rem:.2f} ms`",
            ]
        )
        dominant_group = "Group1" if g1_prefill > g2_prefill else "Group2"
        dominant_gap = abs(g1_prefill - g2_prefill)
        lines.append(
            f"- Prefill 차이가 가장 큰 구간은 `{dominant_group}` 쪽이며, 두 그룹 간 prefill 평균 차이는 `{dominant_gap:.2f} ms`입니다."
        )
        if (g1_block + g1_prefill + g1_rem) > (g2_block + g2_prefill + g2_rem):
            higher_group = "Group1"
            total_gap = (g1_block + g1_prefill + g1_rem) - (g2_block + g2_prefill + g2_rem)
        else:
            higher_group = "Group2"
            total_gap = (g2_block + g2_prefill + g2_rem) - (g1_block + g1_prefill + g1_rem)
        lines.append(
            f"- 평균 total TTFT는 `{higher_group}`가 더 크고, 차이는 `{total_gap:.2f} ms`입니다."
        )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is not installed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points_by_tenant = {tenant_count: {} for tenant_count in TENANT_COUNTS}
    for exp_dir, exp_label in EXPERIMENTS:
        for tenant_count in TENANT_COUNTS:
            points = load_points(exp_dir, tenant_count_filter=tenant_count)
            points_by_tenant[tenant_count][exp_label] = points
            if tenant_count == TENANT_COUNT_FILTER:
                elapsed_xs_ms = build_elapsed_ms(points)
                write_points_csv(
                    points,
                    elapsed_xs_ms,
                    OUT_DIR / f"{exp_label}_ttft_breakdown_timeline_32tenants_compressed_points.csv",
                )
    common_total_max = 0.0
    common_mean_total_max = 0.0
    common_group_mean_total_max = 0.0
    common_group_p95_turn_max = 0.0
    common_group_p99_turn_max = 0.0
    for points_32 in points_by_tenant[TENANT_COUNT_FILTER].values():
        for point in points_32:
            total = point["blocking_ms"] + point["prefill_ms"] + point["remaining_ms"]
            common_total_max = max(common_total_max, total)
        grouped = {turn: [] for turn in range(1, 11)}
        for point in points_32:
            grouped[int(point["turn_index"])].append(point)
        for turn in range(1, 11):
            subset = grouped[turn]
            if not subset:
                continue
            total_mean = sum(
                p["blocking_ms"] + p["prefill_ms"] + p["remaining_ms"] for p in subset
            ) / len(subset)
            common_mean_total_max = max(common_mean_total_max, total_mean)
        for history_limit_tokens in ["8192", "2048"]:
            grouped_by_group = {turn: [] for turn in range(1, 11)}
            for point in points_32:
                if point["history_limit_tokens"] != history_limit_tokens:
                    continue
                grouped_by_group[int(point["turn_index"])].append(point)
            for turn in range(1, 11):
                subset = grouped_by_group[turn]
                if not subset:
                    continue
                total_mean = sum(
                    p["blocking_ms"] + p["prefill_ms"] + p["remaining_ms"] for p in subset
                ) / len(subset)
                common_group_mean_total_max = max(common_group_mean_total_max, total_mean)
                total_p95 = (
                    percentile([p["blocking_ms"] for p in subset], 0.95)
                    + percentile([p["prefill_ms"] for p in subset], 0.95)
                    + percentile([p["remaining_ms"] for p in subset], 0.95)
                )
                total_p99 = (
                    percentile([p["blocking_ms"] for p in subset], 0.99)
                    + percentile([p["prefill_ms"] for p in subset], 0.99)
                    + percentile([p["remaining_ms"] for p in subset], 0.99)
                )
                common_group_p95_turn_max = max(common_group_p95_turn_max, total_p95)
                common_group_p99_turn_max = max(common_group_p99_turn_max, total_p99)

    total_ylim = (0.0, common_total_max * 1.05 if common_total_max > 0 else 1.0)
    mean_ylim = (0.0, common_mean_total_max * 1.05 if common_mean_total_max > 0 else 1.0)
    group_mean_ylim = (
        0.0,
        common_group_mean_total_max * 1.05 if common_group_mean_total_max > 0 else 1.0,
    )
    group_p95_turn_ylim = (
        0.0,
        common_group_p95_turn_max * 1.05 if common_group_p95_turn_max > 0 else 1.0,
    )
    group_p99_turn_ylim = (
        0.0,
        common_group_p99_turn_max * 1.05 if common_group_p99_turn_max > 0 else 1.0,
    )

    for exp_label, points_32 in points_by_tenant[TENANT_COUNT_FILTER].items():
        render_turn_breakdown_by_tenant_continuous(
            plt,
            points_32,
            f"{exp_label}: TTFT Breakdown by Tenant and Turn (tenant_count=32, continuous)",
            OUT_DIR / f"{exp_label}_ttft_breakdown_by_tenant_turn_32tenants_continuous.png",
            y_limits=total_ylim,
        )
        render_turn_breakdown_mean_continuous(
            plt,
            points_32,
            f"{exp_label}: Mean TTFT Breakdown by Turn (tenant_count=32)",
            OUT_DIR / f"{exp_label}_ttft_breakdown_mean_by_turn_32tenants.png",
            y_limits=mean_ylim,
        )
        render_turn_breakdown_group_means(
            plt,
            points_32,
            f"{exp_label}: Group1 vs Group2 Mean TTFT Breakdown by Turn (tenant_count=32)",
            OUT_DIR / f"{exp_label}_ttft_breakdown_group_means_by_turn_32tenants.png",
            y_limits=group_mean_ylim,
        )

    common_group_bar_max = 0.0
    common_group_p95_bar_max = 0.0
    common_group_p99_bar_max = 0.0
    combined_points_by_tenant = {}
    for tenant_count in TENANT_COUNTS:
        combined_points = []
        for points in points_by_tenant[tenant_count].values():
            combined_points.extend(points)
        combined_points_by_tenant[tenant_count] = combined_points
        group1_points = [p for p in combined_points if p["history_limit_tokens"] == "8192"]
        group2_points = [p for p in combined_points if p["history_limit_tokens"] == "2048"]
        common_group_bar_max = max(
            common_group_bar_max,
            mean([p["blocking_ms"] for p in group1_points]),
            mean([p["prefill_ms"] for p in group1_points]),
            mean([p["remaining_ms"] for p in group1_points]),
            mean([p["blocking_ms"] for p in group2_points]),
            mean([p["prefill_ms"] for p in group2_points]),
            mean([p["remaining_ms"] for p in group2_points]),
        )
        common_group_p95_bar_max = max(
            common_group_p95_bar_max,
            percentile([p["blocking_ms"] for p in group1_points], 0.95),
            percentile([p["prefill_ms"] for p in group1_points], 0.95),
            percentile([p["remaining_ms"] for p in group1_points], 0.95),
            percentile([p["blocking_ms"] for p in group2_points], 0.95),
            percentile([p["prefill_ms"] for p in group2_points], 0.95),
            percentile([p["remaining_ms"] for p in group2_points], 0.95),
        )
        common_group_p99_bar_max = max(
            common_group_p99_bar_max,
            percentile([p["blocking_ms"] for p in group1_points], 0.99),
            percentile([p["prefill_ms"] for p in group1_points], 0.99),
            percentile([p["remaining_ms"] for p in group1_points], 0.99),
            percentile([p["blocking_ms"] for p in group2_points], 0.99),
            percentile([p["prefill_ms"] for p in group2_points], 0.99),
            percentile([p["remaining_ms"] for p in group2_points], 0.99),
        )

    group_bar_ylim = (
        0.0,
        common_group_bar_max * 1.05 if common_group_bar_max > 0 else 1.0,
    )
    group_p95_bar_ylim = (
        0.0,
        common_group_p95_bar_max * 1.05 if common_group_p95_bar_max > 0 else 1.0,
    )
    group_p99_bar_ylim = (
        0.0,
        common_group_p99_bar_max * 1.05 if common_group_p99_bar_max > 0 else 1.0,
    )

    for tenant_count in TENANT_COUNTS:
        combined_points = combined_points_by_tenant[tenant_count]
        render_turn_breakdown_group_means(
            plt,
            combined_points,
            f"exp1+exp2+exp3: Group1 vs Group2 Mean TTFT Breakdown by Turn (tenant_count={tenant_count})",
            OUT_DIR / f"exp123_ttft_breakdown_group_means_by_turn_{tenant_count}tenants.png",
            y_limits=group_mean_ylim,
        )
        render_turn_breakdown_group_percentiles(
            plt,
            combined_points,
            f"exp1+exp2+exp3: Group1 vs Group2 P95 TTFT Breakdown by Turn (tenant_count={tenant_count})",
            OUT_DIR / f"exp123_ttft_breakdown_group_p95_by_turn_{tenant_count}tenants.png",
            quantile=0.95,
            y_limits=group_p95_turn_ylim,
        )
        render_turn_breakdown_group_percentiles(
            plt,
            combined_points,
            f"exp1+exp2+exp3: Group1 vs Group2 P99 TTFT Breakdown by Turn (tenant_count={tenant_count})",
            OUT_DIR / f"exp123_ttft_breakdown_group_p99_by_turn_{tenant_count}tenants.png",
            quantile=0.99,
            y_limits=group_p99_turn_ylim,
        )
        render_group_metric_means_bar(
            plt,
            combined_points,
            f"exp1+exp2+exp3: Group1 vs Group2 Mean Breakdown (tenant_count={tenant_count})",
            OUT_DIR / f"exp123_group1_group2_mean_breakdown_bar_{tenant_count}tenants.png",
            y_limits=group_bar_ylim,
        )
        render_group_metric_percentile_bar(
            plt,
            combined_points,
            f"exp1+exp2+exp3: Group1 vs Group2 P95 Breakdown (tenant_count={tenant_count})",
            OUT_DIR / f"exp123_group1_group2_p95_breakdown_bar_{tenant_count}tenants.png",
            quantile=0.95,
            y_limits=group_p95_bar_ylim,
        )
        render_group_metric_percentile_bar(
            plt,
            combined_points,
            f"exp1+exp2+exp3: Group1 vs Group2 P99 Breakdown (tenant_count={tenant_count})",
            OUT_DIR / f"exp123_group1_group2_p99_breakdown_bar_{tenant_count}tenants.png",
            quantile=0.99,
            y_limits=group_p99_bar_ylim,
        )

    write_group_means_by_turn_csv(
        points_by_tenant,
        OUT_DIR / "exp123_ttft_breakdown_group_means_by_turn_all_tenants.csv",
    )
    write_group_mean_breakdown_bar_csv(
        points_by_tenant,
        OUT_DIR / "exp123_group1_group2_mean_breakdown_bar_all_tenants.csv",
    )
    write_group_means_markdown(
        points_by_tenant,
        OUT_DIR / "exp123_group_means_analysis.md",
    )

    print(f"[DONE] plots saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
