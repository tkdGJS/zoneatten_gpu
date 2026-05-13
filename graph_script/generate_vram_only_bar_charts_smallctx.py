#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "result_raw.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "graphs" / Path(__file__).stem

METRICS: List[Tuple[str, str, str]] = [
    ("ttft_ms", "TTFT (ms)", "ttft"),
    ("p95_tbt_ms", "p95(TBT) (ms)", "tbt"),
    ("ttlt_ms", "TTLT (ms)", "ttlt"),
    ("input_tokens", "Input Tokens", "input_tokens"),
    ("output_tokens", "Output Tokens", "output_tokens"),
    ("kv_history_tokens", "KV History Tokens", "kv_history_tokens"),
    ("prefix_hit_tokens", "Prefix Hit Tokens", "prefix_hit_tokens"),
    ("prefix_hit_rate", "Prefix Hit Rate", "prefix_hit_rate"),
    ("blocking_time_ms", "Blocking Time (ms)", "blocking_time"),
]

GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(28)
FONT_AXIS = load_font(24)
FONT_TICK = load_font(18)
FONT_SMALL = load_font(16)


def parse_float(value: str) -> float:
    if value == "":
        return 0.0
    return float(value)


def rotated_text_image(text: str, font: ImageFont.ImageFont, fill=BLACK) -> Image.Image:
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0] + 2)
    height = max(1, bbox[3] - bbox[1] + 2)
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(image)
    image_draw.text((1 - bbox[0], 1 - bbox[1]), text, font=font, fill=fill)
    return image.rotate(90, expand=True)


def draw_centered_text(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font, fill=BLACK) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def format_tick(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.3f}"


def compute_ylim(rows: List[Dict[str, str]], metric_key: str) -> float:
    values = [parse_float(row[metric_key]) for row in rows]
    max_value = max(values) if values else 0.0
    if max_value <= 0:
        return 1.0
    return max_value * 1.1


def render_chart(
    rows: List[Dict[str, str]],
    tenant_count: int,
    metric_key: str,
    metric_label: str,
    metric_slug: str,
    y_max: float,
) -> None:
    group_gap = 14
    bar_width = 16
    left = 140
    right = 40
    top = 70
    bottom = 135
    plot_height = 620
    total_bars = len(rows)
    width = max(1400, left + right + total_bars * bar_width + (tenant_count - 1) * group_gap)
    height = top + plot_height + bottom

    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)

    plot_left = left
    plot_top = top
    plot_bottom = top + plot_height
    plot_right = width - right

    tick_count = 5
    for idx in range(tick_count + 1):
        ratio = idx / tick_count
        y = plot_bottom - ratio * plot_height
        draw.line((plot_left, y, plot_right, y), fill=LIGHT_GRAY, width=1)
        tick_value = y_max * ratio
        draw.text((70, y - 8), format_tick(tick_value), font=FONT_TICK, fill=BLACK)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=BLACK, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=BLACK, width=2)

    x = plot_left + 8
    tenant_centers: Dict[int, List[float]] = defaultdict(list)
    turn_labels: List[Tuple[float, str]] = []
    tenant_boundaries: List[float] = []
    current_tenant = None
    for row in rows:
        tenant_id = int(row["tenant_id"])
        turn_index = int(row["turn_index"])
        if current_tenant is not None and tenant_id != current_tenant:
            tenant_boundaries.append(x + group_gap / 2)
            x += group_gap
        current_tenant = tenant_id
        value = parse_float(row[metric_key])
        bar_height = 0 if y_max == 0 else (value / y_max) * plot_height
        x0 = x
        x1 = x + bar_width - 2
        y0 = plot_bottom - bar_height
        y1 = plot_bottom
        draw.rectangle((x0, y0, x1, y1), fill=GRAY, outline=GRAY)
        center_x = (x0 + x1) / 2
        tenant_centers[tenant_id].append(center_x)
        turn_labels.append((center_x, str(turn_index)))
        x += bar_width

    draw_centered_text(
        draw,
        (width / 2, 24),
        f"tenant_count={tenant_count} | {metric_label}",
        FONT_TITLE,
    )
    ylabel_img = rotated_text_image(metric_label, FONT_AXIS)
    image.paste(ylabel_img, (18, int((height - ylabel_img.height) / 2)), ylabel_img)

    turn_y = plot_bottom + 6
    for center_x, turn_text in turn_labels:
        label_img = rotated_text_image(turn_text, FONT_SMALL)
        image.paste(label_img, (int(center_x - label_img.width / 2), turn_y), label_img)

    tenant_y = plot_bottom + 70
    for tenant_id, centers in tenant_centers.items():
        center_x = sum(centers) / len(centers)
        draw_centered_text(draw, (center_x, tenant_y), str(tenant_id), FONT_TICK)

    for boundary_x in tenant_boundaries:
        draw.line((boundary_x, plot_top, boundary_x, plot_bottom + 88), fill=BLACK, width=1)

    draw_centered_text(draw, (width / 2, height - 18), "Turns / Tenant ID", FONT_AXIS)

    out_path = OUT_DIR / f"tenant_count_{tenant_count}_{metric_slug}.png"
    image.save(out_path)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(RAW_CSV.open("r", encoding="utf-8", newline="")))

    for row in rows:
        if row["prefix_hit_rate"] == "":
            row["prefix_hit_rate"] = "0"

    y_limits = {metric_key: compute_ylim(rows, metric_key) for metric_key, _, _ in METRICS}

    by_tenant_count: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_tenant_count[int(row["tenant_count"])].append(row)

    for tenant_count, group_rows in sorted(by_tenant_count.items()):
        ordered_rows = sorted(
            group_rows,
            key=lambda row: (int(row["tenant_id"]), int(row["turn_index"])),
        )
        for metric_key, metric_label, metric_slug in METRICS:
            render_chart(
                ordered_rows,
                tenant_count,
                metric_key,
                metric_label,
                metric_slug,
                y_limits[metric_key],
            )

    print(f"[DONE] generated {len(by_tenant_count) * len(METRICS)} charts in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
