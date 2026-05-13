#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parents[1] / "vram_only_results_smallctx_mixed_limits_8GB_ori" / "graphs" / Path(__file__).stem
EXPERIMENTS = [
    'block_2048_limit_same_timing1',
    'block_2048_limit_same_timing2',
    'block_2048_limit_same_timing3',
]
TENANT_COUNTS = ['8', '16', '8+16']
BLOCKING_THRESHOLD_MS = 100.0
GROUP_COLORS = {'Group1': '#2563eb', 'Group2': '#f97316'}


def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0


def percentile(xs, q):
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys)-1)*q
    lo = int(pos)
    hi = min(lo+1, len(ys)-1)
    frac = pos - lo
    return ys[lo]*(1-frac) + ys[hi]*frac


def padded_limits(values):
    if not values:
        return (0.0, 1.0)
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = max(1.0, abs(lo)*0.05, 0.1)
        return (lo-pad, hi+pad)
    pad = (hi-lo)*0.05
    return (lo-pad, hi+pad)


def quantile_edges(values, n_bins):
    vals = sorted(values)
    edges = [vals[0]]
    for i in range(1, n_bins):
        edges.append(percentile(vals, i / n_bins))
    edges.append(vals[-1])
    fixed = [edges[0]]
    for e in edges[1:]:
        if e <= fixed[-1]:
            e = fixed[-1] + 1e-9
        fixed.append(e)
    return fixed


def assign_bin(v, edges):
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if i == len(edges) - 2:
            if left <= v <= right:
                return i
        if left <= v < right:
            return i
    return len(edges) - 2


def group_label(row):
    return 'Group1' if row['history_limit_tokens'] == '8192' else 'Group2'


def load_rows():
    rows = []
    for exp in EXPERIMENTS:
        path = ROOT / exp / 'vram_only_results_smallctx_mixed_limits' / 'result_raw.csv'
        with path.open() as f:
            for row in csv.DictReader(f):
                if row['status'] != 'success':
                    continue
                tenant_count = row['tenant_count']
                if tenant_count not in {'8', '16'}:
                    continue
                blocking_ms = float(row['blocking_time_ms'] or 0.0)
                if blocking_ms >= BLOCKING_THRESHOLD_MS:
                    continue
                rows.append({
                    'tenant_count': tenant_count,
                    'group_label': group_label(row),
                    'input_tokens': float(row['input_tokens'] or 0.0),
                    'prefix_hit_rate': float(row['prefix_hit_rate'] or 0.0),
                    'prefix_hit_tokens': float(row['prefix_hit_tokens'] or 0.0),
                    'ttft_ms': float(row['ttft_ms'] or 0.0),
                })
    return rows


def build_binned(rows, tenant_key):
    if tenant_key == '8+16':
        subset = list(rows)
    else:
        subset = [r for r in rows if r['tenant_count'] == tenant_key]
    if not subset:
        return subset, []
    input_edges = quantile_edges([r['input_tokens'] for r in subset], 4)
    hit_edges = [0.0, 0.25, 0.5, 0.75, 1.000001]
    cells = defaultdict(list)
    for r in subset:
        ib = assign_bin(r['input_tokens'], input_edges)
        hb = assign_bin(r['prefix_hit_rate'], hit_edges)
        cells[(ib, hb)].append(r)
    summaries = []
    for ib in range(4):
        for hb in range(4):
            cell = cells.get((ib, hb), [])
            if not cell:
                continue
            summaries.append({
                'tenant_key': tenant_key,
                'input_bin': ib,
                'hit_bin': hb,
                'input_label': f"{input_edges[ib]:.0f}-{input_edges[ib+1]:.0f}",
                'hit_label': ['0-0.25','0.25-0.5','0.5-0.75','0.75-1.0'][hb],
                'mean_ttft_ms': mean([r['ttft_ms'] for r in cell]),
                'p50_ttft_ms': percentile([r['ttft_ms'] for r in cell], 0.5),
                'p95_ttft_ms': percentile([r['ttft_ms'] for r in cell], 0.95),
                'mean_prefix_hit_tokens': mean([r['prefix_hit_tokens'] for r in cell]),
                'count': len(cell),
            })
    return subset, summaries


def save_csv(rows, path):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_input_vs_ttft_by_hit_rate(plt, rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    subsets = {
        '8': [r for r in rows if r['tenant_count'] == '8'],
        '16': [r for r in rows if r['tenant_count'] == '16'],
        '8+16': list(rows),
    }
    all_x = [r['input_tokens'] for r in rows]
    all_y = [r['ttft_ms'] for r in rows]
    all_c = [r['prefix_hit_rate'] for r in rows]
    xlim = padded_limits(all_x)
    ylim = padded_limits(all_y)
    clim = padded_limits(all_c)
    for ax, tenant_key in zip(axes, TENANT_COUNTS):
        sub = subsets[tenant_key]
        sc = ax.scatter([r['input_tokens'] for r in sub], [r['ttft_ms'] for r in sub],
                        c=[r['prefix_hit_rate'] for r in sub], cmap='viridis',
                        vmin=clim[0], vmax=clim[1], s=18, alpha=0.5)
        ax.set_title(f'Input vs TTFT by prefix hit rate\ntenants={tenant_key}, blocking < 100 ms')
        ax.set_xlabel('Input Tokens')
        ax.set_ylabel('TTFT (ms)')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label='Prefix Hit Rate')
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_binned_lines(plt, summaries, metric_key, ylabel, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    all_y = [r[metric_key] for r in summaries]
    ylim = padded_limits(all_y)
    x_positions = [0,1,2,3]
    x_labels = ['0-0.25','0.25-0.5','0.5-0.75','0.75-1.0']
    colors = ['#1d4ed8', '#059669', '#d97706', '#dc2626']
    for ax, tenant_key in zip(axes, TENANT_COUNTS):
        sub = [r for r in summaries if r['tenant_key'] == tenant_key]
        for ib in range(4):
            line = [r for r in sub if r['input_bin'] == ib]
            line = sorted(line, key=lambda r: r['hit_bin'])
            if not line:
                continue
            ax.plot([r['hit_bin'] for r in line], [r[metric_key] for r in line], marker='o', color=colors[ib], label=f"Input {line[0]['input_label']}")
        ax.set_title(f'{ylabel} by prefix-hit bin within input bins\ntenants={tenant_key}, blocking < 100 ms')
        ax.set_xlabel('Prefix Hit Rate Bin')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions, x_labels)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02,1.0), frameon=True, title='Input Bin')
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_md(rows, summaries, path):
    lines = []
    lines.append('# Prefix Hit vs TTFT in Resource-Sufficient Slice')
    lines.append('')
    lines.append('- Data: `exp1+exp2+exp3`')
    lines.append('- Slice: `tenant_count in {8,16}`, `blocking_time_ms < 100`')
    lines.append('- Goal: test whether prefix hit can be tied to TTFT gain after reducing queue-pressure confounding')
    lines.append('')
    for tenant_key in TENANT_COUNTS:
        sub = [r for r in summaries if r['tenant_key'] == tenant_key]
        if not sub:
            continue
        lines.append(f'## tenants={tenant_key}')
        for ib in range(4):
            line = sorted([r for r in sub if r['input_bin'] == ib], key=lambda r: r['hit_bin'])
            if not line:
                continue
            vals = ', '.join(f"{r['hit_label']}=>p50 {r['p50_ttft_ms']:.1f}" for r in line)
            lines.append(f'- Input bin {line[0]["input_label"]}: {vals}')
        lines.append('')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    all_summaries = []
    for tenant_key in TENANT_COUNTS:
        _, summaries = build_binned(rows, tenant_key)
        all_summaries.extend(summaries)
    save_csv(rows, OUT_DIR / 'prefix_hit_ttft_sufficient_rows.csv')
    save_csv(all_summaries, OUT_DIR / 'prefix_hit_ttft_sufficient_binned_summary.csv')
    plot_input_vs_ttft_by_hit_rate(plt, rows, OUT_DIR / 'input_vs_ttft_by_prefix_hit_rate_blocking_lt_100ms_8_16_32combo.png')
    plot_binned_lines(plt, all_summaries, 'p50_ttft_ms', 'P50 TTFT (ms)', OUT_DIR / 'p50_ttft_by_prefix_hit_bin_within_input_bins_blocking_lt_100ms.png')
    plot_binned_lines(plt, all_summaries, 'p95_ttft_ms', 'P95 TTFT (ms)', OUT_DIR / 'p95_ttft_by_prefix_hit_bin_within_input_bins_blocking_lt_100ms.png')
    write_md(rows, all_summaries, OUT_DIR / 'prefix_hit_ttft_sufficient_analysis.md')

if __name__ == '__main__':
    main()
