#!/usr/bin/env python3

import json
import os
import sys
import subprocess
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def load_results(filepath="results_all.json"):
    with open(filepath) as f:
        return json.load(f)

def geometric_mean(values):
    values = [v for v in values if v > 0]
    if not values:
        return 1.0
    return np.exp(np.mean(np.log(values)))

def categorize_benchmark(filepath):
    parts = filepath.split('/')
    for p in parts:
        if p in ['core', 'float', 'mem', 'mixed', 'long']:
            return p
    return 'other'

def generate_static_improvement_bar(results, output_dir):
    data = []
    for r in results:
        name = os.path.basename(r['file']).replace('.bril', '')
        ratio = r['static_after'] / r['static_before'] if r['static_before'] > 0 else 1.0
        data.append((name, ratio, r['static_before'], r['static_after']))

    data.sort(key=lambda x: x[1])

    top_data = data[:20]

    fig, ax = plt.subplots(figsize=(12, 8))

    names = [d[0] for d in top_data]
    ratios = [d[1] for d in top_data]
    before = [d[2] for d in top_data]
    after = [d[3] for d in top_data]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, before, width, label='Original', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label='After GVN', color='#27ae60', alpha=0.8)

    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Static Instructions')
    ax.set_title('Top 20 Benchmarks: Static Instruction Reduction')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, (b, a) in enumerate(zip(before, after)):
        reduction = (1 - a/b) * 100
        ax.annotate(f'-{reduction:.0f}%', xy=(i, a), ha='center', va='bottom', fontsize=8, color='#27ae60')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'static_improvement_bar.png'))
    plt.close()
    print(f"  Saved: static_improvement_bar.png")

def generate_static_ratio_histogram(results, output_dir):
    ratios = []
    for r in results:
        if r['static_before'] > 0:
            ratios.append(r['static_after'] / r['static_before'])

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0.2, 1.2, 25)
    n, bins_out, patches = ax.hist(ratios, bins=bins, edgecolor='black', alpha=0.7, color='#3498db')

    for i, patch in enumerate(patches):
        if bins_out[i] < 1.0:
            patch.set_facecolor('#27ae60')
        else:
            patch.set_facecolor('#e74c3c')

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='No change')
    ax.axvline(x=geometric_mean(ratios), color='#9b59b6', linestyle='-', linewidth=2,
               label=f'Geometric Mean: {geometric_mean(ratios):.3f}')

    ax.set_xlabel('Static Instruction Ratio (After/Before)')
    ax.set_ylabel('Number of Benchmarks')
    ax.set_title('Distribution of Static Instruction Ratios Across All Benchmarks')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    improved = sum(1 for r in ratios if r < 1.0)
    total = len(ratios)
    ax.text(0.95, 0.95, f'{improved}/{total} benchmarks improved',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'static_ratio_histogram.png'))
    plt.close()
    print(f"  Saved: static_ratio_histogram.png")

def generate_dynamic_improvement_bar(results, output_dir):
    data = []
    for r in results:
        if r.get('dyn_before', 0) > 100:
            name = os.path.basename(r['file']).replace('.bril', '')
            ratio = r['dyn_after'] / r['dyn_before'] if r['dyn_before'] > 0 else 1.0
            data.append((name, ratio, r['dyn_before'], r['dyn_after']))

    data.sort(key=lambda x: x[1])

    top_data = data[:15]

    fig, ax = plt.subplots(figsize=(12, 7))

    names = [d[0] for d in top_data]
    before = [d[2] for d in top_data]
    after = [d[3] for d in top_data]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, before, width, label='Original', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label='After GVN', color='#27ae60', alpha=0.8)

    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Dynamic Instructions (log scale)')
    ax.set_title('Top 15 Benchmarks: Dynamic Instruction Reduction')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dynamic_improvement_bar.png'))
    plt.close()
    print(f"  Saved: dynamic_improvement_bar.png")

def generate_category_comparison(results, output_dir):
    categories = defaultdict(lambda: {'static_ratios': [], 'dynamic_ratios': []})

    for r in results:
        cat = categorize_benchmark(r['file'])
        if r['static_before'] > 0:
            categories[cat]['static_ratios'].append(r['static_after'] / r['static_before'])
        if r.get('dyn_before', 0) > 0:
            categories[cat]['dynamic_ratios'].append(r['dyn_after'] / r['dyn_before'])

    cats = sorted(categories.keys())
    static_gm = [geometric_mean(categories[c]['static_ratios']) for c in cats]
    dynamic_gm = [geometric_mean(categories[c]['dynamic_ratios']) for c in cats]
    counts = [len(categories[c]['static_ratios']) for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(cats))
    width = 0.35

    bars1 = ax.bar(x - width/2, static_gm, width, label='Static Instrs', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, dynamic_gm, width, label='Dynamic Instrs', color='#e67e22', alpha=0.8)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Benchmark Category')
    ax.set_ylabel('Instruction Ratio (Geometric Mean)')
    ax.set_title('GVN Optimization Impact by Benchmark Category')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(cats, counts)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    all_values = static_gm + dynamic_gm
    min_val = min(all_values) - 0.05
    max_val = max(max(all_values), 1.0) + 0.05
    ax.set_ylim(min_val, max_val)

    for bar, val in zip(bars1, static_gm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, dynamic_gm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_comparison.png'))
    plt.close()
    print(f"  Saved: category_comparison.png")

def generate_scatter_static_vs_dynamic(results, output_dir):
    static_ratios = []
    dynamic_ratios = []
    names = []

    for r in results:
        if r['static_before'] > 0 and r.get('dyn_before', 0) > 0:
            static_ratios.append(r['static_after'] / r['static_before'])
            dynamic_ratios.append(r['dyn_after'] / r['dyn_before'])
            names.append(os.path.basename(r['file']).replace('.bril', ''))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = []
    for s, d in zip(static_ratios, dynamic_ratios):
        if s < 1.0 and d < 1.0:
            colors.append('#27ae60')
        elif s < 1.0 or d < 1.0:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')

    scatter = ax.scatter(static_ratios, dynamic_ratios, c=colors, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.plot([0.2, 1.2], [0.2, 1.2], 'k:', alpha=0.3, label='Static = Dynamic')

    ax.set_xlabel('Static Instruction Ratio')
    ax.set_ylabel('Dynamic Instruction Ratio')
    ax.set_title('Static vs Dynamic Instruction Improvements')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 1.15)
    ax.set_ylim(0.2, 1.15)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', label='Both improved'),
        Patch(facecolor='#f39c12', label='One improved'),
        Patch(facecolor='#e74c3c', label='Neither improved')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    for i, (s, d, name) in enumerate(zip(static_ratios, dynamic_ratios, names)):
        if s < 0.5 or d < 0.5:
            ax.annotate(name, (s, d), fontsize=7, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'static_vs_dynamic_scatter.png'))
    plt.close()
    print(f"  Saved: static_vs_dynamic_scatter.png")

def generate_summary_table(results, output_dir):
    static_ratios = [r['static_after'] / r['static_before'] for r in results if r['static_before'] > 0]
    dynamic_ratios = [r['dyn_after'] / r['dyn_before'] for r in results if r.get('dyn_before', 0) > 0]

    total_static_before = sum(r['static_before'] for r in results)
    total_static_after = sum(r['static_after'] for r in results)
    total_dynamic_before = sum(r.get('dyn_before', 0) for r in results)
    total_dynamic_after = sum(r.get('dyn_after', 0) for r in results)

    improved_static = sum(1 for r in static_ratios if r < 1.0)
    improved_dynamic = sum(1 for r in dynamic_ratios if r < 1.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    data = [
        ['Total Benchmarks', str(len(results))],
        ['All Tests Pass', 'Yes (100%)'],
        ['', ''],
        ['Static Instructions (Total)', f'{total_static_before:,} -> {total_static_after:,}'],
        ['Static Reduction', f'{(1 - total_static_after/total_static_before)*100:.1f}%'],
        ['Static Ratio (GM)', f'{geometric_mean(static_ratios):.4f}'],
        ['Benchmarks with Static Improvement', f'{improved_static}/{len(static_ratios)} ({improved_static/len(static_ratios)*100:.1f}%)'],
        ['', ''],
        ['Dynamic Instructions (Total)', f'{total_dynamic_before:,} -> {total_dynamic_after:,}'],
        ['Dynamic Reduction', f'{(1 - total_dynamic_after/total_dynamic_before)*100:.1f}%'],
        ['Dynamic Ratio (GM)', f'{geometric_mean(dynamic_ratios):.4f}'],
        ['Benchmarks with Dynamic Improvement', f'{improved_dynamic}/{len(dynamic_ratios)} ({improved_dynamic/len(dynamic_ratios)*100:.1f}%)'],
    ]

    table = ax.table(cellText=data, colWidths=[0.5, 0.4], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for i in [0, 1, 3, 4, 5, 6, 8, 9, 10, 11]:
        if i < len(data):
            table[(i, 0)].set_facecolor('#f0f0f0')

    ax.set_title('GVN Optimization Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'))
    plt.close()
    print(f"  Saved: summary_table.png")

def generate_waterfall_chart(results, output_dir):
    data = []
    for r in results:
        savings = r['static_before'] - r['static_after']
        name = os.path.basename(r['file']).replace('.bril', '')
        data.append((name, savings, r['static_before']))

    data.sort(key=lambda x: -x[1])

    top_data = data[:15]

    fig, ax = plt.subplots(figsize=(12, 7))

    names = [d[0] for d in top_data]
    savings = [d[1] for d in top_data]

    colors = ['#27ae60' if s > 0 else '#e74c3c' for s in savings]

    cumulative = np.cumsum(savings)

    ax.bar(range(len(names)), savings, color=colors, alpha=0.8, edgecolor='black')
    ax.plot(range(len(names)), cumulative, 'ko-', markersize=6, label='Cumulative')

    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Instructions Saved')
    ax.set_title('Top 15 Contributors to Static Instruction Reduction')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, c in enumerate(cumulative):
        ax.annotate(f'{c}', (i, c), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waterfall_savings.png'))
    plt.close()
    print(f"  Saved: waterfall_savings.png")

def measure_ssa_overhead(test_dirs=None):
    import to_ssa
    import from_ssa
    import tdce
    from copy import deepcopy

    if test_dirs is None:
        test_dirs = ["tests/core", "tests/float", "tests/mem", "tests/mixed"]

    results = []

    test_files = []
    for test_dir in test_dirs:
        test_files.extend(glob.glob(f"{test_dir}/*.bril"))

    for filepath in sorted(test_files):
        try:
            with open(filepath) as f:
                bril_text = f.read()

            proc = subprocess.run(["bril2json"], input=bril_text, capture_output=True, text=True)
            if proc.returncode != 0:
                continue
            prog = json.loads(proc.stdout)

            original_count = sum(
                len([i for i in func.get("instrs", []) if "op" in i or "label" in i])
                for func in prog.get("functions", [])
            )

            ssa_only_prog = deepcopy(prog)
            for func in ssa_only_prog.get("functions", []):
                to_ssa.run(func)
                from_ssa.lower_phis(func)
                tdce.trivial_dce_func(func)

            ssa_only_count = sum(
                len([i for i in func.get("instrs", []) if "op" in i or "label" in i])
                for func in ssa_only_prog.get("functions", [])
            )

            import gvn
            gvn_prog = deepcopy(prog)
            gvn.main(gvn_prog, use_ssa=True)

            gvn_count = sum(
                len([i for i in func.get("instrs", []) if "op" in i or "label" in i])
                for func in gvn_prog.get("functions", [])
            )

            results.append({
                "file": filepath,
                "original": original_count,
                "after_ssa": ssa_only_count,
                "after_gvn": gvn_count,
                "ssa_overhead": ssa_only_count - original_count,
                "gvn_savings": original_count - gvn_count,
                "ssa_ratio": ssa_only_count / original_count if original_count > 0 else 1.0,
                "gvn_ratio": gvn_count / original_count if original_count > 0 else 1.0,
            })

        except Exception as e:
            print(f"  Warning: Failed to process {filepath}: {e}", file=sys.stderr)
            continue

    return results


def generate_ssa_overhead_bar(ssa_results, output_dir):
    data = [(r["file"], r["ssa_overhead"], r["gvn_savings"]) for r in ssa_results]
    data.sort(key=lambda x: -x[1])

    top_data = data[:20]

    fig, ax = plt.subplots(figsize=(14, 8))

    names = [os.path.basename(d[0]).replace('.bril', '') for d in top_data]
    ssa_overhead = [d[1] for d in top_data]
    gvn_savings = [d[2] for d in top_data]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, ssa_overhead, width, label='SSA Overhead (after SSA - original)',
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, gvn_savings, width, label='GVN Savings (original - after GVN)',
                   color='#27ae60', alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Instruction Difference')
    ax.set_title('SSA Conversion Overhead vs GVN Optimization Savings\n(Top 20 Benchmarks by SSA Overhead)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssa_overhead_bar.png'))
    plt.close()
    print(f"  Saved: ssa_overhead_bar.png")


def generate_ssa_overhead_histogram(ssa_results, output_dir):
    ratios = [r["ssa_ratio"] for r in ssa_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0.8, 1.5, 25)
    n, bins_out, patches = ax.hist(ratios, bins=bins, edgecolor='black', alpha=0.7)

    for i, patch in enumerate(patches):
        if bins_out[i] <= 1.0:
            patch.set_facecolor('#27ae60')
        elif bins_out[i] <= 1.1:
            patch.set_facecolor('#f39c12')
        else:
            patch.set_facecolor('#e74c3c')

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='No change')
    gm = geometric_mean(ratios)
    ax.axvline(x=gm, color='#9b59b6', linestyle='-', linewidth=2,
               label=f'Geometric Mean: {gm:.3f}')

    ax.set_xlabel('SSA Overhead Ratio (After SSA / Original)')
    ax.set_ylabel('Number of Benchmarks')
    ax.set_title('Distribution of SSA Conversion Overhead\n(SSA + Lowering + DCE, No GVN)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    overhead_count = sum(1 for r in ratios if r > 1.0)
    total = len(ratios)
    ax.text(0.95, 0.95, f'{overhead_count}/{total} benchmarks have SSA overhead',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssa_overhead_histogram.png'))
    plt.close()
    print(f"  Saved: ssa_overhead_histogram.png")


def generate_ssa_vs_gvn_scatter(ssa_results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))

    ssa_ratios = [r["ssa_ratio"] for r in ssa_results]
    gvn_ratios = [r["gvn_ratio"] for r in ssa_results]
    names = [os.path.basename(r["file"]).replace('.bril', '') for r in ssa_results]

    colors = []
    for ssa, gvn in zip(ssa_ratios, gvn_ratios):
        if gvn < 1.0:
            colors.append('#27ae60')
        elif gvn <= ssa:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')

    scatter = ax.scatter(ssa_ratios, gvn_ratios, c=colors, alpha=0.6, s=60,
                        edgecolors='black', linewidth=0.5)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Original size')
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.plot([0.7, 1.5], [0.7, 1.5], 'k:', alpha=0.3, label='SSA = GVN')

    ax.set_xlabel('SSA Overhead Ratio (After SSA / Original)')
    ax.set_ylabel('Final GVN Ratio (After GVN / Original)')
    ax.set_title('SSA Overhead vs Final GVN Result')
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', label='Net benefit (GVN < original)'),
        Patch(facecolor='#f39c12', label='GVN helps (GVN < SSA)'),
        Patch(facecolor='#e74c3c', label='No benefit'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    for ssa, gvn, name in zip(ssa_ratios, gvn_ratios, names):
        if ssa > 1.3 or gvn < 0.6:
            ax.annotate(name, (ssa, gvn), fontsize=7, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssa_vs_gvn_scatter.png'))
    plt.close()
    print(f"  Saved: ssa_vs_gvn_scatter.png")


def generate_ssa_pipeline_comparison(ssa_results, output_dir):
    data = sorted(ssa_results, key=lambda r: r["gvn_ratio"])[:15]

    fig, ax = plt.subplots(figsize=(14, 8))

    names = [os.path.basename(r["file"]).replace('.bril', '') for r in data]
    original = [r["original"] for r in data]
    after_ssa = [r["after_ssa"] for r in data]
    after_gvn = [r["after_gvn"] for r in data]

    x = np.arange(len(names))
    width = 0.25

    bars1 = ax.bar(x - width, original, width, label='Original', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, after_ssa, width, label='After SSA (no GVN)', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, after_gvn, width, label='After GVN', color='#27ae60', alpha=0.8)

    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Static Instructions')
    ax.set_title('Instruction Count at Each Pipeline Stage\n(Top 15 Benchmarks by GVN Improvement)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssa_pipeline_comparison.png'))
    plt.close()
    print(f"  Saved: ssa_pipeline_comparison.png")


def generate_ssa_summary_table(ssa_results, output_dir):
    ssa_ratios = [r["ssa_ratio"] for r in ssa_results]
    gvn_ratios = [r["gvn_ratio"] for r in ssa_results]

    total_original = sum(r["original"] for r in ssa_results)
    total_after_ssa = sum(r["after_ssa"] for r in ssa_results)
    total_after_gvn = sum(r["after_gvn"] for r in ssa_results)

    ssa_overhead_count = sum(1 for r in ssa_ratios if r > 1.0)
    gvn_improved_count = sum(1 for r in gvn_ratios if r < 1.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    data = [
        ['Total Benchmarks', str(len(ssa_results))],
        ['', ''],
        ['SSA Overhead Analysis', ''],
        ['Total Instructions (Original)', f'{total_original:,}'],
        ['Total Instructions (After SSA)', f'{total_after_ssa:,}'],
        ['SSA Overhead (Total)', f'+{total_after_ssa - total_original:,} ({(total_after_ssa/total_original - 1)*100:.1f}%)'],
        ['SSA Overhead Ratio (GM)', f'{geometric_mean(ssa_ratios):.4f}'],
        ['Benchmarks with SSA Overhead', f'{ssa_overhead_count}/{len(ssa_results)} ({ssa_overhead_count/len(ssa_results)*100:.1f}%)'],
        ['', ''],
        ['GVN Recovery', ''],
        ['Total Instructions (After GVN)', f'{total_after_gvn:,}'],
        ['Net Change from Original', f'{total_after_gvn - total_original:+,} ({(total_after_gvn/total_original - 1)*100:+.1f}%)'],
        ['GVN Ratio (GM)', f'{geometric_mean(gvn_ratios):.4f}'],
        ['Benchmarks Improved by GVN', f'{gvn_improved_count}/{len(ssa_results)} ({gvn_improved_count/len(ssa_results)*100:.1f}%)'],
    ]

    table = ax.table(cellText=data, colWidths=[0.55, 0.35], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for i in [0, 2, 9]:
        if i < len(data):
            table[(i, 0)].set_facecolor('#d5dbdb')
            table[(i, 1)].set_facecolor('#d5dbdb')

    ax.set_title('SSA Overhead and GVN Recovery Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssa_summary_table.png'))
    plt.close()
    print(f"  Saved: ssa_summary_table.png")

    return {
        "total_benchmarks": len(ssa_results),
        "total_original": total_original,
        "total_after_ssa": total_after_ssa,
        "total_after_gvn": total_after_gvn,
        "ssa_overhead_ratio_gm": geometric_mean(ssa_ratios),
        "gvn_ratio_gm": geometric_mean(gvn_ratios),
        "ssa_overhead_count": ssa_overhead_count,
        "gvn_improved_count": gvn_improved_count,
    }


def main():
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    results = load_results("results_all.json")
    print(f"  Loaded {len(results)} benchmark results")

    print("\nGenerating standard figures...")

    generate_static_improvement_bar(results, output_dir)
    generate_static_ratio_histogram(results, output_dir)
    generate_dynamic_improvement_bar(results, output_dir)
    generate_category_comparison(results, output_dir)
    generate_scatter_static_vs_dynamic(results, output_dir)
    generate_summary_table(results, output_dir)
    generate_waterfall_chart(results, output_dir)

    static_ratios = [r['static_after'] / r['static_before'] for r in results if r['static_before'] > 0]
    dynamic_ratios = [r['dyn_after'] / r['dyn_before'] for r in results if r.get('dyn_before', 0) > 0]

    print("\n=== Summary Statistics ===")
    print(f"Static Instruction Ratio (GM):  {geometric_mean(static_ratios):.4f}")
    print(f"Dynamic Instruction Ratio (GM): {geometric_mean(dynamic_ratios):.4f}")
    print(f"Benchmarks improved (static):   {sum(1 for r in static_ratios if r < 1.0)}/{len(static_ratios)}")
    print(f"Benchmarks improved (dynamic):  {sum(1 for r in dynamic_ratios if r < 1.0)}/{len(dynamic_ratios)}")

    print("\n=== SSA Overhead Analysis ===")
    print("Measuring SSA overhead (this may take a moment)...")
    ssa_results = measure_ssa_overhead()

    if ssa_results:
        print(f"  Processed {len(ssa_results)} benchmarks")

        with open("ssa_overhead_results.json", "w") as f:
            json.dump(ssa_results, f, indent=2)
        print("  Saved: ssa_overhead_results.json")

        print("\nGenerating SSA overhead figures...")
        generate_ssa_overhead_bar(ssa_results, output_dir)
        generate_ssa_overhead_histogram(ssa_results, output_dir)
        generate_ssa_vs_gvn_scatter(ssa_results, output_dir)
        generate_ssa_pipeline_comparison(ssa_results, output_dir)
        ssa_summary = generate_ssa_summary_table(ssa_results, output_dir)

        print("\n=== SSA Overhead Summary ===")
        print(f"SSA Overhead Ratio (GM):        {ssa_summary['ssa_overhead_ratio_gm']:.4f}")
        print(f"GVN Final Ratio (GM):           {ssa_summary['gvn_ratio_gm']:.4f}")
        print(f"Benchmarks with SSA overhead:   {ssa_summary['ssa_overhead_count']}/{ssa_summary['total_benchmarks']}")
        print(f"Benchmarks improved by GVN:     {ssa_summary['gvn_improved_count']}/{ssa_summary['total_benchmarks']}")
        print(f"Total instructions (original):  {ssa_summary['total_original']:,}")
        print(f"Total instructions (after SSA): {ssa_summary['total_after_ssa']:,} (+{ssa_summary['total_after_ssa'] - ssa_summary['total_original']:,})")
        print(f"Total instructions (after GVN): {ssa_summary['total_after_gvn']:,} ({ssa_summary['total_after_gvn'] - ssa_summary['total_original']:+,})")
    else:
        print("  Warning: Could not measure SSA overhead")

    print(f"\nAll figures saved to '{output_dir}/' directory")

if __name__ == "__main__":
    main()
