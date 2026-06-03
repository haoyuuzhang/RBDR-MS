"""Compare strategies across scale levels × N seeds each.

Results saved to output/result.csv.
"""

from __future__ import annotations

import csv
import sys
import os
from time import perf_counter

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from case_generator import generate_case
from metrics import compute_metrics
from sim_engine import SimulationEngine
from strategies import (HierarchicalStrategy, PureRuleStrategy,
                         ClairvoyantMILPStrategy)

# ═══════════════════════════════════════════════════════════════════════════════
# sweep configuration
# ═══════════════════════════════════════════════════════════════════════════════

N_SEEDS = 5
START_SEED = 41

# (label, shop_rule, penalty_lambda) — PA with λ=0.1 and LBU-approximation with λ=10.0
HIER_CONFIGS = [
    ('PA',  'PA', 0.1),   # penalty-aware, light load tiebreaker
    ('LBU', 'PA', 10.0),  # load-dominated → essentially Least Busy Unit
]
HIER_RULES = [c[0] for c in HIER_CONFIGS]
PURE_RULES = ['FIFO', 'EDD', 'ATC']
CLAIRVOYANT_BASELINE = ['Clairvoyant']
ALL_RULES = HIER_RULES + PURE_RULES

COLORS = {'PA': '#27AE60', 'LBU': '#8E44AD',
          'FIFO': '#95A5A6', 'EDD': '#F39C12', 'ATC': '#1ABC9C',
          'Clairvoyant': "#2357C7"}

# ── Shared machine layout (fixed across all scales) ──
# 5 service units, 18 machines, 4 machine types (M1–M4)
SERVICE_UNITS: dict[str, list[str]] = {
    'U1': ['M1_U1', 'M2_U1', 'M3_U1', 'M4_U1'],
    'U2': ['M1_U2', 'M2_U2', 'M3_U2', 'M4_U2'],
    'U3': ['M1_U3', 'M2_U3', 'M3_U3'],
    'U4': ['M1_U4', 'M2_U4', 'M4_U4'],
    'U5': ['M2_U5', 'M3_U5', 'M4_U5'],
}
N_MACHINES = sum(len(v) for v in SERVICE_UNITS.values())  # 18

# Scale levels: (label, n_initial, n_dynamic) — only job counts vary
SCALES: list[tuple[str, int, int]] = [
    ('Tiny',    5,  3),    #   8 jobs, ratio 0.44
    ('Small',   8,  5),    #  13 jobs, ratio 0.72
    ('Medium', 12,  8),    #  20 jobs, ratio 1.11
    ('Large',  16, 10),    #  26 jobs, ratio 1.44
]

HIER_UNIT_LIMIT = 15   # per-unit limit for hierarchical
HIER_INIT_LIMIT = 30   # initial-plan limit for hierarchical
CLAIRVOYANT_TIME_LIMIT = 120  # one-shot offline-optimal baseline

# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

_METRICS_KEYS = ['makespan', 'penalty', 'lead_time', 'time']

def _run_one_seed(jobs, machines, disruptions):
    """Run one seed across all strategies, return per-strategy metrics."""
    results: dict[str, dict] = {}

    n_jobs = len(jobs)

    for label, shop_rule, lam in HIER_CONFIGS:
        t0 = perf_counter()
        engine = SimulationEngine(jobs, machines,
                                  HierarchicalStrategy(
                                      shop_rule=shop_rule,
                                      unit_time_limit=HIER_UNIT_LIMIT,
                                      init_time_limit=HIER_INIT_LIMIT,
                                      penalty_lambda=lam),
                                  disruptions)
        m = compute_metrics(engine.run(), jobs)
        results[label] = {
            'makespan': m['makespan'],
            'penalty': m['total_penalty'],
            'lead_time': m['total_active_lead_time'] / n_jobs if n_jobs else 0,
            'time': perf_counter() - t0,
        }

    for rule in PURE_RULES:
        t0 = perf_counter()
        engine = SimulationEngine(jobs, machines,
                                  PureRuleStrategy(rule=rule),
                                  disruptions)
        m = compute_metrics(engine.run(), jobs)
        results[rule] = {
            'makespan': m['makespan'],
            'penalty': m['total_penalty'],
            'lead_time': m['total_active_lead_time'] / n_jobs if n_jobs else 0,
            'time': perf_counter() - t0,
        }

    # Clairvoyant (offline-optimal baseline: knows all jobs & disruptions at t=0)
    t0 = perf_counter()
    engine = SimulationEngine(jobs, machines,
                              ClairvoyantMILPStrategy(time_limit=CLAIRVOYANT_TIME_LIMIT),
                              disruptions)
    m = compute_metrics(engine.run(), jobs)
    results['Clairvoyant'] = {
        'makespan': m['makespan'],
        'penalty': m['total_penalty'],
        'lead_time': m['total_active_lead_time'] / n_jobs if n_jobs else 0,
        'time': perf_counter() - t0,
    }

    return results


def _run_scale_experiment(label, n_init, n_dyn):
    """Run N_SEEDS for one scale level, return per-seed results."""

    n_total = n_init + n_dyn
    print(f"  {label} ({n_total} jobs) …", end=" ", flush=True)
    t0 = perf_counter()

    all_seeds: list[dict] = []
    for i in range(N_SEEDS):
        seed = START_SEED + i
        jobs, machines, disruptions = generate_case(
            seed=seed, n_initial=n_init, n_dynamic=n_dyn,
            ops_per_job=3, time_horizon=200.0,
            service_units=SERVICE_UNITS,
        )
        r = _run_one_seed(jobs, machines, disruptions)
        all_seeds.append(r)

    print(f"done ({perf_counter() - t0:.1f}s)")
    return all_seeds

# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

CSV_COLUMNS = ['scale', 'n_jobs', 'seed', 'strategy',
               'makespan', 'penalty', 'lead_time', 'time']


def main():
    strategies = ALL_RULES + CLAIRVOYANT_BASELINE
    job_counts = [n_init + n_dyn for _, n_init, n_dyn in SCALES]

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    # ── Run experiments ──
    print("Running comparison …")
    t_start = perf_counter()

    all_raw: list[list[dict]] = []   # all_raw[scale_idx][seed_idx][strategy][metric]
    for label, n_init, n_dyn in SCALES:
        raw = _run_scale_experiment(label, n_init, n_dyn)
        all_raw.append(raw)

    total_time = perf_counter() - t_start
    print(f"Total: {total_time:.0f}s")

    # ── Save CSV ──
    csv_path = os.path.join(out_dir, 'result.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for si, (label, n_init, n_dyn) in enumerate(SCALES):
            n_total = n_init + n_dyn
            for seed_i in range(N_SEEDS):
                seed = START_SEED + seed_i
                for s in strategies:
                    r = all_raw[si][seed_i][s]
                    writer.writerow({
                        'scale': label,
                        'n_jobs': n_total,
                        'seed': seed,
                        'strategy': s,
                        'makespan': f"{r['makespan']:.2f}",
                        'penalty': f"{r['penalty']:.2f}",
                        'lead_time': f"{r['lead_time']:.2f}",
                        'time': f"{r['time']:.4f}",
                    })

    print(f"Results saved to '{csv_path}'")

    # ── Summary tables (console) ──
    # Compute averages per scale per strategy
    averages: list[dict] = []
    for si in range(len(SCALES)):
        avg = {s: {k: 0.0 for k in _METRICS_KEYS} for s in strategies}
        for seed_i in range(N_SEEDS):
            for s in strategies:
                for k in _METRICS_KEYS:
                    avg[s][k] += all_raw[si][seed_i][s][k]
        for s in strategies:
            for k in _METRICS_KEYS:
                avg[s][k] /= N_SEEDS
        averages.append(avg)

    # TABLE_METRICS = [
    #     ('makespan',  '.0f', 'Makespan'),
    #     ('penalty',   '.0f', 'Total Penalty'),
    #     ('lead_time', '.1f', 'Avg Active Lead Time'),
    #     ('time',      '.2f', 'Wall Time (s)'),
    # ]

    # for metric, fmt, metric_name in TABLE_METRICS:
    #     print(f"\n  {metric_name}")
    #     hdr = f"  {'Scale':<10}" + "".join(f" {s:>12}" for s in strategies)
    #     print(hdr)
    #     print("  " + "-" * (10 + 13 * len(strategies)))
    #     for idx, r in enumerate(averages):
    #         print(f"  {SCALES[idx][0]:<10}" + "".join(
    #             f" {r[s][metric]:>12{fmt}}" for s in strategies))

    # ── Charts ──
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np

    DISPLAY_NAMES = {'PA': 'Hierarchical-PA', 'LBU': 'Hierarchical-LBU',
                     'FIFO': 'FIFO', 'EDD': 'EDD', 'ATC': 'ATC',
                     'Clairvoyant': 'Clairvoyant'}

    n_scales = len(SCALES)
    x_pos = np.arange(n_scales)
    x_labels = [str(jc) for jc in job_counts]

    # ── line chart: 2×2 grid ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = axes.flatten()

    metrics_display = [
        ('makespan',  'Makespan'),
        ('penalty',   'Total Penalty'),
        ('lead_time', 'Avg Active Lead Time'),
        ('time',      'Wall Time (s)'),
    ]
    for col, (metric, metric_title) in enumerate(metrics_display):
        ax = axes_flat[col]
        for s in strategies:
            vals = [averages[i][s][metric] for i in range(n_scales)]
            ax.plot(x_pos, vals, marker='o', markersize=7,
                    color=COLORS[s], linewidth=2.2, alpha=0.9,
                    label=DISPLAY_NAMES[s])
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel('Number of Jobs', fontsize=9)
        ax.grid(alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=len(strategies),
               loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Average Metrics by Scale Level ({N_SEEDS} seeds)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(out_dir, 'scale_comparison_lines.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')

    # ── scatter: Wall Time × Penalty, 2×2 grid ──
    SCATTER_MARKERS = {'PA': 'P', 'LBU': 'D', 'FIFO': 's', 'EDD': 'D', 'ATC': '^',
                       'Clairvoyant': '*'}

    fig_s, axes_s = plt.subplots(2, 2, figsize=(14, 10))
    axes_s_flat = axes_s.flatten()
    for extra in range(n_scales, 4):
        axes_s_flat[extra].set_visible(False)

    for si, (ax, jc) in enumerate(zip(axes_s_flat, job_counts)):
        for s in strategies:
            xs = [all_raw[si][seed][s]['time'] for seed in range(N_SEEDS)]
            ys = [all_raw[si][seed][s]['penalty'] for seed in range(N_SEEDS)]
            ax.scatter(xs, ys, c=COLORS[s], marker=SCATTER_MARKERS[s],
                       label=DISPLAY_NAMES[s], s=50, edgecolors='white',
                       linewidth=0.3, alpha=0.85, zorder=3)
        ax.set_title(f'{jc} Jobs', fontsize=11, fontweight='bold')
        ax.set_xlabel('Wall Time (s)', fontsize=9)
        ax.set_ylabel('Total Penalty', fontsize=9)
        ax.grid(alpha=0.3)

    handles, labels = axes_s_flat[0].get_legend_handles_labels()
    fig_s.legend(handles, labels, fontsize=7, ncol=7,
                 loc='upper center', bbox_to_anchor=(0.5, 0.02))
    fig_s.suptitle('Time–Quality Trade-off: Wall Time vs Penalty (per seed)',
                   fontsize=13, fontweight='bold')
    fig_s.tight_layout()
    path_s = os.path.join(out_dir, 'scale_scatter_time_vs_penalty.png')
    fig_s.savefig(path_s, dpi=150, bbox_inches='tight')

    print(f"\nCharts saved to '{out_dir}/'")
    plt.close('all')


if __name__ == '__main__':
    main()
