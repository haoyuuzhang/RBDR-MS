"""Compare 7 strategies across 4 scale levels × 20 seeds each.

MILP | Hierarchical (ECT) | Pure Rules (FIFO/EDD/ATC) | RightShift | Periodic
"""

from __future__ import annotations

import sys
import os
from time import perf_counter

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from case_generator import generate_case
from metrics import compute_metrics
from sim_engine import SimulationEngine
from strategies import (FullMILPStrategy, HierarchicalStrategy,
                         PureRuleStrategy, RightShiftStrategy)

# ═══════════════════════════════════════════════════════════════════════════════
# sweep configuration
# ═══════════════════════════════════════════════════════════════════════════════

N_SEEDS = 10
START_SEED = 41

HIER_RULES = ['ECT']
PURE_RULES = ['FIFO', 'EDD', 'ATC']
EXTRA_BASELINES = ['RightShift']
ALL_RULES = HIER_RULES + PURE_RULES + EXTRA_BASELINES

COLORS = {'MILP': '#3498DB', 'ECT': '#E74C3C',
          'FIFO': '#95A5A6', 'EDD': '#F39C12', 'ATC': '#1ABC9C',
          'RightShift': '#E67E22'}

# 4 paired scale levels: (label, service_units, n_initial, n_dynamic)
SCALES: list[tuple[str, dict[str, list[str]], int, int]] = [
    # ── Scale 1:  2U,  5M,   8J  (ratio 1.6) ──
    ('Small\n2U-5M / 8J', {
        'U1': ['M1_U1', 'M2_U1', 'M3_U1'],
        'U2': ['M1_U2', 'M2_U2'],
    }, 5, 3),

    # ── Scale 2:  3U,  8M,  15J  (ratio 1.9) ──
    ('Medium\n3U-8M / 10J', {
        'U1': ['M1_U1', 'M2_U1', 'M3_U1'],
        'U2': ['M1_U2', 'M2_U2', 'M3_U2'],
        'U3': ['M1_U3', 'M2_U3'],
    }, 6, 4),

    # ── Scale 3:  4U, 12M,  24J  (ratio 2.0) ──
    ('Large\n4U-12M / 16J', {
        'U1': ['M1_U1', 'M2_U1', 'M3_U1'],
        'U2': ['M1_U2', 'M2_U2', 'M3_U2'],
        'U3': ['M1_U3', 'M2_U3', 'M3_U3'],
        'U4': ['M1_U4', 'M2_U4', 'M3_U4'],
    }, 10, 6),

    # ── Scale 4:  5U, 17M,  35J  (ratio 2.1) ──
    ('Large\n5U-17M / 25J', {
        'U1': ['M1_U1', 'M2_U1', 'M3_U1', 'M4_U1'],
        'U2': ['M1_U2', 'M2_U2', 'M3_U2', 'M4_U2'],
        'U3': ['M1_U3', 'M2_U3', 'M3_U3'],
        'U4': ['M1_U4', 'M2_U4', 'M4_U4'],
        'U5': ['M2_U5', 'M3_U5', 'M4_U5'],
    }, 16, 9),
]

MILP_TIME_LIMIT = 60
HIER_UNIT_LIMIT = 30
HIER_INIT_LIMIT = 60

# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

_METRICS_KEYS = ['makespan', 'penalty', 'lead_time', 'time']

def _run_one_seed(jobs, machines, disruptions):
    """Run one seed across all 5 strategies, return per-strategy metrics."""
    results: dict[str, dict] = {}

    t0 = perf_counter()
    engine = SimulationEngine(jobs, machines,
                              FullMILPStrategy(time_limit=MILP_TIME_LIMIT),
                              disruptions)
    m = compute_metrics(engine.run(), jobs)
    n_jobs = len(jobs)
    results['MILP'] = {
        'makespan': m['makespan'],
        'penalty': m['total_penalty'],
        'lead_time': m['total_active_lead_time'] / n_jobs if n_jobs else 0,
        'time': perf_counter() - t0,
    }

    for rule in HIER_RULES:
        t0 = perf_counter()
        engine = SimulationEngine(jobs, machines,
                                  HierarchicalStrategy(
                                      shop_rule=rule,
                                      unit_time_limit=HIER_UNIT_LIMIT,
                                      init_time_limit=HIER_INIT_LIMIT),
                                  disruptions)
        m = compute_metrics(engine.run(), jobs)
        results[rule] = {
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

    # Right-Shift
    t0 = perf_counter()
    engine = SimulationEngine(jobs, machines,
                              RightShiftStrategy(init_time_limit=MILP_TIME_LIMIT),
                              disruptions)
    m = compute_metrics(engine.run(), jobs)
    results['RightShift'] = {
        'makespan': m['makespan'],
        'penalty': m['total_penalty'],
        'lead_time': m['total_active_lead_time'] / n_jobs if n_jobs else 0,
        'time': perf_counter() - t0,
    }

    return results


def _run_scale_experiment(label, m_units, n_init, n_dyn):
    """Run N_SEEDS for one scale level, return (averages, per_seed_results)."""
    n_machines = sum(len(v) for v in m_units.values())

    accum = {s: {k: 0.0 for k in _METRICS_KEYS} for s in ['MILP'] + ALL_RULES}
    all_seeds: list[dict] = []

    print(f"\n{'='*70}")
    print(f"  {label.replace(chr(10), ' ')}  "
          f"({n_machines} machines, {n_init + n_dyn} jobs)")
    print(f"{'='*70}")

    for i in range(N_SEEDS):
        seed = START_SEED + i
        jobs, machines, disruptions = generate_case(
            seed=seed, n_initial=n_init, n_dynamic=n_dyn,
            ops_per_job=3, time_horizon=200.0,
            service_units=m_units,
        )
        r = _run_one_seed(jobs, machines, disruptions)
        all_seeds.append(r)

        parts = []
        for s in ['MILP'] + ALL_RULES:
            for k in _METRICS_KEYS:
                accum[s][k] += r[s][k]
            parts.append(f"{s} m={r[s]['makespan']:.0f} p={r[s]['penalty']:.0f}")

        print(f"  seed={seed:<4}  " + "  |  ".join(parts))

    for s in accum:
        for k in accum[s]:
            accum[s][k] /= N_SEEDS

    return accum, all_seeds

# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    strategies = ['MILP'] + ALL_RULES
    DISPLAY_NAMES = {'MILP': 'MILP', 'ECT': 'Hierarchical',
                     'FIFO': 'FIFO', 'EDD': 'EDD', 'ATC': 'ATC',
                     'RightShift': 'RightShift'}
    scale_labels_full = [s[0] for s in SCALES]
    job_counts = [n_init + n_dyn for _, _, n_init, n_dyn in SCALES]
    x_labels = [str(jc) for jc in job_counts]

    results: list[dict] = []
    all_raw: list[list[dict]] = []   # all_raw[scale_idx][seed_idx][strategy][metric]

    t_start = perf_counter()
    for label, m_units, n_init, n_dyn in SCALES:
        avg, raw = _run_scale_experiment(label, m_units, n_init, n_dyn)
        results.append(avg)
        all_raw.append(raw)

    total_time = perf_counter() - t_start

    # ═══════════════════════════════════════════════════════════════════════════
    # summary tables
    # ═══════════════════════════════════════════════════════════════════════════

    TABLE_METRICS = [
        ('makespan',  '.0f', 'Makespan'),
        ('penalty',   '.0f', 'Total Penalty'),
        ('lead_time', '.1f', 'Avg Active Lead Time'),
        ('time',      '.2f', 'Wall Time (s)'),
    ]

    for metric, fmt, metric_name in TABLE_METRICS:
        print(f"\n{'─'*80}")
        print(f"  Average {metric_name}")
        print(f"{'─'*80}")
        hdr = f"  {'Scale':<22}" + "".join(f" {s:>12}" for s in strategies)
        print(hdr)
        print("  " + "-" * (22 + 13 * len(strategies)))
        for idx, r in enumerate(results):
            label_flat = scale_labels_full[idx].replace('\n', ' ')
            row = f"  {label_flat:<22}" + "".join(
                f" {r[s][metric]:>12{fmt}}" for s in strategies)
            print(row)

    # ═══════════════════════════════════════════════════════════════════════════
    # charts
    # ═══════════════════════════════════════════════════════════════════════════

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    metrics_display = [
        ('makespan',  'Makespan'),
        ('penalty',   'Total Penalty'),
        ('lead_time', 'Avg Active Lead Time'),
        ('time',      'Wall Time (s)'),
    ]
    n_scales = len(SCALES)
    x_pos = np.arange(n_scales)

    # ── line chart: 2×2 grid ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = axes.flatten()

    for col, (metric, metric_title) in enumerate(metrics_display):
        ax = axes_flat[col]
        for s in strategies:
            vals = [results[i][s][metric] for i in range(n_scales)]
            ax.plot(x_pos, vals, marker='o', markersize=7,
                    color=COLORS[s], linewidth=2.2, alpha=0.9,
                    label=DISPLAY_NAMES[s])
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel('Number of Jobs', fontsize=9)
        ax.grid(alpha=0.3)
        if col == 3:
            ax.legend(fontsize=7, ncol=7, loc='upper left',
                      bbox_to_anchor=(1.01, 1.0))

    fig.suptitle('Average Metrics by Scale Level (20 seeds)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'scale_comparison_lines.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to '{out_path}'")

    # ── scatter: Wall Time × Penalty, 2×2 grid ──
    SCATTER_MARKERS = {'MILP': 'X', 'ECT': 'o', 'FIFO': 's', 'EDD': 'D', 'ATC': '^',
                       'RightShift': 'P'}

    fig_s, axes_s = plt.subplots(2, 2, figsize=(14, 10))
    axes_s_flat = axes_s.flatten()
    # hide unused subplot if n_scales < 4
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

    # single legend at top-right of figure
    handles, labels = axes_s_flat[0].get_legend_handles_labels()
    fig_s.legend(handles, labels, fontsize=7, ncol=7,
                 loc='upper center', bbox_to_anchor=(0.5, 0.02))

    fig_s.suptitle('Time–Quality Trade-off: Wall Time vs Penalty (per seed)',
                   fontsize=13, fontweight='bold')
    fig_s.tight_layout()
    path_s = os.path.join(out_dir, 'scale_scatter_time_vs_penalty.png')
    fig_s.savefig(path_s, dpi=150, bbox_inches='tight')
    print(f"Scatter chart saved to '{path_s}'")

    print(f"\nTotal wall time: {total_time:.0f} s")
    plt.show()


if __name__ == '__main__':
    main()
