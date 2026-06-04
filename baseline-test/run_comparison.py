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
from metrics import compute_metrics, compute_resilience
from sim_engine import SimulationEngine
from strategies import PureRuleStrategy, ClairvoyantMILPStrategy

# ═══════════════════════════════════════════════════════════════════════════════
# sweep configuration
# ═══════════════════════════════════════════════════════════════════════════════

N_SEEDS = 1
START_SEED = 45

PURE_RULES = ['FIFO', 'EDD', 'SPT', 'WINQ', 'PA']
CLAIRVOYANT_BASELINE = ['Clairvoyant']

COLORS = {'FIFO': '#95A5A6', 'EDD': '#F39C12', 'SPT': '#1ABC9C',
          'WINQ': '#E67E22', 'PA': '#27AE60',
          'Clairvoyant': '#2357C7'}

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

CLAIRVOYANT_TIME_LIMIT = 30  # one-shot offline-optimal baseline

# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

_METRICS_KEYS = [
    # Time-based
    'makespan', 'total_flow_time', 'total_waiting_time', 'avg_flow_time',
    'machine_utilization',
    # Quality-based
    'total_penalty', 'max_tardiness', 'avg_tardiness', 'max_earliness',
    'avg_earliness',
    # Resilience-based
    'schedule_deviation', 'sequence_deviation',
    # Computation
    'time',
]

def _run_one_seed(jobs, machines, disruptions):
    """Run one seed across all strategies, return per-strategy metrics.

    Clairvoyant runs first and serves as the reference schedule for
    resilience metrics of the pure dispatching rules.
    """
    results: dict[str, dict] = {}

    # ── Clairvoyant (reference for resilience) ───────────────────────
    t0 = perf_counter()
    engine = SimulationEngine(jobs, machines,
                              ClairvoyantMILPStrategy(time_limit=CLAIRVOYANT_TIME_LIMIT),
                              disruptions, idle_timeout=0.01)
    ref_schedule = engine.run()
    ref_m = compute_metrics(ref_schedule, jobs)
    results['Clairvoyant'] = _pack_result(ref_m, schedule_deviation=0.0,
                                          sequence_deviation=0.0,
                                          elapsed=perf_counter() - t0)

    # ── Pure dispatching rules ───────────────────────────────────────
    for rule in PURE_RULES:
        t0 = perf_counter()
        engine = SimulationEngine(jobs, machines,
                                  PureRuleStrategy(rule=rule),
                                  disruptions)
        schedule = engine.run()
        m = compute_metrics(schedule, jobs)
        r = compute_resilience(schedule, ref_schedule)
        results[rule] = _pack_result(m,
                                     schedule_deviation=r['schedule_deviation'],
                                     sequence_deviation=r['sequence_deviation'],
                                     elapsed=perf_counter() - t0)

    return results


def _pack_result(m: dict, schedule_deviation: float,
                 sequence_deviation: float, elapsed: float) -> dict:
    """Unpack compute_metrics output into a flat dict for CSV / averages."""
    return {
        # Time-based
        'makespan':            m['makespan'],
        'total_flow_time':     m['total_flow_time'],
        'total_waiting_time':  m['total_waiting_time'],
        'avg_flow_time':       m['avg_flow_time'],
        'machine_utilization': m['machine_utilization'],
        # Quality-based
        'total_penalty': m['total_penalty'],
        'max_tardiness': m['max_tardiness'],
        'avg_tardiness': m['avg_tardiness'],
        'max_earliness': m['max_earliness'],
        'avg_earliness': m['avg_earliness'],
        # Resilience
        'schedule_deviation': schedule_deviation,
        'sequence_deviation': sequence_deviation,
        # Computation
        'time': elapsed,
    }


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
               'makespan', 'total_flow_time', 'total_waiting_time',
               'avg_flow_time', 'machine_utilization',
               'total_penalty', 'max_tardiness', 'avg_tardiness',
               'max_earliness', 'avg_earliness',
               'schedule_deviation', 'sequence_deviation', 'time']


def main():
    strategies = PURE_RULES + CLAIRVOYANT_BASELINE
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
                        'total_flow_time': f"{r['total_flow_time']:.2f}",
                        'total_waiting_time': f"{r['total_waiting_time']:.2f}",
                        'avg_flow_time': f"{r['avg_flow_time']:.2f}",
                        'machine_utilization': f"{r['machine_utilization']:.4f}",
                        'total_penalty': f"{r['total_penalty']:.2f}",
                        'max_tardiness': f"{r['max_tardiness']:.2f}",
                        'avg_tardiness': f"{r['avg_tardiness']:.2f}",
                        'max_earliness': f"{r['max_earliness']:.2f}",
                        'avg_earliness': f"{r['avg_earliness']:.2f}",
                        'schedule_deviation': f"{r['schedule_deviation']:.4f}",
                        'sequence_deviation': f"{r['sequence_deviation']:.4f}",
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

    DISPLAY_NAMES = {'FIFO': 'FIFO', 'EDD': 'EDD', 'SPT': 'SPT',
                     'WINQ': 'WINQ', 'PA': 'PA',
                     'Clairvoyant': 'Clairvoyant'}

    n_scales = len(SCALES)
    x_pos = np.arange(n_scales)
    x_labels = [str(jc) for jc in job_counts]

    # ── helper: one subplot per metric ──────────────────────────────────
    def _draw_grid(axes_flat, metric_list, strategies, averages,
                   n_scales, x_pos, x_labels):
        """Fill a grid of axes with line plots, one metric per cell."""
        for col, (metric, metric_title) in enumerate(metric_list):
            if col >= len(axes_flat):
                break
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
        # Hide unused cells
        for extra in range(len(metric_list), len(axes_flat)):
            axes_flat[extra].set_visible(False)

    # ═══════════════════════════════════════════════════════════════════
    # Figure 1 — Time-based metrics (2×3 grid, 5 panels)
    # ═══════════════════════════════════════════════════════════════════
    TIME_METRICS = [
        ('makespan',            'Makespan'),
        ('total_flow_time',     'Total Flow Time'),
        ('total_waiting_time',  'Total Waiting Time'),
        ('avg_flow_time',       'Avg Flow Time'),
        ('machine_utilization', 'Machine Utilization (∑p / Cmax)'),
    ]

    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    _draw_grid(axes1.flatten(), TIME_METRICS, strategies, averages,
               n_scales, x_pos, x_labels)
    handles, labels = axes1[0, 0].get_legend_handles_labels()
    fig1.legend(handles, labels, fontsize=8, ncol=len(strategies),
                loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig1.suptitle('Time-Based Metrics', fontsize=14, fontweight='bold')
    fig1.tight_layout(rect=[0, 0.06, 1, 1])
    fig1.savefig(os.path.join(out_dir, 'fig1_time_based.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ═══════════════════════════════════════════════════════════════════
    # Figure 2 — Quality-based metrics (2×3 grid, 5 panels)
    # ═══════════════════════════════════════════════════════════════════
    QUALITY_METRICS = [
        ('max_tardiness', 'Max Tardiness'),
        ('avg_tardiness', 'Avg Tardiness'),
        ('max_earliness', 'Max Earliness'),
        ('avg_earliness', 'Avg Earliness'),
        ('total_penalty', 'Total Penalty'),
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    _draw_grid(axes2.flatten(), QUALITY_METRICS, strategies, averages,
               n_scales, x_pos, x_labels)
    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, fontsize=8, ncol=len(strategies),
                loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle('Quality-Based Metrics', fontsize=14, fontweight='bold')
    fig2.tight_layout(rect=[0, 0.06, 1, 1])
    fig2.savefig(os.path.join(out_dir, 'fig2_quality_based.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # ═══════════════════════════════════════════════════════════════════
    # Figure 3 — Resilience metrics (1×2 grid, 2 panels)
    # ═══════════════════════════════════════════════════════════════════
    RESILIENCE_METRICS = [
        ('schedule_deviation', 'Schedule Deviation (unit changes / n_ops)'),
        ('sequence_deviation', 'Sequence Deviation (pairwise order reversals)'),
    ]

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    _draw_grid(axes3.flatten(), RESILIENCE_METRICS, strategies,
               averages, n_scales, x_pos, x_labels)
    handles, labels = axes3[0].get_legend_handles_labels()
    fig3.legend(handles, labels, fontsize=8, ncol=len(strategies),
                loc='lower center', bbox_to_anchor=(0.5, -0.04))
    fig3.suptitle('Resilience Metrics  (reference = Clairvoyant)',
                  fontsize=14, fontweight='bold')
    fig3.tight_layout(rect=[0, 0.08, 1, 1])
    fig3.savefig(os.path.join(out_dir, 'fig3_resilience.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # ═══════════════════════════════════════════════════════════════════
    # Figure 4 — Computation efficiency (bar chart, single panel)
    # ═══════════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    bar_width = 0.12
    for i, s in enumerate(strategies):
        vals = [averages[si][s]['time'] for si in range(n_scales)]
        offset = (i - len(strategies) / 2 + 0.5) * bar_width
        ax4.bar(x_pos + offset, vals, bar_width,
                color=COLORS[s], label=DISPLAY_NAMES[s], alpha=0.9)

    ax4.set_title('Computation Time', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_labels, fontsize=10)
    ax4.set_xlabel('Number of Jobs', fontsize=10)
    ax4.set_ylabel('Wall Time (s)', fontsize=10)
    ax4.legend(fontsize=8, ncol=len(strategies))
    ax4.grid(alpha=0.3, axis='y')
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, 'fig4_computation.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig4)

    print(f"\nCharts saved to '{out_dir}/'")
    plt.close('all')


if __name__ == '__main__':
    main()
