"""Main entry point for Hierarchical FJSP simulation.

Runs Experiment 1 (urgent order insertion) and Experiment 2 (machine disruption),
saves results to CSV/JSON files, and displays Gantt charts interactively.

Usage::

    python  case_1/main.py          # run as script
    python -m case_1.main           # run as module
"""

import sys
import os

# Support both ``python case_1/main.py`` and ``python -m case_1.main``
if __package__ is None:
    _PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJ not in sys.path:
        sys.path.insert(0, _PROJ)
    __package__ = 'case_1'

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cases import build_jobs
from .simulation import simulate_dynamic_arrival, simulate_machine_disruption
from .metrics import compute_metrics
from .plotting import plot_gantt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(('.csv', '.png')):
            os.remove(os.path.join(OUTPUT_DIR, f))

    all_jobs = build_jobs()
    jobs_t0 = [j for j in all_jobs if j.arrival_time <= 0]
    jobs_t24 = [j for j in all_jobs if j.arrival_time <= 24]

    # ═════════════════════════════════════════════════════════════════════
    # Experiment 1: Urgent Order Insertion (J4 arrives at t=24)
    # ═════════════════════════════════════════════════════════════════════
    print("Running Experiment 1: Urgent Order Insertion")
    exp1_t0, exp1_t24 = simulate_dynamic_arrival(all_jobs)

    if exp1_t0 is None or exp1_t24 is None:
        print("[Gurobi not available -- MILP solver required]")
        return

    # ═════════════════════════════════════════════════════════════════════
    # Experiment 2: Machine Disruption (continues from Exp 1 t=24 result)
    # ═════════════════════════════════════════════════════════════════════
    print("Running Experiment 2: Machine Processing Time Disruption...")
    disruption_time = 56.0
    exp2_schedule = simulate_machine_disruption(
        all_jobs, exp1_t24,
        disruption_time=disruption_time, disrupted_machine='M1_U2', factor=2.0)

    if exp2_schedule is None:
        print("[Experiment 2: MILP infeasible or Gurobi error]")
        return

    # Write combined results
    _write_combined_csv(OUTPUT_DIR, exp1_t0, jobs_t0, exp1_t24, jobs_t24,
                        exp2_schedule, jobs_t24)

    # ═════════════════════════════════════════════════════════════════════
    # 3-Panel Gantt Chart (interactive display)
    # ═════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16))

    x_max = max(
        max(e.end_time for e in exp1_t0),
        max(e.end_time for e in exp1_t24),
        max(e.end_time for e in exp2_schedule),
    ) * 1.08

    plot_gantt(exp1_t0, 'Exp 1 — Initial Schedule (t=0)', ax=ax1)
    ax1.set_xlim(0, x_max)

    plot_gantt(exp1_t24, 'Exp 1 — After Urgent J4 Insertion (t=24)',
               current_time=24, ax=ax2)
    ax2.set_xlim(0, x_max)

    plot_gantt(exp2_schedule,
               f'Exp 2 — After M1_U2 Disruption (t={disruption_time}, proc.time x2)',
               current_time=disruption_time, ax=ax3)
    ax3.set_xlim(0, x_max)

    fig.suptitle('Hierarchical FJSP — Urgent Order (J4) & Machine Disruption (M1_U2)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(pad=3.0)

    gantt_path = os.path.join(OUTPUT_DIR, "gantt_chart.png")
    fig.savefig(gantt_path, dpi=150, bbox_inches='tight')
    print(f"Gantt chart saved to '{gantt_path}'")


def _write_combined_csv(out_dir, s1, j1, s2, j2, s3, j3):
    """Write all three experiment results into a single case_results.csv."""
    filepath = os.path.join(out_dir, "case_results.csv")

    def _write_section(writer, title, schedule, jobs):
        m = compute_metrics(schedule, jobs)
        writer.writerow([title])
        writer.writerow(['job_id', 'completion_Cj', 'due_date_dj',
                         'earliness_Ej', 'tardiness_Tj', 'penalty'])
        for d in m['details']:
            writer.writerow([
                f"J{d['job_id']}", f"{d['completion']:.1f}",
                f"{d['due_date']:.1f}", f"{d['earliness']:.1f}",
                f"{d['tardiness']:.1f}", f"{d['penalty']:.1f}",
            ])
        writer.writerow([])
        writer.writerow(['total_penalty', f"{m['total_penalty']:.1f}"])
        writer.writerow(['total_active_lead_time',
                         f"{m['total_active_lead_time']:.1f}"])
        writer.writerow([])
        writer.writerow(['job_id', 'op_idx', 'machine', 'service_unit',
                         'start_time', 'end_time', 'duration', 'status'])
        for e in sorted(schedule, key=lambda e: (e.job_id, e.op_idx)):
            writer.writerow([
                f"J{e.job_id}", e.op_idx + 1, e.machine, e.service_unit,
                f"{e.start_time:.1f}", f"{e.end_time:.1f}",
                f"{e.duration:.1f}", 'FIXED' if e.fixed else 'planned',
            ])
        writer.writerow([])

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        _write_section(writer, '=== Exp1: Initial Schedule (t=0) ===', s1, j1)
        _write_section(writer, '=== Exp1: After J4 Insertion (t=24) ===', s2, j2)
        _write_section(writer, '=== Exp2: After M1_U2 Disruption (t=60) ===', s3, j3)

    print(f"Combined results saved to '{filepath}'")


if __name__ == '__main__':
    main()
