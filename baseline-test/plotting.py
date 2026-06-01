"""Gantt chart plotting for baseline strategy comparison.

Auto-detects machine layout and job count from schedule data — no
hard-coded limits on machines or jobs.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

from models import ScheduleEntry

# Colour palette (cycles for >10 jobs)
_TAB20 = list(cm.tab20.colors) + list(cm.tab20b.colors)


def _job_color(job_id: int) -> str:
    return _TAB20[(job_id - 1) % len(_TAB20)]


def _machine_layout(schedule: List[ScheduleEntry]) -> tuple:
    """Return (y_labels, y_positions) auto-detected from *schedule*.

    Machines are grouped by service unit; units are ordered alphabetically.
    """
    machines = sorted({e.machine for e in schedule},
                      key=lambda m: (m.split('_')[1], m.split('_')[0]))
    units_order = sorted({e.service_unit for e in schedule})

    y_labels = []
    y_positions = {}
    unit_boundaries: List[int] = []  # y-indices where a new unit starts

    for unit in units_order:
        unit_start = len(y_labels)
        for m in machines:
            if m.endswith(f"_{unit}"):
                y_labels.append(f"{m}  ({unit})")
                y_positions[m] = len(y_labels) - 1
        if len(y_labels) > unit_start:
            unit_boundaries.append(len(y_labels) - 0.5)

    return y_labels, y_positions, unit_boundaries


def plot_gantt(
    schedule: List[ScheduleEntry],
    title: str = "",
    decision_times: Optional[List[float]] = None,
    disruption_times: Optional[List[float]] = None,
    ref_schedule: Optional[List[ScheduleEntry]] = None,
    ax: Optional[plt.Axes] = None,
):
    """Draw a single Gantt chart for *schedule* on *ax*.

    If *ref_schedule* is given, entries that match it (same job, op,
    machine and start time within 0.01) are drawn with diagonal hatching
    to indicate they follow the original plan.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    if not schedule:
        return

    # Build reference lookup
    ref_map: Dict[tuple, ScheduleEntry] = {}
    if ref_schedule:
        for e in ref_schedule:
            ref_map[(e.job_id, e.op_idx)] = e

    y_labels, y_positions, unit_boundaries = _machine_layout(schedule)

    for entry in schedule:
        y = y_positions[entry.machine]
        color = _job_color(entry.job_id)
        bar = ax.barh(y, entry.duration, left=entry.start_time, height=0.55,
                      color=color, edgecolor='white', linewidth=0.5, alpha=0.9)
        label = f"J{entry.job_id}-{entry.op_idx + 1}"
        ax.text(entry.start_time + entry.duration / 2, y,
                label, ha='center', va='center', fontsize=6,
                fontweight='bold', color='white')

        # Hatch entries that match the reference plan
        matched = False
        if ref_schedule:
            ref_entry = ref_map.get((entry.job_id, entry.op_idx))
            if ref_entry is not None:
                if (entry.machine == ref_entry.machine
                        and abs(entry.start_time - ref_entry.start_time) < 0.01):
                    matched = True
        if entry.fixed or matched:
            bar.patches[0].set_hatch('///')
            bar.patches[0].set_edgecolor('black')
            bar.patches[0].set_linewidth(0.8)

    for sep_y in unit_boundaries:
        ax.axhline(y=sep_y, color='black', linewidth=1.2, linestyle='-')

    ax.set_yticks(list(range(len(y_labels))))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel('Time', fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(left=0, right=max(e.end_time for e in schedule) * 1.08)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    if decision_times:
        for t in decision_times:
            ax.axvline(x=t, color='blue', linestyle='--', linewidth=1.0, alpha=0.5)
    if disruption_times:
        for t in disruption_times:
            ax.axvline(x=t, color='red', linestyle=':', linewidth=1.5, alpha=0.8)


def plot_comparison_grid(
    results: List[dict],
    disruption_times: Optional[List[float]] = None,
    decision_times: Optional[List[float]] = None,
    output_path: str = "gantt_comparison.png",
    dpi: int = 150,
):
    """Create a multi-panel Gantt chart comparing rescheduling strategies.

    The initial schedule is used as a reference for hatching but is not
    drawn as a separate subplot.  Entries that match the initial plan are
    shown with diagonal hatching.
    """
    # Extract initial schedule as reference for hatching
    ref_schedule: Optional[List[ScheduleEntry]] = None
    for r in results:
        if r['name'] == 'Initial Schedule':
            ref_schedule = r['schedule']
            break

    n = len(results)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             constrained_layout=True)

    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = list(axes)
    else:
        axes = list(axes.flat)

    # Collect all job IDs across all schedules for legend
    all_job_ids = sorted({e.job_id for r in results for e in r['schedule']})

    for idx, r in enumerate(results):
        ax = axes[idx]
        penalty = r['metrics']['total_penalty']
        makespan = r['metrics']['makespan']
        title = f"{r['name']}\npen={penalty:.0f}  Cmax={makespan:.0f}"
        # Pass ref_schedule to other strategies for hatching
        rf = ref_schedule if r['name'] != 'Initial Schedule' else None
        plot_gantt(r['schedule'], title=title,
                   decision_times=decision_times,
                   disruption_times=disruption_times,
                   ref_schedule=rf,
                   ax=ax)

    legend_patches = [
        mpatches.Patch(color=_job_color(jid), label=f"J{jid}")
        for jid in all_job_ids
    ]
    fig.legend(handles=legend_patches, loc='outside lower center',
               fontsize=9, ncol=min(8, len(all_job_ids)),
               title='Jobs', title_fontsize=10)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Dynamic Rescheduling Baseline Comparison',
                 fontsize=18, fontweight='bold')
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Comparison Gantt chart saved to '{output_path}'")


