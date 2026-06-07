"""Gantt chart plotting for Kacem FJSP schedules."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
from typing import List, Optional, Set
from models import ScheduleEntry

# ── Service unit layout ──────────────────────────────────────────────────
SERVICE_UNITS = {
    'U1': ['M1', 'M2', 'M3', 'M4'],
    'U2': ['M5', 'M6', 'M7', 'M8'],
}

# All machines in display order (U1 top, then U2)
Y_MACHINES = SERVICE_UNITS['U1'] + SERVICE_UNITS['U2']


def job_color(job_id: int):
    """Return a distinct colour for each job (1-based)."""
    cmap = get_cmap('tab20')
    return cmap((job_id - 1) % 20)


def job_label(job_id: int) -> str:
    return f"J{job_id}"


def plot_gantt(schedule: List[ScheduleEntry],
               title: str,
               current_time: Optional[float] = None,
               highlight_jobs: Optional[Set[int]] = None,
               show_legend: bool = True,
               ax: Optional[plt.Axes] = None):
    """Draw a Gantt chart on *ax* (or create a new figure).

    Parameters
    ----------
    schedule : list of ScheduleEntry
    title : str
    current_time : float or None
        Draw a red dashed vertical line at this time if given.
    highlight_jobs : set of int or None
        Job IDs that should be visually highlighted (e.g. newly arrived jobs).
    show_legend : bool
        If False, suppress the per-panel job legend.
    ax : matplotlib Axes or None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 7))

    if highlight_jobs is None:
        highlight_jobs = set()

    # Build y-position map (fixed by SERVICE_UNITS layout)
    y_positions = {m: i for i, m in enumerate(Y_MACHINES)}

    for entry in schedule:
        y = y_positions[entry.machine]
        base_color = job_color(entry.job_id)
        is_hl = entry.job_id in highlight_jobs

        # Highlighted bars: full opacity, thick border, slightly raised
        if is_hl:
            bar = ax.barh(y, entry.duration, left=entry.start_time, height=0.60,
                          color=base_color, edgecolor='black', linewidth=2.2,
                          alpha=1.0, zorder=3)
        else:
            bar = ax.barh(y, entry.duration, left=entry.start_time, height=0.55,
                          color=base_color, edgecolor='white', linewidth=0.5, alpha=0.85)

        label = f"{job_label(entry.job_id)}-{entry.op_idx + 1}"
        ax.text(entry.start_time + entry.duration / 2, y,
                label, ha='center', va='center', fontsize=6.5,
                fontweight='bold', color='white')

        if entry.fixed:
            bar.patches[0].set_hatch('///')
            bar.patches[0].set_edgecolor('black')
            bar.patches[0].set_linewidth(0.8)

    # ── Separator between U1 and U2 ──────────────────────────────────────
    sep_y = len(SERVICE_UNITS['U1']) - 0.5
    ax.axhline(y=sep_y, color='black', linewidth=1.8, linestyle='-')

    # ── Service unit labels on the right ─────────────────────────────────
    u1_mid = (0 + len(SERVICE_UNITS['U1']) - 1) / 2
    u2_mid = len(SERVICE_UNITS['U1']) + (0 + len(SERVICE_UNITS['U2']) - 1) / 2
    for unit, mid_y in [('U1', u1_mid), ('U2', u2_mid)]:
        ax.text(1.01, (len(Y_MACHINES) - 0.5 - mid_y) / len(Y_MACHINES),
                f'Service\nUnit {unit}',
                transform=ax.transAxes, ha='left', va='center',
                fontsize=9, fontweight='bold', color='#555555')

    # ── Axis labels & ticks ──────────────────────────────────────────────
    y_labels = [f"{m}  ({'U1' if m in SERVICE_UNITS['U1'] else 'U2'})"
                for m in Y_MACHINES]
    ax.set_yticks(list(range(len(y_labels))))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('Machine (Service Unit)', fontsize=11)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    if schedule:
        ax.set_xlim(left=0, right=max(e.end_time for e in schedule) * 1.08)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # ── Decision-point marker ────────────────────────────────────────────
    if current_time is not None:
        ax.axvline(x=current_time, color='red', linestyle='--', linewidth=1.8,
                   alpha=0.7, label=f'Decision point  t = {current_time}')
        ax.legend(loc='upper right', fontsize=9)

    # ── Per-panel legend (can be suppressed for a shared figure legend) ──
    if show_legend:
        job_ids = sorted({e.job_id for e in schedule})
        legend_patches = []
        for jid in job_ids:
            label = f"{job_label(jid)} (Job {jid})"
            if jid in highlight_jobs:
                legend_patches.append(mpatches.Patch(
                    facecolor=job_color(jid), label=label,
                    edgecolor='black', linewidth=2.0))
            else:
                legend_patches.append(mpatches.Patch(
                    facecolor=job_color(jid), label=label))
        ax.legend(handles=legend_patches, loc='upper left', fontsize=7.5,
                  ncol=min(len(job_ids), 5), title='Jobs', title_fontsize=9)
