"""Gantt chart plotting for Hierarchical FJSP schedules."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional
from models import ScheduleEntry
from config import SERVICE_UNITS

JOB_COLORS = {
    1: '#4C72B0',
    2: '#DD8452',
    3: '#55A868',
    4: '#C44E52',
}

JOB_LABELS = {1: 'J1', 2: 'J2', 3: 'J3', 4: 'J4'}


def plot_gantt(schedule: List[ScheduleEntry],
               title: str,
               current_time: Optional[float] = None,
               ax: Optional[plt.Axes] = None):
    """Draw a Gantt chart for the given schedule on the provided axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    y_labels = []
    y_positions = {}
    y_idx = 0

    for unit in ['U1', 'U2']:
        for m in SERVICE_UNITS[unit]:
            y_labels.append(f"{m}  ({unit})")
            y_positions[m] = y_idx
            y_idx += 1

    for entry in schedule:
        y = y_positions[entry.machine]
        color = JOB_COLORS.get(entry.job_id, '#888888')

        bar = ax.barh(y, entry.duration, left=entry.start_time, height=0.55,
                      color=color, edgecolor='white', linewidth=0.5, alpha=0.9)

        label = f"{JOB_LABELS.get(entry.job_id, entry.job_id)}-{entry.op_idx + 1}"
        ax.text(entry.start_time + entry.duration / 2, y,
                label, ha='center', va='center', fontsize=7,
                fontweight='bold', color='white')

        if entry.fixed:
            bar.patches[0].set_hatch('///')
            bar.patches[0].set_edgecolor('black')
            bar.patches[0].set_linewidth(0.8)

    if len(SERVICE_UNITS) > 1:
        sep_y = len(SERVICE_UNITS['U1']) - 0.5
        ax.axhline(y=sep_y, color='black', linewidth=1.5, linestyle='-')

    u1_mid = (0 + len(SERVICE_UNITS['U1']) - 1) / 2
    u2_mid = len(SERVICE_UNITS['U1']) + (0 + len(SERVICE_UNITS['U2']) - 1) / 2
    for unit, mid_y in [('U1', u1_mid), ('U2', u2_mid)]:
        ax.text(1.01, mid_y / len(y_positions), f'Service\nUnit {unit}',
                transform=ax.transAxes, ha='left', va='center',
                fontsize=9, fontweight='bold', color='#555555')

    ax.set_yticks(list(range(len(y_labels))))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('Machine (Service Unit)', fontsize=11)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    if schedule:
        ax.set_xlim(left=0, right=max(e.end_time for e in schedule) * 1.08)
    ax.xaxis.set_major_locator(plt.MultipleLocator(24))
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    if current_time is not None:
        ax.axvline(x=current_time, color='red', linestyle='--', linewidth=1.8,
                   alpha=0.7, label=f'Decision point  t = {current_time}')
        ax.legend(loc='upper right', fontsize=9)

    legend_patches = [mpatches.Patch(color=JOB_COLORS[jid],
                                      label=f"{JOB_LABELS[jid]} (Job {jid})")
                      for jid in sorted(JOB_COLORS)]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8,
              ncol=4, title='Jobs', title_fontsize=9)
