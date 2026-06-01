"""Save schedule and metrics results to a combined CSV file."""

import csv
import os
from typing import List
from .models import Job, ScheduleEntry
from .metrics import compute_metrics


def save_results_csv(schedule: List[ScheduleEntry], jobs: List[Job],
                     filepath: str):
    """Save metrics and schedule together in a single CSV file.

    The file has two sections separated by a blank line:
      1. Per-job metrics + summary (total_penalty, total_active_lead_time)
      2. Schedule entries
    """
    m = compute_metrics(schedule, jobs)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # ── Metrics section ─────────────────────────────────────────────
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

        # ── Schedule section ────────────────────────────────────────────
        writer.writerow(['job_id', 'op_idx', 'machine', 'service_unit',
                         'start_time', 'end_time', 'duration', 'status'])
        for e in sorted(schedule, key=lambda e: (e.job_id, e.op_idx)):
            writer.writerow([
                f"J{e.job_id}", e.op_idx + 1, e.machine, e.service_unit,
                f"{e.start_time:.1f}", f"{e.end_time:.1f}",
                f"{e.duration:.1f}", 'FIXED' if e.fixed else 'planned',
            ])
