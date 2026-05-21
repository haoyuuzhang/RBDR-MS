"""Metrics computation for Hierarchical FJSP schedules."""

from typing import Dict, List
from models import Job, ScheduleEntry


def compute_metrics(schedule: List[ScheduleEntry], jobs: List[Job]) -> dict:
    """Compute completion times, earliness/tardiness penalties, and active lead time."""
    completion: Dict[int, float] = {}
    for e in schedule:
        jid = e.job_id
        if jid not in completion or e.end_time > completion[jid]:
            completion[jid] = e.end_time

    total_penalty = 0.0
    total_active_lead_time = 0.0
    details = []

    for job in jobs:
        Cj = completion.get(job.job_id, 0)
        Ej = max(0, job.due_date - Cj)
        Tj = max(0, Cj - job.due_date)
        penalty = job.alpha * Ej + job.beta * Tj
        total_penalty += penalty
        first_start = min((e.start_time for e in schedule
                           if e.job_id == job.job_id), default=0)
        total_active_lead_time += Cj - first_start
        details.append({
            'job_id': job.job_id,
            'completion': Cj,
            'due_date': job.due_date,
            'earliness': Ej,
            'tardiness': Tj,
            'penalty': penalty,
        })

    return {
        'completion': completion,
        'total_penalty': total_penalty,
        'total_active_lead_time': total_active_lead_time,
        'details': details,
    }
