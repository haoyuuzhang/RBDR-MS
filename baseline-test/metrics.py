"""Metrics computation for schedule evaluation."""

from typing import Dict, List
from models import Job, ScheduleEntry


def compute_metrics(schedule: List[ScheduleEntry], jobs: List[Job]) -> dict:
    """Compute completion times, earliness/tardiness penalties, and makespan."""
    completion: Dict[int, float] = {}
    for e in schedule:
        jid = e.job_id
        if jid not in completion or e.end_time > completion[jid]:
            completion[jid] = e.end_time

    total_penalty = 0.0
    total_active_lead_time = 0.0
    makespan = 0.0
    details = []

    for job in jobs:
        Cj = completion.get(job.job_id, 0)
        Ej = max(0, job.due_date - Cj)
        Tj = max(0, Cj - job.due_date)
        penalty = job.alpha * Ej + job.beta * Tj
        total_penalty += penalty
        if Cj > makespan:
            makespan = Cj

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
        'makespan': makespan,
        'details': details,
    }


def print_comparison_table(results: List[dict]):
    """Pretty-print a comparison table for multiple strategies."""
    header = f"{'Strategy':<28} {'Penalty':>10} {'Makespan':>10} {'Tardiness':>10} {'LeadTime':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        tardiness = sum(d['tardiness'] for d in r['metrics']['details'])
        print(f"{r['name']:<28} {r['metrics']['total_penalty']:>10.1f} "
              f"{r['metrics']['makespan']:>10.1f} {tardiness:>10.1f} "
              f"{r['metrics']['total_active_lead_time']:>10.1f}")
    print(sep)
