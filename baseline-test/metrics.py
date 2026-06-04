"""Metrics computation for schedule evaluation.

Provides two functions:
- ``compute_metrics``     — time-based, quality-based, and efficiency stats
- ``compute_resilience``  — deviation from a reference schedule
"""

from collections import defaultdict
from typing import Dict, List
from models import Job, ScheduleEntry


def compute_metrics(schedule: List[ScheduleEntry], jobs: List[Job]) -> dict:
    """Compute a full suite of schedule-evaluation metrics.

    Returns a dict with keys grouped by category:

    *Time-based*
        makespan, total_flow_time, total_waiting_time, avg_flow_time,
        machine_utilization

    *Quality-based*
        total_penalty, max_tardiness, avg_tardiness, max_earliness,
        avg_earliness

    *Details*
        details — per-job breakdown
    """
    # ── per-job completion ──────────────────────────────────────────────
    completion: Dict[int, float] = {}
    for e in schedule:
        jid = e.job_id
        if jid not in completion or e.end_time > completion[jid]:
            completion[jid] = e.end_time

    # ── per-job actual processing time (on the unit it ran on) ──────────
    job_total_proc: Dict[int, float] = defaultdict(float)
    job_op_units: Dict[int, Dict[int, str]] = defaultdict(dict)
    for e in schedule:
        job_total_proc[e.job_id] += (e.end_time - e.start_time)
        job_op_units[e.job_id][e.op_idx] = e.service_unit

    # ── accumulate ──────────────────────────────────────────────────────
    total_penalty = 0.0
    total_flow_time = 0.0
    total_waiting_time = 0.0
    total_processing_time = 0.0
    makespan = 0.0
    tardiness_list: List[float] = []
    earliness_list: List[float] = []
    details: List[dict] = []
    n_jobs = len(jobs)

    for job in jobs:
        Cj = completion.get(job.job_id, 0.0)
        Ej = max(0.0, job.due_date - Cj)
        Tj = max(0.0, Cj - job.due_date)
        penalty = job.alpha * Ej + job.beta * Tj
        total_penalty += penalty

        flow_time = max(0.0, Cj - job.release_date)
        proc_time = job_total_proc.get(job.job_id, 0.0)
        waiting_time = max(0.0, flow_time - proc_time)

        total_flow_time += flow_time
        total_waiting_time += waiting_time
        total_processing_time += proc_time

        if Cj > makespan:
            makespan = Cj

        tardiness_list.append(Tj)
        earliness_list.append(Ej)

        details.append({
            'job_id': job.job_id,
            'completion': Cj,
            'release_date': job.release_date,
            'due_date': job.due_date,
            'flow_time': flow_time,
            'waiting_time': waiting_time,
            'earliness': Ej,
            'tardiness': Tj,
            'penalty': penalty,
        })

    avg_flow_time = total_flow_time / n_jobs if n_jobs else 0.0
    machine_utilization = total_processing_time / makespan if makespan > 0 else 0.0

    max_tardiness = max(tardiness_list) if tardiness_list else 0.0
    avg_tardiness = sum(tardiness_list) / n_jobs if n_jobs else 0.0
    max_earliness = max(earliness_list) if earliness_list else 0.0
    avg_earliness = sum(earliness_list) / n_jobs if n_jobs else 0.0

    return {
        # Time-based
        'makespan':            makespan,
        'total_flow_time':     total_flow_time,
        'total_waiting_time':  total_waiting_time,
        'avg_flow_time':       avg_flow_time,
        'machine_utilization': machine_utilization,

        # Quality-based
        'total_penalty':  total_penalty,
        'max_tardiness':  max_tardiness,
        'avg_tardiness':  avg_tardiness,
        'max_earliness':  max_earliness,
        'avg_earliness':  avg_earliness,

        # Per-job details
        'details': details,
    }


def compute_resilience(
    actual_schedule: List[ScheduleEntry],
    reference_schedule: List[ScheduleEntry],
) -> dict:
    """Measure deviation of *actual_schedule* from *reference_schedule*.

    Returns
    -------
    schedule_deviation : float
        Fraction of operations whose **service unit** differs from the
        reference assignment.
    sequence_deviation : float
        Fraction of same-machine operation **pairs** whose relative order
        is reversed compared to the reference.
    """
    # ── unit-assignment maps: (job_id, op_idx) → service_unit ──────────
    ref_units: Dict[tuple, str] = {}
    for e in reference_schedule:
        ref_units[(e.job_id, e.op_idx)] = e.service_unit

    act_units: Dict[tuple, str] = {}
    for e in actual_schedule:
        act_units[(e.job_id, e.op_idx)] = e.service_unit

    # Schedule deviation — fraction of ops that changed unit
    unit_changes = 0
    n_ops = len(act_units)
    for key, unit in act_units.items():
        if key in ref_units and ref_units[key] != unit:
            unit_changes += 1
    schedule_deviation = unit_changes / n_ops if n_ops > 0 else 0.0

    # ── per-machine sequences (sorted by start time) ────────────────────
    ref_seq: Dict[str, List[tuple]] = defaultdict(list)
    act_seq: Dict[str, List[tuple]] = defaultdict(list)

    for e in reference_schedule:
        ref_seq[e.machine].append((e.job_id, e.op_idx, e.start_time))
    for e in actual_schedule:
        act_seq[e.machine].append((e.job_id, e.op_idx, e.start_time))

    for mid in ref_seq:
        ref_seq[mid].sort(key=lambda x: x[2])
    for mid in act_seq:
        act_seq[mid].sort(key=lambda x: x[2])

    # Sequence deviation — fraction of pairs whose order is reversed
    total_pairs = 0
    reversed_pairs = 0

    for mid, act_ops in act_seq.items():
        ref_order: Dict[tuple, int] = {}
        for i, (jid, oidx, _) in enumerate(ref_seq.get(mid, [])):
            ref_order[(jid, oidx)] = i

        n = len(act_ops)
        for i in range(n):
            for j in range(i + 1, n):
                ki = (act_ops[i][0], act_ops[i][1])
                kj = (act_ops[j][0], act_ops[j][1])
                if ki in ref_order and kj in ref_order:
                    total_pairs += 1
                    if ref_order[ki] > ref_order[kj]:
                        reversed_pairs += 1

    sequence_deviation = reversed_pairs / total_pairs if total_pairs > 0 else 0.0

    return {
        'schedule_deviation': schedule_deviation,
        'sequence_deviation': sequence_deviation,
    }


def print_comparison_table(results: List[dict]):
    """Pretty-print a comparison table for multiple strategies."""
    header = (f"{'Strategy':<28} {'Penalty':>10} {'Makespan':>10} "
              f"{'Tardiness':>10} {'FlowTime':>10}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        tardiness = sum(d['tardiness'] for d in r['metrics']['details'])
        print(f"{r['name']:<28} {r['metrics']['total_penalty']:>10.1f} "
              f"{r['metrics']['makespan']:>10.1f} {tardiness:>10.1f} "
              f"{r['metrics']['total_flow_time']:>10.1f}")
    print(sep)
