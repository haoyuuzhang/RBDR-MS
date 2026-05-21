"""Simulation scenarios for Hierarchical FJSP.

Experiments:
  Dynamic Arrival    : t=0 initial schedule, t=24 J4 arrives -> reschedule
  Machine Disruption : Machine M1_U2 disruption at t=60 (continues from dynamic arrival)
"""

from typing import List, Optional, Tuple
from models import Job, ScheduleEntry
from scheduler import schedule_milp


def simulate_dynamic_arrival(
    all_jobs: List[Job],
    time_limit: float = 60.0,
) -> Tuple[Optional[List[ScheduleEntry]], Optional[List[ScheduleEntry]]]:
    """
    Two-stage dynamic-arrival simulation using MILP.

    t=0  : MILP optimal initial schedule for visible jobs
    t=24 : J4 arrives -> MILP re-optimization with frozen ops
    """
    visible_jobs_t0 = [j for j in all_jobs if j.arrival_time <= 0]
    initial_schedule = schedule_milp(visible_jobs_t0, [], 0.0, time_limit)
    if initial_schedule is None:
        return None, None

    fixed_at_t24: List[ScheduleEntry] = []
    for e in initial_schedule:
        if e.start_time < 24:
            e_fixed = ScheduleEntry(e.job_id, e.op_idx, e.machine,
                                    e.service_unit,
                                    e.start_time, e.end_time, fixed=True)
            fixed_at_t24.append(e_fixed)

    visible_jobs_t24 = [j for j in all_jobs if j.arrival_time <= 24]
    updated_schedule = schedule_milp(visible_jobs_t24, fixed_at_t24,
                                     24.0, time_limit)
    if updated_schedule is None:
        return None, None

    return initial_schedule, updated_schedule


def simulate_machine_disruption(
    all_jobs: List[Job],
    previous_schedule: List[ScheduleEntry],
    disruption_time: float = 60.0,
    disrupted_machine: str = 'M1_U2',
    factor: float = 2.0,
    time_limit: float = 60.0,
) -> Optional[List[ScheduleEntry]]:
    """
    Machine disruption at t=disruption_time (continues from dynamic-arrival result).

    The disrupted_machine's processing time is multiplied by ``factor``.
    - Completed ops (end <= disruption_time): kept as-is
    - Mid-processing ops (start < disruption_time < end): remaining time scaled
    - Not-yet-started ops: rescheduled with updated time factors
    """
    fixed_entries: List[ScheduleEntry] = []

    for e in previous_schedule:
        if e.end_time <= disruption_time:
            fixed_entries.append(ScheduleEntry(
                e.job_id, e.op_idx, e.machine, e.service_unit,
                e.start_time, e.end_time, fixed=True))
        elif e.start_time < disruption_time:
            if e.machine == disrupted_machine:
                remaining = (e.end_time - disruption_time) * factor
                new_end = disruption_time + remaining
            else:
                new_end = e.end_time
            fixed_entries.append(ScheduleEntry(
                e.job_id, e.op_idx, e.machine, e.service_unit,
                e.start_time, new_end, fixed=True))

    time_factors = {disrupted_machine: factor}
    schedule = schedule_milp(all_jobs, fixed_entries, disruption_time,
                             time_limit, time_factors=time_factors)
    return schedule
