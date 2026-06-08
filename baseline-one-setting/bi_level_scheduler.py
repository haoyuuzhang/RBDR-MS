"""Bi-level FJSP scheduler.

Upper level (Shop): Rule-based operation-to-service-unit assignment (SPT / FIFO / WINQ)
Lower level (Unit):  Gurobi MILP for optimal intra-unit scheduling

The initial schedule at t=0 is computed by a **global** Gurobi MILP (all 8
machines).  When disruptions occur at t=2 (J9/J10 arrivals) and t=6 (M3
breakdown), re-scheduling uses the two-level architecture:
upper-level rule assigns operations to U1 or U2, then each unit solves its
own independent MILP.

Usage
-----
    from bi_level_scheduler import simulate_bi_level, GurobiUnitSolver

    result = simulate_bi_level('SPT', jobs, disruptions, GurobiUnitSolver())
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from models import Job, Operation, ScheduleEntry
from scheduler import schedule_makespan_milp

# ── Service unit definitions ────────────────────────────────────────────────
SERVICE_UNITS: Dict[str, List[str]] = {
    'U1': ['M1', 'M2', 'M3', 'M4'],
    'U2': ['M5', 'M6', 'M7', 'M8'],
}
UNIT_NAMES: List[str] = ['U1', 'U2']
ALL_MACHINES: List[str] = [m for u in UNIT_NAMES for m in SERVICE_UNITS[u]]

# Decision points where re-optimisation is triggered
DECISION_POINTS: List[float] = [0.0, 2.0, 6.0]


# ═════════════════════════════════════════════════════════════════════════════
#  Lower-level solver interface
# ═════════════════════════════════════════════════════════════════════════════

class UnitSolver(ABC):
    """Abstract interface for service-unit-level scheduling solvers.

    Subclass and implement :meth:`solve` to plug in different optimisation
    backends (e.g. Gurobi MILP, immune-inspired negotiation, heuristic search).
    """

    @abstractmethod
    def solve(self,
              jobs: List[Job],
              fixed_entries: List[ScheduleEntry],
              current_time: float,
              machine_deadlines: Optional[Dict[str, float]] = None,
              time_limit: float = 120.0) -> Optional[List[ScheduleEntry]]:
        """Solve the scheduling problem for a single service unit.

        Parameters
        ----------
        jobs : list of Job
            Jobs whose operations need to be scheduled on this unit.
            Operation ``times`` dicts should already be restricted to the
            unit's machines by the caller.
        fixed_entries : list of ScheduleEntry
            Already-committed operations (completed or in-progress).
        current_time : float
            The moment at which the schedule is being computed.
        machine_deadlines : dict or None
            Per-machine hard deadlines (e.g. ``{'M3': 6.0}`` for breakdown).
        time_limit : float
            Solver time limit in seconds.

        Returns
        -------
        list of ScheduleEntry  or  None if infeasible.
        """
        ...


class GurobiUnitSolver(UnitSolver):
    """Unit-level solver backed by Gurobi MILP.

    Delegates directly to :func:`scheduler.schedule_makespan_milp`.
    """

    def solve(self,
              jobs: List[Job],
              fixed_entries: List[ScheduleEntry],
              current_time: float,
              machine_deadlines: Optional[Dict[str, float]] = None,
              time_limit: float = 120.0) -> Optional[List[ScheduleEntry]]:
        return schedule_makespan_milp(
            jobs, fixed_entries,
            current_time=current_time,
            machine_deadlines=machine_deadlines,
            time_limit=time_limit,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _restrict_op_to_unit(op: Operation, unit_machines: List[str]) -> Operation:
    """Return a copy of *op* with only machines in *unit_machines*."""
    restricted = {m: t for m, t in op.times.items() if m in unit_machines}
    return Operation(job_id=op.job_id, op_idx=op.op_idx, times=restricted)


def _restrict_job_to_unit(job: Job, unit_machines: List[str]) -> Job:
    """Return a copy of *job* with operations restricted to *unit_machines*."""
    return Job(
        job_id=job.job_id,
        arrival_time=job.arrival_time,
        operations=[_restrict_op_to_unit(op, unit_machines) for op in job.operations],
    )


def _get_feasible_units(op: Operation) -> List[str]:
    """Return unit names (U1/U2) that have at least one feasible machine."""
    feasible = []
    for unit_name in UNIT_NAMES:
        if any(m in op.times for m in SERVICE_UNITS[unit_name]):
            feasible.append(unit_name)
    return feasible


# ═════════════════════════════════════════════════════════════════════════════
#  Upper-level dispatcher
# ═════════════════════════════════════════════════════════════════════════════

def assign_ops_to_units(
    ready_ops: List[Tuple[int, int]],
    rule_name: str,
    job_map: Dict[int, Job],
    unit_workload: Optional[Dict[str, float]] = None,
    existing_assignments: Optional[Dict[Tuple[int, int], str]] = None,
) -> Dict[Tuple[int, int], str]:
    """Assign ready operations to service units using a dispatching rule.

    Parameters
    ----------
    ready_ops : list of (job_id, op_idx)
        Operations that are ready for dispatch (predecessors completed, job
        arrived).  Processed in list order for FIFO, independently for others.
    rule_name : str
        ``'SPT'``, ``'FIFO'``, or ``'WINQ'``.
    job_map : dict
        ``{job_id: Job}`` lookup.
    unit_workload : dict or None
        Current estimated workload per unit ``{unit_name: total_time}``.
        Used as a tie-breaker / load-balancer.  Mutated in-place.
    existing_assignments : dict or None
        Already-assigned operations ``{(job_id, op_idx): unit_name}``.
        These are **not** reassigned.

    Returns
    -------
    dict
        ``{(job_id, op_idx): unit_name}`` — includes existing + new assignments.
    """
    if unit_workload is None:
        unit_workload = {u: 0.0 for u in UNIT_NAMES}
    if existing_assignments is None:
        existing_assignments = {}

    assignments = dict(existing_assignments)

    for (jid, oidx) in ready_ops:
        if (jid, oidx) in assignments:
            continue

        op = job_map[jid].operations[oidx]
        feasible_units = _get_feasible_units(op)

        if not feasible_units:
            # No feasible machine in any unit — shouldn't happen with Kacem data
            continue

        if len(feasible_units) == 1:
            chosen = feasible_units[0]
        elif rule_name == 'SPT':
            chosen = _assign_spt(op, feasible_units, unit_workload)
        elif rule_name == 'FIFO':
            chosen = _assign_fifo(feasible_units, unit_workload)
        elif rule_name == 'WINQ':
            chosen = _assign_winq(jid, oidx, job_map, feasible_units, unit_workload)
        else:
            raise ValueError(f"Unknown rule: {rule_name}")

        assignments[(jid, oidx)] = chosen
        # Update workload estimate with a rough proxy (min processing time in unit)
        min_t = min(op.times[m] for m in SERVICE_UNITS[chosen] if m in op.times)
        unit_workload[chosen] += min_t

    return assignments


def _assign_spt(op: Operation,
                feasible_units: List[str],
                workload: Dict[str, float]) -> str:
    """SPT rule: pick unit with smallest min processing time."""
    def key(u: str) -> Tuple[float, float]:
        min_t = min(op.times[m] for m in SERVICE_UNITS[u] if m in op.times)
        return (min_t, workload[u])
    return min(feasible_units, key=key)


def _assign_fifo(feasible_units: List[str],
                 workload: Dict[str, float]) -> str:
    """FIFO rule at unit level: assign to less-loaded unit (load balancing)."""
    return min(feasible_units, key=lambda u: workload[u])


def _assign_winq(jid: int,
                 oidx: int,
                 job_map: Dict[int, Job],
                 feasible_units: List[str],
                 workload: Dict[str, float]) -> str:
    """WINQ rule: consider next operation's queue congestion."""
    job = job_map[jid]
    next_oidx = oidx + 1

    # Last operation → assign to less-loaded unit
    if next_oidx >= len(job.operations):
        return min(feasible_units, key=lambda u: workload[u])

    next_op = job.operations[next_oidx]

    def unit_congestion(u: str) -> float:
        next_feasible = [m for m in next_op.feasible_machines
                         if m in SERVICE_UNITS[u]]
        if not next_feasible:
            return float('inf')
        # Congestion proxy: unit workload / number of feasible machines
        # (lower is better — less competition for next operation)
        return workload[u] / len(next_feasible)

    return min(feasible_units, key=lambda u: (unit_congestion(u), workload[u]))


# ═════════════════════════════════════════════════════════════════════════════
#  Schedule execution helpers
# ═════════════════════════════════════════════════════════════════════════════

def _collect_fixed_and_ready(
    schedule: List[ScheduleEntry],
    current_time: float,
    previously_fixed_keys: Set[Tuple[int, int]],
) -> Tuple[List[ScheduleEntry], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Split a schedule at *current_time* into fixed / in-progress / completed.

    Returns
    -------
    fixed_entries : list of ScheduleEntry
        Entries that are either completed (end ≤ now) or in-progress
        (start < now < end).  These block their assigned machines.
    completed_keys : set of (job_id, op_idx)
        Operations that have finished by *current_time*.
    in_progress_keys : set of (job_id, op_idx)
        Operations currently being processed (started but not finished).
    """
    fixed_entries: List[ScheduleEntry] = []
    completed_keys: Set[Tuple[int, int]] = set()
    in_progress_keys: Set[Tuple[int, int]] = set()

    for e in schedule:
        key = (e.job_id, e.op_idx)
        if key in previously_fixed_keys:
            # Already committed in an earlier stage
            fixed_entries.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time, fixed=True))
            completed_keys.add(key)
            continue

        if e.end_time <= current_time:
            # Completed
            fixed_entries.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time, fixed=True))
            completed_keys.add(key)
        elif e.start_time < current_time < e.end_time:
            # In progress — must continue on its assigned machine
            fixed_entries.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time, fixed=True))
            in_progress_keys.add(key)
        # else: future (start_time >= current_time) — not fixed

    return fixed_entries, completed_keys, in_progress_keys


def _get_ready_successors(
    completed_keys: Set[Tuple[int, int]],
    in_progress_keys: Set[Tuple[int, int]],
    job_map: Dict[int, Job],
    fixed_keys: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Return operations whose predecessors are completed and are not fixed.

    Only returns the *first* uncompleted operation of each job where the
    immediately preceding operation is in *completed_keys*.
    """
    ready: List[Tuple[int, int]] = []

    for job in job_map.values():
        for idx, op in enumerate(job.operations):
            key = (job.job_id, op.op_idx)
            if key in fixed_keys or key in in_progress_keys:
                continue
            if key in completed_keys:
                continue
            if idx == 0:
                # First operation: ready if job has arrived
                if job.arrival_time <= max(
                    completed_keys and max(
                        (e[1] for e in completed_keys if e[0] == job.job_id),
                        default=-1) or 0,
                    # Actually: first op is ready if its job has arrived by now
                ):
                    # This check is done at the decision-point level
                    pass
                ready.append(key)
                break  # only first uncompleted op
            else:
                prev_key = (job.job_id, idx - 1)
                if prev_key in completed_keys:
                    # Predecessor done → this op is ready
                    ready.append(key)
                    break
                elif prev_key in fixed_keys or prev_key in in_progress_keys:
                    # Predecessor not done → this op not ready yet
                    break
                # else: predecessor hasn't even been considered yet
                break

    return ready


# ═════════════════════════════════════════════════════════════════════════════
#  Main simulation orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def simulate_bi_level(
    rule_name: str,
    jobs: List[Job],
    disruptions: List[dict],
    unit_solver: UnitSolver,
    initial_schedule: Optional[List[ScheduleEntry]] = None,
) -> Optional[dict]:
    """Run bi-level scheduling simulation across three decision points.

    Parameters
    ----------
    rule_name : str
        ``'SPT'``, ``'FIFO'``, or ``'WINQ'`` — used at t=2 and t=6.
    jobs : list of Job
        All jobs (initial + dynamic).
    disruptions : list of dict
        Each has keys ``'time'`` (float) and ``'machine'`` (str).
    unit_solver : UnitSolver
        Solver for intra-unit scheduling (used at t=2 and t=6).
    initial_schedule : list of ScheduleEntry or None
        Pre-computed optimal schedule for t=0 (e.g. from Baseline A cache).
        If ``None``, a global Gurobi MILP is called at t=0.

    Returns
    -------
    dict or None
        ``{'cmax': float, 'entries': list[dict], 'partial_entries': list[dict],
          'compute_time': float, 'unit_assignments': dict,
          'snapshot_schedules': dict}``
        or ``None`` if any MILP stage fails.
    """
    t_start_total = time.perf_counter()

    job_map: Dict[int, Job] = {j.job_id: j for j in jobs}
    initial_jobs = [j for j in jobs if j.arrival_time <= 0.0]

    # Track which operation keys have been fixed in prior stages
    fixed_keys: Set[Tuple[int, int]] = set()
    # Track unit assignments  (job_id, op_idx) → unit_name
    unit_assignments: Dict[Tuple[int, int], str] = {}
    # Accumulated schedule entries (final consolidated schedule)
    all_entries: List[ScheduleEntry] = []
    # Interrupted (partial) entries
    all_partial: List[ScheduleEntry] = []
    # Snapshot schedules at each decision point (for Gantt charts)
    snapshot_schedules: Dict[str, list] = {}

    # ── Stage 0: t=0 — Global optimal schedule ─────────────────────────────
    if initial_schedule is not None:
        print(f"\n  [{rule_name}] Stage 0 — t=0  Using cached optimal schedule  "
              f"(J1-J{len(initial_jobs)}, all 8 machines)")
        sched_0 = list(initial_schedule)
        dt0 = 0.0
    else:
        print(f"\n  [{rule_name}] Stage 0 — t=0  Global MILP  (J1-J{len(initial_jobs)}, "
              f"all 8 machines) ...")
        t0 = time.perf_counter()
        sched_0 = schedule_makespan_milp(initial_jobs, [], current_time=0.0,
                                          time_limit=120.0)
        dt0 = time.perf_counter() - t0

        if sched_0 is None:
            print("    !! Global MILP failed at t=0")
            return None

    cmax_0 = max(e.end_time for e in sched_0)
    print(f"    OK  C_max = {cmax_0:.3f}  (solve time {dt0:.1f} s)")

    # Mark all t=0 entries as the baseline
    snapshot_schedules['0.0'] = [e for e in sched_0]
    snapshot_schedules['0.0'] = [e for e in sched_0]

    # ═══════════════════════════════════════════════════════════════════════
    #  Stage 1: t=2 — J9/J10 arrive, bi-level re-scheduling
    # ═══════════════════════════════════════════════════════════════════════
    t_event = 2.0
    print(f"\n  [{rule_name}] Stage 1 — t={t_event:.0f}  Bi-level re-scheduling  "
          f"(J9, J10 arrive) ...")

    # --- Step 1: Split t=0 schedule at t=2 ---------------------------------
    fixed_2, completed_2, in_progress_2 = _collect_fixed_and_ready(
        sched_0, t_event, fixed_keys)
    fixed_keys |= {(e.job_id, e.op_idx) for e in fixed_2}

    n_completed = len(completed_2)
    n_in_prog = len(in_progress_2)
    print(f"    Fixed: {n_completed} completed, {n_in_prog} in-progress")

    # --- Step 2: Identify new jobs and ready operations ---------------------
    new_jobs = [j for j in jobs if j.arrival_time == t_event]
    new_job_ids = {j.job_id for j in new_jobs}
    print(f"    New jobs arriving: {sorted(new_job_ids)}")

    # Operations that are ready at t=2:
    # - First ops of newly arrived jobs (J9, J10)
    # - Successors of completed ops
    ready_2: List[Tuple[int, int]] = []

    # New jobs: first operations are ready
    for j in new_jobs:
        ready_2.append((j.job_id, 0))

    # Existing jobs: successors of completed ops
    for (jid, oidx) in completed_2:
        job = job_map[jid]
        next_oidx = oidx + 1
        if next_oidx < len(job.operations):
            nxt_key = (jid, next_oidx)
            if nxt_key not in fixed_keys:
                ready_2.append(nxt_key)

    # Also collect any op from the t=0 schedule that hasn't started yet
    # and isn't ready yet (its predecessor is still in progress)
    unstarted_keys: Set[Tuple[int, int]] = set()
    for e in sched_0:
        key = (e.job_id, e.op_idx)
        if key not in fixed_keys and e.start_time >= t_event:
            unstarted_keys.add(key)

    print(f"    Ready ops: {len(ready_2)},  unstarted: {len(unstarted_keys)}")

    # --- Step 3: Upper-level unit assignment --------------------------------
    unit_workload: Dict[str, float] = {'U1': 0.0, 'U2': 0.0}

    # Compute workload from fixed entries per unit
    for e in fixed_2:
        for u_name, machines in SERVICE_UNITS.items():
            if e.machine in machines:
                unit_workload[u_name] += e.duration

    ready_to_assign = ready_2 + [k for k in unstarted_keys if k not in unit_assignments]
    # Remove duplicates while preserving order
    seen = set()
    ready_to_assign_unique = []
    for k in ready_to_assign:
        if k not in seen:
            seen.add(k)
            ready_to_assign_unique.append(k)

    t_assign_start = time.perf_counter()
    unit_assignments = assign_ops_to_units(
        ready_to_assign_unique, rule_name, job_map,
        unit_workload=unit_workload,
        existing_assignments=unit_assignments,
    )
    dt_assign_2 = time.perf_counter() - t_assign_start

    # --- Step 4: Per-unit MILP ----------------------------------------------
    sched_2_all: List[ScheduleEntry] = list(fixed_2)
    total_dt_milp_2 = 0.0

    for unit_name in UNIT_NAMES:
        unit_machines = SERVICE_UNITS[unit_name]
        other_machines = [m for u in UNIT_NAMES if u != unit_name
                          for m in SERVICE_UNITS[u]]

        # Collect ops assigned to this unit
        unit_op_keys = {k for k, u in unit_assignments.items() if u == unit_name}

        # Build unit-specific jobs — keep ALL operations at original positions
        # so that op_idx matches the list index (required by schedule_makespan_milp).
        # Ops not assigned to this unit get times restricted to the OTHER unit's
        # machines so they are not schedulable here.
        unit_jobs: List[Job] = []
        all_jids_in_play = {k[0] for k in unit_op_keys} | {e.job_id for e in fixed_2}
        for jid in all_jids_in_play:
            original_job = job_map[jid]
            new_ops: List[Operation] = []
            for orig_op in original_job.operations:
                key = (jid, orig_op.op_idx)
                if key in unit_op_keys and key not in fixed_keys:
                    # Assigned to this unit → restrict to unit machines
                    restricted_times = {m: t for m, t in orig_op.times.items()
                                        if m in unit_machines}
                elif key in fixed_keys:
                    # Already fixed → keep this unit's machines only
                    restricted_times = {m: t for m, t in orig_op.times.items()
                                        if m in unit_machines}
                else:
                    # Not assigned to this unit, not fixed →
                    # restrict to OTHER unit's machines (won't be schedulable here)
                    restricted_times = {m: t for m, t in orig_op.times.items()
                                        if m in other_machines}
                new_ops.append(Operation(jid, orig_op.op_idx, restricted_times))
            unit_jobs.append(Job(jid, original_job.arrival_time, new_ops))

        # Pass ALL fixed entries (both units) so the MILP knows about every
        # committed operation.  Entries on the other unit's machines block
        # those machines, which is correct — this unit cannot use them.
        unit_fixed_all = list(fixed_2)

        if not unit_jobs and not unit_fixed_all:
            print(f"    [{unit_name}]  No work — skipping")
            continue

        n_unit_fixed = len([e for e in fixed_2 if e.machine in unit_machines])
        print(f"    [{unit_name}]  {len(unit_jobs)} jobs, {n_unit_fixed} fixed "
              f"(+ {len(fixed_2) - n_unit_fixed} other-unit)  → MILP ...")
        t_milp = time.perf_counter()
        unit_schedule = unit_solver.solve(
            unit_jobs, unit_fixed_all, current_time=t_event, time_limit=120.0)
        dt_milp = time.perf_counter() - t_milp
        total_dt_milp_2 += dt_milp

        if unit_schedule is None:
            print(f"    !! [{unit_name}] MILP failed")
            return None

        # Add non-fixed entries to the consolidated schedule
        for e in unit_schedule:
            if (e.job_id, e.op_idx) not in fixed_keys:
                sched_2_all.append(e)

        cmax_u = max((e.end_time for e in unit_schedule), default=0.0)
        print(f"    [{unit_name}]  C_max = {cmax_u:.3f}  (solve time {dt_milp:.1f} s)")

    cmax_2 = max((e.end_time for e in sched_2_all), default=0.0)
    print(f"    Combined C_max = {cmax_2:.3f}")

    snapshot_schedules['2.0'] = sched_2_all

    # ═══════════════════════════════════════════════════════════════════════
    #  Stage 2: t=6 — M3 breakdown, bi-level re-scheduling
    # ═══════════════════════════════════════════════════════════════════════
    t_event = 6.0

    # Find the disruption info
    disruption = disruptions[0] if disruptions else {}
    broken_machine = disruption.get('machine', 'M3')
    breakdown_time = disruption.get('time', 6.0)

    print(f"\n  [{rule_name}] Stage 2 — t={t_event:.0f}  Bi-level re-scheduling  "
          f"({broken_machine} breakdown) ...")

    # --- Step 1: Split t=2 schedule at t=6 ---------------------------------
    fixed_6, completed_6, in_progress_6 = _collect_fixed_and_ready(
        sched_2_all, t_event, fixed_keys)
    fixed_keys |= {(e.job_id, e.op_idx) for e in fixed_6}
    # Also add partial entries from earlier stages to fixed_keys
    for pe in all_partial:
        fixed_keys.add((pe.job_id, pe.op_idx))

    n_completed_6 = len(completed_6)
    n_in_prog_6 = len(in_progress_6)
    print(f"    Fixed: {n_completed_6} completed, {n_in_prog_6} in-progress")

    # --- Step 2: Handle M3 interruption -------------------------------------
    interrupted_keys: Set[Tuple[int, int]] = set()
    interrupted_entries: List[ScheduleEntry] = []

    for key in list(in_progress_6):
        # Find the schedule entry for this in-progress op
        for e in sched_2_all:
            if (e.job_id, e.op_idx) == key:
                if e.machine == broken_machine:
                    # Interrupted!
                    interrupted_keys.add(key)
                    interrupted_entries.append(ScheduleEntry(
                        job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                        start_time=e.start_time, end_time=breakdown_time,
                        fixed=True,
                    ))
                    # Remove from in_progress and fixed
                    in_progress_6.discard(key)
                    fixed_6 = [fe for fe in fixed_6
                               if (fe.job_id, fe.op_idx) != key]
                    print(f"    Interrupted: J{e.job_id}-{e.op_idx + 1} on {broken_machine}")
                break

    all_partial.extend(interrupted_entries)

    # --- Step 3: Identify ready ops at t=6 ----------------------------------
    ready_6: List[Tuple[int, int]] = []

    # Successors of completed ops
    for (jid, oidx) in completed_6:
        job = job_map[jid]
        next_oidx = oidx + 1
        if next_oidx < len(job.operations):
            nxt_key = (jid, next_oidx)
            if nxt_key not in fixed_keys:
                ready_6.append(nxt_key)

    # Interrupted ops go back to the ready pool
    for key in interrupted_keys:
        if key not in fixed_keys:
            ready_6.append(key)

    # Unstarted ops from t=2 schedule (not fixed, not ready via completion)
    unstarted_6: Set[Tuple[int, int]] = set()
    for e in sched_2_all:
        key = (e.job_id, e.op_idx)
        if key not in fixed_keys and e.start_time >= t_event:
            unstarted_6.add(key)

    # Remove interrupted ops from unstarted (they're already in ready_6)
    unstarted_6 -= interrupted_keys

    print(f"    Ready ops: {len(ready_6)},  unstarted: {len(unstarted_6)},  "
          f"interrupted: {len(interrupted_keys)}")

    # --- Step 4: Upper-level unit assignment (re-assign interrupted ops) ----
    # Reset unit workload based on fixed entries
    unit_workload = {'U1': 0.0, 'U2': 0.0}
    for e in fixed_6:
        for u_name, machines in SERVICE_UNITS.items():
            if e.machine in machines:
                unit_workload[u_name] += e.duration

    # Remove interrupted ops from existing assignments (they get re-assigned)
    for key in interrupted_keys:
        unit_assignments.pop(key, None)

    ready_to_assign_6 = ready_6 + [k for k in unstarted_6 if k not in unit_assignments]
    seen_6 = set()
    ready_to_assign_6_unique = []
    for k in ready_to_assign_6:
        if k not in seen_6:
            seen_6.add(k)
            ready_to_assign_6_unique.append(k)

    t_assign_start = time.perf_counter()
    unit_assignments = assign_ops_to_units(
        ready_to_assign_6_unique, rule_name, job_map,
        unit_workload=unit_workload,
        existing_assignments=unit_assignments,
    )
    dt_assign_6 = time.perf_counter() - t_assign_start

    # --- Step 5: Per-unit MILP ----------------------------------------------
    sched_6_all: List[ScheduleEntry] = list(fixed_6)
    total_dt_milp_6 = 0.0

    # M3 deadline only applies to U1
    machine_deadlines = {broken_machine: breakdown_time}

    for unit_name in UNIT_NAMES:
        unit_machines = SERVICE_UNITS[unit_name]
        other_machines = [m for u in UNIT_NAMES if u != unit_name
                          for m in SERVICE_UNITS[u]]

        # Collect ops assigned to this unit
        unit_op_keys = {k for k, u in unit_assignments.items() if u == unit_name}

        # Build unit-specific jobs — keep ALL operations at original positions
        # so that op_idx matches the list index (required by schedule_makespan_milp).
        unit_jobs: List[Job] = []
        all_jids_in_play_6 = {k[0] for k in unit_op_keys} | {e.job_id for e in fixed_6}
        for jid in all_jids_in_play_6:
            original_job = job_map[jid]
            new_ops: List[Operation] = []
            for orig_op in original_job.operations:
                key = (jid, orig_op.op_idx)
                if key in unit_op_keys and key not in fixed_keys:
                    restricted_times = {m: t for m, t in orig_op.times.items()
                                        if m in unit_machines}
                elif key in fixed_keys:
                    restricted_times = {m: t for m, t in orig_op.times.items()
                                        if m in unit_machines}
                else:
                    restricted_times = {m: t for m, t in orig_op.times.items()
                                        if m in other_machines}
                new_ops.append(Operation(jid, orig_op.op_idx, restricted_times))
            unit_jobs.append(Job(jid, original_job.arrival_time, new_ops))

        # Pass ALL fixed entries (both units) so the MILP knows about every
        # committed operation.
        unit_fixed_all = list(fixed_6)

        # Check if there are any ops in this unit that actually need scheduling
        has_unit_work = any(
            (op.times for job in unit_jobs for op in job.operations
             if (op.job_id, op.op_idx) in unit_op_keys
             and (op.job_id, op.op_idx) not in fixed_keys))

        if not has_unit_work:
            u_cmax = max((e.end_time for e in unit_fixed_all), default=0.0)
            print(f"    [{unit_name}]  No new work, C_max = {u_cmax:.3f}  (skipped MILP)")
            continue

        # Only apply machine_deadlines to the unit containing the broken machine
        unit_deadlines = machine_deadlines if broken_machine in unit_machines else None

        n_unit_fixed = len([e for e in fixed_6 if e.machine in unit_machines])
        print(f"    [{unit_name}]  {len(unit_jobs)} jobs, {n_unit_fixed} fixed"
              f"  (+ {len(fixed_6) - n_unit_fixed} other-unit)"
              f"{'  [M3 deadline]' if unit_deadlines else ''}  → MILP ...")
        t_milp = time.perf_counter()
        unit_schedule = unit_solver.solve(
            unit_jobs, unit_fixed_all, current_time=t_event,
            machine_deadlines=unit_deadlines, time_limit=120.0)
        dt_milp = time.perf_counter() - t_milp
        total_dt_milp_6 += dt_milp

        if unit_schedule is None:
            print(f"    !! [{unit_name}] MILP failed")
            return None

        for e in unit_schedule:
            if (e.job_id, e.op_idx) not in fixed_keys:
                sched_6_all.append(e)

        cmax_u = max((e.end_time for e in unit_schedule), default=0.0)
        print(f"    [{unit_name}]  C_max = {cmax_u:.3f}  (solve time {dt_milp:.1f} s)")

    cmax_6 = max((e.end_time for e in sched_6_all), default=0.0)
    print(f"    Combined C_max = {cmax_6:.3f}")

    snapshot_schedules['6.0'] = sched_6_all

    # ── Assemble final result ──────────────────────────────────────────────
    total_compute = time.perf_counter() - t_start_total

    # Serialise entries for JSON output
    from dataclasses import asdict
    serialised_entries = [asdict(e) for e in sched_6_all]
    serialised_partial = [asdict(e) for e in all_partial]

    # Convert unit_assignments keys from tuple to string for JSON
    serialised_assignments = {
        f"{jid},{oidx}": unit_name
        for (jid, oidx), unit_name in unit_assignments.items()
    }

    # Serialise snapshot schedules
    serialised_snapshots = {}
    for t_str, entries in snapshot_schedules.items():
        serialised_snapshots[t_str] = [asdict(e) for e in entries]

    print(f"\n  [{rule_name}]  Final C_max = {cmax_6:.3f}  "
          f"(total compute = {total_compute:.3f} s)")

    return {
        'cmax': cmax_6,
        'entries': serialised_entries,
        'partial_entries': serialised_partial,
        'compute_time': total_compute,
        'unit_assignments': serialised_assignments,
        'snapshot_schedules': serialised_snapshots,
        'snapshot_cmax': {
            '0.0': cmax_0,
            '2.0': cmax_2,
            '6.0': cmax_6,
        },
    }
