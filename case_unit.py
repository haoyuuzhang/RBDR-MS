"""
Pharmaceutical Production Dynamic Multi-Resource FJSP Simulation
Based on Example 1 from problem.md

Simulates a two-stage decision process:
  t=0  : initial schedule for J1, J2, J3
  t=24 : J4 arrives, J2-Op1 completed, J2-Op2 in progress → reschedule
"""

from dataclasses import dataclass
from itertools import product
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Operation:
    """A single operation within a job."""
    job_id: int
    op_idx: int          # 0-based index within the job
    machine_type: str     # 'M1', 'M2', or 'M3'
    proc_time: float
    no_wait_next: bool = False


@dataclass
class Job:
    """A job (order) composed of an ordered sequence of operations."""
    job_id: int
    release_date: float
    due_date: float
    alpha: float          # earliness penalty weight
    beta: float           # tardiness penalty weight
    operations: List[Operation]
    arrival_time: float = 0.0


@dataclass
class ScheduleEntry:
    """Records the assignment and timing of one operation."""
    job_id: int
    op_idx: int
    machine: str
    start_time: float
    end_time: float
    fixed: bool = False   # True if already started / completed (frozen)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ═══════════════════════════════════════════════════════════════════════════════
# Resource configuration (Example 1)
# ═══════════════════════════════════════════════════════════════════════════════

# 可以修改为service unit
MACHINES_BY_TYPE: Dict[str, List[str]] = {
    'M1': ['M1_1', 'M1_2'],
    'M2': ['M2_1'],
    'M3': ['M3_1'],
} 

ALL_MACHINES: List[str] = [m for ml in MACHINES_BY_TYPE.values() for m in ml]


def compatible_machines(machine_type: str) -> List[str]:
    """Return concrete machine names of `machine_type`."""
    return MACHINES_BY_TYPE.get(machine_type, [])


# ═══════════════════════════════════════════════════════════════════════════════
# Job / operation definitions (Example 1)
# ═══════════════════════════════════════════════════════════════════════════════

def build_jobs() -> List[Job]:
    return [
        Job(job_id=1, release_date=0, due_date=96, alpha=0, beta=1,
            arrival_time=0, operations=[
                Operation(1, 0, 'M1', 10, no_wait_next=True),
                Operation(1, 1, 'M2', 12, no_wait_next=True),
                Operation(1, 2, 'M1', 14),
            ]),
        Job(job_id=2, release_date=0, due_date=72, alpha=0, beta=4,
            arrival_time=0, operations=[
                Operation(2, 0, 'M1', 14),
                Operation(2, 1, 'M2', 22),
                Operation(2, 2, 'M1', 19),
            ]),
        Job(job_id=3, release_date=0, due_date=120, alpha=1, beta=2,
            arrival_time=0, operations=[
                Operation(3, 0, 'M1', 12, no_wait_next=True),
                Operation(3, 1, 'M2', 12, no_wait_next=True),
                Operation(3, 2, 'M1', 18),
            ]),
        Job(job_id=4, release_date=48, due_date=96, alpha=1, beta=6,
            arrival_time=24, operations=[
                Operation(4, 0, 'M3', 12, no_wait_next=True),
                Operation(4, 1, 'M1', 12, no_wait_next=True),
                Operation(4, 2, 'M2', 20),
            ]),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Resource availability tracker
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceTracker:
    """Maintains busy intervals for machines."""

    def __init__(self):
        # machine -> list of (start, end) busy intervals
        self.machine_intervals: Dict[str, List[Tuple[float, float]]] = {
            m: [] for m in ALL_MACHINES}

    def add(self, machine: str, start: float, end: float):
        self.machine_intervals[machine].append((start, end))

    # ------------------------------------------------------------------
    # Machine
    # ------------------------------------------------------------------
    def machine_free_at(self, machine: str, t: float) -> bool:
        return not any(s < t + 1e-9 and e > t + 1e-9
                       for s, e in self.machine_intervals[machine])

    def machine_next_free(self, machine: str, t: float) -> float:
        """Earliest time >= t when `machine` is free."""
        intervals = sorted(self.machine_intervals[machine])
        for s, e in intervals:
            if s < t + 1e-9 and e > t + 1e-9:
                t = e
        return t

# ═══════════════════════════════════════════════════════════════════════════════
# Constructive scheduler  (EDD priority, earliest-feasible insertion)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_penalty(end_time: float, due_date: float,
                     alpha: float, beta: float) -> float:
    """Weighted earliness–tardiness penalty for a candidate completion time."""
    if end_time <= due_date:
        return alpha * (due_date - end_time)
    else:
        return beta * (end_time - due_date)


# 贪心调度器
def schedule_jobs(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float,
) -> List[ScheduleEntry]:
    """
    Build a schedule for all jobs.

    Parameters
    ----------
    jobs : list of Job
        All jobs visible at this decision point.
    fixed_entries : list of ScheduleEntry
        Operations already completed or in-progress (frozen).
    current_time : float
        The decision-point timestamp.
    """
    tracker = ResourceTracker()
    # Pre-populate tracker from fixed entries
    for e in fixed_entries:
        tracker.add(e.machine, e.start_time, e.end_time)

    schedule: List[ScheduleEntry] = list(fixed_entries)

    # Determine which operations remain to be scheduled
    fixed_set = {(e.job_id, e.op_idx) for e in fixed_entries}

    # Gather unscheduled operations
    unscheduled: List[Tuple[Job, Operation]] = []
    for job in jobs:
        for op in job.operations:
            if (job.job_id, op.op_idx) not in fixed_set:
                unscheduled.append((job, op))

    # Sort by EDD then job_id
    unscheduled.sort(key=lambda x: (x[0].due_date, x[0].job_id, x[1].op_idx))

    # Process operations in priority order
    # We may need multiple passes because of precedence within a job
    scheduled_set: set = fixed_set.copy()

    while unscheduled:
        progress = False
        remaining: List[Tuple[Job, Operation]] = []

        for job, op in unscheduled:
            # Can this operation be scheduled now?
            # Check: is the previous op of the same job already scheduled?
            prev_scheduled = True
            prev_end = job.release_date
            if op.op_idx > 0:
                prev_key = (job.job_id, op.op_idx - 1)
                if prev_key not in scheduled_set:
                    prev_scheduled = False

            if not prev_scheduled:
                remaining.append((job, op))
                continue

            # Skip if already scheduled as part of a no-wait pair
            if (job.job_id, op.op_idx) in scheduled_set:
                continue

            # Find the end time of the previous operation
            if op.op_idx > 0:
                for e in schedule:
                    if e.job_id == job.job_id and e.op_idx == op.op_idx - 1:
                        prev_end = e.end_time
                        break

            # Determine if this op starts a no-wait chain
            if op.no_wait_next:
                # Build the full no-wait chain
                chain_ops: List[Operation] = [op]
                idx = op.op_idx
                while (chain_ops[-1].no_wait_next
                       and idx + 1 < len(job.operations)):
                    chain_ops.append(job.operations[idx + 1])
                    idx += 1

                # Does this chain include the last operation?
                chain_target: Optional[float] = None
                if job.alpha > 0:
                    last_in_chain = (chain_ops[-1].op_idx
                                     == len(job.operations) - 1)
                    if last_in_chain:
                        chain_target = job.due_date

                start, machines = _find_no_wait_chain_start(
                    job, chain_ops, prev_end, tracker, current_time,
                    chain_target)
                if start is None or machines is None:
                    remaining.append((job, op))
                    continue

                # Create schedule entries for the whole chain
                offset = 0.0
                for i, chain_op in enumerate(chain_ops):
                    end_t = start + offset + chain_op.proc_time
                    e = ScheduleEntry(job.job_id, chain_op.op_idx,
                                      machines[i],
                                      start + offset, end_t)
                    schedule.append(e)
                    tracker.add(machines[i],
                                e.start_time, e.end_time)
                    scheduled_set.add((job.job_id, chain_op.op_idx))
                    offset += chain_op.proc_time
                progress = True
            else:
                # Single operation
                target_end: Optional[float] = None
                if job.alpha > 0 and op.op_idx == len(job.operations) - 1:
                    target_end = job.due_date

                start, m = _find_single_start(job, op, prev_end, tracker,
                                              current_time, target_end)
                if start is None:
                    remaining.append((job, op))
                    continue

                e = ScheduleEntry(job.job_id, op.op_idx, m,
                                  start, start + op.proc_time)
                schedule.append(e)
                tracker.add(m, e.start_time, e.end_time)
                scheduled_set.add((job.job_id, op.op_idx))
                progress = True

        if not progress:
            # Should not happen for a feasible instance, but guard against loops
            break
        unscheduled = remaining

    # Sort schedule by start time
    schedule.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
    return schedule


def _find_single_start(
    job: Job,
    op: Operation,
    prev_end: float,
    tracker: ResourceTracker,
    current_time: float,
    target_end: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Find a feasible start time and machine for a single operation.

    Uses free-gap analysis: for each candidate machine, enumerates every
    contiguous free window, computes the start time within that window
    that gives the lowest penalty, and picks the best across all machines
    and gaps.
    """
    candidate_machines = compatible_machines(op.machine_type)
    if not candidate_machines:
        return None, None

    best_start = float('inf')
    best_penalty = float('inf')
    best_machine = None
    t0 = max(prev_end, current_time, job.release_date)

    for m in candidate_machines:
        gaps = _free_gaps(m, t0, tracker)

        for gap_start, gap_end in gaps:
            if gap_end - gap_start < op.proc_time - 1e-9:
                continue   # gap too short

            # Optimal t within this gap
            if target_end is not None:
                ideal_t = target_end - op.proc_time
                cand_t = max(gap_start, min(ideal_t, gap_end - op.proc_time))
            else:
                cand_t = gap_start   # earliest-start heuristic

            # Clamp cand_t to the feasible window
            found_t = max(gap_start, min(cand_t, gap_end - op.proc_time))

            end_t = found_t + op.proc_time
            penalty = (_compute_penalty(end_t, target_end, job.alpha, job.beta)
                       if target_end is not None else found_t)

            if penalty < best_penalty - 1e-9:
                best_penalty = penalty
                best_start = found_t
                best_machine = m

    if best_machine is None:
        return None, None
    return best_start, best_machine



def _interval_free_in_tracker(
    machine: str,
    start: float,
    end: float,
    tracker: ResourceTracker,
) -> bool:
    """Verify machine remains free throughout [start, end)."""
    for s, e in tracker.machine_intervals[machine]:
        if s < end and e > start:
            return False
    return True


def _free_gaps(machine: str, t0: float,
               tracker: ResourceTracker) -> List[Tuple[float, float]]:
    """Return free intervals for *machine* starting from *t0*."""
    busy = list(tracker.machine_intervals[machine])
    busy.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    for s, e in busy:
        if merged and s <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    gaps: List[Tuple[float, float]] = []
    cur = t0
    for s, e in merged:
        if s > cur + 1e-9:
            gaps.append((cur, s))
        cur = max(cur, e)
    gaps.append((cur, float('inf')))
    return gaps


def _check_chain_feasible(
    t: float,
    chain_ops: List[Operation],
    machines: Tuple[str, ...],
    tracker: ResourceTracker,
) -> bool:
    """Verify every op in a no-wait chain fits when the chain starts at *t*."""
    offset = 0.0
    for i, op in enumerate(chain_ops):
        m = machines[i]
        start = t + offset
        end = start + op.proc_time
        if not _interval_free_in_tracker(m, start, end, tracker):
            return False
        offset += op.proc_time
    return True


def _find_no_wait_chain_start(
    job: Job,
    chain_ops: List[Operation],
    prev_end: float,
    tracker: ResourceTracker,
    current_time: float,
    target_end: Optional[float] = None,
) -> Tuple[Optional[float], Optional[List[str]]]:
    """
    Find a feasible start for a no-wait chain (≥ 2 consecutive operations).

    Enumerates free gaps of the first machine, then for each machine
    combination picks the *t* that makes completion as close to
    *target_end* as possible while respecting resource availability.
    """
    # Compatible machines for each operation in the chain
    machine_options: List[List[str]] = [
        compatible_machines(op.machine_type)
        for op in chain_ops
    ]
    if any(not opts for opts in machine_options):
        return None, None

    best_start = float('inf')
    best_penalty = float('inf')
    best_machines: Optional[List[str]] = None
    t0 = max(prev_end, current_time, job.release_date)
    total_dur = sum(op.proc_time for op in chain_ops)

    for machines in product(*machine_options):
        m0 = machines[0]
        gaps = _free_gaps(m0, t0, tracker)

        for gap_start, gap_end in gaps:
            if gap_end - gap_start < total_dur - 1e-9:
                continue

            # Optimal t within this gap
            if target_end is not None:
                ideal_t = target_end - total_dur
                cand_t = max(gap_start, min(ideal_t, gap_end - total_dur))
            else:
                cand_t = gap_start

            # Walk within the gap to find a feasible t for the whole chain
            t = cand_t
            found = False
            while t <= gap_end - total_dur + 1e-9:
                if _check_chain_feasible(t, chain_ops, machines, tracker):
                    found = True
                    break
                t += 0.5
            if not found:
                continue

            end_t = t + total_dur
            penalty = (_compute_penalty(end_t, target_end,
                                        job.alpha, job.beta)
                       if target_end is not None else t)

            if penalty < best_penalty - 1e-9:
                best_penalty = penalty
                best_start = t
                best_machines = list(machines)

    if best_machines is None:
        return None, None
    return best_start, best_machines


# ═══════════════════════════════════════════════════════════════════════════════
# MILP-based scheduler (via Gurobi)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import gurobipy as _gp
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False


def schedule_jobs_milp(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float,
    time_limit: float = 60.0,
) -> Optional[List[ScheduleEntry]]:
    """
    Build a schedule via MILP using Gurobi.

    Returns None if Gurobi is not installed or the solver fails.
    """
    if not _HAS_GUROBI:
        return None

    fixed_keys = {(e.job_id, e.op_idx) for e in fixed_entries}

    ops_to_schedule: List[Tuple[Job, Operation]] = []
    for job in jobs:
        for op in job.operations:
            if (job.job_id, op.op_idx) not in fixed_keys:
                ops_to_schedule.append((job, op))

    if not ops_to_schedule:
        return list(fixed_entries)

    env = _gp.Env(params={"OutputFlag": 0})
    model = _gp.Model("FJSP", env=env)
    model.Params.TimeLimit = time_limit

    # ── Variables ──────────────────────────────────────────────────────
    s: Dict[Tuple[int, int], _gp.Var] = {}
    x: Dict[Tuple[Tuple[int, int], str], _gp.Var] = {}

    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        s[key] = model.addVar(
            lb=max(current_time, job.release_date),
            vtype=_gp.GRB.CONTINUOUS,
            name=f"s_{job.job_id}_{op.op_idx}")
        for m in compatible_machines(op.machine_type):
            x[(key, m)] = model.addVar(
                vtype=_gp.GRB.BINARY,
                name=f"x_{job.job_id}_{op.op_idx}_{m}")

    C: Dict[int, _gp.Var] = {}
    E: Dict[int, _gp.Var] = {}
    T: Dict[int, _gp.Var] = {}
    first_start: Dict[int, _gp.Var] = {}
    for job in jobs:
        jid = job.job_id
        C[jid] = model.addVar(lb=0, vtype=_gp.GRB.CONTINUOUS, name=f"C_{jid}")
        E[jid] = model.addVar(lb=0, vtype=_gp.GRB.CONTINUOUS, name=f"E_{jid}")
        T[jid] = model.addVar(lb=0, vtype=_gp.GRB.CONTINUOUS, name=f"T_{jid}")
        first_start[jid] = model.addVar(lb=0, vtype=_gp.GRB.CONTINUOUS,
                                        name=f"fs_{jid}")

    # ── Constraints ────────────────────────────────────────────────────

    # (1) Each operation assigned to exactly one compatible machine
    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        model.addConstr(
            _gp.quicksum(x[(key, m)]
                         for m in compatible_machines(op.machine_type)) == 1)

    # (2) Precedence within each job, including no-wait
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            key_cur = (job.job_id, op_cur.op_idx)
            key_next = (job.job_id, op_next.op_idx)

            if key_cur in fixed_keys:
                cur_end = next(e.end_time for e in fixed_entries
                               if e.job_id == job.job_id
                               and e.op_idx == op_cur.op_idx)
                if key_next in fixed_keys:
                    continue
                if op_cur.no_wait_next:
                    model.addConstr(s[key_next] == cur_end)
                else:
                    model.addConstr(s[key_next] >= cur_end)
            elif key_next not in fixed_keys:
                cur_end_expr = s[key_cur] + op_cur.proc_time
                if op_cur.no_wait_next:
                    model.addConstr(s[key_next] == cur_end_expr)
                else:
                    model.addConstr(s[key_next] >= cur_end_expr)

    # (3) Job completion
    for job in jobs:
        last_key = (job.job_id, len(job.operations) - 1)
        if last_key in fixed_keys:
            last_end = next(e.end_time for e in fixed_entries
                            if e.job_id == job.job_id
                            and e.op_idx == len(job.operations) - 1)
            model.addConstr(C[job.job_id] == last_end)
        else:
            model.addConstr(
                C[job.job_id] >= s[last_key] + job.operations[-1].proc_time)

    # (4) First-start constraint
    for job in jobs:
        first_key = (job.job_id, 0)
        if first_key in fixed_keys:
            fs_val = next(e.start_time for e in fixed_entries
                          if e.job_id == job.job_id and e.op_idx == 0)
            model.addConstr(first_start[job.job_id] == fs_val)
        else:
            model.addConstr(first_start[job.job_id] == s[first_key])
        model.addConstr(C[job.job_id] >= first_start[job.job_id])

    # (5) Earliness / tardiness
    for job in jobs:
        jid = job.job_id
        model.addConstr(E[jid] >= job.due_date - C[jid])
        model.addConstr(T[jid] >= C[jid] - job.due_date)

    # (6) Machine disjunction (big-M)
    BIG_M = 10000.0
    for m in ALL_MACHINES:
        ops_on_m: List[Tuple[Tuple[int, int], Operation]] = []
        for job, op in ops_to_schedule:
            key = (job.job_id, op.op_idx)
            if m in compatible_machines(op.machine_type):
                ops_on_m.append((key, op))

        for i in range(len(ops_on_m)):
            for j in range(i + 1, len(ops_on_m)):
                key_a, op_a = ops_on_m[i]
                key_b, op_b = ops_on_m[j]

                y = model.addVar(
                    vtype=_gp.GRB.BINARY,
                    name=f"y_{key_a[0]}_{key_a[1]}_{key_b[0]}_{key_b[1]}_{m}")

                # a before b  (relaxed when not both assigned to m)
                model.addConstr(
                    s[key_a] + op_a.proc_time <= s[key_b]
                    + BIG_M * (1 - y)
                    + BIG_M * (1 - x[(key_a, m)])
                    + BIG_M * (1 - x[(key_b, m)]))
                # b before a
                model.addConstr(
                    s[key_b] + op_b.proc_time <= s[key_a]
                    + BIG_M * y
                    + BIG_M * (1 - x[(key_a, m)])
                    + BIG_M * (1 - x[(key_b, m)]))

    # (7) Fixed entries block their machines while still in progress
    for e in fixed_entries:
        if e.end_time > current_time:
            for job, op in ops_to_schedule:
                key = (job.job_id, op.op_idx)
                if e.machine in compatible_machines(op.machine_type):
                    model.addConstr(
                        s[key] >= e.end_time
                        - BIG_M * (1 - x[(key, e.machine)]))

    # ── Objective ──────────────────────────────────────────────────────
    model.setObjective(
        _gp.quicksum(job.alpha * E[job.job_id] + job.beta * T[job.job_id]
                     for job in jobs),
        _gp.GRB.MINIMIZE)

    # ── Solve ──────────────────────────────────────────────────────────
    model.optimize()

    if model.Status not in (_gp.GRB.OPTIMAL, _gp.GRB.SUBOPTIMAL,
                            _gp.GRB.TIME_LIMIT):
        return None

    # ── Extract schedule ───────────────────────────────────────────────
    try:
        schedule: List[ScheduleEntry] = list(fixed_entries)
        for job, op in ops_to_schedule:
            key = (job.job_id, op.op_idx)
            start_val = s[key].X
            assigned = None
            for m in compatible_machines(op.machine_type):
                if x[(key, m)].X > 0.5:
                    assigned = m
                    break
            if assigned is None:
                return None
            schedule.append(ScheduleEntry(
                job.job_id, op.op_idx, assigned,
                start_val, start_val + op.proc_time))
        schedule.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
        return schedule
    except Exception:
        return None


def validate_schedule(
    schedule: List[ScheduleEntry],
    jobs: List[Job],
    current_time: float,
    label: str = "",
) -> List[str]:
    """
    Check whether *schedule* satisfies all hard constraints.

    Returns a list of violation messages (empty = feasible).
    """
    errors: List[str] = []

    # ── 1. No missing operations ───────────────────────────────────────
    covered = {(e.job_id, e.op_idx) for e in schedule}
    for job in jobs:
        for op in job.operations:
            if (job.job_id, op.op_idx) not in covered:
                errors.append(f"[{label}] Missing: J{job.job_id}-Op{op.op_idx+1}")

    # ── 2. Start-time lower bounds ─────────────────────────────────────
    for e in schedule:
        job = next(j for j in jobs if j.job_id == e.job_id)
        lb = max(current_time, job.release_date)
        if e.start_time < lb - 1e-6:
            errors.append(
                f"[{label}] J{e.job_id}-Op{e.op_idx+1}: "
                f"start={e.start_time:.2f} < max(current={current_time}, "
                f"release={job.release_date}) = {lb}")

    # ── 3. Precedence & no-wait within each job ────────────────────────
    for job in jobs:
        sorted_ops = sorted(
            [e for e in schedule if e.job_id == job.job_id],
            key=lambda e: e.op_idx)
        for k in range(len(sorted_ops) - 1):
            e_cur = sorted_ops[k]
            e_next = sorted_ops[k + 1]
            if e_next.op_idx != e_cur.op_idx + 1:
                continue  # non-consecutive, skip
            op_cur = job.operations[e_cur.op_idx]
            if op_cur.no_wait_next:
                if abs(e_next.start_time - e_cur.end_time) > 1e-6:
                    errors.append(
                        f"[{label}] J{job.job_id}: no-wait violated between "
                        f"Op{e_cur.op_idx+1}(end={e_cur.end_time:.2f}) and "
                        f"Op{e_next.op_idx+1}(start={e_next.start_time:.2f})")
            else:
                if e_next.start_time < e_cur.end_time - 1e-6:
                    errors.append(
                        f"[{label}] J{job.job_id}: precedence violated: "
                        f"Op{e_next.op_idx+1} start={e_next.start_time:.2f} "
                        f"< Op{e_cur.op_idx+1} end={e_cur.end_time:.2f}")

    # ── 4. Machine exclusivity ─────────────────────────────────────────
    for m in ALL_MACHINES:
        ops_on_m = sorted(
            [e for e in schedule if e.machine == m],
            key=lambda e: e.start_time)
        for k in range(len(ops_on_m) - 1):
            if ops_on_m[k].end_time > ops_on_m[k + 1].start_time + 1e-6:
                errors.append(
                    f"[{label}] Machine {m}: overlap J{ops_on_m[k].job_id}-"
                    f"Op{ops_on_m[k].op_idx+1} [{ops_on_m[k].start_time:.2f},"
                    f"{ops_on_m[k].end_time:.2f}) with J{ops_on_m[k+1].job_id}-"
                    f"Op{ops_on_m[k+1].op_idx+1} [{ops_on_m[k+1].start_time:.2f},"
                    f"{ops_on_m[k+1].end_time:.2f})")

    # ── 5. Machine compatibility ───────────────────────────────────────
    for e in schedule:
        job = next(j for j in jobs if j.job_id == e.job_id)
        op = job.operations[e.op_idx]
        compat = compatible_machines(op.machine_type)
        if e.machine not in compat:
            errors.append(
                f"[{label}] J{e.job_id}-Op{e.op_idx+1}: assigned to "
                f"{e.machine}, not in {compat} (type {op.machine_type})")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation engine
# ═══════════════════════════════════════════════════════════════════════════════

def simulate() -> Tuple[List[ScheduleEntry], List[ScheduleEntry]]:
    """
    Run the two-stage simulation.

    Returns
    -------
    initial_schedule : list of ScheduleEntry   (t=0 plan)
    updated_schedule : list of ScheduleEntry   (t=24 plan)
    """
    all_jobs = build_jobs()

    # ── Stage 1: t = 0 ──────────────────────────────────────────────────
    visible_jobs_t0 = [j for j in all_jobs if j.arrival_time <= 0]
    initial_schedule = schedule_jobs(visible_jobs_t0, [], 0.0)

    # ── Simulation until t = 24 ─────────────────────────────────────────
    # Determine which operations are completed or in-progress at t=24
    fixed_at_t24: List[ScheduleEntry] = []
    for e in initial_schedule:
        if e.start_time < 24:
            # Already started — freeze it
            e_fixed = ScheduleEntry(e.job_id, e.op_idx, e.machine,
                                    e.start_time, e.end_time,
                                    fixed=True)
            fixed_at_t24.append(e_fixed)

    # ── Stage 2: t = 24 ─────────────────────────────────────────────────
    visible_jobs_t24 = [j for j in all_jobs if j.arrival_time <= 24]
    updated_schedule = schedule_jobs(visible_jobs_t24, fixed_at_t24, 24.0)

    return initial_schedule, updated_schedule


def simulate_milp(
    time_limit: float = 60.0,
) -> Tuple[Optional[List[ScheduleEntry]], Optional[List[ScheduleEntry]]]:
    """
    Run the two-stage simulation using MILP (Gurobi).

    Returns (None, None) if Gurobi is not available.
    """
    all_jobs = build_jobs()

    # ── Stage 1: t = 0 ──────────────────────────────────────────────────
    visible_jobs_t0 = [j for j in all_jobs if j.arrival_time <= 0]
    initial_schedule = schedule_jobs_milp(visible_jobs_t0, [], 0.0, time_limit)
    if initial_schedule is None:
        return None, None

    # ── Determine frozen operations at t=24 ─────────────────────────────
    fixed_at_t24: List[ScheduleEntry] = []
    for e in initial_schedule:
        if e.start_time < 24:
            e_fixed = ScheduleEntry(e.job_id, e.op_idx, e.machine,
                                    e.start_time, e.end_time,
                                    fixed=True)
            fixed_at_t24.append(e_fixed)

    # ── Stage 2: t = 24 ─────────────────────────────────────────────────
    visible_jobs_t24 = [j for j in all_jobs if j.arrival_time <= 24]
    updated_schedule = schedule_jobs_milp(visible_jobs_t24, fixed_at_t24,
                                          24.0, time_limit)
    if updated_schedule is None:
        return None, None

    return initial_schedule, updated_schedule


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(schedule: List[ScheduleEntry], jobs: List[Job]) -> dict:
    """Compute completion times, earliness, tardiness, and weighted penalty."""
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
        # Active lead time = Cj - start time of first operation
        first_start = min((e.start_time for e in schedule
                           if e.job_id == job.job_id), default=0)
        alt = Cj - first_start
        total_active_lead_time += alt
        details.append((job.job_id, Cj, Ej, Tj, penalty))

    return {
        'completion': completion,
        'total_penalty': total_penalty,
        'total_active_lead_time': total_active_lead_time,
        'details': details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Gantt chart plotting
# ═══════════════════════════════════════════════════════════════════════════════

JOB_COLORS = {
    1: '#4C72B0',   # blue
    2: '#DD8452',   # orange
    3: '#55A868',   # green
    4: '#C44E52',   # red
}

JOB_LABELS = {
    1: 'J₁',
    2: 'J₂',
    3: 'J₃',
    4: 'J₄',
}


def plot_gantt(schedule: List[ScheduleEntry],
               title: str,
               current_time: Optional[float] = None,
               ax: Optional[plt.Axes] = None):
    """
    Draw a Gantt chart on the given axes.

    Each machine is a row.  Operations are coloured by job.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    machines = ALL_MACHINES
    y_positions = {m: i for i, m in enumerate(machines)}

    for entry in schedule:
        y = y_positions[entry.machine]
        color = JOB_COLORS.get(entry.job_id, '#888888')

        bar = ax.barh(y, entry.duration, left=entry.start_time, height=0.55,
                      color=color, edgecolor='white', linewidth=0.5, alpha=0.9)

        # Label each bar with job + operation index
        label = f"{JOB_LABELS.get(entry.job_id, entry.job_id)}-{entry.op_idx + 1}"
        ax.text(entry.start_time + entry.duration / 2, y,
                label, ha='center', va='center', fontsize=7,
                fontweight='bold', color='white')

        # Mark fixed operations with a hatch pattern
        if entry.fixed:
            bar.patches[0].set_hatch('///')
            bar.patches[0].set_edgecolor('black')
            bar.patches[0].set_linewidth(0.8)

    # Styling
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(machines)
    ax.set_ylabel('Machine', fontsize=11)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(left=0, right=max(e.end_time for e in schedule) * 1.08)
    ax.xaxis.set_major_locator(plt.MultipleLocator(24))
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Decision-point line
    if current_time is not None:
        ax.axvline(x=current_time, color='red', linestyle='--', linewidth=1.8,
                   alpha=0.7, label=f'Decision point  t = {current_time}')
        ax.legend(loc='upper right', fontsize=9)

    # Legend for jobs
    legend_patches = [mpatches.Patch(color=JOB_COLORS[jid],
                                      label=f"{JOB_LABELS[jid]} (Job {jid})")
                      for jid in sorted(JOB_COLORS)]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8,
              ncol=4, title='Jobs', title_fontsize=9)


def print_schedule(schedule: List[ScheduleEntry], title: str):
    """Pretty-print a schedule to the console."""
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")
    print(f"  {'Job':>6s}  {'Op':>4s}  {'Machine':>8s}  "
          f"{'Start':>8s}  {'End':>8s}  {'Dur':>6s}  {'Status':>10s}")
    print(f"  {'─' * 72}")
    for e in schedule:
        status = 'FIXED' if e.fixed else 'planned'
        print(f"  {'J' + str(e.job_id):>6s}  {e.op_idx + 1:>4d}  "
              f"{e.machine:>8s}  "
              f"{e.start_time:>8.1f}  {e.end_time:>8.1f}  "
              f"{e.duration:>6.1f}  {status:>10s}")


def print_metrics(schedule: List[ScheduleEntry], jobs: List[Job], label: str):
    """Print completion times and penalty metrics."""
    metrics = compute_metrics(schedule, jobs)
    print(f"\n  ── {label} Metrics ──")
    header = (f"  {'Job':>6s}  {'Cⱼ':>8s}  {'dⱼ':>8s}  "
              f"{'Eⱼ':>8s}  {'Tⱼ':>8s}  {'αⱼEⱼ+βⱼTⱼ':>12s}")
    print(header)
    print(f"  {'─' * len(header)}")
    for jid, Cj, Ej, Tj, pen in metrics['details']:
        job = next(j for j in jobs if j.job_id == jid)
        print(f"  {'J' + str(jid):>6s}  {Cj:>8.1f}  {job.due_date:>8.1f}  "
              f"{Ej:>8.1f}  {Tj:>8.1f}  {pen:>12.1f}")
    print(f"  {'─' * len(header)}")
    print(f"  Total weighted penalty : {metrics['total_penalty']:.1f}")
    print(f"  Total active lead time : {metrics['total_active_lead_time']:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Pharmaceutical FJSP Rescheduling Simulation — Example 1")
    print("=" * 72)
    print(f"  Resources: 4 machines (M1×2, M2×1, M3×1)")

    all_jobs = build_jobs()
    jobs_t0 = [j for j in all_jobs if j.arrival_time <= 0]
    jobs_t24 = [j for j in all_jobs if j.arrival_time <= 24]

    # ── Method 1: Greedy EDD heuristic ───────────────────────────────────
    print("\n" + "=" * 72)
    print("  METHOD 1: Greedy EDD Heuristic")
    print("=" * 72)
    initial_greedy, updated_greedy = simulate()

    print_schedule(initial_greedy, "Initial Schedule (t = 0)")
    print_metrics(initial_greedy, jobs_t0, "Initial (Greedy)")
    print_schedule(updated_greedy, "Updated Schedule (t = 24, J₄ arrived)")
    print_metrics(updated_greedy, jobs_t24, "Updated (Greedy)")

    # Frozen operations at t=24
    print(f"\n  ── Operations frozen at t=24 ──")
    frozen = [e for e in updated_greedy if e.fixed]
    for e in frozen:
        print(f"    J{e.job_id}-Op{e.op_idx + 1}  on {e.machine}  "
              f"[{e.start_time:.1f} → {e.end_time:.1f}]")

    # ── Method 2: MILP (Gurobi) ──────────────────────────────────────────
    initial_milp, updated_milp = simulate_milp()

    if initial_milp is not None and updated_milp is not None:
        print("\n" + "=" * 72)
        print("  METHOD 2: MILP (Gurobi)")
        print("=" * 72)
        print_schedule(initial_milp, "Initial Schedule (t = 0)")
        print_metrics(initial_milp, jobs_t0, "Initial (MILP)")
        print_schedule(updated_milp, "Updated Schedule (t = 24, J₄ arrived)")
        print_metrics(updated_milp, jobs_t24, "Updated (MILP)")

        # ── Side-by-side comparison ────────────────────────────────────
        m_greedy_init = compute_metrics(initial_greedy, jobs_t0)
        m_greedy_upd  = compute_metrics(updated_greedy, jobs_t24)
        m_milp_init   = compute_metrics(initial_milp, jobs_t0)
        m_milp_upd    = compute_metrics(updated_milp, jobs_t24)

        print(f"\n{'─' * 72}")
        print(f"  Comparison: Greedy vs MILP")
        print(f"{'─' * 72}")
        hdr = (f"  {'Stage':<8s} {'Metric':<22s} "
               f"{'Greedy':>10s}  {'MILP':>10s}  {'Gap':>10s}")
        print(hdr)
        print(f"  {'─' * len(hdr)}")

        for label, gm, mm in [("t=0", m_greedy_init, m_milp_init),
                              ("t=24", m_greedy_upd, m_milp_upd)]:
            for mname, key in [("Total Penalty", "total_penalty"),
                               ("Active Lead Time", "total_active_lead_time")]:
                gv = gm[key]
                mv = mm[key]
                gap = gv - mv
                print(f"  {label:<8s} {mname:<22s} "
                      f"{gv:>10.1f}  {mv:>10.1f}  {gap:>+10.1f}")

        # ── Per-job penalty breakdown ──────────────────────────────────
        print(f"\n{'─' * 72}")
        print(f"  Per-Job Penalty Breakdown  (αⱼEⱼ + βⱼTⱼ)")
        print(f"{'─' * 72}")
        jhdr = (f"  {'Job':>6s} {'dⱼ':>6s} {'αⱼ':>4s} {'βⱼ':>4s} "
                f"{'Greedy Cⱼ':>10s} {'MILP Cⱼ':>10s} "
                f"{'G-Pen':>10s} {'M-Pen':>10s} {'ΔPen':>10s}")
        print(jhdr)
        print(f"  {'─' * len(jhdr)}")
        for jid in sorted(set(j.job_id for j in jobs_t24)):
            job = next(j for j in jobs_t24 if j.job_id == jid)
            gC = m_greedy_upd['completion'].get(jid, 0)
            mC = m_milp_upd['completion'].get(jid, 0)
            gE = max(0, job.due_date - gC)
            gT = max(0, gC - job.due_date)
            mE = max(0, job.due_date - mC)
            mT = max(0, mC - job.due_date)
            gPen = job.alpha * gE + job.beta * gT
            mPen = job.alpha * mE + job.beta * mT
            print(f"  {'J'+str(jid):>6s} {job.due_date:>6.0f} "
                  f"{job.alpha:>4.0f} {job.beta:>4.0f} "
                  f"{gC:>10.1f} {mC:>10.1f} "
                  f"{gPen:>10.1f} {mPen:>10.1f} "
                  f"{gPen - mPen:>+10.1f}")

        # ── Constraint validation ──────────────────────────────────────
        print(f"\n{'─' * 72}")
        print(f"  Constraint Validation")
        print(f"{'─' * 72}")
        all_ok = True
        for sched, lbl in [(initial_greedy, "Greedy-t0"),
                           (updated_greedy, "Greedy-t24"),
                           (initial_milp, "MILP-t0"),
                           (updated_milp, "MILP-t24")]:
            jlist = jobs_t0 if "t0" in lbl else jobs_t24
            ct = 0.0 if "t0" in lbl else 24.0
            errs = validate_schedule(sched, jlist, ct, lbl)
            if errs:
                all_ok = False
                for err in errs:
                    print(f"  ✗ {err}")
            else:
                print(f"  ✓ {lbl}: all constraints satisfied")

        if not all_ok:
            print(f"\n  WARNING: constraint violations detected above.")

        # ── Comparison Gantt chart ─────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))

        t0_max = max(max(e.end_time for e in initial_greedy),
                     max(e.end_time for e in initial_milp)) * 1.08
        t24_max = max(max(e.end_time for e in updated_greedy),
                      max(e.end_time for e in updated_milp)) * 1.08
        x_max = max(t0_max, t24_max)

        plot_gantt(initial_greedy, 'Greedy  —  Initial (t=0)', ax=axes[0, 0])
        axes[0, 0].set_xlim(0, x_max)
        plot_gantt(initial_milp, 'MILP  —  Initial (t=0)', ax=axes[0, 1])
        axes[0, 1].set_xlim(0, x_max)

        plot_gantt(updated_greedy,
                   'Greedy  —  Updated (t=24)  (hatched = frozen)',
                   current_time=24, ax=axes[1, 0])
        axes[1, 0].set_xlim(0, x_max)
        plot_gantt(updated_milp,
                   'MILP  —  Updated (t=24)  (hatched = frozen)',
                   current_time=24, ax=axes[1, 1])
        axes[1, 1].set_xlim(0, x_max)

        fig.tight_layout(pad=4.0)
        plt.savefig('gantt_charts_unit.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n  Gantt charts saved to 'gantt_charts_unit.png'")

    else:
        # Gurobi not available — fallback to greedy-only display
        print("\n  [Gurobi not available — showing greedy results only]\n")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))
        t0_max = max(e.end_time for e in initial_greedy) * 1.08
        t24_max = max(e.end_time for e in updated_greedy) * 1.08
        x_max = max(t0_max, t24_max)

        plot_gantt(initial_greedy,
                   'Initial Schedule  (t = 0)  —  J₁, J₂, J₃',
                   ax=ax1)
        ax1.set_xlim(0, x_max)
        plot_gantt(updated_greedy,
                   'Updated Schedule  (t = 24, J₄ arrives)  —  hatched = frozen',
                   current_time=24, ax=ax2)
        ax2.set_xlim(0, x_max)

        fig.tight_layout(pad=3.0)
        plt.savefig('gantt_charts_unit.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n  Gantt charts saved to 'gantt_charts_unit.png'")


if __name__ == '__main__':
    main()
