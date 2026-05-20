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
    clean_room: str       # 'R1' or 'R2'
    operators_needed: int = 1
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
    clean_room: str
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

CLEAN_ROOMS: List[str] = ['R1', 'R2']

# R1 supports all machines; R2 supports only M1, M2
ROOM_MACHINE_COMPAT: Dict[str, List[str]] = {
    'R1': ['M1', 'M2', 'M3'],
    'R2': ['M1', 'M2'],
}

NUM_OPERATORS: int = 3


def compatible_machines(machine_type: str, room: str) -> List[str]:
    """Return concrete machine names of `machine_type` that fit in `room`."""
    if machine_type not in ROOM_MACHINE_COMPAT.get(room, []):
        return []
    return MACHINES_BY_TYPE[machine_type]


# ═══════════════════════════════════════════════════════════════════════════════
# Job / operation definitions (Example 1)
# ═══════════════════════════════════════════════════════════════════════════════

def build_jobs() -> List[Job]:
    return [
        Job(job_id=1, release_date=0, due_date=96, alpha=0, beta=1,
            arrival_time=0, operations=[
                Operation(1, 0, 'M1', 10, 'R1', no_wait_next=True),
                Operation(1, 1, 'M2', 12, 'R1', no_wait_next=True),
                Operation(1, 2, 'M1', 14, 'R1'),
            ]),
        Job(job_id=2, release_date=0, due_date=72, alpha=0, beta=4,
            arrival_time=0, operations=[
                Operation(2, 0, 'M1', 14, 'R2'),
                Operation(2, 1, 'M2', 22, 'R2', no_wait_next=True),
                Operation(2, 2, 'M1', 19, 'R2'),
            ]),
        Job(job_id=3, release_date=0, due_date=120, alpha=1, beta=2,
            arrival_time=0, operations=[
                Operation(3, 0, 'M1', 12, 'R1', no_wait_next=True),
                Operation(3, 1, 'M2', 12, 'R1', no_wait_next=True),
                Operation(3, 2, 'M1', 18, 'R1'),
            ]),
        Job(job_id=4, release_date=48, due_date=96, alpha=1, beta=6,
            arrival_time=24, operations=[
                Operation(4, 0, 'M1', 12, 'R2'),
                Operation(4, 1, 'M2', 14, 'R2', no_wait_next=True),
                Operation(4, 2, 'M3', 10, 'R1'),
            ]),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Resource availability tracker
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceTracker:
    """Maintains busy intervals for machines, rooms, and operators."""

    def __init__(self):
        # machine -> list of (start, end) busy intervals
        self.machine_intervals: Dict[str, List[Tuple[float, float]]] = {
            m: [] for m in ALL_MACHINES}
        # room -> list of (start, end) busy intervals
        self.room_intervals: Dict[str, List[Tuple[float, float]]] = {
            r: [] for r in CLEAN_ROOMS}
        # operator busy intervals  (flat list)
        self.operator_intervals: List[Tuple[float, float]] = []

    def add(self, machine: str, room: str, start: float, end: float,
            operators: int = 1):
        self.machine_intervals[machine].append((start, end))
        self.room_intervals[room].append((start, end))
        for _ in range(operators):
            self.operator_intervals.append((start, end))

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

    # ------------------------------------------------------------------
    # Room
    # ------------------------------------------------------------------
    def room_free_at(self, room: str, t: float) -> bool:
        return not any(s < t + 1e-9 and e > t + 1e-9
                       for s, e in self.room_intervals[room])

    def room_next_free(self, room: str, t: float) -> float:
        intervals = sorted(self.room_intervals[room])
        for s, e in intervals:
            if s < t + 1e-9 and e > t + 1e-9:
                t = e
        return t

    # ------------------------------------------------------------------
    # Machine + room combo
    # ------------------------------------------------------------------
    def machine_room_next_free(self, machine: str, room: str, t: float) -> float:
        """Earliest time >= t when *both* machine and room are free."""
        t = max(self.machine_next_free(machine, t),
                self.room_next_free(room, t))
        # After jumping, one resource might be inside the other's busy window
        changed = True
        while changed:
            changed = False
            t2 = max(self.machine_next_free(machine, t),
                     self.room_next_free(room, t))
            if t2 > t + 1e-9:
                t = t2
                changed = True
        return t

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------
    def operators_free_count(self, t: float) -> int:
        busy = sum(1 for s, e in self.operator_intervals
                   if s < t + 1e-9 and e > t + 1e-9)
        return NUM_OPERATORS - busy

    def operators_available_throughout(self, start: float, end: float,
                                       needed: int) -> bool:
        """Check that at least `needed` operators are free across [start, end)."""
        # Gather all time points where operator busy status could change
        times = {start, end}
        for s, e in self.operator_intervals:
            if s < end and e > start:
                times.add(s)
                times.add(e)
        times = sorted(times)
        for i in range(len(times) - 1):
            t_mid = (times[i] + times[i + 1]) / 2.0
            if start <= t_mid < end:
                if self.operators_free_count(t_mid) < needed:
                    return False
        return True

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
        tracker.add(e.machine, e.clean_room, e.start_time, e.end_time)

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
                                      machines[i], chain_op.clean_room,
                                      start + offset, end_t)
                    schedule.append(e)
                    tracker.add(machines[i], chain_op.clean_room,
                                e.start_time, e.end_time,
                                chain_op.operators_needed)
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

                e = ScheduleEntry(job.job_id, op.op_idx, m, op.clean_room,
                                  start, start + op.proc_time)
                schedule.append(e)
                tracker.add(m, op.clean_room, e.start_time, e.end_time,
                            op.operators_needed)
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
    contiguous free window of (machine, clean_room), computes the start
    time within that window that gives the lowest penalty, and picks the
    best across all machines and gaps.
    """
    candidate_machines = compatible_machines(op.machine_type, op.clean_room)
    if not candidate_machines:
        return None, None

    best_start = float('inf')
    best_penalty = float('inf')
    best_machine = None
    t0 = max(prev_end, current_time, job.release_date)

    for m in candidate_machines:
        gaps = _free_gaps(m, op.clean_room, t0, tracker)

        for gap_start, gap_end in gaps:
            if gap_end - gap_start < op.proc_time - 1e-9:
                continue   # gap too short

            # Optimal t within this gap
            if target_end is not None:
                ideal_t = target_end - op.proc_time
                cand_t = max(gap_start, min(ideal_t, gap_end - op.proc_time))
            else:
                cand_t = gap_start   # earliest-start heuristic

            # Verify operator availability at cand_t; walk if needed
            found_t = _seek_operator_ok(cand_t, gap_start,
                                        gap_end - op.proc_time,
                                        op, tracker)
            if found_t is None:
                continue

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


def _seek_operator_ok(
    cand_t: float,
    lo: float,
    hi: float,
    op: Operation,
    tracker: ResourceTracker,
    max_steps: int = 40,
) -> Optional[float]:
    """
    Try to land at *cand_t* (clamped to [*lo*, *hi*]).
    If operators are unavailable, walk forward then backward in 0.5 steps
    looking for the nearest feasible instant inside the window.
    """
    cand_t = max(lo, min(cand_t, hi))
    # Forward
    t = cand_t
    for _ in range(max_steps):
        if t > hi:
            break
        if tracker.operators_available_throughout(t, t + op.proc_time,
                                                   op.operators_needed):
            return t
        t += 0.5
    # Backward
    t = cand_t - 0.5
    for _ in range(max_steps):
        if t < lo:
            break
        if tracker.operators_available_throughout(t, t + op.proc_time,
                                                   op.operators_needed):
            return t
        t -= 0.5
    return None


def _interval_free_in_tracker(
    machine: str,
    room: str,
    start: float,
    end: float,
    tracker: ResourceTracker,
) -> bool:
    """Verify machine and room remain free throughout [start, end)."""
    # Check machine — see if any busy interval overlaps (start, end)
    for s, e in tracker.machine_intervals[machine]:
        if s < end and e > start:
            return False
    # Check room
    for s, e in tracker.room_intervals[room]:
        if s < end and e > start:
            return False
    return True


def _free_gaps(machine: str, room: str, t0: float,
               tracker: ResourceTracker) -> List[Tuple[float, float]]:
    """Return free intervals for (*machine*, *room*) starting from *t0*."""
    # Collect busy intervals from machine and room, then merge
    busy = (list(tracker.machine_intervals[machine])
            + list(tracker.room_intervals[room]))
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
        if not _interval_free_in_tracker(m, op.clean_room, start, end, tracker):
            return False
        offset += op.proc_time
    total_dur = sum(op.proc_time for op in chain_ops)
    return tracker.operators_available_throughout(t, t + total_dur,
                                                   chain_ops[0].operators_needed)


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

    Enumerates free gaps of (m₀, room₀), then for each machine combination
    picks the *t* that makes completion as close to *target_end* as possible
    while respecting every phase's resource availability.
    """
    # Compatible machines for each operation in the chain
    machine_options: List[List[str]] = [
        compatible_machines(op.machine_type, op.clean_room)
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
        gaps = _free_gaps(m0, chain_ops[0].clean_room, t0, tracker)

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
                                    e.clean_room, e.start_time, e.end_time,
                                    fixed=True)
            fixed_at_t24.append(e_fixed)

    # ── Stage 2: t = 24 ─────────────────────────────────────────────────
    visible_jobs_t24 = [j for j in all_jobs if j.arrival_time <= 24]
    updated_schedule = schedule_jobs(visible_jobs_t24, fixed_at_t24, 24.0)

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

        # Annotate clean room
        ax.text(entry.start_time + entry.duration / 2, y + 0.22,
                entry.clean_room, ha='center', va='center', fontsize=5.5,
                color='white', alpha=0.85)

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
    print(f"  {'Job':>6s}  {'Op':>4s}  {'Machine':>8s}  {'Room':>4s}  "
          f"{'Start':>8s}  {'End':>8s}  {'Dur':>6s}  {'Status':>10s}")
    print(f"  {'─' * 72}")
    for e in schedule:
        status = 'FIXED' if e.fixed else 'planned'
        print(f"  {'J' + str(e.job_id):>6s}  {e.op_idx + 1:>4d}  "
              f"{e.machine:>8s}  {e.clean_room:>4s}  "
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
    print(f"  Resources: 2 rooms (R1,R2) | "
          f"4 machines (M1×2, M2×1, M3×1) | {NUM_OPERATORS} operators")
    print(f"  Room compatibility: R1→M1,M2,M3  |  R2→M1,M2")

    initial_schedule, updated_schedule = simulate()
    all_jobs = build_jobs()

    # ── Console output ──────────────────────────────────────────────────
    print_schedule(initial_schedule, "Initial Schedule (t = 0)")
    print_metrics(initial_schedule,
                  [j for j in all_jobs if j.arrival_time <= 0],
                  "Initial")

    print_schedule(updated_schedule, "Updated Schedule (t = 24, J₄ arrived)")
    print_metrics(updated_schedule,
                  [j for j in all_jobs if j.arrival_time <= 24],
                  "Updated")

    # Describe frozen operations at t=24
    print(f"\n  ── Operations frozen at t=24 ──")
    frozen = [e for e in updated_schedule if e.fixed]
    for e in frozen:
        print(f"    J{e.job_id}-Op{e.op_idx + 1}  on {e.machine} in "
              f"{e.clean_room}  [{e.start_time:.1f} → {e.end_time:.1f}]")

    # ── Gantt charts ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))

    t0_max = max(e.end_time for e in initial_schedule) * 1.08
    t24_max = max(e.end_time for e in updated_schedule) * 1.08
    x_max = max(t0_max, t24_max)

    plot_gantt(initial_schedule,
               'Initial Schedule  (t = 0)  —  J₁, J₂, J₃',
               ax=ax1)
    ax1.set_xlim(0, x_max)

    plot_gantt(updated_schedule,
               'Updated Schedule  (t = 24, J₄ arrives)  —  hatched = frozen',
               current_time=24,
               ax=ax2)
    ax2.set_xlim(0, x_max)

    fig.tight_layout(pad=3.0)
    plt.savefig('gantt_charts.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n  Gantt charts saved to 'gantt_charts.png'")


if __name__ == '__main__':
    main()
