"""Single-level rule-based FJSP scheduling for the Kacem 8x8 benchmark.

Three dispatching rules are implemented as pure dynamic schedulers:
  SPT  — Shortest Processing Time (choose the operation with the shortest
         processing time on the free machine)
  FIFO — First In First Out (choose the operation that became ready earliest)
  WINQ — Work In Next Queue (choose the operation whose job's *next* operation
         faces the least total queue congestion)

Each rule handles the same dynamic events as the Gurobi baselines:
  - t=0  : J1-J8 arrive
  - t=2  : J9, J10 arrive
  - t=6  : M3 breaks down (Service Unit 1)

The scheduler is single-level — all 8 machines belong to one pool with no
service-unit decomposition.  This serves as a pure rule-based reference
point for comparing against the two-level architecture.

For each rule, three snapshot Gantt charts are produced (t=0, t=2, t=6)
showing the schedule state at each decision point.  Completed and
in-progress operations before the decision point are marked as **fixed**.

Output
------
  output/pure_rule_results.json        — cached simulation results
  output/fig_pure_gantt_spt.png        — SPT  snapshot Gantt  (t=0 | t=2 | t=6)
  output/fig_pure_gantt_fifo.png       — FIFO snapshot Gantt  (t=0 | t=2 | t=6)
  output/fig_pure_gantt_winq.png       — WINQ snapshot Gantt  (t=0 | t=2 | t=6)
  Console summary with C_max and compute-time for each rule.

Usage
------
  python baseline-clairvoyant/run_pure_baselines.py
"""

import heapq
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Tuple

# Support running as a plain script (the directory name has hyphens -> not a
# valid Python package, so we add it to sys.path instead of using relative
# imports).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from models import Job, Operation, ScheduleEntry
from plotting import plot_gantt, job_color, job_label, Y_MACHINES

OUTPUT_DIR = os.path.join(_HERE, "output")
DATA_PATH = os.path.join(_HERE, "kacem_data.json")
CACHE_PATH = os.path.join(OUTPUT_DIR, "pure_rule_results.json")

# All 8 machines in display order (consistent with plotting.py)
ALL_MACHINES = list(Y_MACHINES)  # ['M1','M2','M3','M4','M5','M6','M7','M8']

# Decision points for snapshot Gantt charts
SNAPSHOT_TIMES = [0.0, 2.0, 6.0]


# ═════════════════════════════════════════════════════════════════════════
#  Data loading (shared with run_baselines.py)
# ═════════════════════════════════════════════════════════════════════════

def _build_job(obj: dict) -> Job:
    """Convert a JSON job dict into a ``Job`` instance."""
    ops = [Operation(job_id=obj["job_id"],
                     op_idx=op["op_idx"],
                     times=op["times"])
           for op in obj["operations"]]
    return Job(job_id=obj["job_id"],
               arrival_time=obj.get("arrival_time", 0.0),
               operations=ops)


def load_data() -> tuple:
    """Load *kacem_data.json* and return  (initial_jobs, dynamic_jobs, disruptions)."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    initial = [_build_job(j) for j in data["initial_jobs"]]
    dynamic = [_build_job(j) for j in data.get("dynamic_jobs", [])]
    disruptions = data.get("disruptions", [])
    return initial, dynamic, disruptions


# ═════════════════════════════════════════════════════════════════════════
#  Rescheduling helpers
# ═════════════════════════════════════════════════════════════════════════

def _split_schedule_at_time(
    schedule: List[ScheduleEntry],
    current_time: float,
    previously_fixed_keys: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[List[ScheduleEntry], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Split a schedule at *current_time* into fixed / completed / in-progress.

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
    if previously_fixed_keys is None:
        previously_fixed_keys = set()

    fixed_entries: List[ScheduleEntry] = []
    completed_keys: Set[Tuple[int, int]] = set()
    in_progress_keys: Set[Tuple[int, int]] = set()

    for e in schedule:
        key = (e.job_id, e.op_idx)
        if key in previously_fixed_keys:
            fixed_entries.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time, fixed=True))
            completed_keys.add(key)
            continue

        if e.end_time <= current_time:
            fixed_entries.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time, fixed=True))
            completed_keys.add(key)
        elif e.start_time < current_time < e.end_time:
            fixed_entries.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time, fixed=True))
            in_progress_keys.add(key)

    return fixed_entries, completed_keys, in_progress_keys


def _get_ready_successors(
    completed_keys: Set[Tuple[int, int]],
    in_progress_keys: Set[Tuple[int, int]],
    job_map: Dict[int, Job],
    fixed_keys: Set[Tuple[int, int]],
    new_job_ids: Optional[Set[int]] = None,
) -> List[Tuple[int, int]]:
    """Return operations that are ready for dispatch at a decision point.

    Ready means: the operation's predecessor is completed AND the operation
    itself is not already fixed, completed, or in-progress.

    For newly arrived jobs, ALL operations are returned (the scheduler will
    only pick the first one since successors aren't ready yet).
    """
    if new_job_ids is None:
        new_job_ids = set()

    ready: List[Tuple[int, int]] = []

    for job in job_map.values():
        jid = job.job_id

        # New jobs: only the first operation is ready; successors become
        # ready when their predecessor completes (handled by event loop).
        if jid in new_job_ids:
            first_key = (jid, 0)
            if first_key not in fixed_keys:
                ready.append(first_key)
            continue

        # Existing jobs: find the first operation that isn't completed,
        # in-progress, or fixed — its predecessor must be completed.
        for idx, op in enumerate(job.operations):
            key = (jid, op.op_idx)
            if key in fixed_keys or key in in_progress_keys:
                continue
            if key in completed_keys:
                continue
            if idx == 0:
                ready.append(key)
                break
            else:
                prev_key = (jid, idx - 1)
                if prev_key in completed_keys:
                    ready.append(key)
                    break
                else:
                    # Predecessor not completed → this and later ops blocked
                    break

    return ready


# ═════════════════════════════════════════════════════════════════════════
#  Event-driven simulation engine
# ═════════════════════════════════════════════════════════════════════════

# Internal event types
_EVT_JOB_ARRIVAL = 0    # data = job_id
_EVT_OP_COMPLETE = 1    # data = (job_id, op_idx, machine)
_EVT_MACHINE_BREAK = 2  # data = machine_name

# Tie-breaker: secondary sort key for rule comparison
_TIEBREAK = lambda jid, oidx: (jid, oidx)


def _compute_winq(candidate_jid: int,
                  candidate_op_idx: int,
                  job_map: Dict[int, Job],
                  ready_ops: Set[Tuple[int, int]],
                  machine_queues: Dict[str, List[Tuple[int, int]]]) -> float:
    """Compute WINQ priority value for a candidate operation.

    WINQ = average total processing time of operations currently waiting
    in the queues of machines that can process the *next* operation of
    the candidate's job.

    A **lower** WINQ value means the next operation has less competition,
    so the current operation should be prioritised.

    If the candidate is the last operation of its job, WINQ = 0 (highest
    priority — no downstream congestion at all).
    """
    job = job_map[candidate_jid]
    next_op_idx = candidate_op_idx + 1

    if next_op_idx >= len(job.operations):
        return 0.0   # last operation → always top priority

    next_op = job.operations[next_op_idx]
    feasible_for_next = next_op.feasible_machines

    if not feasible_for_next:
        return 0.0

    total_work = 0.0
    for m in feasible_for_next:
        queue = machine_queues.get(m, [])
        work_on_m = 0.0
        for (q_jid, q_oidx) in queue:
            # Don't count the candidate's own next operation (it isn't
            # ready yet — we're still working on the candidate).
            if (q_jid, q_oidx) == (candidate_jid, next_op_idx):
                continue
            q_job = job_map[q_jid]
            q_op = q_job.operations[q_oidx]
            work_on_m += q_op.times.get(m, 0.0)
        total_work += work_on_m

    return total_work / len(feasible_for_next)


def _build_machine_queues(ready_ops: Set[Tuple[int, int]],
                          job_map: Dict[int, Job],
                          machines: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Build per-machine lists of ready operations that can run on each machine."""
    queues: Dict[str, List[Tuple[int, int]]] = {m: [] for m in machines}
    for (jid, oidx) in ready_ops:
        op = job_map[jid].operations[oidx]
        for m in op.feasible_machines:
            if m in queues:
                queues[m].append((jid, oidx))
    return queues


def simulate_rule(rule_name: str,
                  jobs: List[Job],
                  disruptions: List[dict],
                  current_time: float = 0.0,
                  previous_schedule: Optional[List[ScheduleEntry]] = None,
                  previous_partial_entries: Optional[List[ScheduleEntry]] = None,
                  fixed_keys_in: Optional[Set[Tuple[int, int]]] = None,
                  ) -> dict:
    """Run an event-driven simulation of the FJSP under a single dispatching rule.

    Parameters
    ----------
    rule_name : str
        ``'SPT'``, ``'FIFO'``, or ``'WINQ'``.
    jobs : list of Job
        All jobs (initial + dynamic), each with its arrival time.
    disruptions : list of dict
        Each dict has keys ``'time'`` (float) and ``'machine'`` (str).
    current_time : float
        The moment at which the simulation starts (0.0 = from scratch).
    previous_schedule : list of ScheduleEntry or None
        A prior plan to warm-start from.  Ops completed or in-progress at
        *current_time* are fixed; unstarted ops are thrown back into the
        ready pool for rescheduling.
    previous_partial_entries : list of ScheduleEntry or None
        Partial (interrupted) entries from a prior stage.
    fixed_keys_in : set of (job_id, op_idx) or None
        Keys already fixed in even-earlier stages (passed through to
        :func:`_split_schedule_at_time`).

    Returns
    -------
    dict
        ``{'cmax': float, 'entries': list[dict], 'partial_entries': list[dict],
          'compute_time': float}``
        where ``entries`` is the completed schedule and ``partial_entries``
        records operations that were interrupted by machine breakdowns.
    """
    t_start = time.perf_counter()

    # ── Lookups ─────────────────────────────────────────────────────────
    job_map: Dict[int, Job] = {j.job_id: j for j in jobs}

    # ── State ───────────────────────────────────────────────────────────
    # ready_ops: operations whose predecessors are done AND whose job has arrived
    ready_ops: Set[Tuple[int, int]] = set()

    # job_next_ready: for each job, the next operation index that needs to
    # become ready (starts at 0, incremented when an op is dispatched)
    job_next_ready: Dict[int, int] = {j.job_id: 0 for j in jobs}

    # job_arrived: whether the job has arrived yet
    job_arrived: Dict[int, bool] = {j.job_id: False for j in jobs}

    # op_in_progress: (job_id, op_idx) → machine_name for ops currently running
    op_in_progress: Dict[Tuple[int, int], str] = {}

    # op_start_times: (job_id, op_idx) → (start_time, machine)
    # Tracks when each in-progress operation began, so we can create partial
    # entries when a machine-break interrupts it.
    op_start_times: Dict[Tuple[int, int], Tuple[float, str]] = {}

    # machine_free_at: when each machine finishes its current operation
    machine_free_at: Dict[str, float] = {m: current_time for m in ALL_MACHINES}

    # machine_broken_at: the time when a machine breaks (inf = never)
    machine_broken_at: Dict[str, float] = {m: float('inf') for m in ALL_MACHINES}

    # completed schedule entries
    schedule: List[ScheduleEntry] = []

    # partial entries: operations that were interrupted by a machine break
    partial_entries: List[ScheduleEntry] = (
        list(previous_partial_entries) if previous_partial_entries else []
    )

    # Track fixed keys across stages
    fixed_keys: Set[Tuple[int, int]] = set(fixed_keys_in) if fixed_keys_in else set()

    # ── Event queue ─────────────────────────────────────────────────────
    # Each event is a 4-tuple: (time, tiebreak, event_type, data)
    # tiebreak ensures deterministic ordering of simultaneous events.
    events: List[Tuple[float, int, int, object]] = []
    _event_counter = 0

    def push_event(t: float, etype: int, data: object) -> None:
        nonlocal _event_counter
        heapq.heappush(events, (t, _event_counter, etype, data))
        _event_counter += 1

    # ── Warm-start from previous schedule ───────────────────────────────
    if previous_schedule is not None:
        fixed_now, completed_now, in_progress_now = _split_schedule_at_time(
            previous_schedule, current_time, fixed_keys)

        # Identify new jobs arriving at current_time
        new_job_ids: Set[int] = {
            j.job_id for j in jobs
            if j.arrival_time == current_time and j.arrival_time > 0
        }

        # Apply fixed entries to machine state
        for e in fixed_now:
            key = (e.job_id, e.op_idx)
            fixed_keys.add(key)
            if key in in_progress_now:
                # In-progress → machine busy until it completes.
                # Do NOT add to schedule yet — the completion event will do that.
                machine_free_at[e.machine] = max(machine_free_at[e.machine], e.end_time)
                op_in_progress[key] = e.machine
                op_start_times[key] = (e.start_time, e.machine)
                push_event(e.end_time, _EVT_OP_COMPLETE, (e.job_id, e.op_idx, e.machine))
            else:
                # Already completed → add to schedule directly
                schedule.append(e)
            # Update job_next_ready
            if e.op_idx >= job_next_ready.get(e.job_id, 0):
                job_next_ready[e.job_id] = e.op_idx + 1

        # Determine ready ops at current_time
        ready_ops_list = _get_ready_successors(
            completed_now, in_progress_now, job_map, fixed_keys, new_job_ids)
        ready_ops = set(ready_ops_list)

        # Mark jobs that have arrived by current_time
        for job in jobs:
            if job.arrival_time <= current_time:
                job_arrived[job.job_id] = True
    else:
        # ── Cold start (original behaviour) ─────────────────────────────
        # Mark jobs that arrive at t=0 as arrived, and their first ops as ready
        for job in jobs:
            if job.arrival_time <= 0.0:
                job_arrived[job.job_id] = True
                ready_ops.add((job.job_id, 0))

    # Schedule remaining job arrivals (those after current_time)
    for job in jobs:
        if job.arrival_time > current_time:
            push_event(job.arrival_time, _EVT_JOB_ARRIVAL, job.job_id)

    # Schedule machine breakdowns (those after current_time)
    for d in disruptions:
        if d["time"] >= current_time:
            push_event(d["time"], _EVT_MACHINE_BREAK, d["machine"])
        else:
            # Past disruption — mark machine as already broken
            machine_broken_at[d["machine"]] = d["time"]

    # ── Dispatch function ───────────────────────────────────────────────
    def dispatch_machine(m: str, now: float) -> Optional[Tuple[int, int]]:
        """Try to assign a ready operation to machine *m* at time *now*.

        Returns the selected (job_id, op_idx) or None.
        """
        # Machine must be free and not broken
        if now < machine_free_at[m]:
            return None
        if now >= machine_broken_at[m]:
            return None

        # Find candidates: ready ops that can run on this machine
        candidates: List[Tuple[int, int]] = []
        for (jid, oidx) in ready_ops:
            op = job_map[jid].operations[oidx]
            if m in op.times:
                candidates.append((jid, oidx))

        if not candidates:
            return None

        # ── Apply dispatching rule ───────────────────────────────────
        if rule_name == 'SPT':
            # Shortest processing time on *this* machine
            def spt_key(x: Tuple[int, int]) -> Tuple[float, int, int]:
                jid, oidx = x
                return (job_map[jid].operations[oidx].times[m],) + _TIEBREAK(jid, oidx)
            selected = min(candidates, key=spt_key)

        elif rule_name == 'FIFO':
            # Earliest "ready time": when the previous op finished
            # (or job arrival time for the first operation)
            def fifo_key(x: Tuple[int, int]) -> Tuple[float, int, int]:
                jid, oidx = x
                if oidx == 0:
                    ready_time = job_map[jid].arrival_time
                else:
                    # Find the completion time of the immediately preceding op
                    prev_oidx = oidx - 1
                    ready_time = float('inf')
                    for e in schedule:
                        if e.job_id == jid and e.op_idx == prev_oidx:
                            ready_time = e.end_time
                            break
                    if ready_time == float('inf'):
                        ready_time = job_map[jid].arrival_time
                return (ready_time,) + _TIEBREAK(jid, oidx)
            selected = min(candidates, key=fifo_key)

        elif rule_name == 'WINQ':
            machine_queues = _build_machine_queues(ready_ops, job_map, ALL_MACHINES)
            def winq_key(x: Tuple[int, int]) -> Tuple[float, int, int]:
                jid, oidx = x
                w = _compute_winq(jid, oidx, job_map, ready_ops, machine_queues)
                return (w,) + _TIEBREAK(jid, oidx)
            selected = min(candidates, key=winq_key)

        else:
            raise ValueError(f"Unknown dispatching rule: {rule_name}")

        # ── Check if operation can finish before machine breaks ───────
        jid, oidx = selected
        op = job_map[jid].operations[oidx]
        proc_time = op.times[m]
        start_time = max(now, machine_free_at[m], job_map[jid].arrival_time)

        # Find the end time of the previous operation (precedence constraint)
        if oidx > 0:
            prev_end = 0.0
            for e in schedule:
                if e.job_id == jid and e.op_idx == oidx - 1:
                    prev_end = e.end_time
                    break
            start_time = max(start_time, prev_end)

        end_time = start_time + proc_time

        # If the operation would finish after the machine breaks,
        # it cannot be assigned to this machine
        if end_time > machine_broken_at[m]:
            return None

        # ── Assign ──────────────────────────────────────────────────
        ready_ops.remove(selected)
        op_in_progress[(jid, oidx)] = m
        op_start_times[(jid, oidx)] = (start_time, m)
        machine_free_at[m] = end_time
        push_event(end_time, _EVT_OP_COMPLETE, (jid, oidx, m))
        return selected

    # ── Initial dispatch ────────────────────────────────────────────────
    for m in ALL_MACHINES:
        dispatch_machine(m, current_time)

    # ── Main event loop ─────────────────────────────────────────────────
    while events:
        now, _, etype, data = heapq.heappop(events)

        # --- Machine break -------------------------------------------------
        if etype == _EVT_MACHINE_BREAK:
            machine: str = data
            machine_broken_at[machine] = now

            # If an operation was running on this machine, it is interrupted
            interrupted: List[Tuple[int, int]] = []
            for (jid, oidx), m_name in list(op_in_progress.items()):
                if m_name == machine:
                    interrupted.append((jid, oidx))

            for key in interrupted:
                # Record the partial work as a ScheduleEntry
                start_t, m_name = op_start_times[key]
                partial_entries.append(ScheduleEntry(
                    job_id=key[0],
                    op_idx=key[1],
                    machine=m_name,
                    start_time=start_t,
                    end_time=now,        # interrupted at breakdown time
                    fixed=True,
                ))
                del op_in_progress[key]
                del op_start_times[key]
                # The operation goes back to the ready pool (partial work lost)
                ready_ops.add(key)

        # --- Job arrival ---------------------------------------------------
        elif etype == _EVT_JOB_ARRIVAL:
            jid: int = data
            job_arrived[jid] = True
            # Mark the first operation as ready
            if job_next_ready[jid] == 0:
                ready_ops.add((jid, 0))

        # --- Operation completion ------------------------------------------
        elif etype == _EVT_OP_COMPLETE:
            jid, oidx, machine = data

            # Guard: only process if this operation was actually in progress
            # (a machine-break may have cancelled it)
            if (jid, oidx) not in op_in_progress:
                continue
            if op_in_progress[(jid, oidx)] != machine:
                continue

            del op_in_progress[(jid, oidx)]
            del op_start_times[(jid, oidx)]

            job = job_map[jid]
            op = job.operations[oidx]
            end_time_val = now
            start_time_val = end_time_val - op.times[machine]

            schedule.append(ScheduleEntry(
                job_id=jid,
                op_idx=oidx,
                machine=machine,
                start_time=start_time_val,
                end_time=end_time_val,
            ))

            # Mark the next operation of this job as ready (if any)
            next_oidx = oidx + 1
            if next_oidx < len(job.operations):
                job_next_ready[jid] = next_oidx
                ready_ops.add((jid, next_oidx))

        # --- Dispatch idle machines after any event ------------------------
        for m in ALL_MACHINES:
            while True:
                if now < machine_free_at[m]:
                    break
                if now >= machine_broken_at[m]:
                    break
                result = dispatch_machine(m, now)
                if result is None:
                    break

    # ── Assemble result ─────────────────────────────────────────────────
    compute_time = time.perf_counter() - t_start

    # Sort schedules by start time
    schedule.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
    partial_entries.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))

    cmax = max((e.end_time for e in schedule), default=0.0)

    return {
        "cmax": cmax,
        "entries": [asdict(e) for e in schedule],
        "partial_entries": [asdict(e) for e in partial_entries],
        "compute_time": compute_time,
    }


# ═════════════════════════════════════════════════════════════════════════
#  Snapshot builder
# ═════════════════════════════════════════════════════════════════════════

def build_snapshot(final_entries: List[ScheduleEntry],
                   partial_entries: List[ScheduleEntry],
                   snapshot_time: float,
                   job_arrival_times: Optional[Dict[int, float]] = None) -> List[ScheduleEntry]:
    """Build the schedule as seen at a given decision point.

    The snapshot shows the **complete** schedule — both fixed (already
    committed) and future (yet-to-be-executed) operations.  Each operation
    appears as a **single unbroken bar**; in-progress bars are never split.

    * Completed    (end_time <= T)               → ``fixed=True``
    * In-progress  (start_time <= T < end_time)  → complete bar, ``fixed=True``
    * Future       (start_time > T)              → complete bar, ``fixed=False``
    * Interrupted  (partial entry, start <= T)   → complete bar on broken
      machine, ``fixed=True``.  The rescheduled counterpart (same job+op on
      a different machine) only appears for T >= break_time.

    .. note::
       The simulation that produced *final_entries* / *partial_entries*
       should already be restricted to jobs whose ``arrival_time <=
       snapshot_time``.  This function does **not** filter by job arrival;
       it assumes the caller has already run a scenario-appropriate
       simulation.
    """
    snapshot: List[ScheduleEntry] = []

    # Keys of interrupted ops and the time they were interrupted
    interrupted_keys: Set[Tuple[int, int]] = {
        (e.job_id, e.op_idx) for e in partial_entries
    }
    break_time_of: Dict[Tuple[int, int], float] = {
        (e.job_id, e.op_idx): e.end_time for e in partial_entries
    }

    for e in final_entries:
        key = (e.job_id, e.op_idx)

        if key in interrupted_keys:
            # This final entry is a *rescheduled* version (on a new machine).
            # Only reveal it for snapshots at/after the break time.
            break_t = break_time_of[key]
            if snapshot_time < break_t:
                continue
            is_fixed = (e.end_time <= snapshot_time
                        or e.start_time < snapshot_time < e.end_time)
            snapshot.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time,
                fixed=is_fixed,
            ))
        else:
            # Normal (non-interrupted) operation — single complete bar
            if e.end_time <= snapshot_time:
                # Completed
                snapshot.append(ScheduleEntry(
                    job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                    start_time=e.start_time, end_time=e.end_time,
                    fixed=True,
                ))
            elif e.start_time < snapshot_time < e.end_time:
                # In progress — show as complete bar, fixed=True
                # (strict < for start_time so t=0 has no in-progress ops)
                snapshot.append(ScheduleEntry(
                    job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                    start_time=e.start_time, end_time=e.end_time,
                    fixed=True,
                ))
            else:
                # Future — hasn't started yet
                snapshot.append(ScheduleEntry(
                    job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                    start_time=e.start_time, end_time=e.end_time,
                    fixed=False,
                ))

    # 2. Interrupted (partial) operations — show the failed attempt on the
    #    broken machine.
    for e in partial_entries:
        if e.end_time <= snapshot_time:
            # Interruption already happened and is in the past
            snapshot.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time,
                fixed=True,
            ))
        elif e.start_time < snapshot_time < e.end_time:
            # Currently in progress on the broken machine (not yet interrupted)
            snapshot.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time,
                fixed=True,
            ))
        elif e.start_time >= snapshot_time:
            # Future — this attempted dispatch hasn't happened yet
            snapshot.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time,
                fixed=False,
            ))

    snapshot.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
    return snapshot


# ═════════════════════════════════════════════════════════════════════════
#  Experiment runner
# ═════════════════════════════════════════════════════════════════════════

def run_experiments() -> Optional[dict]:
    """Run SPT, FIFO and WINQ rolling-horizon rescheduling on the Kacem 8x8 benchmark.

    The simulation follows three decision stages, all starting from Baseline A
    (the optimal MILP solution for J1-J8):

    * **t=0** : Initial plan = Baseline A schedule (J1-J8, C_max=14).
      No ops are fixed yet — this is the intended plan before any disruption.
    * **t=2** : J9 and J10 arrive.  Ops completed or in-progress by t=2 from
      the Baseline A plan are **fixed**; remaining ops (including J9/J10)
      are rescheduled with the chosen dispatching rule.
    * **t=6** : M3 breaks down.  Ops completed or in-progress by t=6 from the
      t=2 re-plan are **fixed**; ops running on M3 are **interrupted** and
      go back to the ready pool.  All unfixed ops are rescheduled with the
      chosen dispatching rule (M3 is permanently dead).

    Returns a dict with keys ``"rules"`` (rule_name → snapshot_time →
    cmax + schedule + partial_entries) and ``"metadata"``, or ``None``
    on data-load or baseline-cache failure.

    Results are cached to *output/pure_rule_results.json*.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load Baseline A (optimal t=0 schedule for J1-J8) ──────────────────
    baseline_cache = os.path.join(OUTPUT_DIR, "baseline_results.json")
    if not os.path.exists(baseline_cache):
        print("ERROR: baseline_results.json not found. Run run_baselines.py first.",
              file=sys.stderr)
        return None
    with open(baseline_cache, "r", encoding="utf-8") as f:
        bl = json.load(f)
    baseline_a_entries = [ScheduleEntry(**e) for e in bl["baselines"]["A"]["entries"]]
    baseline_a_cmax = bl["baselines"]["A"]["cmax"]
    baseline_a_time = bl["baselines"]["A"].get("compute_time", 0.0)

    # ── Load problem data ────────────────────────────────────────────────
    initial_jobs, dynamic_jobs, disruptions = load_data()
    all_jobs = initial_jobs + dynamic_jobs

    print("=" * 72)
    print("  Kacem 8x8 FJSP -- Single-Level Rule-Based Rescheduling")
    print("=" * 72)
    print(f"  Initial plan  : Baseline A  (J1-J{len(initial_jobs)}, C_max={baseline_a_cmax})")
    print(f"  Initial jobs  : {len(initial_jobs)}  (J1-J{len(initial_jobs)})")
    print(f"  Dynamic jobs  : {len(dynamic_jobs)}  (arrive at t="
          f"{dynamic_jobs[0].arrival_time if dynamic_jobs else 'N/A'})")
    print(f"  Disruptions   : {len(disruptions)}")
    for d in disruptions:
        print(f"                  {d['machine']} breakdown at t={d['time']}")
    total_ops = sum(len(j.operations) for j in all_jobs)
    print(f"  Total ops     : {total_ops}")
    print(f"  Machines      : {len(ALL_MACHINES)}  ({', '.join(ALL_MACHINES)})")
    print(f"  Rescheduling  : t=2 (J9/J10 arrive)  →  t=6 (M3 breakdown)")
    print("-" * 72)

    rules = ["SPT", "FIFO", "WINQ"]
    results: Dict[str, dict] = {}
    # Per-rule compute times (cumulative)
    rule_total_times: Dict[str, float] = {}

    for rule_name in rules:
        print(f"\n[{rule_name}]")
        rule_results: Dict[str, dict] = {}
        t_start_rule = time.perf_counter()

        # ── Stage 0: t=0 — Baseline A initial plan ────────────────────────
        print(f"  Stage 0 — t=0  Using Baseline A  (J1-J{len(initial_jobs)}, "
              f"C_max={baseline_a_cmax:.3f})")
        rule_results["0.0"] = {
            "cmax": baseline_a_cmax,
            "entries": [asdict(e) for e in baseline_a_entries],
            "partial_entries": [],
            "compute_time": baseline_a_time,
        }

        # ── Stage 1: t=2 — J9/J10 arrive, reschedule from Baseline A ──────
        t_event = 2.0
        print(f"  Stage 1 — t={t_event:.0f}  J9, J10 arrive  →  rule-based rescheduling ...")

        sim_2 = simulate_rule(
            rule_name, all_jobs, [],
            current_time=t_event,
            previous_schedule=baseline_a_entries,
        )

        cmax_2 = sim_2["cmax"]
        dt_2 = sim_2["compute_time"]
        n_entries_2 = len(sim_2["entries"])
        n_jobs_2 = len([j for j in all_jobs if j.arrival_time <= t_event])
        print(f"           |  {n_jobs_2:2d} jobs  |  "
              f"C_max = {cmax_2:7.3f}  |  "
              f"compute = {dt_2*1000:5.1f} ms  |  "
              f"ops scheduled = {n_entries_2}")
        rule_results["2.0"] = sim_2

        sched_2_entries = [ScheduleEntry(**e) for e in sim_2["entries"]]
        partial_2 = [ScheduleEntry(**e) for e in sim_2.get("partial_entries", [])]

        # ── Stage 2: t=6 — M3 breakdown, reschedule from t=2 plan ─────────
        t_event = 6.0
        broken_machine = disruptions[0]["machine"] if disruptions else "M3"
        print(f"  Stage 2 — t={t_event:.0f}  {broken_machine} breakdown  →  rule-based rescheduling ...")

        sim_6 = simulate_rule(
            rule_name, all_jobs, disruptions,
            current_time=t_event,
            previous_schedule=sched_2_entries,
            previous_partial_entries=partial_2,
        )

        cmax_6 = sim_6["cmax"]
        dt_6 = sim_6["compute_time"]
        n_entries_6 = len(sim_6["entries"])
        n_partial_6 = len(sim_6["partial_entries"])
        n_jobs_6 = len([j for j in all_jobs if j.arrival_time <= t_event])
        partial_str = f"  partial (interrupted) = {n_partial_6}" if n_partial_6 else ""
        print(f"           |  {n_jobs_6:2d} jobs  |  "
              f"C_max = {cmax_6:7.3f}  |  "
              f"compute = {dt_6*1000:5.1f} ms  |  "
              f"ops scheduled = {n_entries_6}"
              f"{'  |  ' + partial_str if partial_str else ''}")
        rule_results["6.0"] = sim_6

        results[rule_name] = rule_results
        rule_total_times[rule_name] = time.perf_counter() - t_start_rule

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  Summary  (single-level rule-based, rolling-horizon from Baseline A)")
    print("-" * 72)
    header = f"  {'Rule':6s}"
    for t in SNAPSHOT_TIMES:
        header += f"    {'t=' + str(int(t)):>8s}"
    header += f"    {'Total':>8s}"
    print(header)
    print("  " + "-" * (6 + len(SNAPSHOT_TIMES) * 13 + 9))
    for rule_name in rules:
        line = f"  {rule_name:6s}"
        for t in SNAPSHOT_TIMES:
            line += f"    {results[rule_name][str(t)]['cmax']:8.3f}"
        line += f"    {rule_total_times[rule_name]*1000:5.0f}ms"
        t6 = results[rule_name]["6.0"]
        n_partial = len(t6.get("partial_entries", []))
        if n_partial:
            line += f"  [{n_partial} interrupted]"
        print(line)
    print("-" * 72)

    # ── Comparison with Gurobi baselines ──────────────────────────────────
    print("\n  Comparison with Gurobi MILP baselines (clairvoyant / optimal):")
    print(f"  Baseline A  (J1-J8 only, optimal)        C_max = {bl['baselines']['A']['cmax']:7.3f}")
    print(f"  Baseline B  (J1-J10 clairvoyant)         C_max = {bl['baselines']['B']['cmax']:7.3f}")
    print(f"  Baseline C  (J1-J10 + M3 dead at t=6)    C_max = {bl['baselines']['C']['cmax']:7.3f}")

    # ── Assemble output ─────────────────────────────────────────────────
    job_arrival_times = {j.job_id: j.arrival_time for j in all_jobs}
    output = {
        "rules": results,
        "metadata": {
            "description": (
                "Single-level rule-based FJSP rolling-horizon rescheduling "
                "(SPT / FIFO / WINQ) from Baseline A initial plan"
            ),
            "disruptions": disruptions,
            "num_jobs": len(all_jobs),
            "num_machines": len(ALL_MACHINES),
            "job_arrival_times": job_arrival_times,
            "snapshot_times": SNAPSHOT_TIMES,
            "baseline_a_cmax": baseline_a_cmax,
        },
    }

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Cached results ->  {CACHE_PATH}")

    return output


# ═════════════════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════════════════

def _plot_one_rule(rule_name: str,
                   rule_data: dict,
                   disruptions: List[dict],
                   snapshot_times: List[float],
                   job_arrival_times: Optional[Dict[int, float]] = None) -> None:
    """Generate a single figure with 3 snapshot Gantt panels for one rule.

    Each panel shows the full schedule **as planned** at that decision point:

    * t=0 — Baseline A initial plan (J1-J8, all unfixed)
    * t=2 — Re-plan after J9/J10 arrival (fixed ops from Baseline A at t=2,
      rule-based rescheduling for the rest)
    * t=6 — Re-plan after M3 breakdown (fixed ops from t=2 plan at t=6,
      rule-based rescheduling for the rest)

    Parameters
    ----------
    rule_name : str
        ``'SPT'``, ``'FIFO'``, or ``'WINQ'``.
    rule_data : dict
        ``{str(snapshot_time): {cmax, entries, partial_entries, compute_time}}``
    disruptions : list of dict
        All disruptions (for the figure-level annotation).
    snapshot_times : list of float
        Decision-point times (typically [0, 2, 6]).
    job_arrival_times : dict, optional
        Kept for backward compatibility; no longer used.
    """
    # ── Per-snapshot results ────────────────────────────────────────────
    snapshots: List[List[ScheduleEntry]] = []
    cmax_list: List[float] = []
    all_job_ids: Set[int] = set()

    for t in snapshot_times:
        key = str(t)
        sim = rule_data[key]
        final_entries = [ScheduleEntry(**e) for e in sim["entries"]]
        partial_entries = [ScheduleEntry(**e) for e in sim.get("partial_entries", [])]

        snap_sched = build_snapshot(final_entries, partial_entries, t)
        snapshots.append(snap_sched)
        cmax_list.append(sim["cmax"])
        all_job_ids.update(e.job_id for e in snap_sched)

    all_job_ids_sorted = sorted(all_job_ids)

    # ── Global x-limit: max of all three scenario makespans ──────────────
    x_max = max(cmax_list) * 1.10 if max(cmax_list) > 0 else 30

    # ── Disruption annotation ────────────────────────────────────────────
    disruption_str = ""
    if disruptions:
        d = disruptions[0]
        disruption_str = f"  |  {d['machine']} breakdown at t={d['time']:.0f}"

    # ═════════════════════════════════════════════════════════════════════
    #  Figure: 1 row × 3 columns  (t=0 | t=2 | t=6)
    # ═════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(27, 9))
    gs = GridSpec(1, 3, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    snapshot_labels = [
        f"t = {t:.0f}  (decision point)" for t in snapshot_times
    ]

    for i, (t, snap_sched, ax) in enumerate(zip(snapshot_times, snapshots, axes)):
        n_fixed = sum(1 for e in snap_sched if e.fixed)
        cmax_i = cmax_list[i]
        dt_i = rule_data[str(t)]["compute_time"]

        # Highlight J9/J10 only when they exist in this scenario
        hl_set = {9, 10} if any(jid >= 9 for jid in (e.job_id for e in snap_sched)) else set()

        panel_title = (
            f'{rule_name}  —  {snapshot_labels[i]}\n'
            f'({n_fixed} ops fixed  |  '
            f'C_max = {cmax_i:.3f},  compute = {dt_i*1000:.1f} ms)'
        )
        plot_gantt(snap_sched, panel_title,
                   current_time=t,
                   highlight_jobs=hl_set,
                   show_legend=False,
                   ax=ax)
        ax.set_xlim(0, x_max)

    # ── Shared figure-level legend ──────────────────────────────────────
    legend_patches = []
    for jid in all_job_ids_sorted:
        label = f"{job_label(jid)} (Job {jid})"
        # Highlight J9/J10 if they're in the job set
        hl_set_global = {9, 10} if any(jid >= 9 for jid in all_job_ids_sorted) else set()
        if jid in hl_set_global:
            legend_patches.append(mpatches.Patch(
                facecolor=job_color(jid), label=label,
                edgecolor='black', linewidth=2.0))
        else:
            legend_patches.append(mpatches.Patch(
                facecolor=job_color(jid), label=label))
    fig.legend(handles=legend_patches, loc='upper center',
               ncol=min(len(all_job_ids_sorted), 10), fontsize=9,
               title='Jobs', title_fontsize=10,
               bbox_to_anchor=(0.5, 0.99))

    fig.suptitle(
        f'Kacem 8x8  FJSP — {rule_name} Rule-Based Scheduling  '
        f'(Snapshot Gantt Charts{disruption_str})',
        fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout(pad=3.5, rect=[0, 0, 1, 0.96])

    # ── Save ────────────────────────────────────────────────────────────
    filename = f"fig_pure_gantt_{rule_name.lower()}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Gantt chart saved ->  {out_path}")


def plot_results(results: dict) -> None:
    """Generate per-rule snapshot Gantt charts (3 figures, 3 panels each).

    For each rule (SPT / FIFO / WINQ), produces a figure with panels at
    t=0, t=2, t=6 showing the schedule state at that decision point.

    Each panel is drawn from an **independent** simulation that only had
    access to the information available at that decision point.

    Parameters
    ----------
    results : dict
        The dict returned by :func:`run_experiments`, or loaded from the
        JSON cache file.
    """
    rules_data = results["rules"]
    meta = results["metadata"]
    disruptions = meta.get("disruptions", [])

    print("\n  Generating snapshot Gantt charts ...")

    for rule_name in ["SPT", "FIFO", "WINQ"]:
        _plot_one_rule(rule_name, rules_data[rule_name],
                       disruptions, SNAPSHOT_TIMES)

    print("  All charts generated.\n")


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    results = run_experiments()
    if results is None:
        print("ERROR: Failed to run experiments.", file=sys.stderr)
        sys.exit(1)
    plot_results(results)


if __name__ == "__main__":
    main()
