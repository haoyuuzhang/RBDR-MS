"""
Hierarchical Flexible Job-Shop Scheduling Simulation
Based on the three-layer model from hierarchical_modeling.md

Three-layer architecture:
  Shop Level         : assigns operations -> service units
  Service Unit Level : schedules operations on machines within each unit
  Machine Level      : executes production (no disturbances in this case)

Two-stage simulation:
  t=0  : Gurobi MILP computes optimal initial schedule for J1, J2, J3
  t=24 : J4 arrives -> hierarchical reschedule with frozen ops
"""

from dataclasses import dataclass
from itertools import product
from typing import List, Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Operation:
    """A single operation within a job."""
    job_id: int
    op_idx: int
    machine_type: str          # 'M1', 'M2', or 'M3'
    unit_times: Dict[str, float]   # {unit_name: processing_time}


@dataclass
class Job:
    """A job (order) composed of an ordered sequence of operations."""
    job_id: int
    release_date: float
    due_date: float
    alpha: float
    beta: float
    operations: List[Operation]
    arrival_time: float = 0.0


@dataclass
class ScheduleEntry:
    """Records the assignment and timing of one operation."""
    job_id: int
    op_idx: int
    machine: str
    service_unit: str
    start_time: float
    end_time: float
    fixed: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ═══════════════════════════════════════════════════════════════════════════════
# Resource configuration
# ═══════════════════════════════════════════════════════════════════════════════

SERVICE_UNITS: Dict[str, List[str]] = {
    'U1': ['M1_U1', 'M2_U1', 'M3_U1'],
    'U2': ['M1_U2', 'M2_U2'],
}

ALL_MACHINES: List[str] = [m for ml in SERVICE_UNITS.values() for m in ml]

MACHINE_UNIT: Dict[str, str] = {}
for unit, machines in SERVICE_UNITS.items():
    for m in machines:
        MACHINE_UNIT[m] = unit


def machine_type_of(machine: str) -> str:
    return machine.split('_')[0]


TRANSPORT_TIME = 2      # cross-unit transport time
BIG_M = 10000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Job / operation definitions
# ═══════════════════════════════════════════════════════════════════════════════

def build_jobs() -> List[Job]:
    """
    Machine sequences:
      J1, J2, J3 : Op1->M1, Op2->M2, Op3->M1   (sequence 1-2-1)
      J4         : Op1->M3, Op2->M1, Op3->M2   (sequence 3-1-2)
    """
    return [
        Job(job_id=1, release_date=0, due_date=96, alpha=0, beta=1,
            arrival_time=0, operations=[
                Operation(1, 0, 'M1', {'U1': 10, 'U2': 14}),
                Operation(1, 1, 'M2', {'U1': 12, 'U2': 16}),
                Operation(1, 2, 'M1', {'U1': 14, 'U2': 16}),
            ]),
        Job(job_id=2, release_date=0, due_date=72, alpha=0, beta=4,
            arrival_time=0, operations=[
                Operation(2, 0, 'M1', {'U1': 18, 'U2': 14}),
                Operation(2, 1, 'M2', {'U1': 30, 'U2': 22}),
                Operation(2, 2, 'M1', {'U1': 25, 'U2': 20}),
            ]),
        Job(job_id=3, release_date=0, due_date=120, alpha=1, beta=2,
            arrival_time=0, operations=[
                Operation(3, 0, 'M1', {'U1': 16, 'U2': 12}),
                Operation(3, 1, 'M2', {'U1': 16, 'U2': 12}),
                Operation(3, 2, 'M1', {'U1': 24, 'U2': 18}),
            ]),
        Job(job_id=4, release_date=48, due_date=96, alpha=1, beta=6,
            arrival_time=24, operations=[
                Operation(4, 0, 'M3', {'U1': 12}),
                Operation(4, 1, 'M1', {'U1': 12, 'U2': 18}),
                Operation(4, 2, 'M2', {'U1': 24, 'U2': 18}),
            ]),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Shop-level: operation -> service unit assignment (load balancing)
# ═══════════════════════════════════════════════════════════════════════════════

def shop_level_assignment(
    ops_to_assign: List[Tuple[Job, Operation]],
    fixed_entries: List[ScheduleEntry],
) -> Dict[Tuple[int, int], str]:
    """
    Greedy load-balanced assignment of operations to service units.

    Objective: min max_u (sum X_{i,j,u} * P_{i,j,u}) / m_u
    """
    unit_load: Dict[str, float] = {u: 0.0 for u in SERVICE_UNITS}
    for e in fixed_entries:
        unit_load[e.service_unit] += e.duration

    assignment: Dict[Tuple[int, int], str] = {}
    for e in fixed_entries:
        assignment[(e.job_id, e.op_idx)] = e.service_unit

    sorted_ops = sorted(ops_to_assign, key=lambda x: (
        x[0].due_date, x[0].job_id, x[1].op_idx))

    for job, op in sorted_ops:
        feasible_units = list(op.unit_times.keys())
        if not feasible_units:
            continue

        best_unit = None
        best_intensity = float('inf')
        for u in feasible_units:
            new_load = unit_load[u] + op.unit_times[u]
            intensity = new_load / len(SERVICE_UNITS[u])
            if intensity < best_intensity:
                best_intensity = intensity
                best_unit = u

        if best_unit is not None:
            assignment[(job.job_id, op.op_idx)] = best_unit
            unit_load[best_unit] += op.unit_times[best_unit]

    return assignment


# ═══════════════════════════════════════════════════════════════════════════════
# Resource tracker (for greedy scheduler)
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceTracker:
    def __init__(self):
        self.machine_intervals: Dict[str, List[Tuple[float, float]]] = {
            m: [] for m in ALL_MACHINES}

    def add(self, machine: str, start: float, end: float):
        self.machine_intervals[machine].append((start, end))

    def machine_next_free(self, machine: str, t: float) -> float:
        intervals = sorted(self.machine_intervals[machine])
        for s, e in intervals:
            if s < t + 1e-9 and e > t + 1e-9:
                t = e
        return t


# ═══════════════════════════════════════════════════════════════════════════════
# Unit-level greedy scheduler
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_penalty(end_time: float, due_date: float,
                     alpha: float, beta: float) -> float:
    if end_time <= due_date:
        return alpha * (due_date - end_time)
    else:
        return beta * (end_time - due_date)


def schedule_operations(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float,
    unit_assignment: Dict[Tuple[int, int], str],
) -> List[ScheduleEntry]:
    """Greedy EDD scheduling within assigned service units."""
    tracker = ResourceTracker()
    for e in fixed_entries:
        tracker.add(e.machine, e.start_time, e.end_time)

    schedule: List[ScheduleEntry] = list(fixed_entries)
    fixed_set = {(e.job_id, e.op_idx) for e in fixed_entries}

    unscheduled: List[Tuple[Job, Operation]] = []
    for job in jobs:
        for op in job.operations:
            if (job.job_id, op.op_idx) not in fixed_set:
                unscheduled.append((job, op))

    unscheduled.sort(key=lambda x: (x[0].due_date, x[0].job_id, x[1].op_idx))

    scheduled_set: set = fixed_set.copy()

    while unscheduled:
        progress = False
        remaining: List[Tuple[Job, Operation]] = []

        for job, op in unscheduled:
            if op.op_idx > 0:
                prev_key = (job.job_id, op.op_idx - 1)
                if prev_key not in scheduled_set:
                    remaining.append((job, op))
                    continue

            prev_end = job.release_date
            prev_unit = None
            if op.op_idx > 0:
                for e in schedule:
                    if e.job_id == job.job_id and e.op_idx == op.op_idx - 1:
                        prev_end = e.end_time
                        prev_unit = e.service_unit
                        break

            assigned_unit = unit_assignment.get((job.job_id, op.op_idx))
            if assigned_unit is None:
                remaining.append((job, op))
                continue

            transport = TRANSPORT_TIME if (
                prev_unit is not None and prev_unit != assigned_unit) else 0.0

            candidate_machines = [
                m for m in SERVICE_UNITS[assigned_unit]
                if machine_type_of(m) == op.machine_type
            ]

            if not candidate_machines:
                remaining.append((job, op))
                continue

            t0 = max(prev_end + transport, current_time, job.release_date)

            target_end: Optional[float] = None
            if job.alpha > 0 and op.op_idx == len(job.operations) - 1:
                target_end = job.due_date

            best_start = float('inf')
            best_penalty = float('inf')
            best_machine = None

            for m in candidate_machines:
                start = max(t0, tracker.machine_next_free(m, t0))
                end_t = start + op.unit_times[assigned_unit]
                penalty = (_compute_penalty(end_t, target_end or end_t,
                                            job.alpha, job.beta)
                           if target_end is not None else start)

                if penalty < best_penalty - 1e-9:
                    best_penalty = penalty
                    best_start = start
                    best_machine = m

            if best_machine is None:
                remaining.append((job, op))
                continue

            end_t = best_start + op.unit_times[assigned_unit]
            e = ScheduleEntry(job.job_id, op.op_idx, best_machine,
                              assigned_unit, best_start, end_t)
            schedule.append(e)
            tracker.add(best_machine, e.start_time, e.end_time)
            scheduled_set.add((job.job_id, op.op_idx))
            progress = True

        if not progress:
            break
        unscheduled = remaining

    schedule.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
    return schedule


# ═══════════════════════════════════════════════════════════════════════════════
# Hierarchical scheduler (greedy)
# ═══════════════════════════════════════════════════════════════════════════════

def hierarchical_schedule(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float,
) -> List[ScheduleEntry]:
    fixed_set = {(e.job_id, e.op_idx) for e in fixed_entries}
    ops_to_assign: List[Tuple[Job, Operation]] = []
    for job in jobs:
        for op in job.operations:
            if (job.job_id, op.op_idx) not in fixed_set:
                ops_to_assign.append((job, op))

    unit_assignment = shop_level_assignment(ops_to_assign, fixed_entries)
    return schedule_operations(jobs, fixed_entries, current_time, unit_assignment)


# ═══════════════════════════════════════════════════════════════════════════════
# MILP-based optimal scheduler (Gurobi)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import gurobipy as _gp
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False


def schedule_milp(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float,
    time_limit: float = 60.0,
) -> Optional[List[ScheduleEntry]]:
    """
    MILP that jointly optimizes unit assignment and machine scheduling.

    Decision variables:
      u[(j,o), unit]   : operation assigned to service unit
      s[(j,o)]         : start time
      y[(a,b), unit]   : sequencing on shared machine in unit

    Objective: minimize total weighted earliness-tardiness penalty.
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
    model = _gp.Model("HierarchicalFJSP", env=env)
    model.Params.TimeLimit = time_limit

    # ── Variables ──────────────────────────────────────────────────────
    u: Dict[Tuple[int, int, str], _gp.Var] = {}   # unit assignment
    s: Dict[Tuple[int, int], _gp.Var] = {}        # start time

    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        s[key] = model.addVar(
            lb=max(current_time, job.release_date),
            vtype=_gp.GRB.CONTINUOUS,
            name=f"s_{job.job_id}_{op.op_idx}")
        for unit_name in op.unit_times:
            u[(job.job_id, op.op_idx, unit_name)] = model.addVar(
                vtype=_gp.GRB.BINARY,
                name=f"u_{job.job_id}_{op.op_idx}_{unit_name}")

    # Same-unit indicator for transport: same[(j,o), u] = u[j,o,u] * u[j,o+1,u]
    same_unit: Dict[Tuple[int, int, str], _gp.Var] = {}
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            key_cur = (job.job_id, op_cur.op_idx)
            key_next = (job.job_id, op_next.op_idx)
            if key_cur in fixed_keys or key_next in fixed_keys:
                continue
            common_units = set(op_cur.unit_times) & set(op_next.unit_times)
            for unit_name in common_units:
                var = model.addVar(
                    vtype=_gp.GRB.BINARY,
                    name=f"same_{job.job_id}_{idx}_{unit_name}")
                same_unit[(job.job_id, idx, unit_name)] = var

    # Completion, earliness, tardiness
    C: Dict[int, _gp.Var] = {}
    E: Dict[int, _gp.Var] = {}
    T: Dict[int, _gp.Var] = {}
    for job in jobs:
        jid = job.job_id
        C[jid] = model.addVar(lb=0, vtype=_gp.GRB.CONTINUOUS, name=f"C_{jid}")
        E[jid] = model.addVar(lb=0, vtype=_gp.GRB.CONTINUOUS, name=f"E_{jid}")
        T[jid] = model.addVar(lb=0, vtype=_gp.GRB.CONTINUOUS, name=f"T_{jid}")

    # Sequencing variables: for each (unit, machine_type) and each pair of ops
    y: Dict[Tuple, _gp.Var] = {}
    # Group ops by (unit, machine_type) to find competing pairs
    unit_type_ops: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        for unit_name in op.unit_times:
            ut_key = (unit_name, op.machine_type)
            if ut_key not in unit_type_ops:
                unit_type_ops[ut_key] = []
            unit_type_ops[ut_key].append(key)

    for (unit_name, _mtype), op_list in unit_type_ops.items():
        for i in range(len(op_list)):
            for j in range(i + 1, len(op_list)):
                a, b = op_list[i], op_list[j]
                y_key = (a[0], a[1], b[0], b[1], unit_name)
                y[y_key] = model.addVar(
                    vtype=_gp.GRB.BINARY,
                    name=f"y_{a[0]}_{a[1]}_{b[0]}_{b[1]}_{unit_name}")

    model.update()

    # ── Constraints ────────────────────────────────────────────────────

    # (1) Each operation assigned to exactly one feasible unit
    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        model.addConstr(
            _gp.quicksum(u[(key[0], key[1], unit_name)]
                         for unit_name in op.unit_times) == 1,
            name=f"assign_{key[0]}_{key[1]}")

    # (2) Same-unit linearization
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            key_cur = (job.job_id, op_cur.op_idx)
            key_next = (job.job_id, op_next.op_idx)
            if key_cur in fixed_keys or key_next in fixed_keys:
                continue
            common_units = set(op_cur.unit_times) & set(op_next.unit_times)
            for unit_name in common_units:
                sv = same_unit[(job.job_id, idx, unit_name)]
                u_cur = u[(key_cur[0], key_cur[1], unit_name)]
                u_next = u[(key_next[0], key_next[1], unit_name)]
                model.addConstr(sv <= u_cur, name=f"same1_{job.job_id}_{idx}_{unit_name}")
                model.addConstr(sv <= u_next, name=f"same2_{job.job_id}_{idx}_{unit_name}")
                model.addConstr(sv >= u_cur + u_next - 1,
                                name=f"same3_{job.job_id}_{idx}_{unit_name}")

    # (3) Precedence within each job (with transport)
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            key_cur = (job.job_id, op_cur.op_idx)
            key_next = (job.job_id, op_next.op_idx)
            common_units = set(op_cur.unit_times) & set(op_next.unit_times)

            # Processing time of current operation
            if key_cur in fixed_keys:
                cur_end = next(e.end_time for e in fixed_entries
                               if e.job_id == job.job_id
                               and e.op_idx == op_cur.op_idx)
                if key_next in fixed_keys:
                    continue
                model.addConstr(s[key_next] >= cur_end,
                                name=f"prec_{job.job_id}_{idx}_fixed")
            elif key_next in fixed_keys:
                cur_dur_expr = _gp.quicksum(
                    u[(key_cur[0], key_cur[1], unit_name)] * op_cur.unit_times[unit_name]
                    for unit_name in op_cur.unit_times)
                next_start_fixed = next(e.start_time for e in fixed_entries
                                        if e.job_id == job.job_id
                                        and e.op_idx == op_next.op_idx)
                model.addConstr(next_start_fixed >= s[key_cur] + cur_dur_expr,
                                name=f"prec_{job.job_id}_{idx}_to_fixed")
            else:
                cur_dur_expr = _gp.quicksum(
                    u[(key_cur[0], key_cur[1], unit_name)] * op_cur.unit_times[unit_name]
                    for unit_name in op_cur.unit_times)

                # Transport time: 0 if same unit, LT if different
                if common_units:
                    same_sum = _gp.quicksum(
                        same_unit[(job.job_id, idx, unit_name)]
                        for unit_name in common_units)
                    transport_expr = TRANSPORT_TIME * (1 - same_sum)
                else:
                    transport_expr = TRANSPORT_TIME

                model.addConstr(
                    s[key_next] >= s[key_cur] + cur_dur_expr + transport_expr,
                    name=f"prec_{job.job_id}_{idx}")

    # (4) Job completion
    for job in jobs:
        last_key = (job.job_id, len(job.operations) - 1)
        if last_key in fixed_keys:
            last_end = next(e.end_time for e in fixed_entries
                            if e.job_id == job.job_id
                            and e.op_idx == len(job.operations) - 1)
            model.addConstr(C[job.job_id] == last_end, name=f"comp_{job.job_id}")
        else:
            last_op = job.operations[-1]
            last_dur = _gp.quicksum(
                u[(last_key[0], last_key[1], unit_name)] * last_op.unit_times[unit_name]
                for unit_name in last_op.unit_times)
            model.addConstr(C[job.job_id] >= s[last_key] + last_dur,
                            name=f"comp_{job.job_id}")

    # (5) Earliness / tardiness
    for job in jobs:
        jid = job.job_id
        model.addConstr(E[jid] >= job.due_date - C[jid], name=f"earl_{jid}")
        model.addConstr(T[jid] >= C[jid] - job.due_date, name=f"tard_{jid}")

    # (6) Machine disjunction
    for (unit_name, _mtype), op_list in unit_type_ops.items():
        for i in range(len(op_list)):
            for j in range(i + 1, len(op_list)):
                a_key, b_key = op_list[i], op_list[j]
                a_dur_expr = _gp.quicksum(
                    u[(a_key[0], a_key[1], un)] * _get_op_time(jobs, a_key, un)
                    for un in _get_op_feasible_units(jobs, a_key))
                b_dur_expr = _gp.quicksum(
                    u[(b_key[0], b_key[1], un)] * _get_op_time(jobs, b_key, un)
                    for un in _get_op_feasible_units(jobs, b_key))

                yk = (a_key[0], a_key[1], b_key[0], b_key[1], unit_name)
                y_var = y[yk]
                u_a = u[(a_key[0], a_key[1], unit_name)]
                u_b = u[(b_key[0], b_key[1], unit_name)]

                # a before b when both in unit
                model.addConstr(
                    s[a_key] + a_dur_expr <= s[b_key]
                    + BIG_M * (3 - u_a - u_b - y_var),
                    name=f"disj1_{a_key}_{b_key}_{unit_name}")
                # b before a when both in unit
                model.addConstr(
                    s[b_key] + b_dur_expr <= s[a_key]
                    + BIG_M * (2 - u_a - u_b + y_var),
                    name=f"disj2_{a_key}_{b_key}_{unit_name}")

    # (7) Fixed entries block their machines
    for e in fixed_entries:
        if e.end_time > current_time:
            mtype = machine_type_of(e.machine)
            unit_name = e.service_unit
            for job, op in ops_to_schedule:
                key = (job.job_id, op.op_idx)
                if op.machine_type == mtype and unit_name in op.unit_times:
                    model.addConstr(
                        s[key] >= e.end_time
                        - BIG_M * (1 - u[(key[0], key[1], unit_name)]),
                        name=f"block_{e.job_id}_{e.op_idx}_{key}")

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
            assigned_unit = None
            for unit_name in op.unit_times:
                if u[(key[0], key[1], unit_name)].X > 0.5:
                    assigned_unit = unit_name
                    break
            if assigned_unit is None:
                return None
            machine = f"{op.machine_type}_{assigned_unit}"
            proc_time = op.unit_times[assigned_unit]
            schedule.append(ScheduleEntry(
                job.job_id, op.op_idx, machine, assigned_unit,
                start_val, start_val + proc_time))
        schedule.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
        return schedule
    except Exception:
        return None


def _get_op_time(jobs: List[Job], key: Tuple[int, int], unit_name: str) -> float:
    """Get processing time of an operation on a given unit."""
    for job in jobs:
        if job.job_id == key[0]:
            op = job.operations[key[1]]
            return op.unit_times.get(unit_name, 0.0)
    return 0.0


def _get_op_feasible_units(jobs: List[Job],
                           key: Tuple[int, int]) -> List[str]:
    """Get list of feasible units for an operation."""
    for job in jobs:
        if job.job_id == key[0]:
            op = job.operations[key[1]]
            return list(op.unit_times.keys())
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(schedule: List[ScheduleEntry], jobs: List[Job]) -> dict:
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
    1: '#4C72B0',
    2: '#DD8452',
    3: '#55A868',
    4: '#C44E52',
}

JOB_LABELS = {
    1: 'J1',
    2: 'J2',
    3: 'J3',
    4: 'J4',
}


def plot_gantt(schedule: List[ScheduleEntry],
               title: str,
               current_time: Optional[float] = None,
               ax: Optional[plt.Axes] = None):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_schedule(schedule: List[ScheduleEntry], title: str):
    print(f"\n{'-' * 80}")
    print(f"  {title}")
    print(f"{'-' * 80}")
    print(f"  {'Job':>6s}  {'Op':>4s}  {'Machine':>10s}  {'Unit':>5s}  "
          f"{'Start':>8s}  {'End':>8s}  {'Dur':>6s}  {'Status':>10s}")
    print(f"  {'-' * 80}")
    for e in schedule:
        status = 'FIXED' if e.fixed else 'planned'
        print(f"  {'J' + str(e.job_id):>6s}  {e.op_idx + 1:>4d}  "
              f"{e.machine:>10s}  {e.service_unit:>5s}  "
              f"{e.start_time:>8.1f}  {e.end_time:>8.1f}  "
              f"{e.duration:>6.1f}  {status:>10s}")


def print_metrics(schedule: List[ScheduleEntry], jobs: List[Job], label: str):
    metrics = compute_metrics(schedule, jobs)
    print(f"\n  -- {label} Metrics --")
    header = (f"  {'Job':>6s}  {'Cj':>8s}  {'dj':>8s}  "
              f"{'Ej':>8s}  {'Tj':>8s}  {'a*E+b*T':>12s}")
    print(header)
    print(f"  {'-' * len(header)}")
    for jid, Cj, Ej, Tj, pen in metrics['details']:
        job = next(j for j in jobs if j.job_id == jid)
        print(f"  {'J' + str(jid):>6s}  {Cj:>8.1f}  {job.due_date:>8.1f}  "
              f"{Ej:>8.1f}  {Tj:>8.1f}  {pen:>12.1f}")
    print(f"  {'-' * len(header)}")
    print(f"  Total weighted penalty : {metrics['total_penalty']:.1f}")
    print(f"  Total active lead time : {metrics['total_active_lead_time']:.1f}")

    unit_loads: Dict[str, float] = {}
    for e in schedule:
        unit_loads[e.service_unit] = unit_loads.get(e.service_unit, 0) + e.duration
    print(f"\n  -- Service Unit Load Summary --")
    for u in ['U1', 'U2']:
        load = unit_loads.get(u, 0)
        n_machines = len(SERVICE_UNITS[u])
        print(f"    {u}: total load = {load:.1f},  "
              f"avg load/machine = {load / n_machines:.1f}  "
              f"({n_machines} machines)")


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_schedule(schedule: List[ScheduleEntry], jobs: List[Job],
                      current_time: float, label: str = "") -> List[str]:
    errors: List[str] = []

    covered = {(e.job_id, e.op_idx) for e in schedule}
    for job in jobs:
        for op in job.operations:
            if (job.job_id, op.op_idx) not in covered:
                errors.append(f"[{label}] Missing: J{job.job_id}-Op{op.op_idx + 1}")

    for e in schedule:
        if e.fixed:
            continue
        job = next(j for j in jobs if j.job_id == e.job_id)
        lb = max(current_time, job.release_date)
        if e.start_time < lb - 1e-6:
            errors.append(
                f"[{label}] J{e.job_id}-Op{e.op_idx + 1}: "
                f"start={e.start_time:.2f} < max(current={current_time}, "
                f"release={job.release_date}) = {lb}")

    for job in jobs:
        sorted_ops = sorted(
            [e for e in schedule if e.job_id == job.job_id],
            key=lambda e: e.op_idx)
        for k in range(len(sorted_ops) - 1):
            e_cur = sorted_ops[k]
            e_next = sorted_ops[k + 1]
            if e_next.op_idx != e_cur.op_idx + 1:
                continue
            if e_next.start_time < e_cur.end_time - 1e-6:
                errors.append(
                    f"[{label}] J{job.job_id}: precedence violated: "
                    f"Op{e_next.op_idx + 1} start={e_next.start_time:.2f} "
                    f"< Op{e_cur.op_idx + 1} end={e_cur.end_time:.2f}")

    for m in ALL_MACHINES:
        ops_on_m = sorted(
            [e for e in schedule if e.machine == m],
            key=lambda e: e.start_time)
        for k in range(len(ops_on_m) - 1):
            if ops_on_m[k].end_time > ops_on_m[k + 1].start_time + 1e-6:
                errors.append(
                    f"[{label}] Machine {m}: overlap "
                    f"J{ops_on_m[k].job_id}-Op{ops_on_m[k].op_idx + 1} "
                    f"with J{ops_on_m[k + 1].job_id}-Op{ops_on_m[k + 1].op_idx + 1}")

    for e in schedule:
        job = next(j for j in jobs if j.job_id == e.job_id)
        op = job.operations[e.op_idx]
        if e.service_unit not in op.unit_times:
            errors.append(
                f"[{label}] J{e.job_id}-Op{e.op_idx + 1}: unit "
                f"{e.service_unit} not feasible")
        expected_type = op.machine_type
        actual_type = machine_type_of(e.machine)
        if expected_type != actual_type:
            errors.append(
                f"[{label}] J{e.job_id}-Op{e.op_idx + 1}: machine type "
                f"mismatch, expected {expected_type}, got {actual_type}")
        if e.machine not in SERVICE_UNITS.get(e.service_unit, []):
            errors.append(
                f"[{label}] J{e.job_id}-Op{e.op_idx + 1}: machine "
                f"{e.machine} not in unit {e.service_unit}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_greedy() -> Tuple[List[ScheduleEntry], List[ScheduleEntry]]:
    """Two-stage simulation using greedy hierarchical scheduler."""
    all_jobs = build_jobs()

    visible_jobs_t0 = [j for j in all_jobs if j.arrival_time <= 0]
    initial_schedule = hierarchical_schedule(visible_jobs_t0, [], 0.0)

    fixed_at_t24: List[ScheduleEntry] = []
    for e in initial_schedule:
        if e.start_time < 24:
            e_fixed = ScheduleEntry(e.job_id, e.op_idx, e.machine,
                                    e.service_unit,
                                    e.start_time, e.end_time, fixed=True)
            fixed_at_t24.append(e_fixed)

    visible_jobs_t24 = [j for j in all_jobs if j.arrival_time <= 24]
    updated_schedule = hierarchical_schedule(visible_jobs_t24, fixed_at_t24, 24.0)

    return initial_schedule, updated_schedule


def simulate_milp(
    time_limit: float = 60.0,
) -> Tuple[Optional[List[ScheduleEntry]], Optional[List[ScheduleEntry]]]:
    """
    Two-stage simulation using MILP.

    t=0 : MILP optimal initial schedule
    t=24: MILP re-optimization with frozen ops
    """
    all_jobs = build_jobs()

    # t=0 MILP
    visible_jobs_t0 = [j for j in all_jobs if j.arrival_time <= 0]
    initial_schedule = schedule_milp(visible_jobs_t0, [], 0.0, time_limit)
    if initial_schedule is None:
        return None, None

    # Determine frozen ops at t=24
    fixed_at_t24: List[ScheduleEntry] = []
    for e in initial_schedule:
        if e.start_time < 24:
            e_fixed = ScheduleEntry(e.job_id, e.op_idx, e.machine,
                                    e.service_unit,
                                    e.start_time, e.end_time, fixed=True)
            fixed_at_t24.append(e_fixed)

    # t=24 MILP
    visible_jobs_t24 = [j for j in all_jobs if j.arrival_time <= 24]
    updated_schedule = schedule_milp(visible_jobs_t24, fixed_at_t24,
                                     24.0, time_limit)
    if updated_schedule is None:
        return None, None

    return initial_schedule, updated_schedule


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  Hierarchical FJSP Rescheduling Simulation")
    print("  Three-Layer Model: Shop -> Service Unit -> Machine")
    print("=" * 80)
    print(f"  Service Units: U1 (M1_U1, M2_U1, M3_U1) | "
          f"U2 (M1_U2, M2_U2)")
    print(f"  Jobs: J1, J2, J3 (t=0)  |  J4 arrives at t=24, released at t=48")
    print(f"  Machine sequences: J1,J2,J3: M1->M2->M1  |  J4: M3->M1->M2")

    all_jobs = build_jobs()
    jobs_t0 = [j for j in all_jobs if j.arrival_time <= 0]
    jobs_t24 = [j for j in all_jobs if j.arrival_time <= 24]

    # ── Greedy baseline ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  METHOD 1: Greedy Hierarchical (Load Balancing + EDD)")
    print("=" * 80)
    initial_greedy, updated_greedy = simulate_greedy()

    print_schedule(initial_greedy, "Initial Schedule (t=0) -- Greedy Hierarchical")
    print_metrics(initial_greedy, jobs_t0, "Initial Greedy (t=0)")
    print_schedule(updated_greedy,
                   "Updated Schedule (t=24, J4 arrived) -- Greedy Hierarchical")
    print_metrics(updated_greedy, jobs_t24, "Updated Greedy (t=24)")

    print(f"\n  -- Operations frozen at t=24 (Greedy) --")
    frozen = [e for e in updated_greedy if e.fixed]
    for e in frozen:
        print(f"    J{e.job_id}-Op{e.op_idx + 1}  on {e.machine} "
              f"({e.service_unit})  [{e.start_time:.1f} -> {e.end_time:.1f}]")

    # ── MILP optimal ─────────────────────────────────────────────────────
    initial_milp, updated_milp = simulate_milp()

    if initial_milp is not None and updated_milp is not None:
        print("\n" + "=" * 80)
        print("  METHOD 2: MILP Optimal (Gurobi)")
        print("=" * 80)
        print_schedule(initial_milp, "Initial Schedule (t=0) -- MILP Optimal")
        print_metrics(initial_milp, jobs_t0, "Initial MILP (t=0)")
        print_schedule(updated_milp,
                       "Updated Schedule (t=24, J4 arrived) -- MILP Re-optimized")
        print_metrics(updated_milp, jobs_t24, "Updated MILP (t=24)")

        print(f"\n  -- Operations frozen at t=24 (MILP) --")
        frozen_milp = [e for e in updated_milp if e.fixed]
        for e in frozen_milp:
            print(f"    J{e.job_id}-Op{e.op_idx + 1}  on {e.machine} "
                  f"({e.service_unit})  [{e.start_time:.1f} -> {e.end_time:.1f}]")

        # ── Comparison ──────────────────────────────────────────────────
        m_greedy_init = compute_metrics(initial_greedy, jobs_t0)
        m_greedy_upd  = compute_metrics(updated_greedy, jobs_t24)
        m_milp_init   = compute_metrics(initial_milp, jobs_t0)
        m_milp_upd    = compute_metrics(updated_milp, jobs_t24)

        print(f"\n{'-' * 80}")
        print(f"  Comparison: Greedy vs MILP")
        print(f"{'-' * 80}")
        hdr = (f"  {'Stage':<8s} {'Metric':<22s} "
               f"{'Greedy':>10s}  {'MILP':>10s}  {'Gap':>10s}")
        print(hdr)
        print(f"  {'-' * len(hdr)}")

        for label, gm, mm in [("t=0", m_greedy_init, m_milp_init),
                              ("t=24", m_greedy_upd, m_milp_upd)]:
            for mname, key in [("Total Penalty", "total_penalty"),
                               ("Active Lead Time", "total_active_lead_time")]:
                gv = gm[key]
                mv = mm[key]
                gap = gv - mv
                print(f"  {label:<8s} {mname:<22s} "
                      f"{gv:>10.1f}  {mv:>10.1f}  {gap:>+10.1f}")

        # Per-job penalty breakdown
        print(f"\n{'-' * 80}")
        print(f"  Per-Job Penalty Breakdown (alpha*E + beta*T)")
        print(f"{'-' * 80}")
        jhdr = (f"  {'Job':>6s} {'dj':>6s} {'alpha':>5s} {'beta':>5s} "
                f"{'Greedy Cj':>10s} {'MILP Cj':>10s} "
                f"{'G-Pen':>10s} {'M-Pen':>10s} {'Delta':>10s}")
        print(jhdr)
        print(f"  {'-' * len(jhdr)}")
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
                  f"{job.alpha:>5.0f} {job.beta:>5.0f} "
                  f"{gC:>10.1f} {mC:>10.1f} "
                  f"{gPen:>10.1f} {mPen:>10.1f} "
                  f"{gPen - mPen:>+10.1f}")

        # ── Validation ──────────────────────────────────────────────────
        print(f"\n{'-' * 80}")
        print(f"  Constraint Validation")
        print(f"{'-' * 80}")
        all_ok = True
        for sched, lbl, jlist, ct in [
            (initial_greedy, "Greedy-t0", jobs_t0, 0.0),
            (updated_greedy, "Greedy-t24", jobs_t24, 24.0),
            (initial_milp, "MILP-t0", jobs_t0, 0.0),
            (updated_milp, "MILP-t24", jobs_t24, 24.0),
        ]:
            errs = validate_schedule(sched, jlist, ct, lbl)
            if errs:
                all_ok = False
                for err in errs:
                    print(f"  X {err}")
            else:
                print(f"  OK {lbl}: all constraints satisfied")

        # ── Gantt charts (2x2: Greedy vs MILP) ──────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))

        t0_max = max(max(e.end_time for e in initial_greedy),
                     max(e.end_time for e in initial_milp)) * 1.08
        t24_max = max(max(e.end_time for e in updated_greedy),
                      max(e.end_time for e in updated_milp)) * 1.08
        x_max = max(t0_max, t24_max)

        plot_gantt(initial_greedy,
                   'Greedy Hierarchical  --  Initial (t=0)', ax=axes[0, 0])
        axes[0, 0].set_xlim(0, x_max)
        plot_gantt(initial_milp, 'MILP Optimal  --  Initial (t=0)', ax=axes[0, 1])
        axes[0, 1].set_xlim(0, x_max)

        plot_gantt(updated_greedy,
                   'Greedy Hierarchical  --  Updated (t=24)  (hatched = frozen)',
                   current_time=24, ax=axes[1, 0])
        axes[1, 0].set_xlim(0, x_max)
        plot_gantt(updated_milp,
                   'MILP Re-optimized  --  Updated (t=24)  (hatched = frozen)',
                   current_time=24, ax=axes[1, 1])
        axes[1, 1].set_xlim(0, x_max)

        fig.tight_layout(pad=4.0)
        plt.savefig('gantt_charts_hierarchical.png', dpi=150, bbox_inches='tight')
        print(f"\n  Gantt charts saved to 'gantt_charts_hierarchical.png'")

    else:
        # No Gurobi — fallback to greedy only
        print("\n  [Gurobi not available -- showing greedy results only]\n")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        t0_max = max(e.end_time for e in initial_greedy) * 1.08
        t24_max = max(e.end_time for e in updated_greedy) * 1.08
        x_max = max(t0_max, t24_max)

        plot_gantt(initial_greedy,
                   'Initial Hierarchical Schedule  (t = 0)  --  J1, J2, J3',
                   ax=ax1)
        ax1.set_xlim(0, x_max)
        plot_gantt(updated_greedy,
                   'Updated Hierarchical Schedule  (t = 24, J4 arrives)  '
                   '--  hatched = frozen',
                   current_time=24,
                   ax=ax2)
        ax2.set_xlim(0, x_max)

        fig.tight_layout(pad=3.0)
        plt.savefig('gantt_charts_hierarchical.png', dpi=150, bbox_inches='tight')
        print(f"\n  Gantt charts saved to 'gantt_charts_hierarchical.png'")


if __name__ == '__main__':
    main()
