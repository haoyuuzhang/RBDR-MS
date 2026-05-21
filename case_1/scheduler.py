"""MILP-based optimal scheduler using Gurobi."""

from typing import Dict, List, Optional, Tuple
from models import Job, Operation, ScheduleEntry
from config import TRANSPORT_TIME, machine_type_of

try:
    import gurobipy as _gp
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False


def _get_effective_time(op: Operation, unit_name: str,
                        time_factors: Optional[Dict[str, float]] = None) -> float:
    """Return processing time scaled by time_factors if applicable."""
    base = op.unit_times.get(unit_name, 0.0)
    if time_factors:
        machine = f"{op.machine_type}_{unit_name}"
        return base * time_factors.get(machine, 1.0)
    return base


def _get_op_time(jobs: List[Job], key: Tuple[int, int], unit_name: str,
                 time_factors: Optional[Dict[str, float]] = None) -> float:
    """Get effective processing time of an operation on a given unit."""
    for job in jobs:
        if job.job_id == key[0]:
            op = job.operations[key[1]]
            base = op.unit_times.get(unit_name, 0.0)
            if time_factors:
                machine = f"{op.machine_type}_{unit_name}"
                return base * time_factors.get(machine, 1.0)
            return base
    return 0.0


def _get_op_feasible_units(jobs: List[Job],
                           key: Tuple[int, int]) -> List[str]:
    """Get list of feasible units for an operation."""
    for job in jobs:
        if job.job_id == key[0]:
            op = job.operations[key[1]]
            return list(op.unit_times.keys())
    return []


def schedule_milp(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float,
    time_limit: float = 60.0,
    time_factors: Optional[Dict[str, float]] = None,
) -> Optional[List[ScheduleEntry]]:
    """
    MILP that jointly optimizes unit assignment and machine scheduling.

    Decision variables:
      u[(j,o), unit]   : operation assigned to service unit
      s[(j,o)]         : start time
      y[(a,b), unit]   : sequencing on shared machine in unit

    Objective: minimize total weighted earliness-tardiness penalty.

    time_factors: optional dict mapping machine_name -> multiplier for disrupted
                  processing times (e.g. {'M1_U2': 2.0})
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
    model.Params.MIPGap = 0.0
    model.Params.MIPFocus = 2

    # Tight M: sum of max processing times + transport + disruption slack
    big_m = 0.0
    for job in jobs:
        for op in job.operations:
            big_m += max(op.unit_times.values()) if op.unit_times else 0
    big_m += TRANSPORT_TIME * sum(len(job.operations) for job in jobs)
    big_m = max(big_m * (max(time_factors.values()) if time_factors else 1.0), 500.0)

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

    # Sequencing variables
    y: Dict[Tuple, _gp.Var] = {}
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
                    u[(key_cur[0], key_cur[1], unit_name)] *
                    _get_effective_time(op_cur, unit_name, time_factors)
                    for unit_name in op_cur.unit_times)
                next_start_fixed = next(e.start_time for e in fixed_entries
                                        if e.job_id == job.job_id
                                        and e.op_idx == op_next.op_idx)
                model.addConstr(next_start_fixed >= s[key_cur] + cur_dur_expr,
                                name=f"prec_{job.job_id}_{idx}_to_fixed")
            else:
                cur_dur_expr = _gp.quicksum(
                    u[(key_cur[0], key_cur[1], unit_name)] *
                    _get_effective_time(op_cur, unit_name, time_factors)
                    for unit_name in op_cur.unit_times)

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
                u[(last_key[0], last_key[1], unit_name)] *
                _get_effective_time(last_op, unit_name, time_factors)
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
                    u[(a_key[0], a_key[1], un)] *
                    _get_op_time(jobs, a_key, un, time_factors)
                    for un in _get_op_feasible_units(jobs, a_key))
                b_dur_expr = _gp.quicksum(
                    u[(b_key[0], b_key[1], un)] *
                    _get_op_time(jobs, b_key, un, time_factors)
                    for un in _get_op_feasible_units(jobs, b_key))

                yk = (a_key[0], a_key[1], b_key[0], b_key[1], unit_name)
                y_var = y[yk]
                u_a = u[(a_key[0], a_key[1], unit_name)]
                u_b = u[(b_key[0], b_key[1], unit_name)]

                model.addConstr(
                    s[a_key] + a_dur_expr <= s[b_key]
                    + big_m * (3 - u_a - u_b - y_var),
                    name=f"disj1_{a_key}_{b_key}_{unit_name}")
                model.addConstr(
                    s[b_key] + b_dur_expr <= s[a_key]
                    + big_m * (2 - u_a - u_b + y_var),
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
                        - big_m * (1 - u[(key[0], key[1], unit_name)]),
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
            proc_time = _get_effective_time(op, assigned_unit, time_factors)
            schedule.append(ScheduleEntry(
                job.job_id, op.op_idx, machine, assigned_unit,
                start_val, start_val + proc_time))
        schedule.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
        return schedule
    except Exception:
        return None
