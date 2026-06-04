"""Self-contained MILP scheduler for the baseline-test framework.

Jointly optimizes unit assignment, machine sequencing, and start times.
Does not depend on any files outside this directory.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from models import Job, Operation, ScheduleEntry, Disruption

try:
    import gurobipy as _gp
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

TRANSPORT_TIME = 2.0


def _machine_type_of(machine: str) -> str:
    """Extract machine type from machine ID, e.g. 'M1_U1' -> 'M1'."""
    return machine.rsplit('_', 1)[0]


def schedule_milp(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float,
    time_limit: float = 60.0,
    time_factors: Optional[Dict[str, float]] = None,
) -> Optional[List[ScheduleEntry]]:
    """Solve the MILP and return a complete schedule.

    Objective: minimize Σ (α_j * E_j + β_j * T_j)
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
    model.Params.MIPGap = 0.01          # allow 1% optimality gap → finds feasible solutions faster
    model.Params.MIPFocus = 1           # prioritise finding feasible solutions over proving bounds
    model.Params.Seed = 0
    model.Params.NodefileStart = 0.5    # start swapping nodes to disk at 0.5 GB (prevents OOM for large models)

    # Big-M
    big_m = 0.0
    for job in jobs:
        for op in job.operations:
            big_m += max(op.unit_times.values()) if op.unit_times else 0
    big_m += TRANSPORT_TIME * sum(len(job.operations) for job in jobs)
    if time_factors:
        big_m *= max(time_factors.values())
    big_m = max(big_m, 500.0)

    # ── Variables ────────────────────────────────────────────────────────
    u: Dict[Tuple[int, int, str], _gp.Var] = {}
    s: Dict[Tuple[int, int], _gp.Var] = {}

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

    # Same-unit indicator: same[(j, idx), u] = u[prev] * u[next]
    same_unit: Dict[Tuple[int, int, str], _gp.Var] = {}
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            kc = (job.job_id, op_cur.op_idx)
            kn = (job.job_id, op_next.op_idx)
            if kc in fixed_keys or kn in fixed_keys:
                continue
            for unit_name in set(op_cur.unit_times) & set(op_next.unit_times):
                var = model.addVar(vtype=_gp.GRB.BINARY,
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
    unit_ops: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        for unit_name in op.unit_times:
            ut_key = (unit_name, op.machine_type)
            unit_ops.setdefault(ut_key, []).append(key)

    for (unit_name, _mtype), op_list in unit_ops.items():
        for i in range(len(op_list)):
            for j in range(i + 1, len(op_list)):
                a, b = op_list[i], op_list[j]
                yk = (a[0], a[1], b[0], b[1], unit_name)
                y[yk] = model.addVar(
                    vtype=_gp.GRB.BINARY,
                    name=f"y_{a[0]}_{a[1]}_{b[0]}_{b[1]}_{unit_name}")

    model.update()

    # ── Constraints ──────────────────────────────────────────────────────

    # (1) Each operation assigned to exactly one feasible unit
    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        model.addConstr(
            _gp.quicksum(u[(key[0], key[1], un)]
                         for un in op.unit_times) == 1,
            name=f"assign_{key[0]}_{key[1]}")

    # (2) Same-unit linearization
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            kc = (job.job_id, op_cur.op_idx)
            kn = (job.job_id, op_next.op_idx)
            if kc in fixed_keys or kn in fixed_keys:
                continue
            for unit_name in set(op_cur.unit_times) & set(op_next.unit_times):
                sv = same_unit[(job.job_id, idx, unit_name)]
                uc = u[(kc[0], kc[1], unit_name)]
                un = u[(kn[0], kn[1], unit_name)]
                model.addConstr(sv <= uc, name=f"same1_{job.job_id}_{idx}_{unit_name}")
                model.addConstr(sv <= un, name=f"same2_{job.job_id}_{idx}_{unit_name}")
                model.addConstr(sv >= uc + un - 1,
                                name=f"same3_{job.job_id}_{idx}_{unit_name}")

    # (3) Precedence within each job (with transport)
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            kc = (job.job_id, op_cur.op_idx)
            kn = (job.job_id, op_next.op_idx)
            common = set(op_cur.unit_times) & set(op_next.unit_times)

            if kc in fixed_keys:
                cur_end = next(e.end_time for e in fixed_entries
                               if e.job_id == job.job_id
                               and e.op_idx == op_cur.op_idx)
                if kn not in fixed_keys:
                    model.addConstr(s[kn] >= cur_end,
                                    name=f"prec_{job.job_id}_{idx}_fixed")
            elif kn in fixed_keys:
                cur_dur_expr = _gp.quicksum(
                    u[(kc[0], kc[1], un)] *
                    _effective_time(op_cur, un, time_factors)
                    for un in op_cur.unit_times)
                next_start = next(e.start_time for e in fixed_entries
                                  if e.job_id == job.job_id
                                  and e.op_idx == op_next.op_idx)
                model.addConstr(next_start >= s[kc] + cur_dur_expr,
                                name=f"prec_{job.job_id}_{idx}_to_fixed")
            else:
                cur_dur_expr = _gp.quicksum(
                    u[(kc[0], kc[1], un)] *
                    _effective_time(op_cur, un, time_factors)
                    for un in op_cur.unit_times)
                if common:
                    same_sum = _gp.quicksum(
                        same_unit[(job.job_id, idx, un)] for un in common)
                    transport_expr = TRANSPORT_TIME * (1 - same_sum)
                else:
                    transport_expr = TRANSPORT_TIME
                model.addConstr(
                    s[kn] >= s[kc] + cur_dur_expr + transport_expr,
                    name=f"prec_{job.job_id}_{idx}")

    # (4) Job completion
    for job in jobs:
        last_key = (job.job_id, len(job.operations) - 1)
        if last_key in fixed_keys:
            last_end = next(e.end_time for e in fixed_entries
                            if e.job_id == job.job_id
                            and e.op_idx == len(job.operations) - 1)
            model.addConstr(C[job.job_id] == last_end,
                            name=f"comp_{job.job_id}")
        else:
            last_op = job.operations[-1]
            last_dur = _gp.quicksum(
                u[(last_key[0], last_key[1], un)] *
                _effective_time(last_op, un, time_factors)
                for un in last_op.unit_times)
            model.addConstr(C[job.job_id] >= s[last_key] + last_dur,
                            name=f"comp_{job.job_id}")

    # (5) Earliness / tardiness
    for job in jobs:
        jid = job.job_id
        model.addConstr(E[jid] >= job.due_date - C[jid], name=f"earl_{jid}")
        model.addConstr(T[jid] >= C[jid] - job.due_date, name=f"tard_{jid}")

    # (6) Machine disjunction
    for (unit_name, _mtype), op_list in unit_ops.items():
        for i in range(len(op_list)):
            for j in range(i + 1, len(op_list)):
                a_key, b_key = op_list[i], op_list[j]
                a_dur_expr = _gp.quicksum(
                    u[(a_key[0], a_key[1], un)] *
                    _op_time(jobs, a_key, un, time_factors)
                    for un in _feasible_units(jobs, a_key))
                b_dur_expr = _gp.quicksum(
                    u[(b_key[0], b_key[1], un)] *
                    _op_time(jobs, b_key, un, time_factors)
                    for un in _feasible_units(jobs, b_key))

                yk = (a_key[0], a_key[1], b_key[0], b_key[1], unit_name)
                yv = y[yk]
                ua = u[(a_key[0], a_key[1], unit_name)]
                ub = u[(b_key[0], b_key[1], unit_name)]

                model.addConstr(
                    s[a_key] + a_dur_expr <= s[b_key]
                    + big_m * (3 - ua - ub - yv),
                    name=f"disj1_{a_key}_{b_key}_{unit_name}")
                model.addConstr(
                    s[b_key] + b_dur_expr <= s[a_key]
                    + big_m * (2 - ua - ub + yv),
                    name=f"disj2_{a_key}_{b_key}_{unit_name}")

    # (7) Fixed entries block their machines
    for e in fixed_entries:
        if e.end_time > current_time:
            mtype = _machine_type_of(e.machine)
            unit_name = e.service_unit
            for job, op in ops_to_schedule:
                key = (job.job_id, op.op_idx)
                if op.machine_type == mtype and unit_name in op.unit_times:
                    model.addConstr(
                        s[key] >= e.end_time
                        - big_m * (1 - u[(key[0], key[1], unit_name)]),
                        name=f"block_{e.job_id}_{e.op_idx}_{key}")

    # ── Objective ────────────────────────────────────────────────────────
    model.setObjective(
        _gp.quicksum(job.alpha * E[job.job_id] + job.beta * T[job.job_id]
                     for job in jobs),
        _gp.GRB.MINIMIZE)

    # ── Solve ────────────────────────────────────────────────────────────
    model.optimize()

    if model.Status not in (_gp.GRB.OPTIMAL, _gp.GRB.SUBOPTIMAL,
                            _gp.GRB.TIME_LIMIT):
        return None

    # ── Extract schedule ─────────────────────────────────────────────────
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
            proc_time = _effective_time(op, assigned_unit, time_factors)
            schedule.append(ScheduleEntry(
                job_id=job.job_id, op_idx=op.op_idx,
                machine=machine, service_unit=assigned_unit,
                start_time=start_val, end_time=start_val + proc_time))
        schedule.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
        return schedule
    except Exception:
        return None


# ── helpers ──────────────────────────────────────────────────────────────

def _effective_time(op: Operation, unit_name: str,
                    time_factors: Optional[Dict[str, float]] = None) -> float:
    base = op.unit_times.get(unit_name, 0.0)
    if time_factors:
        machine = f"{op.machine_type}_{unit_name}"
        return base * time_factors.get(machine, 1.0)
    return base


def _op_time(jobs: List[Job], key: Tuple[int, int], unit_name: str,
             time_factors: Optional[Dict[str, float]] = None) -> float:
    for job in jobs:
        if job.job_id == key[0]:
            return _effective_time(job.operations[key[1]], unit_name, time_factors)
    return 0.0


def _feasible_units(jobs: List[Job], key: Tuple[int, int]) -> List[str]:
    for job in jobs:
        if job.job_id == key[0]:
            return list(job.operations[key[1]].unit_times.keys())
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Clairvoyant MILP — disruption-aware via piecewise-linear duration modelling
# ═══════════════════════════════════════════════════════════════════════════════

def schedule_milp_clairvoyant(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float,
    time_limit: float = 120.0,
    disruptions: Optional[List[Disruption]] = None,
    machine_count: int = 1,
) -> Optional[List[ScheduleEntry]]:
    """Clairvoyant MILP: jointly optimise with perfect knowledge of disruptions.

    For operations on disrupted machines, the effective duration is a
    piecewise-linear function of start time (an op may span disruption
    boundaries).  This function is encoded via Gurobi PWL constraints.

    Parameters
    ----------
    jobs:          All jobs (the clairvoyant sees future arrivals).
    fixed_entries: Already-completed / in-progress operations (frozen).
    current_time:  Current simulation time.
    time_limit:    Gurobi time limit per solve.
    disruptions:   Known disruptions with time / duration / factor.
    machine_count: Total number of machines (for horizon estimation).
    """
    if not _HAS_GUROBI:
        return None

    disruptions = disruptions or []

    fixed_keys = {(e.job_id, e.op_idx) for e in fixed_entries}
    ops_to_schedule: List[Tuple[Job, Operation]] = []
    for job in jobs:
        for op in job.operations:
            if (job.job_id, op.op_idx) not in fixed_keys:
                ops_to_schedule.append((job, op))

    if not ops_to_schedule:
        return list(fixed_entries)

    # ── Identify disrupted machines and estimate horizon ───────────────
    disrupted_machines: Dict[str, List[Disruption]] = {}  # machine_id → disruptions
    for d in disruptions:
        disrupted_machines.setdefault(d.machine_id, []).append(d)

    max_release = max((j.release_date for j in jobs), default=0)
    total_work = sum(
        sum(op.unit_times.values()) / max(len(op.unit_times), 1)
        for j in jobs for op in j.operations
    )
    horizon = max_release + total_work / max(machine_count, 1)
    horizon = max(horizon, 200.0) * 1.5  # generous buffer
    n_samples = 200  # PWL sample points (fine-grained for disruption boundaries)

    # ── Pre-compute PWL breakpoints for disrupted machines ─────────────
    # pwl_data[(jid, oidx, unit)] = (xs, ys)  or  None for non-disrupted
    pwl_data: Dict[Tuple[int, int, str], Optional[Tuple[List[float], List[float]]]] = {}
    for job, op in ops_to_schedule:
        for unit_name in op.unit_times:
            machine = f"{op.machine_type}_{unit_name}"
            base_dur = op.unit_times[unit_name]
            if machine in disrupted_machines:
                xs, ys = _make_pwl_breakpoints(
                    base_dur, machine, disruptions, horizon, n_samples)
                pwl_data[(job.job_id, op.op_idx, unit_name)] = (xs, ys)
            else:
                pwl_data[(job.job_id, op.op_idx, unit_name)] = None

    # ── Build model ────────────────────────────────────────────────────
    env = _gp.Env(params={"OutputFlag": 0})
    model = _gp.Model("FJSP_Clairvoyant", env=env)
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.001        # tight gap — clairvoyant must be near-optimal
    model.Params.MIPFocus = 0          # balanced: don't sacrifice optimality for speed
    model.Params.Seed = 0
    model.Params.NodefileStart = 0.5

    # Big-M
    big_m = 0.0
    for job in jobs:
        for op in job.operations:
            big_m += max(op.unit_times.values()) if op.unit_times else 0
    big_m += TRANSPORT_TIME * sum(len(job.operations) for job in jobs)
    # Use max disruption factor for worst-case duration
    max_factor = 1.0
    for d in disruptions:
        max_factor = max(max_factor, d.factor)
    big_m *= max_factor
    big_m = max(big_m, 500.0)

    # ── Variables ──────────────────────────────────────────────────────
    u: Dict[Tuple[int, int, str], _gp.Var] = {}   # unit assignment
    s: Dict[Tuple[int, int], _gp.Var] = {}         # start time
    d_var: Dict[Tuple[int, int, str], _gp.Var] = {}  # effective duration
    ud: Dict[Tuple[int, int, str], _gp.Var] = {}     # linearized u × d

    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        s[key] = model.addVar(
            lb=max(current_time, job.release_date),
            vtype=_gp.GRB.CONTINUOUS,
            name=f"s_{job.job_id}_{op.op_idx}")
        for unit_name in op.unit_times:
            uk = (job.job_id, op.op_idx, unit_name)
            u[uk] = model.addVar(vtype=_gp.GRB.BINARY,
                                 name=f"u_{job.job_id}_{op.op_idx}_{unit_name}")
            # Duration variable (upper bound: base × max_factor for disrupted, base otherwise)
            machine = f"{op.machine_type}_{unit_name}"
            base_dur = op.unit_times[unit_name]
            max_dur = base_dur * max_factor if machine in disrupted_machines else base_dur
            d_var[uk] = model.addVar(lb=0, ub=max_dur,
                                     vtype=_gp.GRB.CONTINUOUS,
                                     name=f"d_{job.job_id}_{op.op_idx}_{unit_name}")
            # Linearized product u × d
            ud[uk] = model.addVar(lb=0, ub=max_dur,
                                  vtype=_gp.GRB.CONTINUOUS,
                                  name=f"ud_{job.job_id}_{op.op_idx}_{unit_name}")

    # Same-unit indicator
    same_unit: Dict[Tuple[int, int, str], _gp.Var] = {}
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            kc = (job.job_id, op_cur.op_idx)
            kn = (job.job_id, op_next.op_idx)
            if kc in fixed_keys or kn in fixed_keys:
                continue
            for unit_name in set(op_cur.unit_times) & set(op_next.unit_times):
                var = model.addVar(vtype=_gp.GRB.BINARY,
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
    unit_ops: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        for unit_name in op.unit_times:
            ut_key = (unit_name, op.machine_type)
            unit_ops.setdefault(ut_key, []).append(key)

    for (_unit_name, _mtype), op_list in unit_ops.items():
        for i in range(len(op_list)):
            for j in range(i + 1, len(op_list)):
                a, b = op_list[i], op_list[j]
                yk = (a[0], a[1], b[0], b[1], _unit_name)
                y[yk] = model.addVar(
                    vtype=_gp.GRB.BINARY,
                    name=f"y_{a[0]}_{a[1]}_{b[0]}_{b[1]}_{_unit_name}")

    model.update()

    # ── PWL constraints (disruption-aware duration) ────────────────────
    for (jid, oidx, unit_name), pwl in pwl_data.items():
        if pwl is not None:
            xs, ys = pwl
            skey = (jid, oidx)
            dkey = (jid, oidx, unit_name)
            model.addGenConstrPWL(s[skey], d_var[dkey], xs, ys,
                                  name=f"pwl_{jid}_{oidx}_{unit_name}")
        else:
            # Non-disrupted machine: duration = base (constant)
            dkey = (jid, oidx, unit_name)
            job = _find_job_by_id(jobs, jid)
            if job:
                base_dur = job.operations[oidx].unit_times.get(unit_name, 0)
                model.addConstr(d_var[dkey] == base_dur,
                                name=f"d_fixed_{jid}_{oidx}_{unit_name}")

    # ── Linearize ud = u × d ───────────────────────────────────────────
    for uk, u_var in u.items():
        dk = d_var[uk]
        udk = ud[uk]
        jid, oidx, unit_name = uk
        job = _find_job_by_id(jobs, jid)
        base_dur = job.operations[oidx].unit_times.get(unit_name, 0) if job else 0
        M_dur = base_dur * max(max_factor, 1.0)
        # ud ≤ M × u
        model.addConstr(udk <= M_dur * u_var, name=f"lin1_{jid}_{oidx}_{unit_name}")
        # ud ≤ d
        model.addConstr(udk <= dk, name=f"lin2_{jid}_{oidx}_{unit_name}")
        # ud ≥ d − M × (1 − u)
        model.addConstr(udk >= dk - M_dur * (1 - u_var),
                        name=f"lin3_{jid}_{oidx}_{unit_name}")

    # ── Constraints (same structure as schedule_milp, using ud) ────────

    # (1) Assignment
    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        model.addConstr(
            _gp.quicksum(u[(key[0], key[1], un)] for un in op.unit_times) == 1,
            name=f"assign_{key[0]}_{key[1]}")

    # (2) Same-unit linearization
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            kc = (job.job_id, op_cur.op_idx)
            kn = (job.job_id, op_next.op_idx)
            if kc in fixed_keys or kn in fixed_keys:
                continue
            for unit_name in set(op_cur.unit_times) & set(op_next.unit_times):
                sv = same_unit[(job.job_id, idx, unit_name)]
                uc = u[(kc[0], kc[1], unit_name)]
                un = u[(kn[0], kn[1], unit_name)]
                model.addConstr(sv <= uc, name=f"same1_{job.job_id}_{idx}_{unit_name}")
                model.addConstr(sv <= un, name=f"same2_{job.job_id}_{idx}_{unit_name}")
                model.addConstr(sv >= uc + un - 1,
                                name=f"same3_{job.job_id}_{idx}_{unit_name}")

    # (3) Precedence (using ud for durations)
    for job in jobs:
        for idx in range(len(job.operations) - 1):
            op_cur = job.operations[idx]
            op_next = job.operations[idx + 1]
            kc = (job.job_id, op_cur.op_idx)
            kn = (job.job_id, op_next.op_idx)
            common = set(op_cur.unit_times) & set(op_next.unit_times)

            if kc in fixed_keys:
                cur_end = next(e.end_time for e in fixed_entries
                               if e.job_id == job.job_id and e.op_idx == op_cur.op_idx)
                if kn not in fixed_keys:
                    model.addConstr(s[kn] >= cur_end,
                                    name=f"prec_{job.job_id}_{idx}_fixed")
            elif kn in fixed_keys:
                cur_dur_sum = _gp.quicksum(
                    ud[(kc[0], kc[1], un)] for un in op_cur.unit_times)
                next_start = next(e.start_time for e in fixed_entries
                                  if e.job_id == job.job_id and e.op_idx == op_next.op_idx)
                model.addConstr(next_start >= s[kc] + cur_dur_sum,
                                name=f"prec_{job.job_id}_{idx}_to_fixed")
            else:
                cur_dur_sum = _gp.quicksum(
                    ud[(kc[0], kc[1], un)] for un in op_cur.unit_times)
                if common:
                    same_sum = _gp.quicksum(
                        same_unit[(job.job_id, idx, un)] for un in common)
                    transport_expr = TRANSPORT_TIME * (1 - same_sum)
                else:
                    transport_expr = TRANSPORT_TIME
                model.addConstr(
                    s[kn] >= s[kc] + cur_dur_sum + transport_expr,
                    name=f"prec_{job.job_id}_{idx}")

    # (4) Job completion
    for job in jobs:
        last_key = (job.job_id, len(job.operations) - 1)
        if last_key in fixed_keys:
            last_end = next(e.end_time for e in fixed_entries
                            if e.job_id == job.job_id
                            and e.op_idx == len(job.operations) - 1)
            model.addConstr(C[job.job_id] == last_end,
                            name=f"comp_{job.job_id}")
        else:
            last_op = job.operations[-1]
            last_dur_sum = _gp.quicksum(
                ud[(last_key[0], last_key[1], un)]
                for un in last_op.unit_times)
            model.addConstr(C[job.job_id] >= s[last_key] + last_dur_sum,
                            name=f"comp_{job.job_id}")

    # (5) Earliness / tardiness
    for job in jobs:
        jid = job.job_id
        model.addConstr(E[jid] >= job.due_date - C[jid], name=f"earl_{jid}")
        model.addConstr(T[jid] >= C[jid] - job.due_date, name=f"tard_{jid}")

    # (6) Machine disjunction (using ud for durations)
    for (_unit_name, _mtype), op_list in unit_ops.items():
        for i in range(len(op_list)):
            for j in range(i + 1, len(op_list)):
                a_key, b_key = op_list[i], op_list[j]
                a_dur_sum = _gp.quicksum(
                    ud[(a_key[0], a_key[1], un)]
                    for un in _feasible_units(jobs, a_key))
                b_dur_sum = _gp.quicksum(
                    ud[(b_key[0], b_key[1], un)]
                    for un in _feasible_units(jobs, b_key))

                yk = (a_key[0], a_key[1], b_key[0], b_key[1], _unit_name)
                yv = y[yk]
                ua = u[(a_key[0], a_key[1], _unit_name)]
                ub = u[(b_key[0], b_key[1], _unit_name)]

                model.addConstr(
                    s[a_key] + a_dur_sum <= s[b_key]
                    + big_m * (3 - ua - ub - yv),
                    name=f"disj1_{a_key}_{b_key}_{_unit_name}")
                model.addConstr(
                    s[b_key] + b_dur_sum <= s[a_key]
                    + big_m * (2 - ua - ub + yv),
                    name=f"disj2_{a_key}_{b_key}_{_unit_name}")

    # (7) Fixed entries block their machines
    for e in fixed_entries:
        if e.end_time > current_time:
            mtype = _machine_type_of(e.machine)
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
            dur_val = d_var[(key[0], key[1], assigned_unit)].X
            schedule.append(ScheduleEntry(
                job_id=job.job_id, op_idx=op.op_idx,
                machine=machine, service_unit=assigned_unit,
                start_time=start_val,
                end_time=start_val + dur_val))
        schedule.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
        return schedule
    except Exception:
        return None


# ── PWL helpers ────────────────────────────────────────────────────────────

def _effective_dur_at(start: float, base: float, machine_id: str,
                      disruptions: List[Disruption]) -> float:
    """Wall-clock duration for an op starting at *start* on a disrupted machine."""
    # Build sorted factor-change events for this machine
    events: List[Tuple[float, float]] = [(0.0, 1.0)]
    for d in disruptions:
        if d.machine_id == machine_id:
            events.append((d.time, d.factor))
            if d.duration > 0:
                events.append((d.time + d.duration, 1.0))
    events.sort()

    # Find current segment
    seg_idx = 0
    for i, (t, _) in enumerate(events):
        if t <= start:
            seg_idx = i

    remaining = base
    t = start

    while remaining > 1e-9 and seg_idx < len(events):
        _, factor = events[seg_idx]
        next_t = events[seg_idx + 1][0] if seg_idx + 1 < len(events) else float('inf')
        seg_len = next_t - t
        work = seg_len / factor if factor > 0 else float('inf')

        if work >= remaining:
            return (t + remaining * factor) - start

        remaining -= work
        t = next_t
        seg_idx += 1

    # Fallback: finish with latest factor
    last_factor = events[-1][1] if events else 1.0
    return (t + remaining * last_factor) - start


def _make_pwl_breakpoints(
    base_dur: float, machine_id: str,
    disruptions: List[Disruption],
    horizon: float, n_samples: int,
) -> Tuple[List[float], List[float]]:
    """Build PWL breakpoints that exactly capture the effective-duration function.

    The function is piecewise-linear with slope changes ONLY at disruption
    onset/recovery times.  By including every disruption boundary as a
    breakpoint, the PWL model becomes *exact* — not an approximation.
    Evenly-spaced samples fill the gaps for smooth coverage.
    """
    # Collect every event time where the factor changes for this machine
    event_times: set[float] = {0.0}
    for d in disruptions:
        if d.machine_id == machine_id:
            event_times.add(d.time)
            if d.duration > 0:
                event_times.add(d.time + d.duration)
    sorted_events = sorted(event_times)

    # Merge: event boundaries + evenly-spaced samples → deduplicated, sorted
    merged: set[float] = set()
    for t in sorted_events:
        merged.add(t)
    for k in range(n_samples + 1):
        merged.add(horizon * k / n_samples)

    xs: List[float] = sorted(merged)
    ys: List[float] = [
        _effective_dur_at(t, base_dur, machine_id, disruptions) for t in xs
    ]
    return xs, ys


def _find_job_by_id(jobs: List[Job], jid: int) -> Optional[Job]:
    for j in jobs:
        if j.job_id == jid:
            return j
    return None
