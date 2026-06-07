"""MILP-based optimal scheduler for pure FJSP (makespan objective) using Gurobi."""

from typing import Dict, List, Optional, Tuple
from models import Job, Operation, ScheduleEntry

try:
    import gurobipy as _gp
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False


def schedule_makespan_milp(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float = 0.0,
    time_factors: Optional[Dict[str, float]] = None,
    machine_deadlines: Optional[Dict[str, float]] = None,
    time_limit: float = 120.0,
) -> Optional[List[ScheduleEntry]]:
    """
    MILP for pure FJSP minimising makespan (C_max).

    Decision variables:
      x[(j,o), m]    : operation assigned to machine m  (binary)
      s[(j,o)]       : start time  (continuous)
      C_max          : makespan  (continuous)
      y[(a,b), m]    : a precedes b on machine m  (binary)

    Parameters
    ----------
    jobs : list of Job
        All jobs visible to the scheduler.
    fixed_entries : list of ScheduleEntry
        Already-fixed operations (completed or mid-processing).
    current_time : float
        The moment at which the schedule is being computed.
    time_factors : dict or None
        Optional per-machine processing-time multipliers  (e.g.  ``{'M3': 2.0}``).
    machine_deadlines : dict or None
        Optional per-machine hard deadlines after which the machine is
        unavailable.  Any operation assigned to *machine* must complete
        by *deadline*:  s + p_m <= deadline + M*(1 - x_m).
    time_limit : float
        Gurobi time limit in seconds.

    Returns
    -------
    list of ScheduleEntry  or  None if infeasible / Gurobi unavailable.
    """
    if not _HAS_GUROBI:
        return None

    # ── Index operations ─────────────────────────────────────────────────
    fixed_keys = {(e.job_id, e.op_idx) for e in fixed_entries}
    ops_to_schedule: List[Tuple[Job, Operation]] = []
    for job in jobs:
        for op in job.operations:
            if (job.job_id, op.op_idx) not in fixed_keys:
                ops_to_schedule.append((job, op))

    if not ops_to_schedule:
        return list(fixed_entries)

    # ── Build lookup: job by id ──────────────────────────────────────────
    job_by_id: Dict[int, Job] = {j.job_id: j for j in jobs}

    # ── Big-M ────────────────────────────────────────────────────────────
    big_m = 0.0
    for job in jobs:
        for op in job.operations:
            big_m += max(op.times.values()) if op.times else 0
    big_m = max(big_m * (max(time_factors.values()) if time_factors else 1.0), 500.0)

    # ── Gurobi environment & model ───────────────────────────────────────
    env = _gp.Env(params={"OutputFlag": 0})
    model = _gp.Model("FJSP_Makespan", env=env)
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.0
    model.Params.MIPFocus = 2

    # ── Variables ────────────────────────────────────────────────────────
    x: Dict[Tuple[int, int, str], _gp.Var] = {}     # assignment  (j_id, o_idx, machine)
    s: Dict[Tuple[int, int], _gp.Var] = {}           # start time  (j_id, o_idx)

    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        s[key] = model.addVar(
            lb=max(current_time, job.arrival_time),
            vtype=_gp.GRB.CONTINUOUS,
            name=f"s_{job.job_id}_{op.op_idx}")
        for m in op.times:
            x[(job.job_id, op.op_idx, m)] = model.addVar(
                vtype=_gp.GRB.BINARY,
                name=f"x_{job.job_id}_{op.op_idx}_{m}")

    # Makespan
    C_max = model.addVar(lb=0, vtype=_gp.GRB.CONTINUOUS, name="C_max")

    # ── Sequencing variables (per-machine ordered pairs) ─────────────────
    # Build index:  machine → list of operation keys that can run on it
    machine_ops: Dict[str, List[Tuple[int, int]]] = {}
    for job, op in ops_to_schedule:
        for m in op.times:
            machine_ops.setdefault(m, []).append((job.job_id, op.op_idx))

    y: Dict[Tuple, _gp.Var] = {}
    for m, op_keys in machine_ops.items():
        for i in range(len(op_keys)):
            for j_idx in range(i + 1, len(op_keys)):
                a_key = op_keys[i]
                b_key = op_keys[j_idx]
                # a → b direction
                y_key_ab = (a_key[0], a_key[1], b_key[0], b_key[1], m)
                y[y_key_ab] = model.addVar(
                    vtype=_gp.GRB.BINARY,
                    name=f"y_{a_key[0]}_{a_key[1]}_to_{b_key[0]}_{b_key[1]}_{m}")
                # b → a direction
                y_key_ba = (b_key[0], b_key[1], a_key[0], a_key[1], m)
                y[y_key_ba] = model.addVar(
                    vtype=_gp.GRB.BINARY,
                    name=f"y_{b_key[0]}_{b_key[1]}_to_{a_key[0]}_{a_key[1]}_{m}")

    model.update()

    # ── Helper: effective processing time ────────────────────────────────
    def _proc_time(j_id: int, o_idx: int, m: str) -> float:
        """Return processing time of an operation on machine m, with optional factor."""
        base = job_by_id[j_id].operations[o_idx].times.get(m, 0.0)
        if time_factors and m in time_factors:
            return base * time_factors[m]
        return base

    def _proc_expr(op_key: Tuple[int, int]) -> _gp.LinExpr:
        """Build  Σ_m  x[op_key,m] * p[op_key,m]   as a linear expression."""
        j_id, o_idx = op_key
        expr = _gp.LinExpr()
        op = job_by_id[j_id].operations[o_idx]
        for m in op.times:
            expr += x[(j_id, o_idx, m)] * _proc_time(j_id, o_idx, m)
        return expr

    # ── Constraints ──────────────────────────────────────────────────────

    # (1) Each operation assigned to exactly one feasible machine
    for job, op in ops_to_schedule:
        key = (job.job_id, op.op_idx)
        model.addConstr(
            _gp.quicksum(x[(key[0], key[1], m)] for m in op.times) == 1,
            name=f"assign_{key[0]}_{key[1]}")

    # (2) Precedence within each job
    for job in jobs:
        ops = job.operations
        for idx in range(len(ops) - 1):
            cur_key = (job.job_id, ops[idx].op_idx)
            nxt_key = (job.job_id, ops[idx + 1].op_idx)

            if cur_key in fixed_keys:
                # Current op is fixed: next must start after fixed end
                cur_end = next(e.end_time for e in fixed_entries
                               if e.job_id == job.job_id
                               and e.op_idx == ops[idx].op_idx)
                if nxt_key not in fixed_keys:
                    model.addConstr(s[nxt_key] >= cur_end,
                                    name=f"prec_{job.job_id}_{idx}_fixed")
            elif nxt_key in fixed_keys:
                # Next op is fixed: current must finish before fixed start
                nxt_start = next(e.start_time for e in fixed_entries
                                 if e.job_id == job.job_id
                                 and e.op_idx == ops[idx + 1].op_idx)
                model.addConstr(s[cur_key] + _proc_expr(cur_key) <= nxt_start,
                                name=f"prec_{job.job_id}_{idx}_to_fixed")
            else:
                # Both free: standard precedence
                model.addConstr(
                    s[nxt_key] >= s[cur_key] + _proc_expr(cur_key),
                    name=f"prec_{job.job_id}_{idx}")

    # (3) Machine disjunction
    for m, op_keys in machine_ops.items():
        for i in range(len(op_keys)):
            for j_idx in range(i + 1, len(op_keys)):
                a_key = op_keys[i]
                b_key = op_keys[j_idx]

                # Shorthand for assignment variables
                x_a = x[(a_key[0], a_key[1], m)]
                x_b = x[(b_key[0], b_key[1], m)]
                y_ab = y[(a_key[0], a_key[1], b_key[0], b_key[1], m)]
                y_ba = y[(b_key[0], b_key[1], a_key[0], a_key[1], m)]

                # y_ab ≤ x_a  and  y_ab ≤ x_b
                model.addConstr(y_ab <= x_a,
                                name=f"y_bound1_{a_key}_{b_key}_{m}")
                model.addConstr(y_ab <= x_b,
                                name=f"y_bound2_{a_key}_{b_key}_{m}")
                model.addConstr(y_ba <= x_a,
                                name=f"y_bound3_{a_key}_{b_key}_{m}")
                model.addConstr(y_ba <= x_b,
                                name=f"y_bound4_{a_key}_{b_key}_{m}")

                # If both on m, one must precede the other
                model.addConstr(
                    y_ab + y_ba >= x_a + x_b - 1,
                    name=f"y_order_{a_key}_{b_key}_{m}")

                # At most one direction
                model.addConstr(
                    y_ab + y_ba <= 1,
                    name=f"y_excl_{a_key}_{b_key}_{m}")

                # Disjunction a→b : if a before b then s_b >= s_a + p_a_m
                p_a_m = _proc_time(a_key[0], a_key[1], m)
                model.addConstr(
                    s[b_key] >= s[a_key] + p_a_m
                    - big_m * (1 - y_ab),
                    name=f"disj_ab_{a_key}_{b_key}_{m}")
                # Disjunction b→a : if b before a then s_a >= s_b + p_b_m
                p_b_m = _proc_time(b_key[0], b_key[1], m)
                model.addConstr(
                    s[a_key] >= s[b_key] + p_b_m
                    - big_m * (1 - y_ba),
                    name=f"disj_ba_{a_key}_{b_key}_{m}")

    # (4) Makespan  ≥  completion of every job's last operation
    for job in jobs:
        last_key = (job.job_id, len(job.operations) - 1)
        if last_key in fixed_keys:
            last_end = next(e.end_time for e in fixed_entries
                           if e.job_id == job.job_id
                           and e.op_idx == len(job.operations) - 1)
            model.addConstr(C_max >= last_end, name=f"cmax_{job.job_id}")
        else:
            model.addConstr(C_max >= s[last_key] + _proc_expr(last_key),
                            name=f"cmax_{job.job_id}")

    # (5) Fixed entries block their assigned machines
    for e in fixed_entries:
        if e.end_time > current_time:
            m = e.machine
            for job, op in ops_to_schedule:
                key = (job.job_id, op.op_idx)
                if m in op.times:
                    model.addConstr(
                        s[key] >= e.end_time
                        - big_m * (1 - x[(key[0], key[1], m)]),
                        name=f"block_{e.job_id}_{e.op_idx}_{key}")

    # (6) Machine unavailability deadlines
    #     If a machine goes down at time *dl*, any operation assigned to it
    #     must complete by *dl*:   s + p_m <= dl + M*(1 - x_m)
    if machine_deadlines:
        for m, dl in machine_deadlines.items():
            for job, op in ops_to_schedule:
                key = (job.job_id, op.op_idx)
                if m not in op.times:
                    continue
                p_m = _proc_time(job.job_id, op.op_idx, m)
                model.addConstr(
                    s[key] + p_m <= dl
                    + big_m * (1 - x[(key[0], key[1], m)]),
                    name=f"deadline_{key[0]}_{key[1]}_{m}")

    # ── Objective ────────────────────────────────────────────────────────
    model.setObjective(C_max, _gp.GRB.MINIMIZE)

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
            assigned_machine = None
            for m in op.times:
                if x[(key[0], key[1], m)].X > 0.5:
                    assigned_machine = m
                    break
            if assigned_machine is None:
                return None
            proc = _proc_time(job.job_id, op.op_idx, assigned_machine)
            schedule.append(ScheduleEntry(
                job.job_id, op.op_idx, assigned_machine,
                start_val, start_val + proc))
        schedule.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
        return schedule
    except Exception:
        return None
