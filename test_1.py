"""
Numerical Example: Three-Layer Hierarchical Scheduling Model for Flexible Job Shop
==================================================================================
Based on hierarchical_modeling.md -- demonstrates the shop / service-unit / machine
layers, load-balancing allocation, and resilience computation under disturbance.
"""

from dataclasses import dataclass
from itertools import product
from math import exp
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================================================================
# 1. PROBLEM DATA
# ==============================================================================

# --- Service Units & Machines ---
# Each unit u has m_u machines; each machine has an importance coefficient alpha
# Format: {unit: {machine: (type, alpha)}}
UNITS: Dict[str, Dict[str, Tuple[str, float]]] = {
    'U1': {
        'M11': ('T1', 1.0),   # T1-capable, high importance
        'M12': ('T2', 0.8),   # T2-capable, medium importance
    },
    'U2': {
        'M21': ('T1', 1.2),   # T1-capable, very high importance
        'M22': ('T2', 0.9),   # T2-capable, medium importance
    },
}

# Flatten for convenience
ALL_MACHINES = [(u, m) for u in UNITS for m in UNITS[u]]
M_TYPE_OF = {(u, m): info[0] for u, m in ALL_MACHINES for info in [UNITS[u][m]]}
ALPHA_OF = {(u, m): info[1] for u, m in ALL_MACHINES for info in [UNITS[u][m]]}
SUM_ALPHA = sum(ALPHA_OF.values())
M_U = {u: len(machines) for u, machines in UNITS.items()}

# --- Jobs & Operations ---
@dataclass
class Operation:
    job_id: int
    op_idx: int      # 0-based
    mtype: str       # 'T1' or 'T2'
    # nominal processing time on each (unit, machine) combo
    p_bar: Dict[Tuple[str, str], float]


@dataclass
class Job:
    """A production job with due-date constraints and penalty weights."""
    job_id: int
    release_date: float   # r_j  -- earliest start time
    due_date: float       # d_j  -- committed delivery date
    alpha: float          # earliness penalty (per unit time early)
    beta: float           # tardiness penalty (per unit time late)
    operations: List[Operation]
    arrival_time: float = 0.0  # when the job becomes known to the scheduler

    @property
    def total_work(self) -> float:
        """Sum of nominal processing times (shortest feasible machine each op)."""
        return sum(min(op.p_bar.values()) for op in self.operations)

# 3 Jobs, 7 operations total
JOBS: List[Job] = [
    Job(1, release_date=0,  due_date=96,  alpha=0, beta=1.0, operations=[
        Operation(1, 0, 'T1', {('U1', 'M11'): 10.0, ('U2', 'M21'): 14.0}),
        Operation(1, 1, 'T2', {('U1', 'M12'):  12.0, ('U2', 'M22'):  16.0}),
        Operation(1, 2, 'T1', {('U1', 'M11'):  14.0, ('U2', 'M21'):  16.0}),
    ]),
    Job(2, release_date=0,  due_date=72,  alpha=0.0, beta=4.0, operations=[
        Operation(2, 0, 'T1', {('U1', 'M11'): 18.0, ('U2', 'M21'): 14.0}),
        Operation(2, 1, 'T2', {('U1', 'M12'): 30.0, ('U2', 'M22'): 22.0}),
        Operation(2, 2, 'T1', {('U1', 'M11'): 25.0, ('U2', 'M21'): 19.0}),
    ]),
    Job(3, release_date=0,  due_date=120,  alpha=1.0, beta=2, operations=[
        Operation(3, 0, 'T1', {('U1', 'M11'):  15.0, ('U2', 'M21'): 12.0}),
        Operation(3, 1, 'T2', {('U1', 'M12'): 16.0, ('U2', 'M22'): 12.0}),
        Operation(3, 2, 'T1', {('U1', 'M11'): 30.0, ('U2', 'M21'): 18.0}),
    ]),
    Job(4, release_date=48,  due_date=96,  alpha=1.0, beta=6, arrival_time=24, operations=[
        Operation(4, 0, 'T1', {('U1', 'M11'):  12.0, ('U2', 'M21'): 20.0}),
        Operation(4, 1, 'T2', {('U1', 'M12'): 12.0, ('U2', 'M22'): 20.0}),
        Operation(4, 2, 'T1', {('U1', 'M11'): 20.0, ('U2', 'M21'): 24.0}),
    ]),
]

ALL_OPS = [(job, op) for job in JOBS for op in job.operations]
JOB_BY_ID = {job.job_id: job for job in JOBS}


# ==============================================================================
# 2. SHOP LEVEL: weighted aggregate processing time & unit assignment
# ==============================================================================

def weighted_agg_time(op: Operation, unit: str,
                      time_factors: Dict[Tuple[str, str], float] = None) -> float:
    """Eq (4): weighted aggregate processing time of *op* on *unit*."""
    tf = time_factors or {}
    machines_in_unit = [m for m in UNITS[unit]]
    compatible = [(u, m) for (u, m) in [(unit, m) for m in machines_in_unit]
                  if M_TYPE_OF[(u, m)] == op.mtype]
    if not compatible:
        return float('inf')  # cannot process on this unit
    num = sum(ALPHA_OF[cm] * op.p_bar[cm] * tf.get(cm, 1.0) for cm in compatible)
    den = sum(ALPHA_OF[cm] for cm in compatible)
    return num / den


def shop_level_assign(
    time_factors: Dict[Tuple[str, str], float] = None,
    visible_jobs: List[Job] = None,
) -> Dict[Tuple[int, int], str]:
    """
    Solve the Shop Level problem: min max_u (sum X_{i,j,u} * P_tilde_{i,j,u} / m_u)
    Eq (5)-(6).  Brute-force enumeration for transparency.
    If visible_jobs is given, only those jobs' operations are considered.
    """
    tf = time_factors or {}
    jobs = visible_jobs if visible_jobs is not None else list(JOBS)
    ops_list = [(job, op) for job in jobs for op in job.operations]

    best_max_load = float('inf')
    best_assign = None

    # Precompute weighted aggregate times
    p_tilde: Dict[Tuple[int, int, str], float] = {}
    for job, op in ops_list:
        for unit in UNITS:
            p_tilde[(job.job_id, op.op_idx, unit)] = weighted_agg_time(op, unit, tf)

    feasible_units = []
    for job, op in ops_list:
        feas = [u for u in UNITS if p_tilde[(job.job_id, op.op_idx, u)] < float('inf')]
        if not feas:
            raise ValueError(f"J{job.job_id}-Op{op.op_idx+1} has no feasible unit!")
        feasible_units.append(feas)

    for choices in product(*feasible_units):
        load = {u: 0.0 for u in UNITS}
        for (job, op), unit in zip(ops_list, choices):
            load[unit] += p_tilde[(job.job_id, op.op_idx, unit)]
        max_load = max(load[u] / M_U[u] for u in UNITS)
        if max_load < best_max_load - 1e-9:
            best_max_load = max_load
            best_assign = {(job.job_id, op.op_idx): u
                          for (job, op), u in zip(ops_list, choices)}

    return best_assign


# ==============================================================================
# 3. UNIT LEVEL: machine selection & sequencing within each unit
# ==============================================================================

ScheduleEntryType = Tuple[str, str, float, float]  # (unit, machine, start, end)

def unit_level_schedule(
    assignment: Dict[Tuple[int, int], str],
    machine_status: Dict[Tuple[str, str], bool] = None,
    time_factors: Dict[Tuple[str, str], float] = None,
    fixed_entries: Dict[Tuple[int, int], ScheduleEntryType] = None,
    current_time: float = 0.0,
    visible_jobs: List[Job] = None,
) -> Tuple[Dict[Tuple[int, int], ScheduleEntryType], float]:
    """
    Returns (schedule, Cmax).

    fixed_entries: ops already completed/in-progress -- pre-load into trackers.
    current_time:  decision-point timestamp (used with release_date lower bound).
    visible_jobs:  subset of JOBS known at this decision point.
    """
    if machine_status is None:
        machine_status = {cm: True for cm in ALL_MACHINES}
    tf = time_factors or {}
    fixed = fixed_entries or {}
    jobs = visible_jobs if visible_jobs is not None else list(JOBS)

    schedule: Dict[Tuple[int, int], ScheduleEntryType] = {}
    machine_free: Dict[Tuple[str, str], float] = {cm: current_time for cm in ALL_MACHINES}
    job_ready: Dict[int, float] = {}

    for job in jobs:
        jid = job.job_id
        # start from max(release_date, current_time), but frozen ops may advance this
        job_ready[jid] = max(job.release_date, current_time)

    # Pre-load fixed entries
    for key, (unit, mach, st, en) in fixed.items():
        schedule[key] = (unit, mach, st, en)
        cm_key = (unit, mach)
        if en > machine_free.get(cm_key, 0.0):
            machine_free[cm_key] = en
        jid = key[0]
        if en > job_ready.get(jid, 0.0):
            job_ready[jid] = en

    # Build list of ops to schedule (visible jobs only, skip fixed)
    ops_to_schedule: List[Tuple[Job, Operation]] = []
    for job in jobs:
        for op in job.operations:
            key = (job.job_id, op.op_idx)
            if key not in fixed:
                ops_to_schedule.append((job, op))

    # Global sort by (job_id, op_idx) to enforce precedence
    ops_to_schedule.sort(key=lambda x: (x[0].job_id, x[1].op_idx))

    for job, op in ops_to_schedule:
        jid = job.job_id
        unit = assignment[(jid, op.op_idx)]

        candidates = [(u, m) for (u, m) in ALL_MACHINES
                     if u == unit
                     and M_TYPE_OF[(u, m)] == op.mtype
                     and machine_status.get((u, m), True)]

        if not candidates:
            schedule[(jid, op.op_idx)] = (unit, 'FAIL', -1.0, -1.0)
            continue

        ready = job_ready[jid]
        best_machine = None
        best_start = float('inf')
        for cm in candidates:
            st = max(ready, machine_free[cm])
            if st < best_start:
                best_start = st
                best_machine = cm

        proc_time = op.p_bar[best_machine] * tf.get(best_machine, 1.0)
        end_time = best_start + proc_time
        schedule[(jid, op.op_idx)] = (unit, best_machine[1],
                                      best_start, end_time)
        machine_free[best_machine] = end_time
        job_ready[jid] = end_time

    valid_ends = [s[3] for s in schedule.values() if s[3] > 0]
    cmax = max(valid_ends) if valid_ends else float('inf')
    return schedule, cmax

def compute_penalty(
    schedule: Dict[Tuple[int, int], Tuple[str, str, float, float]],
    visible_jobs: List[Job] = None,
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Compute earliness/tardiness penalty for each job.
    Returns {job_id: (completion, earliness, tardiness, penalty)}.
    """
    jobs = visible_jobs if visible_jobs is not None else list(JOBS)
    result = {}
    for job in jobs:
        jid = job.job_id
        ends = [schedule[(jid, op.op_idx)][3]
                for op in job.operations
                if (jid, op.op_idx) in schedule]
        Cj = max(ends) if ends else 0.0
        Ej = max(0.0, job.due_date - Cj)
        Tj = max(0.0, Cj - job.due_date)
        penalty = job.alpha * Ej + job.beta * Tj
        result[jid] = (Cj, Ej, Tj, penalty)
    return result


# ==============================================================================
# 4. RESILIENCE METRIC
# ==============================================================================

def resilience(phi: float, cmax_nominal: float, cmax_disturbed: float
               ) -> Tuple[float, float, float]:
    """
    Compute R(xi) = exp(-(phi + eta))   -- Eq (1)-(3)

    phi = resource damage ratio (weighted by alpha * degradation degree)
    eta = completion delay ratio
    Returns (R, phi, eta) for display convenience.
    """
    eta = (cmax_disturbed - cmax_nominal) / cmax_nominal if cmax_nominal > 0 else 0.0
    return exp(-(phi + eta)), phi, eta


# ==============================================================================
# 5. HELPER: compute loads from an assignment
# ==============================================================================

def compute_loads(assignment: Dict[Tuple[int, int], str],
                  time_factors: Dict[Tuple[str, str], float] = None,
                  visible_jobs: List[Job] = None) -> Dict[str, float]:
    """Return {unit: total_P_tilde} for the given shop-level assignment."""
    tf = time_factors or {}
    jobs = visible_jobs if visible_jobs is not None else list(JOBS)
    loads = {u: 0.0 for u in UNITS}
    for job in jobs:
        for op in job.operations:
            u = assignment[(job.job_id, op.op_idx)]
            loads[u] += weighted_agg_time(op, u, tf)
    return loads


# ==============================================================================
# 6. PRINTING & DISPLAY FUNCTIONS
# ==============================================================================

def print_separator(title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def print_configuration():
    """Print problem setup: units, machines, jobs with their attributes."""
    print_separator("CONFIGURATION")
    print(f"  Service Units: {len(UNITS)}  |  Total Machines: {len(ALL_MACHINES)}")
    for u in UNITS:
        machines = [f"{m}({info[0]}, alpha={info[1]})" for m, info in UNITS[u].items()]
        print(f"    {u}: {', '.join(machines)}  (m_{u.lower()} = {M_U[u]})")
    print(f"  Sum(alpha)  = {SUM_ALPHA:.1f}")
    print(f"\n  Jobs: {len(JOBS)}  |  Operations: {len(ALL_OPS)}")
    print(f"  {'Job':>5s}  {'r_j':>5s}  {'d_j':>5s}  {'arr':>4s}  "
          f"{'alpha':>5s}  {'beta':>5s}  {'Ops':>20s}")
    print(f"  {'-' * 60}")
    for job in sorted(JOBS, key=lambda j: j.job_id):
        op_desc = ", ".join(f"O{job.job_id}{i+1}({op.mtype})"
                          for i, op in enumerate(job.operations))
        print(f"  {'J' + str(job.job_id):>5s}  {job.release_date:>5.0f}  "
              f"{job.due_date:>5.0f}  {job.arrival_time:>4.0f}  "
              f"{job.alpha:>5.1f}  {job.beta:>5.1f}  {op_desc}")


def print_shop_level_table(assignment: Dict[Tuple[int, int], str],
                           loads: Dict[str, float],
                           visible_jobs: List[Job] = None):
    """Print P_tilde table, per-unit loads, and f1 objective value."""
    jobs = visible_jobs if visible_jobs is not None else list(JOBS)
    ops_list = [(job, op) for job in jobs for op in job.operations]

    print("\n  Weighted aggregate processing times  P_tilde_{i,j,u}  (Eq 4):")
    header = f"  {'Op':>12s}"
    for u in UNITS:
        header += f"  {'P_tilde(' + u + ')':>13s}"
    header += f"  {'Assigned':>10s}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for job, op in ops_list:
        u = assignment[(job.job_id, op.op_idx)]
        vals = ""
        for unit in UNITS:
            v = weighted_agg_time(op, unit)
            if v < float('inf'):
                vals += f"  {v:>13.1f}"
            else:
                vals += f"  {'N/A':>13s}"
        print(f"  {'O' + str(job.job_id) + str(op.op_idx+1):>12s}"
              f"{vals}  -> {u:>8s}")

    print(f"  {'-' * len(header)}")
    load_str = "  " + " " * 12
    for u in UNITS:
        load_str += f"  sum={loads[u]:>11.1f}"
    print(load_str)
    load_per_m = "  " + " " * 12
    for u in UNITS:
        load_per_m += f"  per_mc={loads[u]/M_U[u]:>7.1f}"
    print(load_per_m)

    max_load = max(loads[u] / M_U[u] for u in UNITS)
    print(f"\n  Objective f1 = min max load/machine = {max_load:.1f}")


def print_schedule_table(
    schedule: Dict[Tuple[int, int], Tuple[str, str, float, float]],
    cmax: float,
    affected_keys: List[Tuple[int, int]] = None,
    fixed_keys: List[Tuple[int, int]] = None,
    visible_jobs: List[Job] = None,
):
    """Print a schedule as a table sorted by start time, with optional markers."""
    affected_keys = affected_keys or []
    fixed_keys = fixed_keys or []
    jobs = visible_jobs if visible_jobs is not None else list(JOBS)
    ops_list = [(job, op) for job in jobs for op in job.operations]

    print(f"\n  {'Job':>6s}  {'Op':>4s}  {'Unit':>5s}  {'Mach':>5s}  "
          f"{'Start':>8s}  {'End':>8s}  {'Dur':>6s}  {'Status':>10s}")
    print(f"  {'-' * 68}")
    sorted_entries = sorted(
        [(job.job_id, op.op_idx) for job, op in ops_list],
        key=lambda k: schedule[k][2])
    for key in sorted_entries:
        u, m, st, en = schedule[key]
        marker = " <-- migrated" if key in affected_keys else ""
        is_fixed = key in fixed_keys
        if st < 0:
            print(f"  {'J' + str(key[0]):>6s}  {key[1]+1:>4d}  {u:>5s}  "
                  f"{'FAIL':>5s}  {'--':>8s}  {'--':>8s}  {'--':>6s}  "
                  f"{'FAIL':>10s}{marker}")
        else:
            dur = en - st
            print(f"  {'J' + str(key[0]):>6s}  {key[1]+1:>4d}  {u:>5s}  {m:>5s}  "
                  f"{st:>8.1f}  {en:>8.1f}  {dur:>6.1f}  "
                  f"{'FIXED' if is_fixed else 'planned':>10s}"
                  f"{marker}")
    print(f"\n  Makespan C_max = {cmax:.1f}")


def print_penalty_table(penalties: Dict[int, Tuple[float, float, float, float]],
                        visible_jobs: List[Job] = None):
    """Print per-job completion, earliness, tardiness, and weighted penalty."""
    jobs = visible_jobs if visible_jobs is not None else list(JOBS)
    print(f"\n  {'Job':>5s}  {'r_j':>5s}  {'d_j':>5s}  {'C_j':>8s}  "
          f"{'E_j':>8s}  {'T_j':>8s}  {'alpha*E+beta*T':>15s}")
    print(f"  {'-' * 65}")
    total = 0.0
    for job in jobs:
        Cj, Ej, Tj, pen = penalties[job.job_id]
        total += pen
        print(f"  {'J' + str(job.job_id):>5s}  {job.release_date:>5.0f}  "
              f"{job.due_date:>5.0f}  {Cj:>8.1f}  {Ej:>8.1f}  {Tj:>8.1f}  "
              f"{pen:>15.1f}")
    print(f"  {'-' * 65}")
    print(f"  Total weighted penalty: {total:.1f}")
    return total


def print_disturbance_info(
    degraded_machines: List[Tuple[str, str]],
    time_factors: Dict[Tuple[str, str], float],
    affected_keys: List[Tuple[int, int]],
    migrated_keys: List[Tuple[int, int]],
    old_loads: Dict[str, float],
    new_loads: Dict[str, float],
):
    """Print which machine is degraded, which ops are affected/migrated, and load change."""
    for cm in degraded_machines:
        factor = time_factors.get(cm, 1.0)
        degradation = (factor - 1.0) * 100
        print(f"\n  Degraded machine: {cm[0]}-{cm[1]} "
              f"(alpha = {ALPHA_OF[cm]}, processing time +{degradation:.0f}%)")
    print(f"  Affected operations (processed on degraded machine):")
    for key in affected_keys:
        print(f"    O{key[0]}{key[1]+1}")
    if migrated_keys:
        print(f"  Migrated operations (re-assigned to backup unit):")
        for key in migrated_keys:
            print(f"    O{key[0]}{key[1]+1}")
    else:
        print(f"  No operations migrated -- all stay on degraded machine.")
    print(f"\n  Unit load before/after shop-level re-optimization:")
    for u in UNITS:
        old_l, new_l = old_loads[u], new_loads[u]
        print(f"    {u}: sum(P_tilde) = {old_l:.1f} -> {new_l:.1f}  "
              f"(per-machine: {old_l/M_U[u]:.1f} -> {new_l/M_U[u]:.1f})")


def print_resilience_metric(
    R: float, phi: float, eta: float,
    cmax_nom: float, cmax_dist: float,
    degraded_alpha: float, degradation_ratio: float,
):
    """Print step-by-step resilience calculation and interpretation."""
    print(f"\n  phi  = Sum(alpha_i * degradation_i) / Sum(alpha_all)")
    print(f"       = ({degraded_alpha:.1f} * {degradation_ratio:.1f}) / {SUM_ALPHA:.1f}")
    print(f"       = {phi:.4f}")
    print(f"\n  eta  = (C'_max - C_max) / C_max")
    print(f"       = ({cmax_dist:.1f} - {cmax_nom:.1f}) / {cmax_nom:.1f}")
    print(f"       = {eta:.4f}")
    print(f"\n  R(xi) = exp(-(phi + eta))")
    print(f"         = exp(-({phi:.4f} + {eta:.4f}))")
    print(f"         = exp(-{phi + eta:.4f})")
    print(f"         = {R:.4f}")

    print(f"\n  -- Interpretation --")
    print(f"  Resource damage:  {phi*100:.1f}% (weighted by machine importance)")
    print(f"  Schedule delay:   {eta*100:.1f}% (relative makespan increase)")
    print(f"  System resilience: {R:.4f}")
    if R > 0.7:
        print(f"  => System is HIGHLY resilient -- absorbs disturbance well.")
    elif R > 0.4:
        print(f"  => System is MODERATELY resilient -- some degradation accepted.")
    else:
        print(f"  => System has LOW resilience -- significant impact from disturbance.")


def print_summary_table(
    cmax_nom: float, cmax_dist: float,
    phi: float, eta: float, R: float,
    pen_nom: float, pen_dist: float,
    loads_nom: Dict[str, float], loads_dist: Dict[str, float],
):
    """Print side-by-side comparison of nominal vs. disturbed scenarios."""
    print_separator("SUMMARY")
    print(f"\n  {'Metric':<30s}  {'Nominal (xi=0)':>15s}  {'Disturbed (xi)':>15s}")
    print(f"  {'-' * 62}")
    print(f"  {'Makespan C_max':<30s}  {cmax_nom:>15.1f}  {cmax_dist:>15.1f}")
    print(f"  {'Resource damage phi':<30s}  {'--':>15s}  {phi:>15.4f}")
    print(f"  {'Delay ratio eta':<30s}  {'--':>15s}  {eta:>15.4f}")
    print(f"  {'Resilience R(xi)':<30s}  {'--':>15s}  {R:>15.4f}")
    print(f"  {'Total penalty':<30s}  {pen_nom:>15.1f}  {pen_dist:>15.1f}")
    for u in UNITS:
        print(f"  {'Unit load ' + u + ' (per m/c)':<30s}  "
              f"{loads_nom[u]/M_U[u]:>15.1f}  {loads_dist[u]/M_U[u]:>15.1f}")
    print()


# ==============================================================================
# 8. GANTT CHART VISUALIZATION
# ==============================================================================

# Color scheme: one color per job
JOB_COLORS = {1: '#4C72B0', 2: '#DD8452', 3: '#55A868', 4: '#C44E52'}
JOB_LABELS = {1: 'J1', 2: 'J2', 3: 'J3', 4: 'J4'}
MACHINE_ORDER = ['M11', 'M12', 'M21', 'M22']


def plot_gantt_dynamic(
    schedule_t0: Dict[Tuple[int, int], Tuple[str, str, float, float]],
    schedule_t24: Dict[Tuple[int, int], Tuple[str, str, float, float]],
    frozen_keys: List[Tuple[int, int]],
    j4_keys: List[Tuple[int, int]],
    cmax_t0: float,
    cmax_t24: float,
    current_time: float = 24.0,
    save_path: str = 'gantt_dynamic.png',
):
    """
    Two-panel Gantt chart for dynamic scheduling:
      Top   : Stage 1 (t=0) initial plan for J1-J3
      Bottom: Stage 2 (t=24) frozen ops (hatched) + J4 (new) + replanned
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 8.5))

    x_max = max(cmax_t0, cmax_t24) * 1.10
    y_pos = {m: i for i, m in enumerate(MACHINE_ORDER)}

    # ---- Top: Stage 1 schedule ----
    bars_t0 = []
    for key, (_u, mach, st, en) in schedule_t0.items():
        if st >= 0:
            bars_t0.append((mach, st, en - st, key[0], key[1],
                           key in frozen_keys, False))
    bars_t0.sort(key=lambda b: b[1])

    for mach, st, dur, jid, op_idx, will_freeze, _is_j4 in bars_t0:
        y = y_pos[mach]
        color = JOB_COLORS.get(jid, '#888888')
        bar = ax_top.barh(y, dur, left=st, height=0.55,
                          color=color, edgecolor='white', linewidth=0.5, alpha=0.9)
        label = f"{JOB_LABELS.get(jid, jid)}-{op_idx + 1}"
        ax_top.text(st + dur / 2, y, label, ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color='white')
        if will_freeze:
            bar.patches[0].set_edgecolor('#FF6B6B')
            bar.patches[0].set_linewidth(2.5)
            bar.patches[0].set_linestyle('--')
            ax_top.text(st + dur / 2, y + 0.30, 'FROZEN', ha='center', va='center',
                        fontsize=5, color='yellow', fontweight='bold')

    ax_top.set_yticks(list(y_pos.values()))
    ax_top.set_yticklabels(MACHINE_ORDER)
    ax_top.set_ylabel('Machine', fontsize=11)
    ax_top.set_title('Stage 1  (t = 0)  --  Initial plan for J1, J2, J3',
                     fontsize=13, fontweight='bold')
    ax_top.set_xlim(0, x_max)
    ax_top.invert_yaxis()
    ax_top.grid(axis='x', alpha=0.3, linestyle='--')
    ax_top.axvline(x=cmax_t0, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax_top.text(cmax_t0 + 1, -0.4, f'Cmax={cmax_t0:.0f}', fontsize=9,
                color='black', va='bottom')

    legend = [mpatches.Patch(color=JOB_COLORS[jid],
                              label=f"{JOB_LABELS[jid]} (Job {jid})")
              for jid in sorted(JOB_COLORS)]
    legend.append(mpatches.Patch(facecolor='white', edgecolor='#FF6B6B',
                                  linewidth=2.5, linestyle='--', hatch='///',
                                  label='Frozen (started < 24)'))
    ax_top.legend(handles=legend, loc='upper right', fontsize=7.5,
                  ncol=3, title='Jobs', title_fontsize=9)

    # ---- Bottom: Stage 2 schedule ----
    bars_t24 = []
    for key, (_u, mach, st, en) in schedule_t24.items():
        if st >= 0:
            bars_t24.append((mach, st, en - st, key[0], key[1],
                            key in frozen_keys, key in j4_keys))
    bars_t24.sort(key=lambda b: b[1])

    for mach, st, dur, jid, op_idx, is_frozen, is_j4 in bars_t24:
        y = y_pos[mach]
        color = JOB_COLORS.get(jid, '#888888')
        # Frozen ops: hatched; J4 ops: slightly more saturated
        hatch = '///' if is_frozen else ''
        alpha_val = 0.95 if is_j4 else 0.85
        bar = ax_bot.barh(y, dur, left=st, height=0.55,
                          color=color, edgecolor='white', linewidth=0.5,
                          alpha=alpha_val, hatch=hatch)
        label = f"{JOB_LABELS.get(jid, jid)}-{op_idx + 1}"
        ax_bot.text(st + dur / 2, y, label, ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color='white')
        if is_frozen:
            bar.patches[0].set_edgecolor('#FF6B6B')
            bar.patches[0].set_linewidth(2)
            bar.patches[0].set_linestyle('--')
            ax_bot.text(st + dur / 2, y + 0.30, 'FROZEN', ha='center', va='center',
                        fontsize=5, color='yellow', fontweight='bold')
        if is_j4:
            ax_bot.text(st + dur / 2, y - 0.28, 'NEW', ha='center', va='center',
                        fontsize=5, color='white', fontweight='bold',
                        bbox=dict(facecolor='#C44E52', alpha=0.7, pad=1,
                                  boxstyle='round,pad=0.15'))

    # Decision-point line
    ax_bot.axvline(x=current_time, color='red', linestyle='--', linewidth=2.0, alpha=0.7)
    ax_bot.text(current_time + 0.5, len(MACHINE_ORDER) - 0.25,
                f't = {current_time:.0f}  (J4 arrives)',
                fontsize=9, color='red', fontweight='bold', va='bottom')

    ax_bot.set_yticks(list(y_pos.values()))
    ax_bot.set_yticklabels(MACHINE_ORDER)
    ax_bot.set_ylabel('Machine', fontsize=11)
    ax_bot.set_xlabel('Time', fontsize=11)
    ax_bot.set_title('Stage 2  (t = 24)  --  Frozen ops + J4 arrival + replan',
                     fontsize=13, fontweight='bold')
    ax_bot.set_xlim(0, x_max)
    ax_bot.invert_yaxis()
    ax_bot.grid(axis='x', alpha=0.3, linestyle='--')
    ax_bot.axvline(x=cmax_t24, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax_bot.text(cmax_t24 + 1, -0.4, f'Cmax={cmax_t24:.0f}', fontsize=9,
                color='black', va='bottom')

    frozen_labels = [f"J{k[0]}-Op{k[1]+1}" for k in frozen_keys]
    j4_labels = [f"J{k[0]}-Op{k[1]+1}" for k in j4_keys]
    fig.suptitle('Hierarchical Scheduling -- Dynamic Rescheduling\n'
                 f'Frozen: {", ".join(frozen_labels)}  |  '
                 f'New (J4): {", ".join(j4_labels)}',
                 fontsize=11, fontweight='bold', y=1.01)

    fig.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Gantt chart saved to '{save_path}'")


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main():
    # -- Configuration --
    print_configuration()

    # ==========================================================================
    # STAGE 1: t = 0  --  J1, J2, J3 are known
    # ==========================================================================
    print_separator("STAGE 1 (t=0): Initial plan for J1, J2, J3")
    visible_t0 = [j for j in JOBS if j.arrival_time <= 0]

    # Shop level
    assignment_t0 = shop_level_assign(visible_jobs=visible_t0)
    loads_t0 = compute_loads(assignment_t0, visible_jobs=visible_t0)
    print_shop_level_table(assignment_t0, loads_t0, visible_jobs=visible_t0)

    # Unit level
    schedule_t0, cmax_t0 = unit_level_schedule(assignment_t0, visible_jobs=visible_t0)
    print_schedule_table(schedule_t0, cmax_t0, visible_jobs=visible_t0)
    pen_t0 = compute_penalty(schedule_t0, visible_jobs=visible_t0)
    total_pen_t0 = print_penalty_table(pen_t0, visible_jobs=visible_t0)

    # ==========================================================================
    # STAGE 2: t = 24  --  J4 arrives, freeze in-progress ops, reschedule
    # ==========================================================================
    print_separator("STAGE 2 (t=24): J4 arrives -- freeze + reschedule")

    # 2a. Identify frozen operations: start_time < 24
    frozen_keys = [key for key in schedule_t0
                   if schedule_t0[key][2] >= 0 and schedule_t0[key][2] < 24]
    frozen_entries = {key: schedule_t0[key] for key in frozen_keys}

    print(f"\n  Frozen operations (started before t=24):")
    for key in frozen_keys:
        u, m, st, en = frozen_entries[key]
        print(f"    O{key[0]}{key[1]+1}: {u}-{m}  [{st:.1f}, {en:.1f})  -- FIXED")

    # 2b. Visible jobs at t=24
    visible_t24 = [j for j in JOBS if j.arrival_time <= 24]

    # 2c. Assign J4 ops to units (shop level for newly arrived job only)
    j4 = JOB_BY_ID[4]
    assignment_j4 = shop_level_assign(visible_jobs=[j4])
    print(f"\n  J4 unit assignment (shop-level):")
    for op in j4.operations:
        u = assignment_j4[(4, op.op_idx)]
        vals = "  ".join(f"P_tilde({unit})={weighted_agg_time(op, unit):.1f}"
                         for unit in UNITS)
        print(f"    O4{op.op_idx+1}: {vals}  -> {u}")

    # 2d. Combine assignments: frozen + unfrozen J1-J3 + J4
    combined_assignment = {}
    # frozen and unfrozen J1-J3 keep their t=0 assignments
    for key, unit in assignment_t0.items():
        combined_assignment[key] = unit
    # J4 gets new assignments
    for key, unit in assignment_j4.items():
        combined_assignment[key] = unit

    # 2e. Unit-level reschedule with frozen entries
    schedule_t24, cmax_t24 = unit_level_schedule(
        combined_assignment,
        fixed_entries=frozen_entries,
        current_time=24.0,
        visible_jobs=visible_t24,
    )
    print_schedule_table(schedule_t24, cmax_t24,
                         fixed_keys=list(frozen_entries.keys()),
                         visible_jobs=visible_t24)
    pen_t24 = compute_penalty(schedule_t24, visible_jobs=visible_t24)
    total_pen_t24 = print_penalty_table(pen_t24, visible_jobs=visible_t24)

    # ==========================================================================
    # Comparison
    # ==========================================================================
    print_separator("COMPARISON: Stage 1 vs Stage 2")
    print(f"\n  {'Metric':<30s}  {'Stage 1 (t=0)':>15s}  {'Stage 2 (t=24)':>15s}")
    print(f"  {'-' * 62}")
    print(f"  {'Makespan C_max':<30s}  {cmax_t0:>15.1f}  {cmax_t24:>15.1f}")
    print(f"  {'Total penalty':<30s}  {total_pen_t0:>15.1f}  {total_pen_t24:>15.1f}")
    for u in UNITS:
        # Recompute loads for stage 2
        pass  # loads not central to this comparison
    job_keys = sorted(set(k[0] for k in schedule_t24.keys()))
    for jid in job_keys:
        job = JOB_BY_ID[jid]
        c0 = pen_t0.get(jid, (0, 0, 0, 0))[0]
        p0 = pen_t0.get(jid, (0, 0, 0, 0))[3]
        c24 = pen_t24.get(jid, (0, 0, 0, 0))[0]
        p24 = pen_t24.get(jid, (0, 0, 0, 0))[3]
        print(f"  {'J' + str(jid) + ' completion / penalty':<30s}  "
              f"{c0:>8.1f} / {p0:>4.1f}  {c24:>8.1f} / {p24:>4.1f}")
    print()

    # -- Gantt Chart --
    j4_keys = [(4, op.op_idx) for op in JOB_BY_ID[4].operations]
    plot_gantt_dynamic(schedule_t0, schedule_t24,
                       frozen_keys=frozen_keys,
                       j4_keys=j4_keys,
                       cmax_t0=cmax_t0, cmax_t24=cmax_t24)


if __name__ == '__main__':
    main()