"""
Numerical Example: Three-Layer Hierarchical Scheduling Model for Flexible Job Shop
==================================================================================
Based on hierarchical_modeling.md -- demonstrates the shop / service-unit / machine
layers, load-balancing allocation, and resilience computation under disturbance.
"""

from dataclasses import dataclass
from itertools import product
from math import exp
from typing import Dict, List, Tuple

# ==============================================================================
# 1. PROBLEM DATA
# ==============================================================================

# --- Service Units & Machines ---
# Each unit u has m_u machines; each machine has an importance coefficient alpha
# Format: {unit: {machine: (type, alpha)}}
UNITS: Dict[str, Dict[str, Tuple[str, float]]] = {
    'U1': {
        'MA': ('T1', 1.0),   # T1-capable, high importance
        'MB': ('T2', 0.8),   # T2-capable, medium importance
    },
    'U2': {
        'MC': ('T1', 1.2),   # T1-capable, very high importance
        'MD': ('T2', 0.9),   # T2-capable, medium importance
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

# 3 Jobs, 7 operations total
JOBS: Dict[int, List[Operation]] = {
    1: [
        Operation(1, 0, 'T1', {('U1', 'MA'): 10.0, ('U2', 'MC'): 12.0}),
        Operation(1, 1, 'T2', {('U1', 'MB'):  8.0, ('U2', 'MD'):  9.0}),
    ],
    2: [
        Operation(2, 0, 'T1', {('U1', 'MA'): 15.0, ('U2', 'MC'): 12.0}),
        Operation(2, 1, 'T2', {('U1', 'MB'): 12.0, ('U2', 'MD'): 10.0}),
    ],
    3: [
        Operation(3, 0, 'T1', {('U1', 'MA'):  9.0, ('U2', 'MC'): 10.0}),
        Operation(3, 1, 'T2', {('U1', 'MB'): 14.0, ('U2', 'MD'): 13.0}),
        Operation(3, 2, 'T1', {('U1', 'MA'): 11.0, ('U2', 'MC'): 13.0}),
    ],
}

ALL_OPS = [(jid, op) for jid in JOBS for op in JOBS[jid]]


# ==============================================================================
# 2. SHOP LEVEL: weighted aggregate processing time & unit assignment
# ==============================================================================

def weighted_agg_time(op: Operation, unit: str) -> float:
    """Eq (4): weighted aggregate processing time of *op* on *unit*."""
    machines_in_unit = [m for m in UNITS[unit]]
    compatible = [(u, m) for (u, m) in [(unit, m) for m in machines_in_unit]
                  if M_TYPE_OF[(u, m)] == op.mtype]
    if not compatible:
        return float('inf')  # cannot process on this unit
    num = sum(ALPHA_OF[cm] * op.p_bar[cm] for cm in compatible)
    den = sum(ALPHA_OF[cm] for cm in compatible)
    return num / den


def shop_level_assign() -> Dict[Tuple[int, int], str]:
    """
    Solve the Shop Level problem: min max_u (sum X_{i,j,u} * P_tilde_{i,j,u} / m_u)
    Eq (5)-(6).  Brute-force enumeration (2^7 = 128 combinations) for transparency.
    """
    best_max_load = float('inf')
    best_assign = None

    # Precompute weighted aggregate times
    p_tilde: Dict[Tuple[int, int, str], float] = {}
    for jid, op in ALL_OPS:
        for unit in UNITS:
            p_tilde[(jid, op.op_idx, unit)] = weighted_agg_time(op, unit)

    # Each operation: list of feasible units
    feasible_units = []
    for jid, op in ALL_OPS:
        feas = [u for u in UNITS if p_tilde[(jid, op.op_idx, u)] < float('inf')]
        if not feas:
            raise ValueError(f"J{jid}-Op{op.op_idx+1} has no feasible unit!")
        feasible_units.append(feas)

    for choices in product(*feasible_units):
        load = {u: 0.0 for u in UNITS}
        for (jid, op), unit in zip(ALL_OPS, choices):
            load[unit] += p_tilde[(jid, op.op_idx, unit)]
        # per-machine load
        max_load = max(load[u] / M_U[u] for u in UNITS)
        if max_load < best_max_load - 1e-9:
            best_max_load = max_load
            best_assign = {(jid, op.op_idx): u
                          for (jid, op), u in zip(ALL_OPS, choices)}

    return best_assign


# ==============================================================================
# 3. UNIT LEVEL: machine selection & sequencing within each unit
# ==============================================================================

def unit_level_schedule(
    assignment: Dict[Tuple[int, int], str],
    machine_status: Dict[Tuple[str, str], bool] = None,
) -> Tuple[Dict[Tuple[int, int], Tuple[str, str, float, float]], float]:
    """
    For each unit, assign ops to specific machines and sequence them.
    Returns (schedule, Cmax).

    schedule: {(job_id, op_idx): (unit, machine, start, end)}
    machine_status: {(unit, machine): is_operational} -- default all True
    """
    if machine_status is None:
        machine_status = {cm: True for cm in ALL_MACHINES}

    schedule: Dict[Tuple[int, int], Tuple[str, str, float, float]] = {}
    # Track when each machine becomes free
    machine_free: Dict[Tuple[str, str], float] = {cm: 0.0 for cm in ALL_MACHINES}
    # Track job-wise precedence: when the previous operation of a job finishes
    job_ready: Dict[int, float] = {jid: 0.0 for jid in JOBS}

    # Group operations by unit
    unit_ops: Dict[str, List[Tuple[int, Operation]]] = {u: [] for u in UNITS}
    for jid, op in ALL_OPS:
        u = assignment[(jid, op.op_idx)]
        unit_ops[u].append((jid, op))

    # Within each unit, sort by job precedence (a simple list scheduling)
    for unit in UNITS:
        ops = unit_ops[unit]
        # Sort by job_id, then op_idx (preserving precedence order)
        ops.sort(key=lambda x: (x[0], x[1].op_idx))

        for jid, op in ops:
            # Find compatible, operational machines in this unit
            candidates = [(u, m) for (u, m) in ALL_MACHINES
                         if u == unit
                         and M_TYPE_OF[(u, m)] == op.mtype
                         and machine_status.get((u, m), True)]

            if not candidates:
                # This operation cannot be processed -- mark as unassigned
                schedule[(jid, op.op_idx)] = (unit, 'FAIL', -1.0, -1.0)
                continue

            # Earliest start = max(prev op end, machine free time)
            ready = job_ready[jid]
            best_machine = None
            best_start = float('inf')
            for cm in candidates:
                st = max(ready, machine_free[cm])
                if st < best_start:
                    best_start = st
                    best_machine = cm

            proc_time = op.p_bar[best_machine]
            end_time = best_start + proc_time
            schedule[(jid, op.op_idx)] = (unit, best_machine[1],
                                          best_start, end_time)
            machine_free[best_machine] = end_time
            job_ready[jid] = end_time

    # Compute Cmax
    valid_ends = [s[3] for s in schedule.values() if s[3] > 0]
    cmax = max(valid_ends) if valid_ends else float('inf')

    return schedule, cmax


# ==============================================================================
# 4. RESILIENCE METRIC
# ==============================================================================

def resilience(cmax_nominal: float, cmax_disturbed: float,
               failed_machines: List[Tuple[str, str]]) -> Tuple[float, float, float]:
    """
    Compute R(xi) = exp(-(phi + eta))   -- Eq (1)-(3)

    phi = resource damage ratio (weighted by alpha)
    eta = completion delay ratio
    """
    phi = sum(ALPHA_OF[cm] for cm in failed_machines) / SUM_ALPHA
    eta = (cmax_disturbed - cmax_nominal) / cmax_nominal if cmax_nominal > 0 else 0.0
    return exp(-(phi + eta)), phi, eta


# ==============================================================================
# 5. PRETTY PRINTING
# ==============================================================================

def print_separator(title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def main():
    print("=" * 72)
    print("  THREE-LAYER HIERARCHICAL SCHEDULING -- NUMERICAL EXAMPLE")
    print("  Flexible Job Shop with Resilience under Machine Failure")
    print("=" * 72)

    # -- Configuration summary -------------------------------------------------
    print_separator("CONFIGURATION")
    print(f"  Service Units: {len(UNITS)}  |  Total Machines: {len(ALL_MACHINES)}")
    for u in UNITS:
        machines = [f"{m}({info[0]}, alpha={info[1]})" for m, info in UNITS[u].items()]
        print(f"    {u}: {', '.join(machines)}  (m_{u.lower()} = {M_U[u]})")
    print(f"  Sum(alpha)  = {SUM_ALPHA:.1f}")
    print(f"\n  Jobs: {len(JOBS)}  |  Operations: {len(ALL_OPS)}")
    for jid, ops in JOBS.items():
        op_desc = ", ".join(f"O{jid}{i+1}({op.mtype})" for i, op in enumerate(ops))
        print(f"    J{jid}: {op_desc}")

    # -- Step 1: Shop Level ----------------------------------------------------
    print_separator("STEP 1 -- SHOP LEVEL: Weighted Aggregate Times & Unit Assignment")

    print("\n  Weighted aggregate processing times  P_tilde_{i,j,u}  (Eq 4):")
    header = f"  {'Op':>12s}"
    for u in UNITS:
        header += f"  {'P_tilde(' + u + ')':>13s}"
    header += f"  {'Assigned':>10s}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    assignment = shop_level_assign()

    # Compute loads for display
    load = {u: 0.0 for u in UNITS}
    for jid, op in ALL_OPS:
        u = assignment[(jid, op.op_idx)]
        pt = weighted_agg_time(op, u)
        load[u] += pt
        vals = ""
        for unit in UNITS:
            v = weighted_agg_time(op, unit)
            if v < float('inf'):
                vals += f"  {v:>13.1f}"
            else:
                vals += f"  {'N/A':>13s}"
        print(f"  {'J' + str(jid) + '-Op' + str(op.op_idx+1):>12s}{vals}  -> {u:>8s}")

    print(f"  {'-' * len(header)}")
    load_str = "  " + " " * 12
    for u in UNITS:
        load_str += f"  sum={load[u]:>11.1f}"
    print(load_str)
    load_per_m = "  " + " " * 12
    for u in UNITS:
        load_per_m += f"  per_mc={load[u]/M_U[u]:>7.1f}"
    print(load_per_m)

    max_load_per_machine = max(load[u] / M_U[u] for u in UNITS)
    print(f"\n  Objective f1 = min max load/machine = {max_load_per_machine:.1f}")

    # -- Step 2: Unit Level (Nominal) ------------------------------------------
    print_separator("STEP 2 -- UNIT LEVEL: Machine Selection & Sequencing (Nominal xi=0)")

    nominal_schedule, cmax_nominal = unit_level_schedule(assignment)
    print(f"\n  Nominal makespan  C_max = {cmax_nominal:.1f}")
    print(f"\n  {'Job':>6s}  {'Op':>4s}  {'Unit':>5s}  {'Mach':>5s}  "
          f"{'Start':>8s}  {'End':>8s}  {'Dur':>6s}")
    print(f"  {'-' * 56}")
    # Sort by start time
    sorted_entries = sorted(
        [(jid, op.op_idx) for jid, op in ALL_OPS],
        key=lambda k: nominal_schedule[k][2])
    for key in sorted_entries:
        u, m, st, en = nominal_schedule[key]
        dur = en - st
        print(f"  {'J' + str(key[0]):>6s}  {key[1]+1:>4d}  {u:>5s}  {m:>5s}  "
              f"{st:>8.1f}  {en:>8.1f}  {dur:>6.1f}")

    # Per-job completion
    job_completion = {}
    for jid in JOBS:
        ends = [nominal_schedule[(jid, op.op_idx)][3]
                for op in JOBS[jid]]
        job_completion[jid] = max(ends)
    print(f"\n  Job completions: " +
          "  ".join(f"J{jid}: C={job_completion[jid]:.1f}"
                    for jid in sorted(job_completion)))

    # -- Step 3: Disturbance ---------------------------------------------------
    print_separator("STEP 3 -- DISTURBANCE SCENARIO: Machine MA in U1 fails")

    # Machine MA in U1 fails at time 0
    failed = [('U1', 'MA')]
    machine_status = {cm: True for cm in ALL_MACHINES}
    for cm in failed:
        machine_status[cm] = False

    # Affected operations: those assigned to U1-MA in nominal schedule
    print(f"\n  Failed machine: U1-MA (alpha = {ALPHA_OF[('U1', 'MA')]})")
    print(f"  Affected operations (originally on U1-MA):")
    affected = []
    for jid, op in ALL_OPS:
        u, m, st, en = nominal_schedule[(jid, op.op_idx)]
        if u == 'U1' and m == 'MA':
            print(f"    J{jid}-Op{op.op_idx+1}: originally [{st:.1f}, {en:.1f})")
            affected.append((jid, op.op_idx))

    # Re-assign affected ops to U2-MC (the only other T1 machine)
    new_assignment = dict(assignment)
    for key in affected:
        new_assignment[key] = 'U2'

    print(f"\n  Re-assignment: {len(affected)} operations migrated U1 -> U2")
    print(f"  Unit load after migration:")
    new_load = {u: 0.0 for u in UNITS}
    for jid, op in ALL_OPS:
        u = new_assignment[(jid, op.op_idx)]
        pt = weighted_agg_time(op, u)
        new_load[u] += pt
    for u in UNITS:
        old_l = load[u]
        new_l = new_load[u]
        print(f"    {u}: sum(P_tilde) = {old_l:.1f} -> {new_l:.1f}  "
              f"(per-machine: {old_l/M_U[u]:.1f} -> {new_l/M_U[u]:.1f})")

    # Unit-level reschedule
    disturbed_schedule, cmax_disturbed = unit_level_schedule(new_assignment,
                                                              machine_status)
    print(f"\n  Disturbed makespan  C'_max = {cmax_disturbed:.1f}")
    print(f"\n  {'Job':>6s}  {'Op':>4s}  {'Unit':>5s}  {'Mach':>5s}  "
          f"{'Start':>8s}  {'End':>8s}  {'Dur':>6s}")
    print(f"  {'-' * 56}")
    sorted_dist = sorted(
        [(jid, op.op_idx) for jid, op in ALL_OPS],
        key=lambda k: disturbed_schedule[k][2])
    for key in sorted_dist:
        u, m, st, en = disturbed_schedule[key]
        dur = en - st if st >= 0 else float('nan')
        marker = " <-- migrated" if key in affected else ""
        if st < 0:
            print(f"  {'J' + str(key[0]):>6s}  {key[1]+1:>4d}  {u:>5s}  "
                  f"{'FAIL':>5s}  {'--':>8s}  {'--':>8s}  {'--':>6s}{marker}")
        else:
            print(f"  {'J' + str(key[0]):>6s}  {key[1]+1:>4d}  {u:>5s}  {m:>5s}  "
                  f"{st:>8.1f}  {en:>8.1f}  {dur:>6.1f}{marker}")

    # -- Step 4: Resilience Computation ----------------------------------------
    print_separator("STEP 4 -- RESILIENCE METRIC  R(xi)")

    R, phi, eta = resilience(cmax_nominal, cmax_disturbed, failed)
    print(f"\n  phi  = Sum(alpha_failed) / Sum(alpha_all)")
    print(f"       = {ALPHA_OF[('U1', 'MA')]:.1f} / {SUM_ALPHA:.1f}")
    print(f"       = {phi:.4f}")
    print(f"\n  eta  = (C'_max - C_max) / C_max")
    print(f"       = ({cmax_disturbed:.1f} - {cmax_nominal:.1f}) / {cmax_nominal:.1f}")
    print(f"       = {eta:.4f}")
    print(f"\n  R(xi) = exp(-(phi + eta))")
    print(f"         = exp(-({phi:.4f} + {eta:.4f}))")
    print(f"         = exp(-{phi + eta:.4f})")
    print(f"         = {R:.4f}")

    # Interpretation
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

    # -- Summary table ---------------------------------------------------------
    print_separator("SUMMARY")
    print(f"\n  {'Metric':<30s}  {'Nominal (xi=0)':>15s}  {'Disturbed (xi)':>15s}")
    print(f"  {'-' * 62}")
    print(f"  {'Makespan C_max':<30s}  {cmax_nominal:>15.1f}  {cmax_disturbed:>15.1f}")
    print(f"  {'Resource damage phi':<30s}  {'--':>15s}  {phi:>15.4f}")
    print(f"  {'Delay ratio eta':<30s}  {'--':>15s}  {eta:>15.4f}")
    print(f"  {'Resilience R(xi)':<30s}  {'--':>15s}  {R:>15.4f}")
    print(f"  {'Unit load U1 (per m/c)':<30s}  {load['U1']/M_U['U1']:>15.1f}  "
          f"{new_load['U1']/M_U['U1']:>15.1f}")
    print(f"  {'Unit load U2 (per m/c)':<30s}  {load['U2']/M_U['U2']:>15.1f}  "
          f"{new_load['U2']/M_U['U2']:>15.1f}")
    print()


if __name__ == '__main__':
    main()