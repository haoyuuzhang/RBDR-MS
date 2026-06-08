"""GA experiments for the Kacem 8×8 FJSP benchmark.

Two families of experiments:

1. **Global GA** (single-level)
   Solves the full FJSP at each decision point using only information
   available at that time (no clairvoyance).  Analogous to the pure
   rule-based baselines (SPT/FIFO/WINQ) but with a GA optimizer.

2. **Bi-level GA** (SPT-MILP → SPT-GA, etc.)
   Replaces the Gurobi MILP lower-level solver with a GA.  The upper-level
   dispatch rule (SPT/FIFO/WINQ) is unchanged; each service unit is solved
   by its own GA instance restricted to that unit's machines.

Output
------
  output/ga_global_results.json      — global GA results
  output/ga_bi_level_results.json    — bi-level GA results
  output/fig_comprehensive_scatter.png  — comprehensive scatter plot (Cmax vs Time, all methods)
  output/fig_ga_global_gantt.png     — Gantt for best global GA schedule
  output/fig_ga_bi_level_gantt_*.png — Gantt per bi-level GA rule
  Console summary with C_max and computation times.

Usage
-----
  python baseline-one-setting/run_ga_experiments.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Tuple

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
from scheduler import schedule_makespan_milp, _HAS_GUROBI
from bi_level_scheduler import (
    SERVICE_UNITS, UNIT_NAMES, ALL_MACHINES,
    assign_ops_to_units,
    _collect_fixed_and_ready,
    _restrict_op_to_unit,
)
from ga_scheduler import (
    FJSPGaSolver, GAUnitSolver, schedule_makespan_ga,
)

OUTPUT_DIR = os.path.join(_HERE, "output")
DATA_PATH = os.path.join(_HERE, "kacem_data.json")
BASELINE_CACHE = os.path.join(OUTPUT_DIR, "baseline_results.json")
RULE_CACHE = os.path.join(OUTPUT_DIR, "pure_rule_results.json")
BI_LEVEL_CACHE = os.path.join(OUTPUT_DIR, "bi_level_results.json")
GA_GLOBAL_CACHE = os.path.join(OUTPUT_DIR, "ga_global_results.json")
GA_BI_LEVEL_CACHE = os.path.join(OUTPUT_DIR, "ga_bi_level_results.json")

SNAPSHOT_TIMES = [0.0, 2.0, 6.0]

# GA hyper-parameters
GA_GLOBAL_KWARGS = dict(pop_size=400, n_generations=500, time_limit=120.0,
                         local_search=True, n_restarts=2)
GA_UNIT_KWARGS = dict(pop_size=200, n_generations=300, time_limit=60.0,
                       local_search=True, n_restarts=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _build_job(obj: dict) -> Job:
    ops = [Operation(job_id=obj["job_id"], op_idx=op["op_idx"], times=op["times"])
           for op in obj["operations"]]
    return Job(job_id=obj["job_id"],
               arrival_time=obj.get("arrival_time", 0.0),
               operations=ops)


def load_data() -> tuple:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    initial = [_build_job(j) for j in data["initial_jobs"]]
    dynamic = [_build_job(j) for j in data.get("dynamic_jobs", [])]
    disruptions = data.get("disruptions", [])
    return initial, dynamic, disruptions


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  Global GA experiments  (single-level)
# ═══════════════════════════════════════════════════════════════════════════════

def run_global_ga() -> Optional[dict]:
    """Run global GA at each decision point (t=0, t=2, t=6).

    Each decision point uses only the information available at that time.
    This is the GA analogue of the single-level rule-based baselines.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    initial_jobs, dynamic_jobs, disruptions = load_data()
    all_jobs = initial_jobs + dynamic_jobs
    disruption = disruptions[0] if disruptions else {}
    broken_machine = disruption.get("machine", "M3")
    breakdown_time = disruption.get("time", 6.0)

    print("=" * 72)
    print("  Kacem 8x8 FJSP — Global GA (Single-Level)")
    print("=" * 72)
    print(f"  Population: {GA_GLOBAL_KWARGS['pop_size']},  "
          f"Generations: {GA_GLOBAL_KWARGS['n_generations']}")
    print(f"  Jobs: {len(all_jobs)}  ({len(initial_jobs)} initial + "
          f"{len(dynamic_jobs)} dynamic)")
    print(f"  Machines: {len(Y_MACHINES)}  ({', '.join(Y_MACHINES)})")
    print("-" * 72)

    results: Dict[str, dict] = {}

    for t in SNAPSHOT_TIMES:
        print(f"\n[Global GA  t={t:.0f}]")

        # Filter to known information
        known_jobs = [j for j in all_jobs if j.arrival_time <= t]
        known_disruptions = [d for d in disruptions if d["time"] <= t]
        machine_deadlines: Optional[Dict[str, float]] = None
        if known_disruptions:
            machine_deadlines = {d["machine"]: d["time"] for d in known_disruptions}

        # For t>0, build fixed entries from  the previous stage's schedule
        fixed_entries: List[ScheduleEntry] = []
        if t > 0:
            prev_key = str(SNAPSHOT_TIMES[SNAPSHOT_TIMES.index(t) - 1])
            if prev_key in results:
                prev_sched = [ScheduleEntry(**e) for e in results[prev_key]["entries"]]
                # Fix ops that start before current time
                for e in prev_sched:
                    if e.start_time < t:
                        fixed_entries.append(ScheduleEntry(
                            job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                            start_time=e.start_time, end_time=e.end_time,
                            fixed=True,
                        ))

        t0 = time.perf_counter()
        solver = FJSPGaSolver(
            pop_size=GA_GLOBAL_KWARGS['pop_size'],
            n_generations=GA_GLOBAL_KWARGS['n_generations'],
            time_limit=GA_GLOBAL_KWARGS['time_limit'],
            local_search=GA_GLOBAL_KWARGS.get('local_search', True),
            n_restarts=GA_GLOBAL_KWARGS.get('n_restarts', 1),
            seed=42 + int(t),
            verbose=True,
        )
        schedule, stats = solver.solve(
            known_jobs, fixed_entries, current_time=t,
            machine_deadlines=machine_deadlines,
        )
        dt = time.perf_counter() - t0

        if schedule is None:
            print(f"  !! Global GA failed at t={t:.0f}")
            return None

        cmax = max((e.end_time for e in schedule), default=0.0)
        n_ops = len([e for e in schedule if not e.fixed
                     or e.start_time >= t])
        print(f"  C_max = {cmax:.3f}  |  ops scheduled = {n_ops}  |  "
              f"wall time = {dt:.1f}s  |  evals = {stats.get('evaluations', 0)}")

        results[str(t)] = {
            "cmax": cmax,
            "entries": [asdict(e) for e in schedule],
            "compute_time": dt,
            "evaluations": stats.get('evaluations', 0),
            "generations": stats.get('generations', 0),
        }

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  Global GA Summary")
    print("-" * 72)
    header = "  {:6s}".format("t")
    header += "    {:>10s}  {:>10s}  {:>12s}  {:>10s}".format(
        "C_max", "Time(s)", "Evals", "Gens")
    print(header)
    for t in SNAPSHOT_TIMES:
        r = results[str(t)]
        print(f"  {t:<6.0f}    {r['cmax']:10.3f}  {r['compute_time']:10.1f}  "
              f"{r['evaluations']:>12d}  {r['generations']:>10d}")

    # Compare with Gurobi baselines
    if os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE, "r", encoding="utf-8") as f:
            bl = json.load(f)
        print("\n  Comparison with Gurobi clairvoyant baselines:")
        for key, label in [("A", "J1-J8 optimal"),
                           ("B", "J1-J10 clairvoyant"),
                           ("C", "J1-J10+M3 clairvoyant")]:
            bl_cmax = bl["baselines"][key]["cmax"]
            t_map = {"A": 0.0, "B": 2.0, "C": 6.0}
            ga_cmax = results[str(t_map[key])]["cmax"]
            gap = (ga_cmax - bl_cmax) / bl_cmax * 100
            sign = "+" if gap > 0 else ""
            print(f"  Baseline {key} ({label:25s})  C_max = {bl_cmax:.1f}  "
                  f"GA = {ga_cmax:.3f}  (gap: {sign}{gap:.2f}%)")

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "global_ga": results,
        "metadata": {
            "description": "Global GA single-level FJSP scheduling",
            "ga_params": GA_GLOBAL_KWARGS,
            "disruptions": disruptions,
            "num_jobs": len(all_jobs),
            "num_machines": len(Y_MACHINES),
            "snapshot_times": SNAPSHOT_TIMES,
        },
    }
    with open(GA_GLOBAL_CACHE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Cached → {GA_GLOBAL_CACHE}")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  Bi-level GA simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_bi_level_ga(
    rule_name: str,
    jobs: List[Job],
    disruptions: List[dict],
    initial_schedule: Optional[List[ScheduleEntry]] = None,
    ga_kwargs: Optional[dict] = None,
) -> Optional[dict]:
    """Run bi-level scheduling with GA unit solvers.

    Mirrors :func:`bi_level_scheduler.simulate_bi_level` but uses per-unit
    GA solvers instead of a shared Gurobi MILP solver.

    Parameters
    ----------
    rule_name : str
        'SPT', 'FIFO', or 'WINQ'.
    jobs : list of Job
        All jobs (initial + dynamic).
    disruptions : list of dict
        Disruption events.
    initial_schedule : list of ScheduleEntry or None
        Cached Baseline A for t=0.  If None, a global GA is used at t=0.
    ga_kwargs : dict or None
        Passed to :class:`GAUnitSolver` for unit-level solves.

    Returns
    -------
    dict  or  None
    """
    if ga_kwargs is None:
        ga_kwargs = GA_UNIT_KWARGS

    t_start_total = time.perf_counter()

    # Track per-stage compute times
    dt_stage_0 = 0.0   # t=0 global solve
    dt_stage_1 = 0.0   # t=2 bi-level
    dt_stage_2 = 0.0   # t=6 bi-level

    job_map: Dict[int, Job] = {j.job_id: j for j in jobs}
    initial_jobs = [j for j in jobs if j.arrival_time <= 0.0]

    fixed_keys: Set[Tuple[int, int]] = set()
    unit_assignments: Dict[Tuple[int, int], str] = {}
    all_entries: List[ScheduleEntry] = []
    all_partial: List[ScheduleEntry] = []
    snapshot_schedules: Dict[str, list] = {}

    # ── Stage 0: t=0 — Global optimal schedule ─────────────────────────────
    t_stage_0_start = time.perf_counter()
    if initial_schedule is not None:
        print(f"\n  [{rule_name}] Stage 0 — t=0  Using cached Baseline A")
        sched_0 = list(initial_schedule)
    else:
        print(f"\n  [{rule_name}] Stage 0 — t=0  Global GA  (J1-J{len(initial_jobs)}, "
              f"all 8 machines) ...")
        t0 = time.perf_counter()
        solver = FJSPGaSolver(
            pop_size=GA_GLOBAL_KWARGS['pop_size'],
            n_generations=GA_GLOBAL_KWARGS['n_generations'],
            time_limit=GA_GLOBAL_KWARGS['time_limit'],
            local_search=GA_GLOBAL_KWARGS.get('local_search', True),
            n_restarts=GA_GLOBAL_KWARGS.get('n_restarts', 1),
            seed=42, verbose=True,
        )
        sched_0, _ = solver.solve(initial_jobs, [], current_time=0.0)
        if sched_0 is None:
            print("    !! Global GA failed at t=0")
            return None

    cmax_0 = max(e.end_time for e in sched_0)
    dt_stage_0 = time.perf_counter() - t_stage_0_start
    print(f"    OK  C_max = {cmax_0:.3f}")
    snapshot_schedules['0.0'] = [e for e in sched_0]

    # ── Stage 1: t=2 — J9/J10 arrive, bi-level re-scheduling ───────────────
    t_stage_1_start = time.perf_counter()
    t_event = 2.0
    print(f"\n  [{rule_name}] Stage 1 — t={t_event:.0f}  Bi-level GA  "
          f"(J9, J10 arrive) ...")

    fixed_2, completed_2, in_progress_2 = _collect_fixed_and_ready(
        sched_0, t_event, fixed_keys)
    fixed_keys |= {(e.job_id, e.op_idx) for e in fixed_2}

    new_jobs = [j for j in jobs if j.arrival_time == t_event]
    print(f"    Fixed: {len(completed_2)} completed, {len(in_progress_2)} in-progress  "
          f"New jobs: {sorted([j.job_id for j in new_jobs])}")

    # Ready ops
    ready_2: List[Tuple[int, int]] = []
    for j in new_jobs:
        ready_2.append((j.job_id, 0))
    for (jid, oidx) in completed_2:
        job = job_map[jid]
        next_oidx = oidx + 1
        if next_oidx < len(job.operations):
            nxt_key = (jid, next_oidx)
            if nxt_key not in fixed_keys:
                ready_2.append(nxt_key)

    unstarted_keys: Set[Tuple[int, int]] = set()
    for e in sched_0:
        key = (e.job_id, e.op_idx)
        if key not in fixed_keys and e.start_time >= t_event:
            unstarted_keys.add(key)

    # Upper-level assignment
    unit_workload = {'U1': 0.0, 'U2': 0.0}
    for e in fixed_2:
        for u_name, machines in SERVICE_UNITS.items():
            if e.machine in machines:
                unit_workload[u_name] += e.duration

    ready_to_assign = ready_2 + [k for k in unstarted_keys
                                  if k not in unit_assignments]
    seen = set()
    ready_to_assign_unique = []
    for k in ready_to_assign:
        if k not in seen:
            seen.add(k)
            ready_to_assign_unique.append(k)

    unit_assignments = assign_ops_to_units(
        ready_to_assign_unique, rule_name, job_map,
        unit_workload=unit_workload,
        existing_assignments=unit_assignments,
    )

    # Per-unit GA
    sched_2_all, all_partial = _run_unit_ga_stage(
        fixed_2, fixed_keys, unit_assignments, job_map,
        current_time=t_event,
        all_partial=all_partial,
        ga_kwargs=ga_kwargs,
        label_prefix=f"[{rule_name}]",
    )
    if sched_2_all is None:
        return None

    cmax_2 = max((e.end_time for e in sched_2_all), default=0.0)
    dt_stage_1 = time.perf_counter() - t_stage_1_start
    print(f"    Combined C_max = {cmax_2:.3f}")
    snapshot_schedules['2.0'] = sched_2_all

    # ── Stage 2: t=6 — M3 breakdown, bi-level re-scheduling ────────────────
    t_stage_2_start = time.perf_counter()
    t_event = 6.0
    disruption = disruptions[0] if disruptions else {}
    broken_machine = disruption.get('machine', 'M3')

    print(f"\n  [{rule_name}] Stage 2 — t={t_event:.0f}  Bi-level GA  "
          f"({broken_machine} breakdown) ...")

    fixed_6, completed_6, in_progress_6 = _collect_fixed_and_ready(
        sched_2_all, t_event, fixed_keys)
    fixed_keys |= {(e.job_id, e.op_idx) for e in fixed_6}
    for pe in all_partial:
        fixed_keys.add((pe.job_id, pe.op_idx))

    # Handle M3 interruption
    interrupted_keys, interrupted_entries, all_partial = _handle_interruption(
        sched_2_all, in_progress_6, broken_machine, t_event,
        fixed_6, all_partial)
    fixed_keys |= {(e.job_id, e.op_idx) for e in interrupted_entries}

    in_progress_6 -= interrupted_keys

    # Ready ops at t=6
    ready_6: List[Tuple[int, int]] = []
    for (jid, oidx) in completed_6:
        job = job_map[jid]
        next_oidx = oidx + 1
        if next_oidx < len(job.operations):
            nxt_key = (jid, next_oidx)
            if nxt_key not in fixed_keys:
                ready_6.append(nxt_key)
    for key in interrupted_keys:
        if key not in fixed_keys:
            ready_6.append(key)

    unstarted_6: Set[Tuple[int, int]] = set()
    for e in sched_2_all:
        key = (e.job_id, e.op_idx)
        if key not in fixed_keys and e.start_time >= t_event:
            unstarted_6.add(key)
    unstarted_6 -= interrupted_keys

    # Upper-level reassignment
    unit_workload = {'U1': 0.0, 'U2': 0.0}
    for e in fixed_6:
        for u_name, machines in SERVICE_UNITS.items():
            if e.machine in machines:
                unit_workload[u_name] += e.duration
    for key in interrupted_keys:
        unit_assignments.pop(key, None)

    ready_to_assign_6 = ready_6 + [k for k in unstarted_6
                                    if k not in unit_assignments]
    seen_6 = set()
    ready_to_assign_6_unique = []
    for k in ready_to_assign_6:
        if k not in seen_6:
            seen_6.add(k)
            ready_to_assign_6_unique.append(k)

    unit_assignments = assign_ops_to_units(
        ready_to_assign_6_unique, rule_name, job_map,
        unit_workload=unit_workload,
        existing_assignments=unit_assignments,
    )

    # Per-unit GA with M3 deadline
    machine_deadlines = {broken_machine: t_event}
    sched_6_all, all_partial = _run_unit_ga_stage(
        fixed_6, fixed_keys, unit_assignments, job_map,
        current_time=t_event,
        machine_deadlines=machine_deadlines,
        all_partial=all_partial,
        ga_kwargs=ga_kwargs,
        label_prefix=f"[{rule_name}]",
    )
    if sched_6_all is None:
        return None

    cmax_6 = max((e.end_time for e in sched_6_all), default=0.0)
    dt_stage_2 = time.perf_counter() - t_stage_2_start
    print(f"    Combined C_max = {cmax_6:.3f}")
    snapshot_schedules['6.0'] = sched_6_all

    # ── Assemble result ─────────────────────────────────────────────────────
    total_compute = time.perf_counter() - t_start_total

    serialised_entries = [asdict(e) for e in sched_6_all]
    serialised_partial = [asdict(e) for e in all_partial]
    serialised_assignments = {
        f"{jid},{oidx}": unit_name
        for (jid, oidx), unit_name in unit_assignments.items()
    }
    serialised_snapshots = {}
    for t_str, entries in snapshot_schedules.items():
        serialised_snapshots[t_str] = [asdict(e) for e in entries]

    print(f"\n  [{rule_name}]  Final C_max = {cmax_6:.3f}  "
          f"(total compute = {total_compute:.1f}s)")

    return {
        'cmax': cmax_6,
        'entries': serialised_entries,
        'partial_entries': serialised_partial,
        'compute_time': total_compute,
        'stage_times': {
            '0.0': dt_stage_0,
            '2.0': dt_stage_1,
            '6.0': dt_stage_2,
        },
        'unit_assignments': serialised_assignments,
        'snapshot_schedules': serialised_snapshots,
        'snapshot_cmax': {
            '0.0': cmax_0,
            '2.0': cmax_2,
            '6.0': cmax_6,
        },
    }


def _run_unit_ga_stage(
    fixed_entries: List[ScheduleEntry],
    fixed_keys: Set[Tuple[int, int]],
    unit_assignments: Dict[Tuple[int, int], str],
    job_map: Dict[int, Job],
    current_time: float,
    machine_deadlines: Optional[Dict[str, float]] = None,
    all_partial: Optional[List[ScheduleEntry]] = None,
    ga_kwargs: Optional[dict] = None,
    label_prefix: str = "",
) -> Tuple[Optional[List[ScheduleEntry]], List[ScheduleEntry]]:
    """Run per-unit GA for one decision stage.

    Returns (combined_schedule, partial_entries) or (None, []) on failure.
    """
    if ga_kwargs is None:
        ga_kwargs = GA_UNIT_KWARGS
    if all_partial is None:
        all_partial = []

    sched_all: List[ScheduleEntry] = list(fixed_entries)

    for unit_name in UNIT_NAMES:
        unit_machines = SERVICE_UNITS[unit_name]
        other_machines = [m for u in UNIT_NAMES if u != unit_name
                          for m in SERVICE_UNITS[u]]

        unit_op_keys = {k for k, u in unit_assignments.items() if u == unit_name}

        # Build unit-specific jobs (same construction as bi_level_scheduler)
        unit_jobs: List[Job] = []
        all_jids = {k[0] for k in unit_op_keys} | {e.job_id for e in fixed_entries}
        for jid in all_jids:
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

        # Check if there's actual work for this unit
        has_work = any(
            key in unit_op_keys and key not in fixed_keys
            for job in unit_jobs for op in job.operations
            for key in [(job.job_id, op.op_idx)]
        )
        if not has_work:
            u_cmax = max((e.end_time for e in sched_all
                         if e.machine in unit_machines), default=0.0)
            n_fixed = len([e for e in fixed_entries if e.machine in unit_machines])
            print(f"    {label_prefix} [{unit_name}]  No new work — C_max = {u_cmax:.3f}  "
                  f"({n_fixed} fixed ops)")
            continue

        unit_deadlines: Optional[Dict[str, float]] = None
        if machine_deadlines:
            unit_deadlines = {m: dl for m, dl in machine_deadlines.items()
                             if m in unit_machines}

        n_fixed = len([e for e in fixed_entries if e.machine in unit_machines])
        n_unit_ops = len([k for k in unit_op_keys if k not in fixed_keys])
        deadline_str = f"  [{', '.join(f'{m}→{dl:.0f}' for m, dl in (unit_deadlines or {}).items())}]" if unit_deadlines else ""
        print(f"    {label_prefix} [{unit_name}]  "
              f"{len(unit_jobs)} jobs, {n_fixed} fixed, "
              f"{n_unit_ops} ops to schedule{deadline_str}  → GA ...")

        solver = GAUnitSolver(
            unit_machines=unit_machines,
            pop_size=ga_kwargs.get('pop_size', 150),
            n_generations=ga_kwargs.get('n_generations', 300),
            time_limit=ga_kwargs.get('time_limit', 60.0),
            local_search=ga_kwargs.get('local_search', True),
            n_restarts=ga_kwargs.get('n_restarts', 1),
            seed=42 + int(current_time) + (1 if unit_name == 'U1' else 2),
        )
        unit_schedule = solver.solve(
            unit_jobs, list(fixed_entries), current_time=current_time,
            machine_deadlines=unit_deadlines,
        )

        if unit_schedule is None:
            print(f"    {label_prefix} !! [{unit_name}] GA failed")
            return None, all_partial

        for e in unit_schedule:
            if (e.job_id, e.op_idx) not in fixed_keys:
                sched_all.append(e)

        cmax_u = max((e.end_time for e in unit_schedule), default=0.0)
        n_new = len([e for e in unit_schedule
                     if (e.job_id, e.op_idx) not in fixed_keys])
        print(f"    {label_prefix} [{unit_name}]  C_max = {cmax_u:.3f}  "
              f"({n_new} new entries)")

    return sched_all, all_partial


def _handle_interruption(
    schedule: List[ScheduleEntry],
    in_progress_keys: Set[Tuple[int, int]],
    broken_machine: str,
    breakdown_time: float,
    fixed_entries: List[ScheduleEntry],
    all_partial: List[ScheduleEntry],
) -> Tuple[Set[Tuple[int, int]], List[ScheduleEntry], List[ScheduleEntry]]:
    """Handle M3 breakdown: find interrupted ops, create partial entries."""
    interrupted_keys: Set[Tuple[int, int]] = set()
    interrupted_entries: List[ScheduleEntry] = []

    for key in list(in_progress_keys):
        for e in schedule:
            if (e.job_id, e.op_idx) == key and e.machine == broken_machine:
                interrupted_keys.add(key)
                interrupted_entries.append(ScheduleEntry(
                    job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                    start_time=e.start_time, end_time=breakdown_time,
                    fixed=True,
                ))
                # Remove from fixed entries (will be rescheduled)
                fixed_entries[:] = [fe for fe in fixed_entries
                                    if (fe.job_id, fe.op_idx) != key]
                print(f"    Interrupted: J{e.job_id}-{e.op_idx + 1} on "
                      f"{broken_machine}  (back to ready pool)")
                break

    all_partial.extend(interrupted_entries)
    return interrupted_keys, interrupted_entries, all_partial


def run_bi_level_ga() -> Optional[dict]:
    """Run bi-level GA experiments for SPT, FIFO, WINQ."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    initial_jobs, dynamic_jobs, disruptions = load_data()
    all_jobs = initial_jobs + dynamic_jobs

    print("\n" + "=" * 72)
    print("  Kacem 8x8 FJSP — Bi-Level GA Scheduling")
    print("  (Global GA at t=0, Bi-level GA at t=2, t=6)")
    print("=" * 72)
    print(f"  Unit GA params: pop={GA_UNIT_KWARGS['pop_size']},  "
          f"gen={GA_UNIT_KWARGS['n_generations']}")
    print(f"  Rules: SPT / FIFO / WINQ")
    print(f"  Units: U1 = {SERVICE_UNITS['U1']}  |  U2 = {SERVICE_UNITS['U2']}")
    print("-" * 72)

    # Load cached Baseline A for t=0 (same starting point as MILP bi-level)
    initial_schedule: Optional[List[ScheduleEntry]] = None
    if os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE, "r", encoding="utf-8") as f:
            bl_cache = json.load(f)
        baseline_a_entries = bl_cache["baselines"]["A"]["entries"]
        initial_schedule = [ScheduleEntry(**e) for e in baseline_a_entries]
        print(f"\n  Using cached Baseline A for t=0  "
              f"(C_max = {bl_cache['baselines']['A']['cmax']:.3f})")
    else:
        print("\n  No cached baseline — will use global GA at t=0")

    rules = ["SPT", "FIFO", "WINQ"]
    results: Dict[str, dict] = {}

    for rule_name in rules:
        print(f"\n{'─' * 72}")
        print(f"  Running:  {rule_name}-GA  (bi-level)")
        print(f"{'─' * 72}")

        sim_result = simulate_bi_level_ga(
            rule_name, all_jobs, disruptions,
            initial_schedule=initial_schedule,
            ga_kwargs=GA_UNIT_KWARGS,
        )

        if sim_result is None:
            print(f"  !! {rule_name}-GA failed")
            return None

        results[rule_name] = sim_result

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Bi-Level GA Summary")
    print("=" * 72)
    header = f"  {'Rule':12s}"
    for t in SNAPSHOT_TIMES:
        header += f"    {'t=' + str(int(t)):>10s}"
    header += f"    {'Final':>10s}  {'Time':>8s}"
    print(header)
    print("  " + "-" * (12 + len(SNAPSHOT_TIMES) * 14 + 22))
    for rule_name in rules:
        r = results[rule_name]
        line = f"  {rule_name + '-GA':12s}"
        for t in SNAPSHOT_TIMES:
            line += f"    {r['snapshot_cmax'][str(t)]:10.3f}"
        line += f"    {r['cmax']:10.3f}  {r['compute_time']:7.1f}s"
        n_partial = len(r['partial_entries'])
        if n_partial:
            line += f"  [{n_partial} interrupted]"
        print(line)
    print("-" * 72)

    # ── Comparison ──────────────────────────────────────────────────────────
    if os.path.exists(BI_LEVEL_CACHE):
        with open(BI_LEVEL_CACHE, "r", encoding="utf-8") as f:
            bl_data = json.load(f)
        print("\n  Comparison: GA vs MILP lower-level (bi-level C_max):")
        for rule_name in rules:
            ga_cmax = results[rule_name]['cmax']
            milp_cmax = bl_data['rules'][rule_name]['cmax']
            delta = (ga_cmax - milp_cmax) / milp_cmax * 100
            sign = '+' if delta > 0 else ''
            print(f"  {rule_name:6s}  MILP = {milp_cmax:.3f}  "
                  f"GA = {ga_cmax:.3f}  ({sign}{delta:.1f}%)")

    if os.path.exists(RULE_CACHE):
        with open(RULE_CACHE, "r", encoding="utf-8") as f:
            rule_data = json.load(f)
        print("\n  Comparison: GA bi-level vs single-level rules (t=6 C_max):")
        for rule_name in rules:
            ga_cmax = results[rule_name]['cmax']
            sl_cmax = rule_data['rules'][rule_name]['6.0']['cmax']
            delta = (ga_cmax - sl_cmax) / sl_cmax * 100
            sign = '+' if delta > 0 else ''
            print(f"  {rule_name:6s}  Rule = {sl_cmax:.3f}  "
                  f"GA bi-level = {ga_cmax:.3f}  ({sign}{delta:.1f}%)")

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "rules": results,
        "metadata": {
            "description": "Bi-level GA scheduling (upper: SPT/FIFO/WINQ, lower: GA)",
            "ga_params": GA_UNIT_KWARGS,
            "disruptions": disruptions,
            "num_jobs": len(all_jobs),
            "num_machines": len(ALL_MACHINES),
            "service_units": SERVICE_UNITS,
            "snapshot_times": SNAPSHOT_TIMES,
        },
    }
    with open(GA_BI_LEVEL_CACHE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Cached → {GA_BI_LEVEL_CACHE}")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  Comprehensive scatter plot  (C_max vs Computation Time)
# ═══════════════════════════════════════════════════════════════════════════════

# Colour per decision time-point
_TIME_COLOR = {0.0: '#2196F3', 2.0: '#FF9800', 6.0: '#4CAF50'}

# Marker per rule / method variant
_RULE_MARKER: Dict[str, str] = {
    'SPT': 'o', 'FIFO': 's', 'WINQ': '^',   # dispatching rules
    'GA': 'D',                                 # global GA
    'Baseline': '*',                           # clairvoyant baseline
}


def _build_scatter_series(
    bl_raw: dict,
    rule_raw: dict,
    milp_bi_raw: dict,
    ga_global: Optional[dict],
    ga_bi_level: Optional[dict],
) -> List[dict]:
    """Return a flat list of scatter-point dicts.

    Each dict:
      x, y, label, time, marker, filled, rule_name
    """
    series: List[dict] = []

    def _time_ms(ct) -> float:
        if ct is None:
            return float('nan')
        # Floor at 0.001 ms so cache‑hit / near‑zero times remain visible
        # on the log‑scale y‑axis (single‑level rules are ~0.2–0.8 ms).
        return max(ct * 1000.0, 0.001)

    # ── Clairvoyant Baselines  (filled stars) ─────────────────────────────
    bl_time_map = {'A': 0.0, 'B': 2.0, 'C': 6.0}
    if bl_raw:
        for key in ['A', 'B', 'C']:
            bl = bl_raw["baselines"][key]
            t = bl_time_map[key]
            series.append({
                'x': bl["cmax"], 'y': _time_ms(bl.get("compute_time")),
                'label': 'Baseline', 'time': t,
                'marker': 'Baseline', 'filled': False,
                'rule_name': 'Baseline',
            })

    # ── Single-level rules  (hollow markers) ──────────────────────────────
    if rule_raw:
        for rn in ['SPT', 'FIFO', 'WINQ']:
            r = rule_raw["rules"][rn]
            for t_str in ['0.0', '2.0', '6.0']:
                t = float(t_str)
                sim = r[t_str]
                series.append({
                    'x': sim["cmax"], 'y': _time_ms(sim["compute_time"]),
                    'label': f'{rn} (single-level)', 'time': t,
                    'marker': rn, 'filled': False,
                    'rule_name': rn,
                })

    # ── Bi-level MILP  (filled markers, thick black border) ───────────────
    # Bi-level at t=0 reuses cached Baseline A → stage_time ≈ 0.
    # Use Baseline A's actual Gurobi solve time as fallback.
    bl_a_ct = None
    if bl_raw:
        bl_a_ct = bl_raw["baselines"]["A"].get("compute_time")
    if milp_bi_raw:
        for rn in ['SPT', 'FIFO', 'WINQ']:
            if rn not in milp_bi_raw.get("rules", {}):
                continue
            r = milp_bi_raw["rules"][rn]
            sc = r["snapshot_cmax"]
            st = r.get("stage_times", {})
            for t_str in ['0.0', '2.0', '6.0']:
                t = float(t_str)
                ct = st.get(t_str)
                if t == 0.0 and (ct is None or ct < 1e-6):
                    ct = bl_a_ct   # fallback to Baseline A solve time
                series.append({
                    'x': sc.get(t_str, r["cmax"]),
                    'y': _time_ms(ct),
                    'label': f'{rn}-MILP (bi-level)', 'time': t,
                    'marker': rn, 'filled': True,
                    'rule_name': rn,
                    'edge_lw': 2.2,
                })

    # ── Global GA  (hollow diamond — single-level) ────────────────────────
    if ga_global:
        gga = ga_global.get("global_ga", ga_global)
        for t_str in ['0.0', '2.0', '6.0']:
            if t_str not in gga:
                continue
            t = float(t_str)
            series.append({
                'x': gga[t_str]["cmax"], 'y': _time_ms(gga[t_str]["compute_time"]),
                'label': 'Global GA (single-level)', 'time': t,
                'marker': 'GA', 'filled': False,
                'rule_name': 'GA',
            })

    # ── Bi-level GA  (filled marker) ──────────────────────────────────────
    # Same fallback: t=0 reuses cached Baseline A.
    if ga_bi_level and "rules" in ga_bi_level:
        for rn in ['SPT', 'FIFO', 'WINQ']:
            if rn not in ga_bi_level["rules"]:
                continue
            r = ga_bi_level["rules"][rn]
            sc = r["snapshot_cmax"]
            st = r.get("stage_times", {})
            for t_str in ['0.0', '2.0', '6.0']:
                t = float(t_str)
                ct = st.get(t_str)
                if t == 0.0 and (ct is None or ct < 1e-6):
                    ct = bl_a_ct
                series.append({
                    'x': sc.get(t_str, r["cmax"]),
                    'y': _time_ms(ct),
                    'label': f'{rn}-GA (bi-level)', 'time': t,
                    'marker': rn, 'filled': True,
                    'rule_name': rn,
                    'edge_lw': 0.8,
                })

    return series


def plot_metrics_table_figure(
    ga_global: Optional[dict] = None,
    ga_bi_level: Optional[dict] = None,
) -> None:
    """Comprehensive scatter plot: Makespan (C_max) vs Computation Time.

    Includes **all** methods across all decision points (t=0, 2, 6).

    Visual encoding
    ---------------
    - **Colour**  →  decision time-point  (blue=0, orange=2, green=6)
    - **Marker**  →  dispatching rule  (●=SPT,  ■=FIFO,  ▲=WINQ,  ◆=GA,  ★=Baseline)
    - **Fill**    →  architecture level  (hollow=single-level,  filled=bi-level)

    Legend is placed in the upper-right corner.

    Saved to ``output/fig_comprehensive_scatter.png``.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load all cached data ──────────────────────────────────────────────
    bl_raw = {}
    if os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE, "r", encoding="utf-8") as f:
            bl_raw = json.load(f)

    rule_raw = {}
    if os.path.exists(RULE_CACHE):
        with open(RULE_CACHE, "r", encoding="utf-8") as f:
            rule_raw = json.load(f)

    milp_bi_raw = {}
    if os.path.exists(BI_LEVEL_CACHE):
        with open(BI_LEVEL_CACHE, "r", encoding="utf-8") as f:
            milp_bi_raw = json.load(f)

    if ga_global is None and os.path.exists(GA_GLOBAL_CACHE):
        with open(GA_GLOBAL_CACHE, "r", encoding="utf-8") as f:
            ga_global = json.load(f)
    if ga_bi_level is None and os.path.exists(GA_BI_LEVEL_CACHE):
        with open(GA_BI_LEVEL_CACHE, "r", encoding="utf-8") as f:
            ga_bi_level = json.load(f)

    series = _build_scatter_series(
        bl_raw, rule_raw, milp_bi_raw, ga_global, ga_bi_level)

    if not series:
        print("  No data available for scatter plot.")
        return

    # ═══════════════════════════════════════════════════════════════════════
    #  Build the figure
    # ═══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(16, 9))

    # ── Plot: group points by (label, filled) to draw connecting lines ────
    from collections import defaultdict
    groups: Dict[Tuple[str, bool], List[dict]] = defaultdict(list)
    for pt in series:
        groups[(pt['label'], pt['filled'])].append(pt)

    for _pts in groups.values():
        pts_sorted = sorted(_pts, key=lambda p: p['time'])

        # Faint connecting line between same-method points across time
        if len(pts_sorted) > 1:
            xs = [p['x'] for p in pts_sorted]
            ys = [p['y'] for p in pts_sorted]
            valid = [(x, y) for x, y in zip(xs, ys)
                     if not (x != x or y != y)]
            if len(valid) >= 2:
                vx, vy = zip(*valid)
                ax.plot(vx, vy, color='#999999', alpha=0.22, linewidth=1.0,
                        linestyle='-', zorder=2)

        # Draw each point
        for pt in pts_sorted:
            if pt['x'] != pt['x'] or pt['y'] != pt['y']:
                continue
            t = pt['time']
            time_color = _TIME_COLOR[t]
            marker_key = pt['marker']
            marker = _RULE_MARKER.get(marker_key, 'o')

            if pt['filled']:
                # Bi-level → filled marker  (MILP gets thick black border)
                elw = pt.get('edge_lw', 0.8)
                ax.scatter(
                    pt['x'], pt['y'],
                    c=time_color, marker=marker, s=130,
                    edgecolors='black', linewidths=elw,
                    zorder=4, alpha=0.88,
                )
            else:
                # Single-level → hollow marker
                ax.scatter(
                    pt['x'], pt['y'],
                    c='none', marker=marker, s=110,
                    edgecolors=time_color, linewidths=1.6,
                    zorder=4, alpha=0.88,
                )

    # ── Axis setup ────────────────────────────────────────────────────────
    ax.set_xlabel('Makespan  (C_max)  [hours]', fontsize=14)
    ax.set_ylabel('Computation Time  [ms]  (log scale)', fontsize=14)
    all_x = [p['x'] for p in series if p['x'] == p['x']]
    if all_x:
        ax.set_xlim(min(all_x) * 0.88, max(all_x) * 1.10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, linestyle='--')

    # ═══════════════════════════════════════════════════════════════════════
    #  Legend  (upper-right)
    # ═══════════════════════════════════════════════════════════════════════
    from matplotlib.lines import Line2D

    # -- Time (colour) --
    time_handles = [
        Line2D([], [], marker='o', color=_TIME_COLOR[0.0], markersize=10,
               markerfacecolor=_TIME_COLOR[0.0], linestyle='None',
               label='t = 0  (initial jobs)'),
        Line2D([], [], marker='o', color=_TIME_COLOR[2.0], markersize=10,
               markerfacecolor=_TIME_COLOR[2.0], linestyle='None',
               label='t = 2  (+J9, J10 arrive)'),
        Line2D([], [], marker='o', color=_TIME_COLOR[6.0], markersize=10,
               markerfacecolor=_TIME_COLOR[6.0], linestyle='None',
               label='t = 6  (+M3 breakdown)'),
    ]

    # -- Marker shape = Rule --
    marker_handles = [
        Line2D([], [], marker='o', color='#555555', markersize=9,
               markerfacecolor='#555555', linestyle='None', label='SPT'),
        Line2D([], [], marker='s', color='#555555', markersize=9,
               markerfacecolor='#555555', linestyle='None', label='FIFO'),
        Line2D([], [], marker='^', color='#555555', markersize=9,
               markerfacecolor='#555555', linestyle='None', label='WINQ'),
        Line2D([], [], marker='D', color='#555555', markersize=9,
               markerfacecolor='#555555', linestyle='None', label='Global GA'),
        Line2D([], [], marker='*', color='#555555', markersize=13,
               markerfacecolor='#555555', linestyle='None', label='Baseline'),
    ]

    # -- Fill = architecture level --
    fill_handles = [
        Line2D([], [], marker='o', color='#555555', markersize=10,
               markerfacecolor='none', markeredgewidth=1.6,
               linestyle='None', label='Hollow  =  single-level'),
        Line2D([], [], marker='o', color='#555555', markersize=10,
               markerfacecolor='#555555', markeredgewidth=0.8,
               linestyle='None', label='Filled  =  bi-level'),
    ]

    all_handles = (
        time_handles +
        [Line2D([], [], color='none', label='')] +
        marker_handles +
        [Line2D([], [], color='none', label='')] +
        fill_handles
    )

    ax.legend(
        handles=all_handles, loc='upper right',
        fontsize=8.5,
        title='Colour → Time     Shape → Rule     Fill → Level',
        title_fontsize=9.5, framealpha=0.9, ncol=1,
    )

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title(
        'FJSP Comprehensive Comparison:  Makespan vs Computation Time\n'
        'Kacem 8×8  —  All Methods  ·  All Decision Points  (t = 0, 2, 6)',
        fontsize=15, fontweight='bold', pad=14)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "fig_comprehensive_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comprehensive scatter plot saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  Gantt charts for GA schedules
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ga_gantt_charts(
    ga_global: Optional[dict] = None,
    ga_bi_level: Optional[dict] = None,
) -> None:
    """Generate Gantt charts for the best GA schedules."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if ga_global is None and os.path.exists(GA_GLOBAL_CACHE):
        with open(GA_GLOBAL_CACHE, "r", encoding="utf-8") as f:
            ga_global = json.load(f)
    if ga_bi_level is None and os.path.exists(GA_BI_LEVEL_CACHE):
        with open(GA_BI_LEVEL_CACHE, "r", encoding="utf-8") as f:
            ga_bi_level = json.load(f)

    # ── Global GA Gantt (3 panels: t=0, t=2, t=6) ──────────────────────────
    if ga_global:
        _plot_global_ga_gantt(ga_global)

    # ── Bi-level GA Gantt (one figure per rule, 3 panels each) ─────────────
    if ga_bi_level and 'rules' in ga_bi_level:
        disruptions_meta = ga_bi_level.get('metadata', {}).get('disruptions', [])
        for rule_name in ['SPT', 'FIFO', 'WINQ']:
            if rule_name in ga_bi_level['rules']:
                _plot_bi_level_ga_gantt(
                    rule_name, ga_bi_level['rules'][rule_name], disruptions_meta)


def _plot_global_ga_gantt(results: dict) -> None:
    """3-panel Gantt for global GA at t=0, t=2, t=6."""
    ga_data = results.get("global_ga", results)
    snapshots: List[List[ScheduleEntry]] = []
    cmax_list: List[float] = []
    all_job_ids: Set[int] = set()

    for t in SNAPSHOT_TIMES:
        t_str = str(t)
        if t_str not in ga_data:
            continue
        entries = [ScheduleEntry(**e) for e in ga_data[t_str]["entries"]]
        # Mark fixed vs future
        snap_sched: List[ScheduleEntry] = []
        for e in entries:
            if e.end_time <= t:
                snap_sched.append(ScheduleEntry(
                    e.job_id, e.op_idx, e.machine,
                    e.start_time, e.end_time, fixed=True))
            elif e.start_time < t < e.end_time:
                snap_sched.append(ScheduleEntry(
                    e.job_id, e.op_idx, e.machine,
                    e.start_time, e.end_time, fixed=True))
            else:
                snap_sched.append(ScheduleEntry(
                    e.job_id, e.op_idx, e.machine,
                    e.start_time, e.end_time, fixed=False))
        snapshots.append(snap_sched)
        cmax_list.append(ga_data[t_str]["cmax"])
        all_job_ids.update(e.job_id for e in snap_sched)

    x_max = max(cmax_list) * 1.10 if cmax_list else 30
    all_jids_sorted = sorted(all_job_ids)

    fig = plt.figure(figsize=(27, 9))
    gs = GridSpec(1, 3, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    labels = [
        "t = 0  (Global GA, J1-J8 only)",
        "t = 2  (Global GA, + J9, J10)",
        "t = 6  (Global GA, + M3 breakdown)",
    ]
    for i, (t, snap, ax) in enumerate(zip(SNAPSHOT_TIMES, snapshots, axes)):
        hl = {9, 10} if t >= 2.0 else set()
        dt = ga_data[str(t)]["compute_time"]
        plot_gantt(snap,
                   f'Global GA  —  {labels[i]}\n'
                   f'C_max = {cmax_list[i]:.3f},  time = {dt:.1f}s',
                   current_time=t, highlight_jobs=hl,
                   show_legend=False, ax=ax)
        ax.set_xlim(0, x_max)

    # Shared legend
    legend_patches = []
    hl_global = {9, 10} if 9 in all_jids_sorted else set()
    for jid in all_jids_sorted:
        label = f"J{jid}"
        if jid in hl_global:
            legend_patches.append(mpatches.Patch(
                facecolor=job_color(jid), label=label,
                edgecolor='black', linewidth=2.0))
        else:
            legend_patches.append(mpatches.Patch(
                facecolor=job_color(jid), label=label))
    fig.legend(handles=legend_patches, loc='upper center',
               ncol=min(len(all_jids_sorted), 10), fontsize=9,
               title='Jobs', title_fontsize=10,
               bbox_to_anchor=(0.5, 0.99))

    fig.suptitle('Kacem 8×8  FJSP — Global GA Scheduling  (Snapshot Gantt Charts)',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout(pad=3.5, rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUTPUT_DIR, "fig_ga_global_gantt.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Global GA Gantt → {out_path}")


def _plot_bi_level_ga_gantt(
    rule_name: str, rule_data: dict, disruptions: List[dict]
) -> None:
    """3-panel Gantt for one bi-level GA rule."""
    snapshots: List[List[ScheduleEntry]] = []
    cmax_list: List[float] = []
    all_job_ids: Set[int] = set()

    snapshot_scheds = rule_data.get("snapshot_schedules", {})
    partial_entries = [ScheduleEntry(**e)
                       for e in rule_data.get("partial_entries", [])]

    for t in SNAPSHOT_TIMES:
        t_str = str(t)
        if t_str in snapshot_scheds:
            entries = [ScheduleEntry(**e) for e in snapshot_scheds[t_str]]
        else:
            entries = [ScheduleEntry(**e) for e in rule_data["entries"]]

        # Build display snapshot
        interrupted_keys = {(e.job_id, e.op_idx) for e in partial_entries}
        snap: List[ScheduleEntry] = []
        for e in entries:
            key = (e.job_id, e.op_idx)
            if key in interrupted_keys and t < 6.0:
                continue
            is_fixed = (e.end_time <= t or e.start_time < t < e.end_time)
            snap.append(ScheduleEntry(e.job_id, e.op_idx, e.machine,
                                      e.start_time, e.end_time, fixed=is_fixed))
        # Add partial entries that show the interruption
        for pe in partial_entries:
            if pe.start_time <= t:
                snap.append(pe)

        snapshots.append(snap)
        cmax_list.append(rule_data["snapshot_cmax"].get(t_str, rule_data["cmax"]))
        all_job_ids.update(e.job_id for e in snap)

    x_max = max(cmax_list) * 1.10 if cmax_list else 30
    all_jids_sorted = sorted(all_job_ids)

    disruption_str = ""
    if disruptions:
        d = disruptions[0]
        disruption_str = f"  |  {d['machine']} breakdown at t={d['time']:.0f}"

    fig = plt.figure(figsize=(27, 9))
    gs = GridSpec(1, 3, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    labels = [
        "t = 0  (Global optimal plan)",
        "t = 2  (Bi-level GA: +J9, J10)",
        "t = 6  (Bi-level GA: +M3 breakdown)",
    ]
    for i, (t, snap, ax) in enumerate(zip(SNAPSHOT_TIMES, snapshots, axes)):
        hl = {9, 10} if t >= 2.0 else set()
        n_fixed = sum(1 for e in snap if e.fixed)
        plot_gantt(snap,
                   f'{rule_name}-GA  —  {labels[i]}\n'
                   f'({n_fixed} ops fixed  |  C_max = {cmax_list[i]:.3f})',
                   current_time=t, highlight_jobs=hl,
                   show_legend=False, ax=ax)
        ax.set_xlim(0, x_max)

    # Shared legend
    legend_patches = []
    hl_global = {9, 10} if 9 in all_jids_sorted else set()
    for jid in all_jids_sorted:
        label = f"J{jid}"
        if jid in hl_global:
            legend_patches.append(mpatches.Patch(
                facecolor=job_color(jid), label=label,
                edgecolor='black', linewidth=2.0))
        else:
            legend_patches.append(mpatches.Patch(
                facecolor=job_color(jid), label=label))
    fig.legend(handles=legend_patches, loc='upper center',
               ncol=min(len(all_jids_sorted), 10), fontsize=9,
               title='Jobs', title_fontsize=10,
               bbox_to_anchor=(0.5, 0.99))

    fig.suptitle(
        f'Kacem 8×8  FJSP — {rule_name}-GA Bi-Level Scheduling  '
        f'(Snapshot Gantt Charts{disruption_str})',
        fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout(pad=3.5, rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUTPUT_DIR, f"fig_ga_bi_level_gantt_{rule_name.lower()}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Bi-level GA Gantt [{rule_name}] → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  Resilience table with GA methods added
# ═══════════════════════════════════════════════════════════════════════════════

def print_metrics_table(
    ga_global: Optional[dict] = None,
    ga_bi_level: Optional[dict] = None,
) -> str:
    """Print and return comprehensive metrics table.

    Baselines A/B/C are shown as a single row, with each corresponding to
    the decision point that has the same information available:
      - Baseline A (J1-J8 optimal)                   → t=0 column
      - Baseline B (J1-J10 clairvoyant)              → t=2 column
      - Baseline C (J1-J10 + M3 clairvoyant)         → t=6 column

    Includes Gurobi baselines, single-level rules, bi-level MILP, bi-level GA,
    and global GA — all with per-stage C_max and computation time.
    """
    # Load all data
    bl_raw = {}
    if os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE, "r", encoding="utf-8") as f:
            bl_raw = json.load(f)

    rule_raw = {}
    if os.path.exists(RULE_CACHE):
        with open(RULE_CACHE, "r", encoding="utf-8") as f:
            rule_raw = json.load(f)

    milp_bi_raw = {}
    if os.path.exists(BI_LEVEL_CACHE):
        with open(BI_LEVEL_CACHE, "r", encoding="utf-8") as f:
            milp_bi_raw = json.load(f)

    if ga_global is None and os.path.exists(GA_GLOBAL_CACHE):
        with open(GA_GLOBAL_CACHE, "r", encoding="utf-8") as f:
            ga_global = json.load(f)
    if ga_bi_level is None and os.path.exists(GA_BI_LEVEL_CACHE):
        with open(GA_BI_LEVEL_CACHE, "r", encoding="utf-8") as f:
            ga_bi_level = json.load(f)

    SEP = "  "
    HL = "-" * 140

    lines = [
        "",
        "Metrics Table: Kacem 8×8 FJSP — C_max & Computation Time",
        "=" * 80,
        "  Baseline A → t=0   Baseline B → t=2   Baseline C → t=6",
        "  Bi-level methods show per-stage solve times (t=0 cached Baseline A).",
        "",
    ]

    # ── Header ───────────────────────────────────────────────────────────────
    h = (f"{'Method':<20s}{SEP}"
         f"{'t=0 Cmax':>10s}  {'t=0 Time':>10s}{SEP}"
         f"{'t=2 Cmax':>10s}  {'t=2 Time':>10s}{SEP}"
         f"{'t=6 Cmax':>10s}  {'t=6 Time':>10s}{SEP}"
         f"{'Final Cmax':>12s}")
    lines.append(h)
    lines.append(HL)

    def _fmt_time(ct):
        """Format a computation time value (seconds) for display."""
        if ct is None:
            return "--"
        if ct < 0.001:
            return "<1ms"
        if ct < 1.0:
            return f"{ct*1000:.0f}ms"
        return f"{ct:.1f}s"

    def add_row(name, d0, d2, d6, final_cmax=None):
        def cell(data):
            if data is None:
                return f"{'--':>10s}  {'--':>10s}"
            cmax, ct = data
            if cmax is None:
                return f"{'--':>10s}  {'--':>10s}"
            ct_str = _fmt_time(ct)
            return f"{cmax:10.3f}  {ct_str:>10s}"
        if final_cmax is None:
            for d in [d6, d2, d0]:
                if d is not None and d[0] is not None:
                    final_cmax = d[0]
                    break
        c0 = cell(d0) if d0 is not None else f"{'--':>10s}  {'--':>10s}"
        c2 = cell(d2) if d2 is not None else f"{'--':>10s}  {'--':>10s}"
        c6 = cell(d6) if d6 is not None else f"{'--':>10s}  {'--':>10s}"
        fc_str = f"{final_cmax:>12.3f}" if final_cmax is not None else f"{'--':>12s}"
        lines.append(f"{name:<20s}{SEP}{c0}{SEP}{c2}{SEP}{c6}{SEP}{fc_str}")

    # ── Clairvoyant Baselines (single row: A→t=0, B→t=2, C→t=6) ─────────────
    if bl_raw:
        bls = bl_raw["baselines"]
        add_row("Baseline (clairvoyant)",
                (bls["A"]["cmax"], bls["A"].get("compute_time")),
                (bls["B"]["cmax"], bls["B"].get("compute_time")),
                (bls["C"]["cmax"], bls["C"].get("compute_time")),
                bls["C"]["cmax"])
        lines.append(HL)

    # ── Single-level rules ───────────────────────────────────────────────────
    if rule_raw:
        for rn in ["SPT", "FIFO", "WINQ"]:
            r = rule_raw["rules"][rn]
            add_row(f"{rn} (rule)",
                    (r["0.0"]["cmax"], r["0.0"]["compute_time"]),
                    (r["2.0"]["cmax"], r["2.0"]["compute_time"]),
                    (r["6.0"]["cmax"], r["6.0"]["compute_time"]))
        lines.append(HL)

    # ── Bi-level MILP ────────────────────────────────────────────────────────
    if milp_bi_raw:
        milp_rules = milp_bi_raw.get("rules", {})
        for rn in ["SPT", "FIFO", "WINQ"]:
            if rn not in milp_rules:
                continue
            r = milp_rules[rn]
            sc = r["snapshot_cmax"]
            st = r.get("stage_times", {})
            # Fallback: if stage_times not available, put total compute_time at t=6
            t0_time = st.get("0.0") if st else None
            t2_time = st.get("2.0") if st else None
            t6_time = st.get("6.0") if st else r.get("compute_time")
            add_row(f"{rn}-MILP (bi-lev)",
                    (sc["0.0"], t0_time),
                    (sc["2.0"], t2_time),
                    (sc["6.0"], t6_time),
                    r["cmax"])
        lines.append(HL)

    # ── Global GA ────────────────────────────────────────────────────────────
    if ga_global:
        gga = ga_global.get("global_ga", ga_global)
        add_row("Global GA",
                (gga["0.0"]["cmax"], gga["0.0"]["compute_time"]),
                (gga["2.0"]["cmax"], gga["2.0"]["compute_time"]),
                (gga["6.0"]["cmax"], gga["6.0"]["compute_time"]))
        lines.append(HL)

    # ── Bi-level GA ──────────────────────────────────────────────────────────
    if ga_bi_level and "rules" in ga_bi_level:
        for rn in ["SPT", "FIFO", "WINQ"]:
            if rn not in ga_bi_level["rules"]:
                continue
            r = ga_bi_level["rules"][rn]
            sc = r["snapshot_cmax"]
            st = r.get("stage_times", {})
            add_row(f"{rn}-GA (bi-lev)",
                    (sc["0.0"], st.get("0.0")),
                    (sc["2.0"], st.get("2.0")),
                    (sc["6.0"], st.get("6.0")),
                    r["cmax"])
        lines.append(HL)

    # ── Gap analysis ─────────────────────────────────────────────────────────
    if bl_raw:
        lines.append("")
        lines.append("Gap Analysis (C_max vs. clairvoyant baseline at each stage):")
        lines.append("-" * 80)
        bl_cmax = {
            "0.0": bl_raw["baselines"]["A"]["cmax"],
            "2.0": bl_raw["baselines"]["B"]["cmax"],
            "6.0": bl_raw["baselines"]["C"]["cmax"],
        }
        bl_final = bl_raw["baselines"]["C"]["cmax"]

        def _gap(method_cmax, bl):
            return (method_cmax - bl) / bl * 100

        # Global GA
        if ga_global:
            gga = ga_global.get("global_ga", ga_global)
            for t_str in ["0.0", "2.0", "6.0"]:
                t = float(t_str)
                gap = _gap(gga[t_str]["cmax"], bl_cmax[t_str])
                lines.append(f"  Global GA       t={t:.0f}:  "
                             f"Cmax={gga[t_str]['cmax']:.3f}  "
                             f"Opt={bl_cmax[t_str]:.1f}  gap={gap:+.2f}%")

        # Bi-level MILP
        if milp_bi_raw:
            for rn in ["SPT", "FIFO", "WINQ"]:
                if rn in milp_bi_raw.get("rules", {}):
                    r = milp_bi_raw["rules"][rn]
                    gap = _gap(r["cmax"], bl_final)
                    lines.append(f"  {rn}-MILP bi-level  final:  "
                                 f"Cmax={r['cmax']:.3f}  "
                                 f"Opt={bl_final:.1f}  gap={gap:+.2f}%")

        # Bi-level GA
        if ga_bi_level and "rules" in ga_bi_level:
            for rn in ["SPT", "FIFO", "WINQ"]:
                if rn in ga_bi_level["rules"]:
                    r = ga_bi_level["rules"][rn]
                    gap = _gap(r["cmax"], bl_final)
                    lines.append(f"  {rn}-GA   bi-level  final:  "
                                 f"Cmax={r['cmax']:.3f}  "
                                 f"Opt={bl_final:.1f}  gap={gap:+.2f}%")

    lines.append(HL)
    lines.append("")

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(OUTPUT_DIR, "metrics_table.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"  Metrics table saved → {out_path}")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GA experiments for Kacem 8x8 FJSP")
    parser.add_argument("--skip-global", action="store_true",
                        help="Skip global GA experiments")
    parser.add_argument("--skip-bi-level", action="store_true",
                        help="Skip bi-level GA experiments")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only generate plots from cached results")
    args = parser.parse_args()

    t_total = time.perf_counter()

    ga_global = None
    ga_bi_level = None

    if not args.plots_only:
        if not args.skip_global:
            print("\n" + "█" * 72)
            print("  PHASE 1:  Global GA (Single-Level)")
            print("█" * 72)
            ga_global = run_global_ga()
            if ga_global is None:
                print("ERROR: Global GA experiments failed.", file=sys.stderr)
                sys.exit(1)

        if not args.skip_bi_level:
            print("\n" + "█" * 72)
            print("  PHASE 2:  Bi-Level GA")
            print("█" * 72)
            ga_bi_level = run_bi_level_ga()
            if ga_bi_level is None:
                print("ERROR: Bi-level GA experiments failed.", file=sys.stderr)
                sys.exit(1)

    # ── Plots and summary (always run) ─────────────────────────────────────
    print("\n" + "█" * 72)
    print("  PHASE 3:  Plots & Summary")
    print("█" * 72)

    plot_metrics_table_figure(ga_global, ga_bi_level)
    plot_ga_gantt_charts(ga_global, ga_bi_level)
    print_metrics_table(ga_global, ga_bi_level)

    elapsed = time.perf_counter() - t_total
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
