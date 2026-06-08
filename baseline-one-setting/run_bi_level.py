"""Bi-level FJSP scheduling experiments for the Kacem 8x8 benchmark.

Three upper-level rules (SPT / FIFO / WINQ) are tested with a Gurobi MILP
lower-level solver.  The initial schedule at t=0 is a global MILP (all 8
machines); re-scheduling at t=2 (J9/J10 arrivals) and t=6 (M3 breakdown)
uses the two-level architecture.

For each rule, a 3-panel snapshot Gantt chart (t=0 | t=2 | t=6) is produced,
along with cache files and a scatter-plot comparison against single-level
rules and Gurobi baselines.

Output
------
  output/bi_level_results.json          — cached experiment results
  output/fig_bi_level_gantt_spt.png     — SPT-MILP snapshot Gantt
  output/fig_bi_level_gantt_fifo.png    — FIFO-MILP snapshot Gantt
  output/fig_bi_level_gantt_winq.png    — WINQ-MILP snapshot Gantt

Usage
-----
  python baseline-one-setting/run_bi_level.py
"""

import json
import os
import sys
import time
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
from scheduler import _HAS_GUROBI
from bi_level_scheduler import (
    simulate_bi_level,
    GurobiUnitSolver,
    SERVICE_UNITS,
    UNIT_NAMES,
    ALL_MACHINES,
)

OUTPUT_DIR = os.path.join(_HERE, "output")
DATA_PATH = os.path.join(_HERE, "kacem_data.json")
CACHE_PATH = os.path.join(OUTPUT_DIR, "bi_level_results.json")
BASELINE_CACHE = os.path.join(OUTPUT_DIR, "baseline_results.json")
RULE_CACHE = os.path.join(OUTPUT_DIR, "pure_rule_results.json")

SNAPSHOT_TIMES = [0.0, 2.0, 6.0]


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════════

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
    """Load *kacem_data.json* and return (initial_jobs, dynamic_jobs, disruptions)."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    initial = [_build_job(j) for j in data["initial_jobs"]]
    dynamic = [_build_job(j) for j in data.get("dynamic_jobs", [])]
    disruptions = data.get("disruptions", [])
    return initial, dynamic, disruptions


# ═════════════════════════════════════════════════════════════════════════════
#  Snapshot builder (adapted from run_pure_baselines.py)
# ═════════════════════════════════════════════════════════════════════════════

def build_snapshot(
    final_entries: List[ScheduleEntry],
    partial_entries: List[ScheduleEntry],
    snapshot_time: float,
) -> List[ScheduleEntry]:
    """Build the schedule as seen at a given decision point.

    * Completed    (end_time <= T)               → ``fixed=True``
    * In-progress  (start_time < T < end_time)   → ``fixed=True``
    * Future       (start_time >= T)             → ``fixed=False``
    * Interrupted  (partial entry, start < T)    → ``fixed=True`` on broken machine
    """
    snapshot: List[ScheduleEntry] = []

    interrupted_keys: Set[Tuple[int, int]] = {
        (e.job_id, e.op_idx) for e in partial_entries
    }
    break_time_of: Dict[Tuple[int, int], float] = {
        (e.job_id, e.op_idx): e.end_time for e in partial_entries
    }

    for e in final_entries:
        key = (e.job_id, e.op_idx)

        if key in interrupted_keys:
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
            if e.end_time <= snapshot_time:
                snapshot.append(ScheduleEntry(
                    job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                    start_time=e.start_time, end_time=e.end_time,
                    fixed=True,
                ))
            elif e.start_time < snapshot_time < e.end_time:
                snapshot.append(ScheduleEntry(
                    job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                    start_time=e.start_time, end_time=e.end_time,
                    fixed=True,
                ))
            else:
                snapshot.append(ScheduleEntry(
                    job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                    start_time=e.start_time, end_time=e.end_time,
                    fixed=False,
                ))

    # Interrupted (partial) operations
    for e in partial_entries:
        if e.end_time <= snapshot_time:
            snapshot.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time,
                fixed=True,
            ))
        elif e.start_time < snapshot_time < e.end_time:
            snapshot.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time,
                fixed=True,
            ))
        elif e.start_time >= snapshot_time:
            snapshot.append(ScheduleEntry(
                job_id=e.job_id, op_idx=e.op_idx, machine=e.machine,
                start_time=e.start_time, end_time=e.end_time,
                fixed=False,
            ))

    snapshot.sort(key=lambda e: (e.start_time, e.job_id, e.op_idx))
    return snapshot


# ═════════════════════════════════════════════════════════════════════════════
#  Experiment runner
# ═════════════════════════════════════════════════════════════════════════════

def run_experiments() -> Optional[dict]:
    """Run bi-level simulations for SPT, FIFO, WINQ.

    Returns a dict with keys ``"rules"`` (rule_name → simulation result) and
    ``"metadata"``, or ``None`` on failure.

    Results are cached to *output/bi_level_results.json*.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    initial_jobs, dynamic_jobs, disruptions = load_data()
    all_jobs = initial_jobs + dynamic_jobs

    print("=" * 72)
    print("  Kacem 8x8 FJSP — Bi-Level Scheduling")
    print("  (Global MILP at t=0, Bi-level re-scheduling at t=2, t=6)")
    print("=" * 72)
    print(f"  Initial jobs : {len(initial_jobs)}  (J1-J{len(initial_jobs)})")
    print(f"  Dynamic jobs : {len(dynamic_jobs)}  (arrive at t="
          f"{dynamic_jobs[0].arrival_time if dynamic_jobs else 'N/A'})")
    print(f"  Disruptions  : {len(disruptions)}")
    for d in disruptions:
        print(f"                 {d['machine']} breakdown at t={d['time']}")
    total_ops = sum(len(j.operations) for j in all_jobs)
    print(f"  Total ops    : {total_ops}")
    print(f"  Machines     : {len(ALL_MACHINES)}  "
          f"(U1: {', '.join(SERVICE_UNITS['U1'])} | "
          f"U2: {', '.join(SERVICE_UNITS['U2'])})")
    print("-" * 72)

    # ── Load cached Baseline A schedule for t=0 ──────────────────────────
    initial_schedule: Optional[List[ScheduleEntry]] = None
    if os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE, "r", encoding="utf-8") as f:
            bl_cache = json.load(f)
        baseline_a_entries = bl_cache["baselines"]["A"]["entries"]
        initial_schedule = [ScheduleEntry(**e) for e in baseline_a_entries]
        print(f"\n  Using cached Baseline A for t=0  "
              f"(C_max = {bl_cache['baselines']['A']['cmax']:.3f})")
    else:
        print("\n  No cached baseline — will attempt Gurobi global solve at t=0")

    if not _HAS_GUROBI:
        print("\n  !! Gurobi not available — bi-level re-scheduling at t=2/t=6"
              " cannot run.")
        print("  Install gurobipy and configure a license to run the full"
              " experiment.")
        print("  The t=0 stage (global MILP) can use the cached Baseline A.\n")
        return None

    rules = ["SPT", "FIFO", "WINQ"]
    solver = GurobiUnitSolver()
    results: Dict[str, dict] = {}

    for rule_name in rules:
        print(f"\n{'─' * 72}")
        print(f"  Running:  {rule_name}-MILP")
        print(f"{'─' * 72}")

        sim_result = simulate_bi_level(rule_name, all_jobs, disruptions, solver,
                                        initial_schedule=initial_schedule)

        if sim_result is None:
            print(f"  !! {rule_name}-MILP failed")
            return None

        results[rule_name] = sim_result

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Bi-Level Scheduling Summary")
    print("=" * 72)
    header = f"  {'Rule':10s}"
    for t in SNAPSHOT_TIMES:
        header += f"    {'t=' + str(int(t)):>10s}"
    header += f"    {'Total':>10s}"
    print(header)
    print("  " + "-" * (10 + len(SNAPSHOT_TIMES) * 15 + 10))
    for rule_name in rules:
        r = results[rule_name]
        line = f"  {rule_name + '-MILP':10s}"
        for t in SNAPSHOT_TIMES:
            line += f"    {r['snapshot_cmax'][str(t)]:10.3f}"
        line += f"    {r['cmax']:10.3f}"
        dt = r['compute_time']
        n_partial = len(r['partial_entries'])
        line += f"  ({dt:.1f}s"
        if n_partial:
            line += f", {n_partial} interrupted"
        line += ")"
        print(line)
    print("-" * 72)

    # ── Comparison with existing baselines ─────────────────────────────────
    if os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE, "r", encoding="utf-8") as f:
            bl = json.load(f)
        print("\n  Comparison with Gurobi MILP clairvoyant baselines:")
        print(f"  Baseline A  (J1-J8 optimal)                C_max = {bl['baselines']['A']['cmax']:7.3f}")
        print(f"  Baseline B  (J1-J10 clairvoyant)           C_max = {bl['baselines']['B']['cmax']:7.3f}")
        print(f"  Baseline C  (J1-J10 + M3 clairvoyant)      C_max = {bl['baselines']['C']['cmax']:7.3f}")

    if os.path.exists(RULE_CACHE):
        with open(RULE_CACHE, "r", encoding="utf-8") as f:
            pr = json.load(f)
        print("\n  Comparison with single-level rule-based (t=6 snapshots):")
        for rn in rules:
            sl_cmax = pr['rules'][rn]['6.0']['cmax']
            bl_cmax = results[rn]['cmax']
            delta = (bl_cmax - sl_cmax) / sl_cmax * 100
            sign = '+' if delta > 0 else ''
            print(f"  {rn:6s}  single-level = {sl_cmax:7.3f}  "
                  f"bi-level = {bl_cmax:7.3f}  ({sign}{delta:.1f}%)")

    # ── Assemble output ───────────────────────────────────────────────────
    output = {
        "rules": results,
        "metadata": {
            "description": "Bi-level FJSP scheduling (upper: SPT/FIFO/WINQ, lower: Gurobi MILP)",
            "disruptions": disruptions,
            "num_jobs": len(all_jobs),
            "num_machines": len(ALL_MACHINES),
            "service_units": SERVICE_UNITS,
            "snapshot_times": SNAPSHOT_TIMES,
        },
    }

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Cached results →  {CACHE_PATH}")

    return output


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting — Snapshot Gantt charts
# ═════════════════════════════════════════════════════════════════════════════

def _plot_one_rule(rule_name: str, rule_data: dict, disruptions: List[dict]) -> None:
    """Generate a 3-panel snapshot Gantt chart for one bi-level rule.

    Panels: t=0 (global optimal) | t=2 (bi-level re-plan) | t=6 (bi-level re-plan)
    """
    snapshots: List[List[ScheduleEntry]] = []
    cmax_list: List[float] = []
    all_job_ids: Set[int] = set()

    # Load full schedule entries from the final stage
    final_entries = [ScheduleEntry(**e) for e in rule_data["entries"]]
    partial_entries = [ScheduleEntry(**e) for e in rule_data.get("partial_entries", [])]

    # Get per-snapshot schedules
    snapshot_scheds = rule_data.get("snapshot_schedules", {})

    for t in SNAPSHOT_TIMES:
        t_str = str(t)
        if t_str in snapshot_scheds:
            snap_entries = [ScheduleEntry(**e) for e in snapshot_scheds[t_str]]
            # Build display snapshot (fixed vs future distinction)
            snap = build_snapshot(snap_entries, partial_entries, t)
        else:
            snap = build_snapshot(final_entries, partial_entries, t)

        snapshots.append(snap)
        cmax_list.append(rule_data["snapshot_cmax"].get(t_str, rule_data["cmax"]))
        all_job_ids.update(e.job_id for e in snap)

    all_job_ids_sorted = sorted(all_job_ids)
    x_max = max(cmax_list) * 1.10 if max(cmax_list) > 0 else 30

    # ── Disruption annotation ──────────────────────────────────────────────
    disruption_str = ""
    if disruptions:
        d = disruptions[0]
        disruption_str = f"  |  {d['machine']} breakdown at t={d['time']:.0f}"

    # ═══════════════════════════════════════════════════════════════════════
    #  Figure: 1 row × 3 columns
    # ═══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(27, 9))
    gs = GridSpec(1, 3, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    snapshot_labels = [
        "t = 0  (Global optimal plan)",
        "t = 2  (Bi-level: +J9, J10)",
        "t = 6  (Bi-level: +M3 breakdown)",
    ]

    for i, (t, snap_sched, ax) in enumerate(zip(SNAPSHOT_TIMES, snapshots, axes)):
        n_fixed = sum(1 for e in snap_sched if e.fixed)
        cmax_i = cmax_list[i]

        # Highlight J9/J10 in t=2 and t=6 panels
        hl_set: Set[int] = set()
        if t >= 2.0 and any(jid >= 9 for jid in (e.job_id for e in snap_sched)):
            hl_set = {9, 10}

        panel_title = (
            f'{rule_name}-MILP  —  {snapshot_labels[i]}\n'
            f'({n_fixed} ops fixed  |  C_max = {cmax_i:.3f})'
        )
        plot_gantt(snap_sched, panel_title,
                   current_time=t,
                   highlight_jobs=hl_set,
                   show_legend=False,
                   ax=ax)
        ax.set_xlim(0, x_max)

    # ── Shared figure-level legend ────────────────────────────────────────
    legend_patches = []
    hl_set_global = {9, 10} if any(jid >= 9 for jid in all_job_ids_sorted) else set()
    for jid in all_job_ids_sorted:
        label = f"J{jid} (Job {jid})"
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
        f'Kacem 8x8  FJSP — {rule_name}-MILP Bi-Level Scheduling  '
        f'(Snapshot Gantt Charts{disruption_str})',
        fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout(pad=3.5, rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUTPUT_DIR, f"fig_bi_level_gantt_{rule_name.lower()}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Gantt chart saved →  {out_path}")


def plot_results(results: dict) -> None:
    """Generate per-rule snapshot Gantt charts."""
    rules_data = results["rules"]
    meta = results["metadata"]
    disruptions = meta.get("disruptions", [])

    print("\n  Generating bi-level snapshot Gantt charts ...")

    for rule_name in ["SPT", "FIFO", "WINQ"]:
        _plot_one_rule(rule_name, rules_data[rule_name], disruptions)

    print("  All bi-level Gantt charts generated.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    results = run_experiments()
    if results is None:
        print("ERROR: Bi-level experiments failed.", file=sys.stderr)
        sys.exit(1)
    plot_results(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
