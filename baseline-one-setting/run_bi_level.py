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
  output/fig_bi_level_scatter.png       — comparison scatter plot

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
#  Comparison scatter plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_comparison(results: dict) -> None:
    """Scatter plot: bi-level vs single-level rules vs Gurobi baselines."""
    # ── Load existing data ─────────────────────────────────────────────────
    bl_data: dict = {}
    if os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE, "r", encoding="utf-8") as f:
            bl_raw = json.load(f)
        for key in ["A", "B", "C"]:
            bl = bl_raw["baselines"][key]
            bl_data[key] = {"cmax": bl["cmax"],
                            "compute_time": bl.get("compute_time", None)}

    rule_data: dict = {}
    if os.path.exists(RULE_CACHE):
        with open(RULE_CACHE, "r", encoding="utf-8") as f:
            rule_raw = json.load(f)
        for rn in ["SPT", "FIFO", "WINQ"]:
            rule_data[rn] = {}
            for t_str in ["0.0", "2.0", "6.0"]:
                sim = rule_raw["rules"][rn][t_str]
                rule_data[rn][t_str] = {
                    "cmax": sim["cmax"],
                    "compute_time": sim["compute_time"],
                }

    # ── Colours & markers ──────────────────────────────────────────────────
    TIME_COLORS = {0.0: '#2196F3', 2.0: '#FF9800', 6.0: '#4CAF50'}
    RULE_MARKERS = {'SPT': 'o', 'FIFO': 's', 'WINQ': '^'}
    BI_MARKERS = {'SPT': 'D', 'FIFO': 'D', 'WINQ': 'D'}  # diamond for bi-level
    BASELINE_MARKER = '*'
    BASELINE_TIME_MAP = {'A': 0.0, 'B': 2.0, 'C': 6.0}

    # ── Compute ranges ────────────────────────────────────────────────────
    all_x = []
    for rn in ["SPT", "FIFO", "WINQ"]:
        for vals in rule_data.get(rn, {}).values():
            all_x.append(vals["cmax"])
        if rn in results["rules"]:
            r = results["rules"][rn]
            for t in SNAPSHOT_TIMES:
                all_x.append(r["snapshot_cmax"][str(t)])
    for bl in bl_data.values():
        all_x.append(bl["cmax"])
    x_min = min(all_x) * 0.88
    x_max = max(all_x) * 1.10

    # ═══════════════════════════════════════════════════════════════════════
    #  Figure
    # ═══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 8))

    # ── Gurobi baselines (stars) ─────────────────────────────────────────
    bl_labels = {
        'A': 'Baseline A\n(t=0, J1-J8)',
        'B': 'Baseline B\n(t=0 clairvoyant, J1-J10)',
        'C': 'Baseline C\n(t=0 clairvoyant, +M3)',
    }
    for key in ['A', 'B', 'C']:
        bl = bl_data[key]
        x = bl['cmax']
        t_ref = BASELINE_TIME_MAP[key]
        ct = bl.get('compute_time')
        if ct is not None:
            y_ms = ct * 1000.0
        else:
            y_ms = 200.0  # placeholder
        ax.scatter(x, y_ms, c=TIME_COLORS[t_ref], marker=BASELINE_MARKER, s=350,
                   edgecolors='black', linewidth=1.5, zorder=5)
        ct_str = f"{ct*1000:.0f} ms" if ct else "N/A"
        ax.annotate(
            f"{bl_labels[key]}\nC_max={x:.1f}, {ct_str}",
            (x, y_ms), textcoords="offset points", xytext=(10, 8),
            fontsize=7.5, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=0.8),
        )

    # ── Single-level rules (hollow markers) ──────────────────────────────
    for rn in ['SPT', 'FIFO', 'WINQ']:
        marker = RULE_MARKERS[rn]
        for t_str, vals in rule_data.get(rn, {}).items():
            t = float(t_str)
            x = vals['cmax']
            y_ms = vals['compute_time'] * 1000.0
            ax.scatter(x, y_ms, c='none', edgecolors=TIME_COLORS[t],
                       marker=marker, s=100, linewidth=1.2, zorder=3,
                       label='_' * 20)  # hide from legend

    # ── Bi-level rules (filled diamonds) ─────────────────────────────────
    for rn in ['SPT', 'FIFO', 'WINQ']:
        if rn not in results["rules"]:
            continue
        r = results["rules"][rn]
        for t in SNAPSHOT_TIMES:
            t_str = str(t)
            if t_str in r["snapshot_cmax"]:
                x = r["snapshot_cmax"][t_str]
            else:
                x = r["cmax"]
            y_ms = r['compute_time'] * 1000.0
            color = TIME_COLORS[t]
            ax.scatter(x, y_ms, c=color, marker='D', s=140,
                       edgecolors='black', linewidth=1.2, zorder=4)
            ax.annotate(
                f"{rn}-MILP  t={t:.0f}",
                (x, y_ms), textcoords="offset points", xytext=(8, -12),
                fontsize=7.0, alpha=0.85,
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=0.6),
            )

    # ── Axis setup ────────────────────────────────────────────────────────
    ax.set_xlabel('Makespan  (C_max)  [hours]', fontsize=13)
    ax.set_ylabel('Computation Time  [ms]  (log scale)', fontsize=13)
    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.25, linestyle='--')

    # ── Legend ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TIME_COLORS[0.0],
               markersize=10, label='t = 0  (initial jobs)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TIME_COLORS[2.0],
               markersize=10, label='t = 2  (+J9, J10 arrive)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TIME_COLORS[6.0],
               markersize=10, label='t = 6  (+M3 breakdown)'),
        Line2D([], [], color='none', label=''),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
               markersize=9, markeredgecolor='#888888', label='Single-level SPT'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
               markersize=9, markeredgecolor='#888888', label='Single-level FIFO'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#888888',
               markersize=9, markeredgecolor='#888888', label='Single-level WINQ'),
        Line2D([], [], color='none', label=''),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#555555',
               markersize=9, label='Bi-level (SPT/FIFO/WINQ-MILP)'),
        Line2D([], [], color='none', label=''),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#333333',
               markersize=14, label='Gurobi Baseline'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              title=' Legend', title_fontsize=10, framealpha=0.85, ncol=1)

    ax.set_title(
        'FJSP Comparison:  Makespan vs Computation Time\n'
        'Kacem 8×8  —  Gurobi Baselines  ·  Single-Level Rules  ·  Bi-Level (Rule+MILP)',
        fontsize=14, fontweight='bold', pad=12)

    out_path = os.path.join(OUTPUT_DIR, "fig_bi_level_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Scatter plot saved →  {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    results = run_experiments()
    if results is None:
        print("ERROR: Bi-level experiments failed.", file=sys.stderr)
        sys.exit(1)
    plot_results(results)
    plot_comparison(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
