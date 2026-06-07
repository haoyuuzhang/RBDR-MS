"""Compute Gurobi MILP baselines for the Kacem 8x8 FJSP benchmark.

Three baselines are computed, all solved at t=0 with increasing information:

  Baseline A — Optimal Original Plan
      t=0, only J1-J8 known.  The best possible makespan with just the
      initial eight jobs.

  Baseline B — Clairvoyant Plan (arrivals)
      t=0, J1-J10 all known (J9/J10 released at t=2).  Perfect foresight
      of all job arrivals — absolute lower bound for any dynamic strategy.

  Baseline C — Clairvoyant Plan (arrivals + disruption)
      t=0, J1-J10 known + M3 breakdown at t=6 known in advance.  The
      solver knows M3 is unavailable after t=6 and optimises accordingly.
      Absolute lower bound with perfect information of all events.

Output
------
  output/baseline_results.json        — cached experiment results
  output/fig_gurobi_baselines.png     — 3-panel Gantt chart
  Console summary with C_max values and solver statistics.

Usage
-----
  python baseline-clairvoyant/run_baselines.py
"""

import json
import os
import sys
import time
from dataclasses import asdict
from typing import List, Optional

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
from scheduler import schedule_makespan_milp
from plotting import plot_gantt, job_color, job_label

OUTPUT_DIR = os.path.join(_HERE, "output")
DATA_PATH = os.path.join(_HERE, "kacem_data.json")
CACHE_PATH = os.path.join(OUTPUT_DIR, "baseline_results.json")


# ═════════════════════════════════════════════════════════════════════════
#  Data loading
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
#  Experiment runner
# ═════════════════════════════════════════════════════════════════════════

def run_experiments() -> Optional[dict]:
    """Run the three Gurobi MILP baselines.

    Returns a dict with keys ``"baselines"`` (A/B/C → cmax + schedule) and
    ``"metadata"`` (disruption info), or ``None`` if any solve fails.

    The result is also cached to *output/baseline_results.json* so the
    plotting stage can be iterated on without re-solving.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    initial_jobs, dynamic_jobs, disruptions = load_data()
    all_jobs = initial_jobs + dynamic_jobs

    print("=" * 72)
    print("  Kacem 8x8 FJSP -- Gurobi MILP Baselines")
    print("=" * 72)
    print(f"  Initial jobs : {len(initial_jobs)}  (J1-J{len(initial_jobs)})")
    print(f"  Dynamic jobs : {len(dynamic_jobs)}  (arrive at t="
          f"{dynamic_jobs[0].arrival_time if dynamic_jobs else 'N/A'})")
    print(f"  Disruptions  : {len(disruptions)}")
    total_ops = sum(len(j.operations) for j in all_jobs)
    print(f"  Total ops    : {total_ops}")
    print("-" * 72)

    # ── Baseline A: Initial optimal plan (J1–J8 only) ───────────────────
    print("\n[Baseline A]  Optimal Original Plan  (J1-J8 at t=0) ...")
    t0 = time.perf_counter()
    sched_A = schedule_makespan_milp(initial_jobs, [], current_time=0.0,
                                     time_limit=120.0)
    dt_A = time.perf_counter() - t0

    if sched_A is None:
        print("  !! Gurobi unavailable or MILP infeasible -- cannot proceed.")
        return None

    cmax_A = max(e.end_time for e in sched_A)
    print(f"  OK  C_max = {cmax_A:.3f}    (solve time {dt_A:.1f} s)")

    # ── Baseline B: Clairvoyant plan (J1–J10 known from t=0) ────────────
    print("\n[Baseline B]  Clairvoyant Plan  (J1-J10 known at t=0, "
          "J9/J10 released at t=2) ...")
    t0 = time.perf_counter()
    sched_B = schedule_makespan_milp(all_jobs, [], current_time=0.0,
                                     time_limit=120.0)
    dt_B = time.perf_counter() - t0

    if sched_B is None:
        print("  !! MILP failed -- cannot proceed.")
        return None

    cmax_B = max(e.end_time for e in sched_B)
    print(f"  OK  C_max = {cmax_B:.3f}    (solve time {dt_B:.1f} s)")

    # ── Baseline C: Clairvoyant plan + M3 breakdown known from t=0 ──────
    disruption = disruptions[0] if disruptions else {}
    disruption_time = disruption.get("time", 6.0)
    disrupted_machine = disruption.get("machine", "M3")
    print(f"\n[Baseline C]  Clairvoyant Plan + Disruption  (t=0 knows: "
          f"J1-J10, {disrupted_machine} dead at t={disruption_time:.0f}) ...")
    t0 = time.perf_counter()
    sched_C = schedule_makespan_milp(
        all_jobs, [], current_time=0.0,
        machine_deadlines={disrupted_machine: disruption_time},
        time_limit=120.0)
    dt_C = time.perf_counter() - t0

    if sched_C is None:
        print("  !! MILP failed -- cannot proceed.")
        return None

    cmax_C = max(e.end_time for e in sched_C)
    print(f"  OK  C_max = {cmax_C:.3f}    (solve time {dt_C:.1f} s)")

    def _serialise_schedule(schedule: List[ScheduleEntry]) -> list:
        """Convert a list of ScheduleEntry to a JSON-serialisable list."""
        return [asdict(e) for e in schedule]

    results = {
        "baselines": {
            "A": {"cmax": cmax_A, "entries": _serialise_schedule(sched_A)},
            "B": {"cmax": cmax_B, "entries": _serialise_schedule(sched_B)},
            "C": {"cmax": cmax_C, "entries": _serialise_schedule(sched_C)},
        },
        "metadata": {
            "disrupted_machine": disrupted_machine,
            "disruption_time": disruption_time,
        },
    }

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Cached results ->  {CACHE_PATH}")

    return results


# ═════════════════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════════════════

def plot_results(results: dict) -> None:
    """Generate the 3-panel Gantt chart from experiment results.

    Parameters
    ----------
    results : dict
        The dict returned by :func:`run_experiments`, or loaded from the
        JSON cache file.
    """
    baselines = results["baselines"]
    meta = results["metadata"]

    def _deserialise_schedule(entries: list) -> List[ScheduleEntry]:
        """Reconstruct ScheduleEntry objects from a list of plain dicts."""
        return [ScheduleEntry(**e) for e in entries]

    sched_A = _deserialise_schedule(baselines["A"]["entries"])
    sched_B = _deserialise_schedule(baselines["B"]["entries"])
    sched_C = _deserialise_schedule(baselines["C"]["entries"])

    cmax_A = baselines["A"]["cmax"]
    cmax_B = baselines["B"]["cmax"]
    cmax_C = baselines["C"]["cmax"]

    disrupted_machine = meta["disrupted_machine"]
    disruption_time = meta["disruption_time"]

    # ═════════════════════════════════════════════════════════════════════
    #  3-Panel Gantt Chart  (row 1: A full-width; row 2: B | C)
    #  Shared legend at top, no per-panel legend.
    # ═════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, 1:3])   # top row, centered (same width as B/C)
    ax2 = fig.add_subplot(gs[1, 0:2])   # bottom-left
    ax3 = fig.add_subplot(gs[1, 2:4])   # bottom-right

    x_max = max(cmax_A, cmax_B, cmax_C) * 1.08

    plot_gantt(sched_A,
               f'Baseline A -- Optimal Original Schedule  (J1-J8 only, t=0)\n'
               f'C_max = {cmax_A:.3f}',
               show_legend=False,
               ax=ax1)
    ax1.set_xlim(0, x_max)

    plot_gantt(sched_B,
               f'Baseline B -- Clairvoyant Schedule  (J1-J10 all known at t=0, '
               f'J9/J10 released t=2)\n'
               f'C_max = {cmax_B:.3f}',
               highlight_jobs={9, 10}, show_legend=False,
               ax=ax2)
    ax2.set_xlim(0, x_max)

    plot_gantt(sched_C,
               f'Baseline C -- Clairvoyant Schedule  (J1-J10 + '
               f'{disrupted_machine} down at t={disruption_time:.0f}, '
               f'all known at t=0)\n'
               f'C_max = {cmax_C:.3f}',
               highlight_jobs={9, 10}, show_legend=False,
               ax=ax3)
    ax3.set_xlim(0, x_max)

    # ── Shared figure-level legend (top, horizontal) ────────────────────
    all_job_ids = sorted({e.job_id for s in (sched_A, sched_B, sched_C)
                          for e in s})
    hl_set = {9, 10}
    legend_patches = []
    for jid in all_job_ids:
        label = f"{job_label(jid)} (Job {jid})"
        if jid in hl_set:
            legend_patches.append(mpatches.Patch(
                facecolor=job_color(jid), label=label,
                edgecolor='black', linewidth=2.0))
        else:
            legend_patches.append(mpatches.Patch(
                facecolor=job_color(jid), label=label))
    fig.legend(handles=legend_patches, loc='upper center',
               ncol=min(len(all_job_ids), 10), fontsize=9,
               title='Jobs', title_fontsize=10,
               bbox_to_anchor=(0.5, 0.97))

    fig.suptitle('Kacem 8x8  FJSP -- Gurobi MILP Optimal Baselines  (all solved at t=0)',
                 fontsize=15, fontweight='bold', y=0.99)
    fig.tight_layout(pad=3.5, rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "fig_gurobi_baselines.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Gantt chart saved ->  {out_path}")


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    # results = run_experiments()   # 注释掉，跳过 MILP
    with open(CACHE_PATH, "r") as f:
        results = json.load(f)
    plot_results(results)


if __name__ == "__main__":
    main()
