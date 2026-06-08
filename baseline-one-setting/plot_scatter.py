"""Scatter plot: Makespan vs Computation Time for baselines and rule-based methods.

Three time periods (t=0, t=2, t=6) are distinguished by colour.
Gurobi MILP baselines (A/B/C) are shown as stars; rule-based methods
(SPT/FIFO/WINQ) as distinct marker shapes.

Output
------
  output/fig_scatter_comparison.png
"""

import json
import os
import sys
import time
from typing import Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(_HERE, "output")
BASELINE_CACHE = os.path.join(OUTPUT_DIR, "baseline_results.json")
RULE_CACHE = os.path.join(OUTPUT_DIR, "pure_rule_results.json")
BI_LEVEL_CACHE = os.path.join(OUTPUT_DIR, "bi_level_results.json")


# ═════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════

def load_rule_data() -> Dict:
    """Load rule-based results.  Returns dict:
        {rule_name: {snapshot_str: {cmax, compute_time}}}
    """
    with open(RULE_CACHE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rules_data: Dict[str, Dict[str, dict]] = {}
    for rule_name, snapshots in raw["rules"].items():
        rules_data[rule_name] = {}
        for t_str, sim in snapshots.items():
            rules_data[rule_name][t_str] = {
                "cmax": sim["cmax"],
                "compute_time": sim["compute_time"],
            }
    return rules_data


def load_bi_level_data() -> Dict:
    """Load bi-level results.  Returns dict:
        {rule_name: {snapshot_str: {cmax, compute_time}}}
    """
    if not os.path.exists(BI_LEVEL_CACHE):
        return {}
    with open(BI_LEVEL_CACHE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    bi_data: Dict[str, Dict[str, dict]] = {}
    for rule_name, rule_result in raw["rules"].items():
        bi_data[rule_name] = {}
        for t_str in ["0.0", "2.0", "6.0"]:
            cmax = rule_result["snapshot_cmax"].get(t_str, rule_result["cmax"])
            bi_data[rule_name][t_str] = {
                "cmax": cmax,
                "compute_time": rule_result["compute_time"],
            }
    return bi_data


def load_baseline_data() -> Dict[str, dict]:
    """Load Gurobi baseline results.  If compute_time is missing, attempt a
    lightweight re-solve.  Returns dict:
        {label: {cmax, compute_time}}
    """
    with open(BASELINE_CACHE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    baselines: Dict[str, dict] = {}
    needs_solve = False

    for key in ["A", "B", "C"]:
        bl = raw["baselines"][key]
        entry = {"cmax": bl["cmax"]}
        if "compute_time" in bl:
            entry["compute_time"] = bl["compute_time"]
        else:
            needs_solve = True
            entry["compute_time"] = None
        baselines[key] = entry

    if needs_solve:
        _fill_baseline_compute_times(baselines, raw)

    return baselines


def _fill_baseline_compute_times(baselines: Dict, raw: dict) -> None:
    """Re-solve baselines to record compute_time, then update cache.

    Only persists valid results (schedule_makespan_milp returns non-None).
    If Gurobi is unavailable, compute_time remains None and baselines are
    placed at a fixed y-offset with an "N/A" marker.
    """
    try:
        from models import Job, Operation
        from scheduler import schedule_makespan_milp, _HAS_GUROBI
    except ImportError:
        return

    if not _HAS_GUROBI:
        print("  Gurobi not available — baseline compute_time marked as N/A.")
        return

    data_path = os.path.join(_HERE, "kacem_data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _build(job_dict):
        return Job(
            job_id=job_dict["job_id"],
            arrival_time=job_dict.get("arrival_time", 0.0),
            operations=[Operation(job_id=job_dict["job_id"],
                                  op_idx=op["op_idx"],
                                  times=op["times"])
                        for op in job_dict["operations"]],
        )

    initial = [_build(j) for j in data["initial_jobs"]]
    dynamic = [_build(j) for j in data.get("dynamic_jobs", [])]
    all_jobs = initial + dynamic
    disruptions = data.get("disruptions", [])

    print("  Re-solving Gurobi baselines to get compute_time ...")

    updated = False

    # Baseline A
    t0 = time.perf_counter()
    sched = schedule_makespan_milp(initial, [], current_time=0.0, time_limit=120.0)
    dt_a = time.perf_counter() - t0
    if sched is not None:
        baselines["A"]["compute_time"] = dt_a
        raw["baselines"]["A"]["compute_time"] = dt_a
        updated = True
        print(f"    A  compute_time = {dt_a:.3f} s")
    else:
        print("    A  Gurobi unavailable — skipping")

    # Baseline B
    t0 = time.perf_counter()
    sched = schedule_makespan_milp(all_jobs, [], current_time=0.0, time_limit=120.0)
    dt_b = time.perf_counter() - t0
    if sched is not None:
        baselines["B"]["compute_time"] = dt_b
        raw["baselines"]["B"]["compute_time"] = dt_b
        updated = True
        print(f"    B  compute_time = {dt_b:.3f} s")
    else:
        print("    B  Gurobi unavailable — skipping")

    # Baseline C
    disruption = disruptions[0] if disruptions else {}
    dt = disruption.get("time", 6.0)
    dm = disruption.get("machine", "M3")
    t0 = time.perf_counter()
    sched = schedule_makespan_milp(
        all_jobs, [], current_time=0.0,
        machine_deadlines={dm: dt},
        time_limit=120.0)
    dt_c = time.perf_counter() - t0
    if sched is not None:
        baselines["C"]["compute_time"] = dt_c
        raw["baselines"]["C"]["compute_time"] = dt_c
        updated = True
        print(f"    C  compute_time = {dt_c:.3f} s")
    else:
        print("    C  Gurobi unavailable — skipping")

    if updated:
        with open(BASELINE_CACHE, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        print(f"  Updated {BASELINE_CACHE}")


# ═════════════════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════════════════

# Colour map for time periods
TIME_COLORS = {
    0.0: '#2196F3',   # blue
    2.0: '#FF9800',   # orange
    6.0: '#4CAF50',   # green
}

# Marker map for rule-based methods
RULE_MARKERS = {
    'SPT':  'o',   # circle
    'FIFO': 's',   # square
    'WINQ': '^',   # triangle
}

# Baseline marker (star)
BASELINE_MARKER = '*'


def plot_scatter(baselines: Dict,
                 rules: Dict,
                 bi_level: Optional[Dict] = None,
                 log_y: bool = True) -> None:
    """Generate the scatter plot using a two-panel layout.

    **Top panel** — Gurobi MILP baseline reference markers.
      Placed in a shaded band; the y-axis here is NOT quantitative (the
      baselines are aligned horizontally for readability).  Each baseline
      shows its makespan (C_max) value.

    **Bottom panel** — Rule-based methods (SPT / FIFO / WINQ).
      Standard scatter plot:  x = makespan,  y = computation time [ms],
      log scale.  Colour encodes the decision point (t=0 / t=2 / t=6),
      marker shape encodes the rule.
    """
    has_baseline_ct = any(b.get('compute_time') is not None
                          for b in baselines.values())

    # ═════════════════════════════════════════════════════════════════════
    #  Figure layout: 2 rows  (top = baselines, bottom = rules)
    # ═════════════════════════════════════════════════════════════════════
    if has_baseline_ct:
        # Single-panel: all data on one scatter plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax_top = None
    else:
        # Two-panel: top for baseline reference, bottom for rules
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(14, 9))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 3.0],
                      hspace=0.08)
        ax_top = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # ── Compute shared x-range ──────────────────────────────────────────
    rule_x_vals = []
    for rule_name in ['SPT', 'FIFO', 'WINQ']:
        for vals in rules[rule_name].values():
            rule_x_vals.append(vals['cmax'])
    baseline_x_vals = [b['cmax'] for b in baselines.values()]
    all_x_vals = rule_x_vals + baseline_x_vals
    if bi_level:
        for rule_name in ['SPT', 'FIFO', 'WINQ']:
            if rule_name in bi_level:
                for vals in bi_level[rule_name].values():
                    all_x_vals.append(vals['cmax'])
    x_min = min(all_x_vals) * 0.88
    x_max = max(all_x_vals) * 1.10

    # ═════════════════════════════════════════════════════════════════════
    #  TOP PANEL  —  Gurobi MILP Baselines
    # ═════════════════════════════════════════════════════════════════════
    # Each baseline corresponds to the information available at a decision point:
    #   A — only t=0 info (J1–J8)
    #   B — clairvoyant about t=2  (J1–J10 arrivals)
    #   C — clairvoyant about t=6  (M3 breakdown)
    BASELINE_TIME_MAP = {'A': 0.0, 'B': 2.0, 'C': 6.0}

    baseline_labels_map = {
        'A': 'Baseline A  (t=0)\nOptimal plan\n(J1–J8 only)',
        'B': 'Baseline B  (t=2)\nClairvoyant arrivals\n(J1–J10 known)',
        'C': 'Baseline C  (t=6)\nClairvoyant + disruption\n(J1–J10 + M3↓ known)',
    }

    if ax_top is not None:
        # ── Shaded baseline reference band ──────────────────────────────
        ax_top.set_facecolor('#F5F5F5')
        ax_top.axhline(y=0.5, color='#E0E0E0', linewidth=0.8, zorder=0)

        y_positions = [0.75, 0.50, 0.25]  # stack vertically to avoid overlap
        for i, key in enumerate(['A', 'B', 'C']):
            bl = baselines[key]
            x = bl['cmax']
            y = y_positions[i]

            color = TIME_COLORS[BASELINE_TIME_MAP[key]]
            ax_top.scatter(x, y, c=color, marker=BASELINE_MARKER, s=350,
                           edgecolors='black', linewidth=1.5, zorder=5)

            # Annotate with baseline label + cmax
            bl_label = baseline_labels_map[key]
            ax_top.annotate(
                f"{bl_label}\nC_max = {x:.1f} h",
                (x, y),
                textcoords="offset points",
                xytext=(12, 0),
                fontsize=8.5,
                fontweight='bold',
                va='center',
                color='#333333',
            )

        ax_top.set_ylim(0.0, 1.05)
        ax_top.set_yticks([])
        ax_top.set_ylabel('Gurobi\nMILP\nBaselines',
                          fontsize=10, fontweight='bold',
                          color='#555555', rotation=0,
                          labelpad=25, va='center')
        ax_top.yaxis.set_label_position('right')
        ax_top.set_xlim(x_min, x_max)
        ax_top.tick_params(axis='x', which='both', bottom=False, top=False,
                           labelbottom=False)
        ax_top.grid(False)
        for spine in ax_top.spines.values():
            spine.set_visible(False)

        # Dashed vertical lines extending down from each baseline
        for key in ['A', 'B', 'C']:
            x = baselines[key]['cmax']
            t_ref = BASELINE_TIME_MAP[key]
            ax_top.axvline(x=x, color=TIME_COLORS[t_ref], linestyle='--',
                           alpha=0.35, linewidth=1.2, zorder=1)
            ax.axvline(x=x, color=TIME_COLORS[t_ref], linestyle='--',
                       alpha=0.25, linewidth=1.0, zorder=1)

        ax_top.set_title(
            'FJSP Baseline Comparison:  Makespan vs Computation Time\n'
            'Kacem 8×8  —  Gurobi MILP baselines  ·  Rule-based methods',
            fontsize=14, fontweight='bold', pad=12)

    else:
        # Single-panel mode: baselines have compute_time
        for i, key in enumerate(['A', 'B', 'C']):
            bl = baselines[key]
            x = bl['cmax']
            y_ms = bl['compute_time'] * 1000.0
            t_ref = BASELINE_TIME_MAP[key]
            color = TIME_COLORS[t_ref]
            ax.scatter(x, y_ms, c=color, marker=BASELINE_MARKER, s=350,
                       edgecolors='black', linewidth=1.5, zorder=5)
            bl_label = baseline_labels_map[key]
            ax.annotate(
                f"{bl_label}\nC_max={x:.1f},  {y_ms:.0f} ms",
                (x, y_ms),
                textcoords="offset points",
                xytext=(10, 8),
                fontsize=8.0,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=0.8),
            )

    # ═════════════════════════════════════════════════════════════════════
    #  BOTTOM PANEL  —  Rule-based methods  (scatter)
    # ═════════════════════════════════════════════════════════════════════

    # ── Plot each rule × decision-point ─────────────────────────────────
    for rule_name in ['SPT', 'FIFO', 'WINQ']:
        marker = RULE_MARKERS[rule_name]
        rule_data = rules[rule_name]
        for t_str, vals in rule_data.items():
            t = float(t_str)
            x = vals['cmax']
            y_ms = vals['compute_time'] * 1000.0

            color = TIME_COLORS[t]
            label_short = f"{rule_name}"

            ax.scatter(x, y_ms, c=color, marker=marker, s=110,
                       edgecolors='black', linewidth=0.8, zorder=4)

            # Compact annotation: rule name + time
            t_label = f"t={t:.0f}"
            ax.annotate(
                f"{rule_name}  {t_label}",
                (x, y_ms),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=7.2,
                alpha=0.85,
                arrowprops=dict(arrowstyle='->', color='gray',
                                alpha=0.4, lw=0.6),
            )

    # ── Bi-level methods (diamond markers) ────────────────────────────────
    if bi_level:
        BI_MARKERS = {'SPT': 'D', 'FIFO': 'D', 'WINQ': 'D'}
        for rule_name in ['SPT', 'FIFO', 'WINQ']:
            if rule_name not in bi_level:
                continue
            bi_data = bi_level[rule_name]
            marker = BI_MARKERS[rule_name]
            for t_str, vals in bi_data.items():
                t = float(t_str)
                x = vals['cmax']
                y_ms = vals['compute_time'] * 1000.0

                color = TIME_COLORS[t]
                ax.scatter(x, y_ms, c=color, marker=marker, s=140,
                           edgecolors='black', linewidth=1.2, zorder=4)

                ax.annotate(
                    f"{rule_name}-MILP  t={t:.0f}",
                    (x, y_ms),
                    textcoords="offset points",
                    xytext=(8, -12),
                    fontsize=7.0,
                    alpha=0.85,
                    arrowprops=dict(arrowstyle='->', color='gray',
                                    alpha=0.4, lw=0.6),
                )

    # ── Axis setup ──────────────────────────────────────────────────────
    ax.set_xlabel('Makespan  (C_max)  [hours]', fontsize=13)
    ax.set_xlim(x_min, x_max)

    if log_y:
        ax.set_yscale('log')
        ax.set_ylabel('Computation Time  [ms]  (log scale)', fontsize=13)
    else:
        ax.set_ylabel('Computation Time  [ms]', fontsize=13)

    # ── Legend ──────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=TIME_COLORS[0.0], markersize=10,
               label='t = 0  (initial jobs, J1–J8)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=TIME_COLORS[2.0], markersize=10,
               label='t = 2  (+J9, J10 arrive)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=TIME_COLORS[6.0], markersize=10,
               label='t = 6  (+M3 breakdown)'),
        Line2D([], [], color='none', label=''),   # spacer
        Line2D([0], [0], marker=RULE_MARKERS['SPT'], color='w',
               markerfacecolor='#555555', markersize=9,
               label='SPT'),
        Line2D([0], [0], marker=RULE_MARKERS['FIFO'], color='w',
               markerfacecolor='#555555', markersize=9,
               label='FIFO'),
        Line2D([0], [0], marker=RULE_MARKERS['WINQ'], color='w',
               markerfacecolor='#555555', markersize=9,
               label='WINQ  (single-level)'),
        Line2D([], [], color='none', label=''),   # spacer
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor='#333333', markersize=10,
               label='SPT/FIFO/WINQ-MILP  (bi-level)'),
        Line2D([], [], color='none', label=''),   # spacer
        Line2D([0], [0], marker=BASELINE_MARKER, color='w',
               markerfacecolor=TIME_COLORS[0.0], markersize=14,
               label='Gurobi Baseline A  (t=0)'),
        Line2D([0], [0], marker=BASELINE_MARKER, color='w',
               markerfacecolor=TIME_COLORS[2.0], markersize=14,
               label='Gurobi Baseline B  (t=2)'),
        Line2D([0], [0], marker=BASELINE_MARKER, color='w',
               markerfacecolor=TIME_COLORS[6.0], markersize=14,
               label='Gurobi Baseline C  (t=6)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              title=' Legend', title_fontsize=10, framealpha=0.85,
              ncol=1)

    # ── Grid & finishing ────────────────────────────────────────────────
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # ── Layout note if baselines lack compute_time ──────────────────────
    if ax_top is not None:
        fig.text(0.5, 0.99,
                 '※  Gurobi MILP baselines: compute_time unavailable '
                 '(Gurobi solver not accessible in current environment).  '
                 'Rule-based methods: event-driven simulation, ~0.2–0.8 ms.',
                 ha='center', va='top', fontsize=8.5,
                 fontstyle='italic', color='#888888')

    # ── Save ────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "fig_scatter_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Scatter plot saved ->  {out_path}")


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading rule-based data ...")
    rules = load_rule_data()

    print("Loading baseline data ...")
    baselines = load_baseline_data()

    print("Loading bi-level data ...")
    bi_level = load_bi_level_data()

    # Print summary table
    print("\n" + "=" * 60)
    print("  Data summary")
    print("=" * 60)
    for rule_name in ['SPT', 'FIFO', 'WINQ']:
        for t_str in ['0.0', '2.0', '6.0']:
            d = rules[rule_name][t_str]
            print(f"  {rule_name:6s}  t={float(t_str):.0f}  "
                  f"cmax={d['cmax']:6.1f}  "
                  f"compute={d['compute_time']*1000:8.3f} ms")
    if bi_level:
        for rule_name in ['SPT', 'FIFO', 'WINQ']:
            if rule_name in bi_level:
                for t_str in ['0.0', '2.0', '6.0']:
                    d = bi_level[rule_name][t_str]
                    print(f"  {rule_name + '-MILP':6s}  t={float(t_str):.0f}  "
                          f"cmax={d['cmax']:6.1f}  "
                          f"compute={d['compute_time']*1000:8.3f} ms")
    for key in ['A', 'B', 'C']:
        b = baselines[key]
        ct_str = f"{b['compute_time']*1000:8.3f} ms" if b.get('compute_time') else "N/A"
        print(f"  Baseline {key}     "
              f"cmax={b['cmax']:6.1f}  "
              f"compute={ct_str}")
    print("-" * 60)

    plot_scatter(baselines, rules, bi_level, log_y=True)


if __name__ == "__main__":
    main()
