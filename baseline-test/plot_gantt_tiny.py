"""Plot Gantt charts for all strategies on the Tiny case.

Reads the exact same seed, case config, and strategy setup as
``run_comparison.py``, so results are guaranteed to match.
"""
import sys, os

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from case_generator import generate_case
from metrics import compute_metrics
from sim_engine import SimulationEngine
from plotting import plot_comparison_grid

# ── Import config directly from run_comparison (single source of truth) ──
from run_comparison import (
    SERVICE_UNITS, SCALES, START_SEED,
    PURE_RULES, CLAIRVOYANT_TIME_LIMIT,
)
from strategies import PureRuleStrategy, ClairvoyantMILPStrategy

# ── Generate the SAME Tiny case as run_comparison ────────────────────────
tiny_label, n_init, n_dyn = SCALES[0]  # ('Tiny', 5, 3)
seed = START_SEED

jobs, machines, disruptions = generate_case(
    seed=seed, n_initial=n_init, n_dynamic=n_dyn,
    ops_per_job=3, time_horizon=200.0,
    service_units=SERVICE_UNITS,
)

disruption_times = sorted({d.time for d in disruptions})

# ── Run all strategies ───────────────────────────────────────────────────
results = []

for rule in PURE_RULES:
    engine = SimulationEngine(jobs, machines,
                              PureRuleStrategy(rule=rule),
                              disruptions)
    schedule = engine.run()
    m = compute_metrics(schedule, jobs)
    results.append({
        'name': rule,
        'schedule': schedule,
        'metrics': m,
    })

engine = SimulationEngine(jobs, machines,
                          ClairvoyantMILPStrategy(time_limit=CLAIRVOYANT_TIME_LIMIT),
                          disruptions, idle_timeout=0.01)
schedule = engine.run()
m = compute_metrics(schedule, jobs)
results.append({
    'name': 'Clairvoyant',
    'schedule': schedule,
    'metrics': m,
})

# ── Print results ────────────────────────────────────────────────────────
n_jobs = len(jobs)
print(f"Tiny case (seed={seed}): {n_jobs} jobs, {len(machines)} machines, "
      f"{len(disruptions)} disruptions")
print(f"  {'Strategy':<14} {'Penalty':>8} {'Makespan':>8}")
print(f"  {'-'*14} {'-'*8} {'-'*8}")
for r in results:
    print(f"  {r['name']:<14} {r['metrics']['total_penalty']:>8.1f} "
          f"{r['metrics']['makespan']:>7.1f}")

# ── Plot ─────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), 'output',
                        'gantt_tiny_all_strategies.png')
plot_comparison_grid(results,
                     disruption_times=disruption_times,
                     output_path=out_path)
