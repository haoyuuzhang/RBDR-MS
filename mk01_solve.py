"""
Optimal Flexible Job-Shop Scheduling for mk01 Benchmark (Gurobi MIP)
Minimizes makespan. No hierarchical or dynamic considerations.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ──────────────────────────────────────────────────────────────
# 1. Load mk01 data
# ──────────────────────────────────────────────────────────────
with open('mk01.json', 'r') as f:
    raw = json.load(f)

num_machines = raw['machines']  # 6

# Flatten operations to sequential IDs
op_ids = []                     # all operation IDs 0..N-1
op_job = {}                     # op_id -> job_idx
op_pos = {}                     # op_id -> position within job
job_ops = defaultdict(list)     # job_idx -> [op_ids in order]
op_machines = {}                # op_id -> {machine: proc_time}

oid = 0
for jidx, job_data in enumerate(raw['jobs']):
    for oidx, alts in enumerate(job_data):
        op_ids.append(oid)
        op_job[oid] = jidx
        op_pos[oid] = oidx
        job_ops[jidx].append(oid)
        op_machines[oid] = {}
        for alt in alts:
            op_machines[oid][alt['machine']] = alt['processing']
        oid += 1

N = len(op_ids)                 # total operations
M = num_machines                # total machines

print(f"Loaded mk01: {len(raw['jobs'])} jobs, {M} machines, {N} operations")

# ──────────────────────────────────────────────────────────────
# 2. Build and solve the Gurobi MIP
# ──────────────────────────────────────────────────────────────
import gurobipy as gp
from gurobipy import GRB

# Big-M: sum of max processing times across all ops
BIG_M = sum(max(op_machines[o].values()) for o in op_ids)

env = gp.Env(params={"OutputFlag": 0})
model = gp.Model("mk01_FJSP", env=env)
model.Params.TimeLimit = 300.0
model.Params.MIPGap = 0.0

# -- Decision variables --
s = {}          # s[o] = start time of operation o (continuous)
x = {}          # x[o,k] = 1 if operation o assigned to machine k
C_max = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Cmax")

for o in op_ids:
    s[o] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{o}")
    for k in op_machines[o]:
        x[(o, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{o}_{k}")

# Sequencing: y[(a,b)] = 1 if a precedes b (only matters when same machine)
y = {}
for i in range(N):
    for j in range(i + 1, N):
        a, b = op_ids[i], op_ids[j]
        # Only need y if a and b share at least one feasible machine
        common = set(op_machines[a].keys()) & set(op_machines[b].keys())
        if common:
            y[(a, b)] = model.addVar(vtype=GRB.BINARY, name=f"y_{a}_{b}")

model.update()

# -- Constraints --

# (1) Each operation assigned to exactly one feasible machine
for o in op_ids:
    model.addConstr(
        gp.quicksum(x[(o, k)] for k in op_machines[o]) == 1,
        name=f"assign_{o}"
    )

# (2) Precedence within each job
for jidx, olist in job_ops.items():
    for p in range(len(olist) - 1):
        o_cur = olist[p]
        o_next = olist[p + 1]
        proc_cur = gp.quicksum(
            op_machines[o_cur][k] * x[(o_cur, k)] for k in op_machines[o_cur]
        )
        model.addConstr(
            s[o_next] >= s[o_cur] + proc_cur,
            name=f"prec_{o_cur}_{o_next}"
        )

# (3) Disjunctive constraints (no overlap on same machine)
for (a, b), y_var in y.items():
    common = set(op_machines[a].keys()) & set(op_machines[b].keys())
    for k in sorted(common):
        p_ak = op_machines[a][k]
        p_bk = op_machines[b][k]
        x_a = x[(a, k)]
        x_b = x[(b, k)]

        # a before b
        model.addConstr(
            s[b] >= s[a] + p_ak - BIG_M * (3 - x_a - x_b - y_var),
            name=f"disj_a_{a}_{b}_k{k}"
        )
        # b before a
        model.addConstr(
            s[a] >= s[b] + p_bk - BIG_M * (2 - x_a - x_b + y_var),
            name=f"disj_b_{a}_{b}_k{k}"
        )

# (4) Makespan bounds completion of every operation
for o in op_ids:
    proc_o = gp.quicksum(
        op_machines[o][k] * x[(o, k)] for k in op_machines[o]
    )
    model.addConstr(C_max >= s[o] + proc_o, name=f"cmax_{o}")

# -- Objective --
model.setObjective(C_max, GRB.MINIMIZE)

# -- Solve --
print(f"\nSolving mk01 FJSP... ({N} ops, {len(y)} sequencing pairs)")
model.optimize()

status_map = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.UNBOUNDED: "UNBOUNDED",
}
print(f"Status: {status_map.get(model.Status, model.Status)}")
print(f"Objective (C_max): {model.ObjVal:.2f}")
if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
    print(f"MIP Gap: {model.MIPGap:.4%}")

# ──────────────────────────────────────────────────────────────
# 3. Extract schedule
# ──────────────────────────────────────────────────────────────
schedule = []  # list of (job, op_pos, machine, start, duration)
for o in op_ids:
    start_t = s[o].X
    assigned_k = None
    proc_t = None
    for k in op_machines[o]:
        if x[(o, k)].X > 0.5:
            assigned_k = k
            proc_t = op_machines[o][k]
            break
    schedule.append((op_job[o], op_pos[o], assigned_k, start_t, proc_t))

# Sort by job and op for display
schedule.sort(key=lambda e: (e[0], e[1]))

print(f"\n{'Job':>4s} {'Op':>4s} {'Mach':>5s} {'Start':>8s} {'End':>8s} {'Dur':>6s}")
print("-" * 42)
for job, op_idx, mach, st, dur in schedule:
    print(f"  J{job + 1:>2d}  {op_idx + 1:>2d}    M{mach + 1}   "
          f"{st:>6.1f}  {st + dur:>6.1f}  {dur:>5.1f}")

cmax_val = max(st + dur for _, _, _, st, dur in schedule)
print(f"\nOptimal makespan: {cmax_val:.1f}")

# ──────────────────────────────────────────────────────────────
# 4. Gantt chart
# ──────────────────────────────────────────────────────────────
import numpy as np

# Color palette for 10 jobs
JOB_COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
    '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
]

fig, ax = plt.subplots(figsize=(18, 8))

machine_labels = [f'M{m + 1}' for m in range(M)]
y_ticks = list(range(M))

for job, op_idx, mach, st, dur in schedule:
    color = JOB_COLORS[job % len(JOB_COLORS)]
    bar = ax.barh(mach, dur, left=st, height=0.6,
                  color=color, edgecolor='white', linewidth=0.8, alpha=0.92)
    label = f'J{job + 1}-{op_idx + 1}'
    # Place label inside bar if it fits, otherwise to the right
    if dur > 2.0:
        ax.text(st + dur / 2, mach, label, ha='center', va='center',
                fontsize=6.5, fontweight='bold', color='white')
    else:
        ax.text(st + dur + 0.3, mach, label, ha='left', va='center',
                fontsize=6, fontweight='bold', color=color)

ax.set_yticks(y_ticks)
ax.set_yticklabels(machine_labels)
ax.set_ylabel('Machine', fontsize=12)
ax.set_xlabel('Time', fontsize=12)
ax.set_title(f'mk01 Flexible Job-Shop — Gurobi Optimal Schedule (C_max = {cmax_val:.1f})',
             fontsize=14, fontweight='bold')
ax.set_xlim(0, cmax_val * 1.06)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')

legend_patches = [
    mpatches.Patch(color=JOB_COLORS[j], label=f'Job {j + 1}')
    for j in range(len(job_ops))
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=7.5,
          ncol=5, title='Jobs', title_fontsize=9)

plt.tight_layout()
out_path = 'mk01_gantt.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nGantt chart saved to '{out_path}'")
