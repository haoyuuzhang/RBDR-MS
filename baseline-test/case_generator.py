"""Randomised case generator for dynamic FJSP rescheduling benchmarks.

Generates instances large enough to differentiate dispatching rules
(FIFO / EDD / SPT / ATC / CR) while keeping the t=0 MILP tractable.

Usage::

    from case_generator import generate_case
    jobs, machines, disruptions = generate_case(seed=42)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from models import Job, Operation, MachineConfig, Disruption


# ── public API ───────────────────────────────────────────────────────────

def generate_case(
    seed: int = 42,
    n_initial: int = 8,
    n_dynamic: int = 5,
    ops_per_job: int = 3,
    time_horizon: float = 200.0,
    service_units: Optional[Dict[str, List[str]]] = None,
) -> tuple:
    """Generate a random dynamic FJSP case.

    Parameters
    ----------
    seed:           Random seed for reproducibility.
    n_initial:      Jobs known at t=0.
    n_dynamic:      Jobs that arrive during the time horizon.
    ops_per_job:    Number of operations per job.
    time_horizon:   Maximum simulation time (dynamic jobs arrive within this).
    service_units:  Machine layout; defaults to an expanded 3-unit config.

    Returns
    -------
    (jobs, machines, disruptions)
    """
    rng = random.Random(seed)

    if service_units is None:
        service_units = _expanded_units()

    machines = _build_machines(service_units)
    machine_types = sorted({m.machine_type for m in machines.values()})
    unit_names = sorted({m.unit for m in machines.values()})

    # ── build all jobs with temporary IDs ──────────────────────────────
    unsorted: List[Job] = []

    # initial jobs (arrive at t=0)
    for _ in range(n_initial):
        unsorted.append(_make_job(
            jid=0, arrival=0.0, release=0.0,
            ops_per_job=ops_per_job,
            machine_types=machine_types,
            unit_names=unit_names,
            service_units=service_units,
            time_horizon=time_horizon,
            rng=rng,
        ))

    # dynamic jobs (staggered arrivals)
    arrival_window = time_horizon * 0.6
    arrival_start = time_horizon * 0.1
    for _ in range(n_dynamic):
        arrival = round(arrival_start + rng.random() * arrival_window, 1)
        release = arrival + rng.choice([0, rng.uniform(5, 20)])
        unsorted.append(_make_job(
            jid=0, arrival=arrival, release=release,
            ops_per_job=ops_per_job,
            machine_types=machine_types,
            unit_names=unit_names,
            service_units=service_units,
            time_horizon=time_horizon,
            rng=rng,
        ))

    # ── sort by arrival time, reassign IDs in that order ───────────────
    unsorted.sort(key=lambda j: (j.arrival_time, j.due_date))
    all_jobs: List[Job] = []
    for new_id, job in enumerate(unsorted, start=1):
        all_jobs.append(_renumber_job(job, new_id))

    # ── disruptions ───────────────────────────────────────────────────
    disruptions = _make_disruptions(all_jobs, machines, time_horizon, rng)

    return all_jobs, machines, disruptions


# ── helpers ──────────────────────────────────────────────────────────────

def _renumber_job(job: Job, new_id: int) -> Job:
    """Return a copy of *job* with job_id updated in both the Job and its Operations."""
    return Job(
        job_id=new_id,
        release_date=job.release_date,
        due_date=job.due_date,
        alpha=job.alpha,
        beta=job.beta,
        operations=[
            Operation(new_id, op.op_idx, op.machine_type, dict(op.unit_times))
            for op in job.operations
        ],
        arrival_time=job.arrival_time,
    )


def _expanded_units() -> Dict[str, List[str]]:
    """5 service units, 18 machines, 4 machine types."""
    return {
        'U1': ['M1_U1', 'M2_U1', 'M3_U1', 'M4_U1'],
        'U2': ['M1_U2', 'M2_U2', 'M3_U2', 'M4_U2'],
        'U3': ['M1_U3', 'M2_U3', 'M3_U3'],
        'U4': ['M1_U4', 'M2_U4', 'M4_U4'],
        'U5': ['M2_U5', 'M3_U5', 'M4_U5'],
    }


def _build_machines(service_units: Dict[str, List[str]]) -> Dict[str, MachineConfig]:
    machines: Dict[str, MachineConfig] = {}
    for unit, mlist in service_units.items():
        for mid in mlist:
            mtype = mid.split('_')[0]
            machines[mid] = MachineConfig(mid, mtype, unit)
    return machines


def _make_job(
    jid: int,
    arrival: float,
    release: float,
    ops_per_job: int,
    machine_types: List[str],
    unit_names: List[str],
    service_units: Dict[str, List[str]],
    time_horizon: float,
    rng: random.Random,
) -> Job:
    """Create a single job with random operations and due-date."""

    op_mtypes: List[str] = []
    for i in range(ops_per_job):
        candidates = [m for m in machine_types if not op_mtypes or m != op_mtypes[-1]]
        if not candidates:
            candidates = machine_types
        op_mtypes.append(rng.choice(candidates))

    operations: List[Operation] = []
    total_min_time = 0.0
    for idx, mtype in enumerate(op_mtypes):
        eligible = [u for u in unit_names if any(
            mid.startswith(mtype) for mid in service_units.get(u, [])
        )]
        if not eligible:
            eligible = unit_names[:1]

        # Assign processing times per eligible unit
        base = rng.uniform(8, 30)
        unit_times: Dict[str, float] = {}
        for u in eligible:
            # ±25% variation across units
            variation = 1.0 + rng.uniform(-0.25, 0.25)
            unit_times[u] = round(base * variation, 1)

        operations.append(Operation(jid, idx, mtype, unit_times))
        total_min_time += min(unit_times.values())

    # Due-date: random tightness (biased toward loose to create earliness-penalty jobs)
    tightness = rng.choices(
        ['tight', 'medium', 'loose', 'very_loose'],
        weights=[0.15, 0.25, 0.40, 0.20],
        k=1,
    )[0]
    if tightness == 'tight':
        dd_factor = rng.uniform(0.9, 1.3)
    elif tightness == 'medium':
        dd_factor = rng.uniform(1.3, 1.8)
    elif tightness == 'loose':
        dd_factor = rng.uniform(1.8, 3.0)
    else:  # very_loose
        dd_factor = rng.uniform(3.0, 4.5)

    due_date = round(max(release, arrival) + total_min_time * dd_factor, 1)

    # Penalty weights
    # alpha (earliness) — higher for jobs with wide due-date windows:
    #   finishing way early when you had plenty of time wastes resources.
    #   Designed so that PA shop rule can exploit the V-shaped penalty curve.
    slack_ratio = (due_date - max(release, arrival)) / max(total_min_time, 0.1)
    if slack_ratio > 3.0:          # very loose due-date → dominant earliness penalty
        alpha = rng.choice([4, 5, 6, 7, 8])
    elif slack_ratio > 2.2:        # loose due-date → high earliness penalty
        alpha = rng.choice([2, 3, 4, 5])
    elif slack_ratio > 1.5:        # moderately loose
        alpha = rng.choice([1, 2, 3, 4])
    elif slack_ratio > 1.1:        # medium window
        alpha = rng.choice([0, 1, 2, 3])
    else:                           # tight window
        alpha = rng.choice([0, 1, 2])

    # beta (tardiness) — range narrowed relative to alpha so that earliness
    # can dominate for loose jobs (α ≫ β creates the V-shape that PA exploits)
    if slack_ratio > 2.2:
        beta = rng.choice([1, 2, 3])
    else:
        beta = rng.choice([1, 2, 3, 4, 5])

    return Job(
        job_id=jid,
        release_date=round(release, 1),
        due_date=due_date,
        alpha=alpha,
        beta=beta,
        operations=operations,
        arrival_time=round(arrival, 1),
    )


def _make_disruptions(
    jobs: List[Job],
    machines: Dict[str, MachineConfig],
    time_horizon: float,
    rng: random.Random,
) -> List[Disruption]:
    """Generate 1-2 random machine disruptions with finite duration."""
    disruptions: List[Disruption] = []
    n_disruptions = rng.choice([1, 2])

    machine_ids = list(machines.keys())
    used_times: List[float] = []

    for _ in range(n_disruptions):
        mid = rng.choice(machine_ids)
        # Place disruption in the middle of the horizon
        d_time = round(time_horizon * 0.25 + rng.random() * time_horizon * 0.4, 1)
        # Avoid clustering disruptions at the same time
        while any(abs(d_time - t) < 10 for t in used_times):
            d_time = round(time_horizon * 0.25 + rng.random() * time_horizon * 0.4, 1)
        used_times.append(d_time)
        factor = rng.choice([1.5, 2.0, 2.5])
        # Duration: 20–60 time units, must not extend past the horizon
        max_dur = time_horizon - d_time
        dur = round(rng.uniform(20, min(60, max_dur)), 1)
        description = (f"{mid} factor=×{factor} for {dur}tu"
                       if dur > 0 else f"{mid} factor=×{factor} (permanent)")
        disruptions.append(Disruption(
            time=d_time,
            machine_id=mid,
            factor=factor,
            duration=dur,
            description=description,
        ))

    return sorted(disruptions, key=lambda d: d.time)


def case_to_dict(jobs: List[Job],
                 machines: Dict[str, MachineConfig],
                 disruptions: List[Disruption],
                 seed: int) -> dict:
    """Serialize a generated case to a plain dict (suitable for JSON)."""
    return {
        'seed': seed,
        'n_jobs': len(jobs),
        'n_machines': len(machines),
        'n_disruptions': len(disruptions),
        'machines': {
            mid: {'type': mc.machine_type, 'unit': mc.unit}
            for mid, mc in machines.items()
        },
        'jobs': [
            {
                'job_id': j.job_id,
                'arrival_time': j.arrival_time,
                'release_date': j.release_date,
                'due_date': j.due_date,
                'alpha': j.alpha,
                'beta': j.beta,
                'operations': [
                    {
                        'op_idx': op.op_idx,
                        'machine_type': op.machine_type,
                        'unit_times': dict(op.unit_times),
                    }
                    for op in j.operations
                ],
            }
            for j in sorted(jobs, key=lambda j: j.job_id)
        ],
        'disruptions': [
            {
                'time': d.time,
                'machine_id': d.machine_id,
                'factor': d.factor,
                'duration': d.duration,
                'description': d.description,
            }
            for d in disruptions
        ],
    }


def save_case_json(jobs: List[Job],
                   machines: Dict[str, MachineConfig],
                   disruptions: List[Disruption],
                   seed: int,
                   output_path: str):
    """Write a generated case to a JSON file."""
    import json
    data = case_to_dict(jobs, machines, disruptions, seed)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Case data saved to '{output_path}'")