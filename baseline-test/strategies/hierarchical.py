"""Hierarchical scheduling strategy.

Two-stage decomposition based on the three-layer elastic scheduling model:

1. **Shop level**: Greedy earliest-completion-time (ECT) rule assigns each
   operation to the service unit where it would finish soonest, considering
   machine availability, job precedence, and cross-unit transport.

2. **Service-unit level**: MILP schedules operations within each unit —
   machine assignment + sequencing + start times — minimising weighted
   earliness-tardiness penalty.

The strategy re-runs the full two-stage pipeline on every job arrival /
disruption event.  The initial t=0 plan comes from the shared base-class
MILP (``_build_initial_plan``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from milp_scheduler import schedule_milp, TRANSPORT_TIME
from strategies.base import BaseStrategy
from models import Job, Operation, ScheduleEntry, Disruption


# Shop-level dispatching rules
_SHOP_RULES = ['ECT', 'EST', 'SPT', 'LBU']


class HierarchicalStrategy(BaseStrategy):
    """Hierarchical two-stage scheduling strategy.

    Parameters
    ----------
    shop_rule:       Shop-level assignment rule:
                     'ECT' - Earliest Completion Time (default)
                     'EST' - Earliest Start Time
                     'SPT' - Shortest Processing Time
                     'LBU' - Least Busy Unit (earliest machine free time)
    unit_time_limit: Gurobi time limit for the unit-level scheduling MILP.
    init_time_limit: Gurobi time limit for the initial t=0 MILP plan.
    """

    def __init__(self, shop_rule: str = 'ECT',
                 unit_time_limit: float = 30.0,
                 init_time_limit: float = 60.0):
        super().__init__()
        if shop_rule not in _SHOP_RULES:
            raise ValueError(f"shop_rule must be one of {_SHOP_RULES}, got {shop_rule!r}")
        self.shop_rule = shop_rule
        self.unit_time_limit = unit_time_limit
        self.init_time_limit = init_time_limit
        self._plan: Dict[Tuple[int, int, str], ScheduleEntry] = {}
        self._time_factors: Dict[str, float] = {}
        self._plan_valid = False
        self._unit_assignments: Dict[Tuple[int, int], str] = {}

    # -- BaseStrategy interface -------------------------------------------

    def attach(self, engine):
        super().attach(engine)
        self._plan = self._build_initial_plan(self.init_time_limit)
        self._plan_valid = bool(self._plan)

    def select_operation(
        self, machine_id: str, machine_type: str, unit: str,
        current_time: float, ready_ops: List[Tuple[Job, Operation]],
    ) -> Optional[Tuple[int, int, str, float]]:
        return self._lookup_plan(machine_id, self._plan) if self._plan_valid else None

    def on_job_arrival(self, job: Job):
        self._run_hierarchical(self.engine.env.now)

    def on_disruption(self, disruption: Disruption):
        self._time_factors[disruption.machine_id] = disruption.factor
        self._run_hierarchical(self.engine.env.now)

    # -- two-stage pipeline -----------------------------------------------

    def _run_hierarchical(self, current_time: float):
        """Execute the full two-stage hierarchical pipeline."""
        if self.engine is None:
            return

        visible = list(self.engine.visible_jobs)
        if not visible:
            self._plan_valid = True
            return

        frozen = self._collect_frozen()

        # Stage 1: Shop-level assignment (greedy ECT)
        self._unit_assignments = self._solve_shop_level(visible, frozen,
                                                        current_time)
        if not self._unit_assignments:
            self._plan_valid = False
            return

        # Stage 2: Restricted unit-level scheduling (MILP)
        restricted_jobs = self._build_restricted_jobs(visible, frozen)
        tf = dict(self._time_factors) if self._time_factors else None

        schedule = schedule_milp(restricted_jobs, frozen, current_time,
                                 self.unit_time_limit, time_factors=tf)
        if schedule is None:
            self._plan_valid = False
            return

        self._plan.clear()
        for e in schedule:
            if not e.fixed:
                self._plan[(e.job_id, e.op_idx, e.machine)] = e
        self._plan_valid = True

    # -- shop-level assignment (greedy rules) -----------------------------

    def _solve_shop_level(
        self, visible: List[Job], frozen: List[ScheduleEntry],
        current_time: float,
    ) -> Dict[Tuple[int, int], str]:
        """Assign each operation to a service unit via a greedy rule.

        Jobs are processed in release-date order; within each job operations
        are assigned sequentially so that precedence is honoured.  The unit
        is selected by *self.shop_rule* (ECT / EST / SPT / LBU), with
        completion time used as tie-breaker for all rules.
        """
        frozen_by_key: Dict[Tuple[int, int], ScheduleEntry] = {}
        for e in frozen:
            frozen_by_key[(e.job_id, e.op_idx)] = e

        machine_free: Dict[str, float] = {}
        for mid in self.engine.machines:
            busy = self.engine.machine_busy_until.get(mid, 0)
            machine_free[mid] = max(busy, current_time)

        assignments: Dict[Tuple[int, int], str] = {}

        for job in sorted(visible, key=lambda j: (j.release_date, j.job_id)):
            prev_completion = max(current_time, job.release_date)
            prev_unit: Optional[str] = None

            for op in job.operations:
                key = (job.job_id, op.op_idx)

                if key in frozen_by_key:
                    fe = frozen_by_key[key]
                    prev_completion = fe.end_time
                    prev_unit = fe.service_unit
                    continue

                best_unit: Optional[str] = None
                best_score = float('inf')
                best_completion = float('inf')   # tie-breaker

                for unit_name, base_time in op.unit_times.items():
                    mid = f"{op.machine_type}_{unit_name}"
                    if mid not in machine_free:
                        continue

                    factor = self._time_factors.get(mid, 1.0)
                    proc_time = base_time * factor
                    machine_ready = machine_free[mid]
                    transport = (TRANSPORT_TIME if prev_unit is not None
                                 and prev_unit != unit_name else 0.0)
                    start = max(machine_ready, prev_completion + transport)
                    completion = start + proc_time

                    # Score according to the selected rule
                    if self.shop_rule == 'ECT':
                        score = completion
                    elif self.shop_rule == 'EST':
                        score = start
                    elif self.shop_rule == 'SPT':
                        score = proc_time
                    elif self.shop_rule == 'LBU':
                        score = machine_ready

                    if (score < best_score
                            or (score == best_score and completion < best_completion)):
                        best_score = score
                        best_completion = completion
                        best_unit = unit_name

                if best_unit is None:
                    return {}

                assignments[key] = best_unit

                mid = f"{op.machine_type}_{best_unit}"
                machine_free[mid] = best_completion
                prev_completion = best_completion
                prev_unit = best_unit

        return assignments

    # -- helpers ----------------------------------------------------------

    def _collect_frozen(self) -> List[ScheduleEntry]:
        """Build frozen entries from completed and in-progress operations."""
        frozen: List[ScheduleEntry] = []
        for (jid, oidx), status in self.engine.op_status.items():
            if status in ('completed', 'in_progress'):
                for e in self.engine.completed_entries:
                    if e.job_id == jid and e.op_idx == oidx:
                        frozen.append(ScheduleEntry(
                            e.job_id, e.op_idx, e.machine,
                            e.service_unit, e.start_time,
                            e.end_time, fixed=True))
                        break
                else:
                    cur = self.engine._current_op
                    for mid, (cjid, coidx, cunit, cstart, cdur) in cur.items():
                        if cjid == jid and coidx == oidx:
                            end = self.engine.machine_busy_until.get(
                                mid, cstart + cdur)
                            frozen.append(ScheduleEntry(
                                jid, oidx, mid, cunit,
                                cstart, end, fixed=True))
                            break
        return frozen

    def _build_restricted_jobs(
        self, visible: List[Job], frozen: List[ScheduleEntry],
    ) -> List[Job]:
        """Return copies of visible jobs with unit_times restricted to the
        shop-level assignment for each operation."""
        frozen_keys = {(e.job_id, e.op_idx) for e in frozen}
        restricted: List[Job] = []
        for job in visible:
            ops: List[Operation] = []
            for op in job.operations:
                key = (job.job_id, op.op_idx)
                if key in frozen_keys:
                    # Keep original (already fixed / won't be rescheduled)
                    ut = dict(op.unit_times)
                elif key in self._unit_assignments:
                    assigned = self._unit_assignments[key]
                    ut = {assigned: op.unit_times[assigned]}
                else:
                    ut = dict(op.unit_times)
                ops.append(Operation(
                    job.job_id, op.op_idx, op.machine_type, ut))
            restricted.append(Job(
                job.job_id, job.release_date, job.due_date,
                job.alpha, job.beta, ops, job.arrival_time))
        return restricted
