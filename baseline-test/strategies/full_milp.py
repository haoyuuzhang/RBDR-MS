"""Full MILP Rescheduling strategy.

At each decision point (t=0, job arrival, disruption) the strategy calls
Gurobi to jointly optimise unit assignment, machine sequencing, and start
times for all remaining operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from milp_scheduler import schedule_milp

from strategies.base import BaseStrategy
from models import Job, Operation, ScheduleEntry, Disruption


class FullMILPStrategy(BaseStrategy):
    """Predictive-reactive strategy that re-optimises via MILP at every event.

    Parameters
    ----------
    time_limit: Gurobi time limit per MILP call (seconds).
    """

    def __init__(self, time_limit: float = 60.0):
        super().__init__()
        self.time_limit = time_limit
        self._plan: Dict[Tuple[int, int, str], ScheduleEntry] = {}
        self._time_factors: Dict[str, float] = {}
        self._plan_valid = False

    # -- BaseStrategy interface -------------------------------------------

    def attach(self, engine):
        super().attach(engine)
        self._plan = self._build_initial_plan(self.time_limit)
        self._plan_valid = bool(self._plan)

    def select_operation(
        self, machine_id: str, machine_type: str, unit: str,
        current_time: float, ready_ops: List[Tuple[Job, Operation]],
    ) -> Optional[Tuple[int, int, str, float]]:
        return self._lookup_plan(machine_id, self._plan) if self._plan_valid else None

    def on_job_arrival(self, job: Job):
        self._replan(self.engine.env.now)

    def on_disruption(self, disruption: Disruption):
        self._time_factors[disruption.machine_id] = disruption.factor
        self._replan(self.engine.env.now)

    # -- internal ---------------------------------------------------------

    def _replan(self, current_time: float):
        """Run MILP and store the resulting plan."""
        if self.engine is None:
            return

        visible = list(self.engine.visible_jobs)
        if not visible:
            self._plan_valid = True
            return

        # Frozen entries: already-completed or in-progress operations
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
                            end = self.engine.machine_busy_until.get(mid, cstart + cdur)
                            frozen.append(ScheduleEntry(
                                jid, oidx, mid, cunit,
                                cstart, end, fixed=True))
                            break

        tf = dict(self._time_factors) if self._time_factors else None

        schedule = schedule_milp(visible, frozen, current_time,
                                 self.time_limit, time_factors=tf)
        if schedule is None:
            self._plan_valid = False
            return

        self._plan.clear()
        for e in schedule:
            if not e.fixed:
                self._plan[(e.job_id, e.op_idx, e.machine)] = e
        self._plan_valid = True
