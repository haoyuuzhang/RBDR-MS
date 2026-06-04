"""Clairvoyant (offline-optimal) MILP strategy.

At t=0 the strategy builds a plan using ALL jobs AND perfect knowledge of
disruption time windows.  The MILP uses piecewise-linear constraints to
model operations that span disruption boundaries — the part before/after
a disruption uses factor=1.0, the part during uses factor×base_time.

This yields a true offline-optimal baseline: no reactive strategy can beat
a plan made with perfect foresight of both arrivals and disruptions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from milp_scheduler import schedule_milp_clairvoyant
from strategies.base import BaseStrategy
from models import Job, Operation, ScheduleEntry, Disruption


class ClairvoyantMILPStrategy(BaseStrategy):
    """Offline-optimal strategy: knows all jobs AND disruptions at t=0.

    Parameters
    ----------
    time_limit: Gurobi time limit for the one-shot t=0 MILP (seconds).
    """

    def __init__(self, time_limit: float = 120.0):
        super().__init__()
        self.time_limit = time_limit
        self._plan: Dict[Tuple[int, int, str], ScheduleEntry] = {}
        self._plan_valid = False

    # -- BaseStrategy interface -------------------------------------------

    def attach(self, engine):
        super().attach(engine)
        self._plan = self._build_clairvoyant_plan()
        self._plan_valid = bool(self._plan)

    def select_operation(
        self, machine_id: str, machine_type: str, unit: str,
        current_time: float, ready_ops: List[Tuple[Job, Operation]],
    ) -> Optional[Tuple[int, int, str, float]]:
        if not self._plan_valid:
            return self._greedy_ect(machine_id, unit, ready_ops)

        # Follow the plan's unit assignment and machine sequence.
        # Do NOT enforce MILP-planned start times: the MILP builds
        # TRANSPORT_TIME into its precedence gaps, and the simulation
        # engine independently adds TRANSPORT_TIME.  Enforcing start
        # times would double-count transport, causing a systematic +2.0
        # time-unit shift for every operation after the first in each job.
        #
        # The PWL breakpoints include all disruption boundaries, making
        # duration calculations exact at any start time — so starting at
        # a slightly different time than planned does not degrade accuracy.
        result = self._lookup_plan(machine_id, self._plan)
        if result is not None:
            return result  # start immediately — plan gives unit + sequence

        if self._has_future_entry(machine_id):
            return None

        return self._greedy_ect(machine_id, unit, ready_ops)

    def on_job_arrival(self, job: Job):
        pass  # Already accounted for in the clairvoyant plan

    def on_disruption(self, disruption: Disruption):
        pass  # Already accounted for in the clairvoyant plan (via PWL)

    # -- internal ---------------------------------------------------------

    def _build_clairvoyant_plan(self) -> Dict[Tuple[int, int, str], ScheduleEntry]:
        """Run the disruption-aware clairvoyant MILP at t=0."""
        if self.engine is None:
            return {}

        all_jobs = list(self.engine.jobs)
        if not all_jobs:
            return {}

        schedule = schedule_milp_clairvoyant(
            jobs=all_jobs,
            fixed_entries=[],
            current_time=0.0,
            time_limit=self.time_limit,
            disruptions=list(self.engine.disruptions),
            machine_count=len(self.engine.machines),
        )
        if schedule is None:
            print("[clairvoyant] MILP returned None (infeasible/failed)")
            return {}

        plan: Dict[Tuple[int, int, str], ScheduleEntry] = {}
        for e in schedule:
            plan[(e.job_id, e.op_idx, e.machine)] = e
        return plan

    def _has_future_entry(self, machine_id: str) -> bool:
        """Check whether a future (not-yet-ready) plan entry exists for this machine."""
        for (jid, oidx, mid), _entry in self._plan.items():
            if mid != machine_id:
                continue
            status = self.engine.op_status.get((jid, oidx))
            if status in ('pending',):
                return True
        return False
