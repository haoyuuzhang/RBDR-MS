"""Initial Schedule strategy.

Executes the common t=0 MILP schedule to completion.  New jobs and
disruptions are ignored — the schedule never adapts. Serves as the
baseline reference for measuring schedule deviation of other strategies.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from strategies.base import BaseStrategy
from models import Job, Operation


class InitialScheduleStrategy(BaseStrategy):
    """Run the initial t=0 plan to completion — never replan."""

    def __init__(self, time_limit: float = 60.0):
        super().__init__()
        self.time_limit = time_limit

    def attach(self, engine):
        super().attach(engine)
        self._initial_plan = self._build_initial_plan(self.time_limit)

    def select_operation(
        self, machine_id: str, machine_type: str, unit: str,
        current_time: float, ready_ops: List[Tuple[Job, Operation]],
    ) -> Optional[Tuple[int, int, str, float]]:
        return self._lookup_plan(machine_id, self._initial_plan)

    def on_job_arrival(self, job: Job):
        """Mark unplanned jobs as completed so simulation can finish."""
        planned_keys = {(jid, oidx) for (jid, oidx, _mid) in self._initial_plan}
        for op in job.operations:
            if (job.job_id, op.op_idx) not in planned_keys:
                self.engine.op_status[(job.job_id, op.op_idx)] = 'completed'
