"""Abstract base class for scheduling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from models import Job, Operation, ScheduleEntry


class BaseStrategy(ABC):
    """Interface that every scheduling strategy must implement.

    Lifecycle
    ---------
    1. ``attach(engine)`` — called once before simulation starts.
    2. ``select_operation(...)`` — called whenever a machine is free and
       ready work exists.  Returns the next operation to process, or *None*
       to leave the machine idle.
    3. ``on_job_arrival(job)`` — called when a new job enters the system.
    4. ``on_disruption(disruption)`` — called when a machine disruption occurs.
    5. ``on_operation_complete(entry)`` — called after each op finishes.

    All strategies share a common t=0 MILP initial plan (via
    ``_build_initial_plan()``).  They diverge only in how they respond
    to disruptions after t=0.
    """

    def __init__(self):
        self.engine = None          # set via attach()
        self._initial_plan: Dict[Tuple[int, int, str], any] = {}
        # (job_id, op_idx, machine_id) -> case_1 ScheduleEntry

    def attach(self, engine):
        self.engine = engine

    # ── shared MILP initial-plan builder ──────────────────────────────────

    def _build_initial_plan(self, time_limit: float = 60.0) -> Dict:
        """Generate the t=0 MILP optimal schedule for currently visible jobs."""
        from milp_scheduler import schedule_milp

        if self.engine is None:
            return {}

        if not self.engine.visible_jobs:
            return {}

        schedule = schedule_milp(
            list(self.engine.visible_jobs), [], 0.0, time_limit)
        if schedule is None:
            print("[_build_initial_plan] MILP returned None (infeasible/failed)")
            return {}

        # Diagnostic: check every operation got scheduled
        expected_ops = set()
        for job in self.engine.visible_jobs:
            for op in job.operations:
                expected_ops.add((job.job_id, op.op_idx))
        scheduled_ops = {(e.job_id, e.op_idx) for e in schedule}
        missing = expected_ops - scheduled_ops
        if missing:
            print(f"[_build_initial_plan] WARNING: {len(missing)} ops missing "
                  f"from schedule: {sorted(missing)}")

        plan: Dict[Tuple[int, int, str], any] = {}
        for e in schedule:
            plan[(e.job_id, e.op_idx, e.machine)] = e
        return plan

    # ── plan lookup helper ────────────────────────────────────────────────

    def _lookup_plan(
        self, machine_id: str, plan: Dict
    ) -> Optional[Tuple[int, int, str, float]]:
        """Return the next planned operation for *machine_id* that is ready.

        Returns ``(job_id, op_idx, unit_name, duration)`` or *None*.
        """
        for (jid, oidx, mid), entry in plan.items():
            if mid != machine_id:
                continue
            if self.engine.op_status.get((jid, oidx)) != 'ready':
                continue
            job = self.engine._find_job(jid)
            if job is None:
                continue
            op = job.operations[oidx]
            duration = op.unit_times.get(entry.service_unit, 0)
            return (jid, oidx, entry.service_unit, duration)
        return None

    # ── abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def select_operation(
        self,
        machine_id: str,
        machine_type: str,
        unit: str,
        current_time: float,
        ready_ops: List[Tuple[Job, Operation]],
    ) -> Optional[Tuple[int, int, str, float]]:
        """Return (job_id, op_idx, unit_name, duration) or None."""
        ...

    def on_job_arrival(self, job: Job):
        """Called when *job* arrives (may be before its release_date)."""

    def on_disruption(self, disruption):
        """Called when a machine disruption is applied."""

    def on_operation_complete(self, entry: ScheduleEntry):
        """Called after an operation finishes."""
