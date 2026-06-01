"""Right-Shift rescheduling strategy.

At t=0 a full MILP plan is built.  When a disruption occurs the existing
schedule is *repaired* by right-shifting all affected operations while
preserving the original machine assignment and operation sequence.
New jobs are appended greedily without re-optimisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from strategies.base import BaseStrategy
from models import Job, Operation, ScheduleEntry, Disruption


class RightShiftStrategy(BaseStrategy):
    """Lazy reactive strategy: one MILP at t=0, then right-shift on disruption.

    Parameters
    ----------
    init_time_limit: Gurobi time limit for the initial t=0 MILP.
    """

    def __init__(self, init_time_limit: float = 60.0):
        super().__init__()
        self.init_time_limit = init_time_limit
        self._plan: Dict[Tuple[int, int, str], ScheduleEntry] = {}
        self._base_dur: Dict[Tuple[int, int, str], float] = {}
        self._plan_valid = False
        self._factors: Dict[str, float] = {}

    # -- BaseStrategy interface -------------------------------------------

    def attach(self, engine):
        super().attach(engine)
        raw_plan = self._build_initial_plan(self.init_time_limit)
        if raw_plan:
            for key, entry in raw_plan.items():
                self._base_dur[key] = entry.duration  # factor=1.0 at t=0
            self._plan = dict(raw_plan)
        self._plan_valid = bool(self._plan)

    def select_operation(
        self, machine_id: str, machine_type: str, unit: str,
        current_time: float, ready_ops: List[Tuple[Job, Operation]],
    ) -> Optional[Tuple[int, int, str, float]]:
        result = self._lookup_plan(machine_id, self._plan) if self._plan_valid else None
        if result is not None:
            return result

        # Fallback for unplanned ops (e.g. new jobs): greedy ECT
        if not ready_ops:
            return None
        best_dur = float('inf')
        best_key: Optional[Tuple[int, int, str, float]] = None
        for job, op in ready_ops:
            p = op.unit_times.get(unit)
            if p is None:
                continue
            dur = p * self._factors.get(machine_id, 1.0)
            if dur < best_dur:
                best_dur = dur
                best_key = (job.job_id, op.op_idx, unit, p)
        return best_key

    def on_job_arrival(self, job: Job):
        pass

    def on_disruption(self, disruption: Disruption):
        self._factors[disruption.machine_id] = disruption.factor
        self._right_shift_repair(disruption)

    # -- schedule repair -------------------------------------------------

    def _right_shift_repair(self, disruption: Disruption):
        """Right-shift all affected entries while preserving sequence."""
        if not self._plan_valid or self.engine is None:
            return

        entries = sorted(self._plan.values(),
                         key=lambda e: (e.start_time, e.op_idx))

        machine_ready: Dict[str, float] = {}
        for mid in self.engine.machines:
            busy = self.engine.machine_busy_until.get(mid, 0)
            machine_ready[mid] = max(busy, self.engine.env.now)

        job_ready: Dict[int, float] = {}
        for (jid, oidx), status in self.engine.op_status.items():
            if status == 'completed':
                for e in self.engine.completed_entries:
                    if e.job_id == jid and e.op_idx == oidx:
                        job_ready[jid] = max(job_ready.get(jid, 0), e.end_time)
                        break
            elif status == 'in_progress':
                for _mid, (cjid, coidx, _cunit, _cstart, _cdur) in \
                        self.engine._current_op.items():
                    if cjid == jid and coidx == oidx:
                        busy = self.engine.machine_busy_until.get(_mid, 0)
                        job_ready[jid] = max(job_ready.get(jid, 0), busy)
                        break

        new_plan: Dict[Tuple[int, int, str], ScheduleEntry] = {}
        for e in entries:
            mid = e.machine
            jid, oidx = e.job_id, e.op_idx
            key = (jid, oidx, mid)

            status = self.engine.op_status.get((jid, oidx), 'pending')
            if status in ('completed', 'in_progress'):
                new_plan[key] = e
                continue

            base = self._base_dur.get(key, e.duration)
            dur = base * self._factors.get(mid, 1.0)
            start = max(machine_ready.get(mid, 0), job_ready.get(jid, 0))

            new_plan[key] = ScheduleEntry(
                jid, oidx, mid, e.service_unit, start, start + dur)

            machine_ready[mid] = start + dur
            job_ready[jid] = start + dur

        self._plan = new_plan
