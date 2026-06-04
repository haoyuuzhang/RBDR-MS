"""Pure dispatching-rule strategy — no MILP at any level.

Used as a lower-bound baseline to show that rules alone are insufficient:
they are fast but produce poor-quality schedules compared to hierarchical MILP.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from strategies.base import BaseStrategy
from models import Job, Operation, ScheduleEntry, Disruption

_RULES = ['FIFO', 'EDD', 'SPT', 'WINQ', 'PA']


class PureRuleStrategy(BaseStrategy):
    """Fully reactive strategy driven entirely by a single dispatching rule.

    No initial MILP plan.  On every ``select_operation`` call the rule
    scores the current ready-queue and picks greedily.

    Parameters
    ----------
    rule: 'FIFO' | 'EDD' | 'SPT' | 'WINQ' | 'PA'
    """

    def __init__(self, rule: str = 'FIFO'):
        super().__init__()
        if rule not in _RULES:
            raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")
        self.rule = rule

    # -- BaseStrategy interface -------------------------------------------

    def attach(self, engine):
        super().attach(engine)
        # No initial plan — pure rules decide on the fly.

    def select_operation(
        self,
        machine_id: str,
        machine_type: str,
        unit: str,
        current_time: float,
        ready_ops: List[Tuple[Job, Operation]],
    ) -> Optional[Tuple[int, int, str, float]]:
        if not ready_ops:
            return None

        best: Optional[Tuple[Job, Operation]] = None
        best_score: float | tuple = None  # sentinel
        best_duration = 0.0

        for job, op in ready_ops:
            p = op.unit_times.get(unit)
            if p is None:
                continue

            if self.rule == 'FIFO':
                score: float | tuple = (job.arrival_time, job.job_id)
            elif self.rule == 'EDD':
                score = (job.due_date, job.job_id)
            elif self.rule == 'SPT':
                score = (p, job.job_id)
            elif self.rule == 'WINQ':
                score = (self._winq_score(job, op), job.job_id)
            elif self.rule == 'PA':
                score = (self._pa_score(job, op, p, current_time), job.job_id)

            if best_score is None or score < best_score:
                best_score = score
                best = (job, op)
                best_duration = p

        if best is None:
            return None

        job, op = best
        return (job.job_id, op.op_idx, unit, best_duration)

    def _winq_score(self, job: Job, op: Operation) -> int:
        """Count ready operations queued for the next operation's machine type.

        A lower score means less downstream congestion — the job's next step
        has a shorter queue, reducing the risk of creating a bottleneck.
        Returns 0 when this is the job's last operation (no downstream wait).
        """
        if op.op_idx + 1 >= len(job.operations):
            return 0  # last op — best possible score
        next_machine_type = job.operations[op.op_idx + 1].machine_type
        if self.engine is None:
            return 0
        return len(self.engine._get_ready_ops(machine_type=next_machine_type))

    def _pa_score(self, job: Job, op: Operation, p: float,
                  current_time: float) -> float:
        """Penalty-Aware score: estimated earliness-tardiness penalty.

        Projects the job's total completion time by summing *p* plus the
        average processing time of all remaining later operations (ignoring
        queue waiting).  From the projected completion, computes the
        weighted α·E + β·T penalty — lower is better.

        Jobs with higher α/β naturally get higher priority when they are
        at risk of violating their due date.
        """
        # Sum average unit time for each remaining later operation
        later_work = 0.0
        for later_op in job.operations[op.op_idx + 1:]:
            if later_op.unit_times:
                avg = sum(later_op.unit_times.values()) / len(later_op.unit_times)
            else:
                avg = 0.0
            later_work += avg

        Cj_est = current_time + p + later_work
        E = max(0.0, job.due_date - Cj_est)
        T = max(0.0, Cj_est - job.due_date)
        return job.alpha * E + job.beta * T

    def on_job_arrival(self, job: Job):
        pass  # No pre-computation needed

    def on_disruption(self, disruption: Disruption):
        pass  # Rules adapt implicitly on next decision

    def on_operation_complete(self, entry: ScheduleEntry):
        pass
