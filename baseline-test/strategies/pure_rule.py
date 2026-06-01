"""Pure dispatching-rule strategy — no MILP at any level.

Used as a lower-bound baseline to show that rules alone are insufficient:
they are fast but produce poor-quality schedules compared to hierarchical MILP.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from strategies.base import BaseStrategy
from models import Job, Operation, ScheduleEntry, Disruption

_RULES = ['FIFO', 'EDD', 'ATC']


class PureRuleStrategy(BaseStrategy):
    """Fully reactive strategy driven entirely by a single dispatching rule.

    No initial MILP plan.  On every ``select_operation`` call the rule
    scores the current ready-queue and picks greedily.

    Parameters
    ----------
    rule: 'FIFO' | 'EDD' | 'ATC'
    atc_k: Look-ahead scaling parameter for ATC (default 2.0).
    """

    def __init__(self, rule: str = 'FIFO', atc_k: float = 2.0):
        super().__init__()
        if rule not in _RULES:
            raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")
        self.rule = rule
        self.atc_k = atc_k

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

        # Pre-compute average processing time for ATC
        avg_p = 1.0
        if self.rule == 'ATC':
            total_p = sum(
                op.unit_times.get(unit, sum(op.unit_times.values()) / max(len(op.unit_times), 1))
                for _, op in ready_ops
            )
            avg_p = max(total_p / len(ready_ops), 0.1)

        for job, op in ready_ops:
            p = op.unit_times.get(unit)
            if p is None:
                continue

            if self.rule == 'FIFO':
                score: float | tuple = (job.arrival_time, job.job_id)
            elif self.rule == 'EDD':
                score = (job.due_date, job.job_id)
            elif self.rule == 'ATC':
                slack = max(job.due_date - current_time - p, 0.0)
                score = (job.beta / max(p, 0.1)) * math.exp(
                    -slack / (self.atc_k * avg_p))

            if best_score is None or \
               (self.rule == 'ATC' and score > best_score) or \
               (self.rule != 'ATC' and score < best_score):
                best_score = score
                best = (job, op)
                best_duration = p

        if best is None:
            return None

        job, op = best
        return (job.job_id, op.op_idx, unit, best_duration)

    def on_job_arrival(self, job: Job):
        pass  # No pre-computation needed

    def on_disruption(self, disruption: Disruption):
        pass  # Rules adapt implicitly on next decision

    def on_operation_complete(self, entry: ScheduleEntry):
        pass
