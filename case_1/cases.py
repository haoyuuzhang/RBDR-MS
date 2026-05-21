"""Job/operation definitions (case data) for Hierarchical FJSP.

Machine sequences:
  J1, J2, J3 : Op1->M1, Op2->M2, Op3->M1   (sequence 1-2-1)
  J4         : Op1->M3, Op2->M1, Op3->M2   (sequence 3-1-2)
"""

from typing import List
from models import Job, Operation


def build_jobs() -> List[Job]:
    """Return the default set of jobs (J1-J4)."""
    return [
        Job(job_id=1, release_date=0, due_date=40, alpha=0, beta=1,
            arrival_time=0, operations=[
                Operation(1, 0, 'M1', {'U1': 10, 'U2': 14}),
                Operation(1, 1, 'M2', {'U1': 12, 'U2': 16}),
                Operation(1, 2, 'M1', {'U1': 14, 'U2': 16}),
            ]),
        Job(job_id=2, release_date=0, due_date=72, alpha=0, beta=4,
            arrival_time=0, operations=[
                Operation(2, 0, 'M1', {'U1': 18, 'U2': 14}),
                Operation(2, 1, 'M2', {'U1': 30, 'U2': 22}),
                Operation(2, 2, 'M1', {'U1': 15, 'U2': 20}),
            ]),
        Job(job_id=3, release_date=0, due_date=72, alpha=1, beta=3,
            arrival_time=0, operations=[
                Operation(3, 0, 'M1', {'U1': 24, 'U2': 18}),
                Operation(3, 1, 'M2', {'U1': 24, 'U2': 28}),
                Operation(3, 2, 'M1', {'U1': 12, 'U2': 16}),
            ]),
        Job(job_id=4, release_date=48, due_date=96, alpha=1, beta=12,
            arrival_time=24, operations=[
                Operation(4, 0, 'M3', {'U1': 12}),
                Operation(4, 1, 'M1', {'U1': 20, 'U2': 12}),
                Operation(4, 2, 'M2', {'U1': 16, 'U2': 16}),
            ]),
    ]
