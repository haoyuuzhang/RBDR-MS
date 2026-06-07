"""Data classes for pure FJSP (Kacem benchmark)."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Operation:
    """A single operation within a job, with direct machine→time mapping."""
    job_id: int
    op_idx: int
    times: Dict[str, float]   # {machine_name: processing_time}

    @property
    def feasible_machines(self) -> List[str]:
        return list(self.times.keys())


@dataclass
class Job:
    """A job composed of an ordered sequence of operations."""
    job_id: int
    arrival_time: float
    operations: List[Operation]


@dataclass
class ScheduleEntry:
    """Records the assignment and timing of one operation."""
    job_id: int
    op_idx: int
    machine: str
    start_time: float
    end_time: float
    fixed: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
